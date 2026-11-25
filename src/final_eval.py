"""
final_eval.py
-------------
Every number the paper reports, from ONE pipeline:
  - detector trained on train+val, scored on the untouched test set
  - thresholds tuned on out-of-fold (OOF) predictions only
  - bootstrap 95% CIs on the test set
Predictions are cached in outputs/predictions.npz, so reruns skip training.
Delete that file if you change the models or the data split.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src import config
from src.imbalance import apply_strategy
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.oof_sweep import (FN_COSTS, HIGH_GRID, LOW_GRID, LOW_RISK_CUTOFF,
                           out_of_fold, search, triage_cost)
from src.preprocess import run_preprocessing_pipeline
from src.uncertainty import get_ensemble_disagreement

CACHE = os.path.join(config.OUTPUTS_DIR, "predictions.npz")
N_BOOT = 1000        # cheap bootstraps (cost differences)
N_BOOT_SLOW = 300    # metric bootstraps (AUC etc.)
BUDGETS = [0.0025, 0.005, 0.01, 0.02, 0.05]
QUANTILES = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
rng = np.random.default_rng(config.RANDOM_STATE)


def get_predictions():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["p_oof"], d["u_oof"], d["y_oof"], d["p_test"], d["u_test"], d["y_test"]

    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_preprocessing_pipeline()
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    p_oof, u_oof = out_of_fold(X_all, y_all)

    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_all, y_all)
    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)
    p_test, u_test = get_ensemble_disagreement(xgb, lgbm, cat, X_test)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    np.savez(CACHE, p_oof=p_oof, u_oof=u_oof, y_oof=y_all.values,
             p_test=p_test, u_test=u_test, y_test=y_test.values)
    return p_oof, u_oof, y_all.values, p_test, u_test, y_test.values


def metric_cis(p, y):
    pred = (p >= 0.5).astype(int)
    point = {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
    }
    boots = {k: [] for k in point}
    n = len(y)
    for _ in range(N_BOOT_SLOW):
        i = rng.integers(0, n, n)
        yi, pi, qi = y[i], p[i], pred[i]
        if yi.sum() == 0:
            continue
        boots["ROC-AUC"].append(roc_auc_score(yi, pi))
        boots["PR-AUC"].append(average_precision_score(yi, pi))
        boots["Precision"].append(precision_score(yi, qi, zero_division=0))
        boots["Recall"].append(recall_score(yi, qi))
        boots["F1"].append(f1_score(yi, qi, zero_division=0))
    for k, v in point.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        print(f"  {k:<10}{v:>8.4f}   95% CI [{lo:.4f}, {hi:.4f}]")


def uncertainty_quality(p, u, y):
    mask = p < LOW_RISK_CUTOFF
    yy, pp, uu = y[mask], p[mask], u[mask]
    scores = {"p (probability)": pp, "u (disagreement)": uu, "u/p (relative)": uu / (pp + 1e-6)}
    print(f"  {int(yy.sum())} frauds among {int(mask.sum())} low-risk cases")
    n = len(yy)
    idxs = [rng.integers(0, n, n) for _ in range(200)]
    for name, s in scores.items():
        auc = roc_auc_score(yy, s)
        boots = [roc_auc_score(yy[i], s[i]) for i in idxs if yy[i].sum() > 0]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  AUROC of {name:<18}{auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")


def per_transaction_cost(approve, block, review, y, fn):
    return (fn * (approve & (y == 1))
            + config.COST_FALSE_POSITIVE * (block & (y == 0))
            + config.COST_ESCALATION * review).astype(float)


def cost_regimes(p_oof, u_oof, y_oof, p_test, u_test, y_test, cands):
    print(f"\n{'FN cost':>8}{'B cost':>10}{'D cost':>10}{'D vs B':>9}{'95% CI':>20}"
          f"{'D review%':>11}{'missed B/D':>12}{'D unc cutoff':>14}")
    for fn in FN_COSTS:
        bB = search(p_oof, u_oof, y_oof, [np.inf], fn)
        bD = search(p_oof, u_oof, y_oof, cands, fn)
        print(f"    thresholds -> B: low={bB['low']:.4f} high={bB['high']:.2f} | D: low={bD['low']:.4f} high={bD['high']:.2f}")        
        costs, stats = {}, {}
        for tag, b in (("B", bB), ("D", bD)):
            _, ap, bl, rv = triage_cost(p_test, u_test, y_test, b["low"], b["high"], b["unc"], fn)
            costs[tag] = per_transaction_cost(ap, bl, rv, y_test, fn)
            stats[tag] = (int(np.sum(ap & (y_test == 1))), 100 * rv.mean())

        d = costs["D"] - costs["B"]
        n = len(d)
        boots = np.array([d[rng.integers(0, n, n)].sum() for _ in range(N_BOOT)])
        base = costs["B"].sum()
        lo, hi = 100 * np.percentile(boots, [2.5, 97.5]) / base
        ci_str = f"[{lo:.1f}%, {hi:.1f}%]"
        miss_str = f"{stats['B'][0]}/{stats['D'][0]}"
        flag = "  *" if (hi < 0 or lo > 0) else ""
        print(f"{fn:>8}{base:>10.0f}{costs['D'].sum():>10.0f}{100 * d.sum() / base:>8.1f}%"
              f"{ci_str:>20}{stats['D'][1]:>10.2f}%{miss_str:>12}{bD['unc']:>14.5f}{flag}")


def search_budget(p, u, y, unc_candidates, budget):
    """Fewest missed frauds (then fewest false blocks) with review rate <= budget."""
    best = None
    for unc in unc_candidates:
        for low in LOW_GRID:
            for high in HIGH_GRID:
                if low >= high:
                    continue
                _, ap, bl, rv = triage_cost(p, u, y, low, high, unc, 1)
                if rv.mean() > budget:
                    continue
                key = (int(np.sum(ap & (y == 1))), int(np.sum(bl & (y == 0))))
                if best is None or key < best["key"]:
                    best = {"key": key, "low": low, "high": high, "unc": unc}
    return best


def budget_analysis(p_oof, u_oof, y_oof, p_test, u_test, y_test, cands):
    print(f"\n{'Budget':>8}{'method':>8}{'review%':>10}{'missed':>9}{'false blocks':>14}{'unc cutoff':>12}")
    for budget in BUDGETS:
        for tag, c in (("B", [np.inf]), ("D", cands)):
            b = search_budget(p_oof, u_oof, y_oof, c, budget)
            if b is None:
                print(f"{budget * 100:>7.2f}%{tag:>8}   infeasible under the threshold grid")
                continue
            _, ap, bl, rv = triage_cost(p_test, u_test, y_test, b["low"], b["high"], b["unc"], 1)
            print(f"{budget * 100:>7.2f}%{tag:>8}{100 * rv.mean():>10.2f}"
                  f"{int(np.sum(ap & (y_test == 1))):>9}{int(np.sum(bl & (y_test == 0))):>14}"
                  f"{b['unc']:>12.5f}")


if __name__ == "__main__":
    p_oof, u_oof, y_oof, p_test, u_test, y_test = get_predictions()
    cands = np.concatenate(([np.inf], np.unique(np.quantile(u_oof, QUANTILES))))

    print(f"\n[1] Detector quality on TEST ({int(y_test.sum())} frauds, "
          f"no-skill PR-AUC = {y_test.mean():.5f}), threshold 0.5")
    metric_cis(p_test, y_test)

    print("\n[2] Does disagreement flag the frauds the model would auto-approve? (OOF)")
    uncertainty_quality(p_oof, u_oof, y_oof)

    print("\n[3] Cost regimes. B = probability only, D = probability + uncertainty (* = CI excludes 0)")
    cost_regimes(p_oof, u_oof, y_oof, p_test, u_test, y_test, cands)

    print("\n[4] Review-budget comparison (tuned on OOF, reported on test)")
    budget_analysis(p_oof, u_oof, y_oof, p_test, u_test, y_test, cands)