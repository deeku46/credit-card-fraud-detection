"""
multi_seed.py
-------------
Confirmatory run: the protocol is FROZEN (same grids, candidates and cost
model as final_eval.py) and repeated on fresh random splits.
Per seed: new train/val/test split, OOF tuning on train+val, final models
on train+val, evaluation on that seed's test set.
Results are saved to outputs/multiseed_results.json after every seed;
finished seeds are skipped on rerun, so an interruption loses nothing.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src import config
from src.final_eval import BUDGETS, QUANTILES, per_transaction_cost, search_budget
from src.imbalance import apply_strategy
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.oof_sweep import FN_COSTS, out_of_fold, search, triage_cost
from src.preprocess import run_preprocessing_pipeline
from src.uncertainty import get_ensemble_disagreement

SEEDS = [11, 22, 33, 44, 55]   # fresh seeds; 42 was already inspected
RESULTS = os.path.join(config.OUTPUTS_DIR, "multiseed_results.json")


def set_seed_everywhere(seed):
    config.RANDOM_STATE = seed
    config.XGB_PARAMS["random_state"] = seed
    config.LGBM_PARAMS["random_state"] = seed
    config.CATBOOST_PARAMS["random_seed"] = seed


def run_seed(seed):
    set_seed_everywhere(seed)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_preprocessing_pipeline()
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    p_oof, u_oof = out_of_fold(X_all, y_all)

    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_all, y_all)
    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)
    p_test, u_test = get_ensemble_disagreement(xgb, lgbm, cat, X_test)

    y_oof, y_te = y_all.values, y_test.values
    cands = np.concatenate(([np.inf], np.unique(np.quantile(u_oof, QUANTILES))))
    pred = (p_test >= 0.5).astype(int)

    out = {
        "seed": seed,
        "n_test_frauds": int(y_te.sum()),
        "metrics": {
            "roc_auc": float(roc_auc_score(y_te, p_test)),
            "pr_auc": float(average_precision_score(y_te, p_test)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
        },
        "regimes": {},
        "budgets": {},
    }

    for fn in FN_COSTS:
        bB = search(p_oof, u_oof, y_oof, [np.inf], fn)
        bD = search(p_oof, u_oof, y_oof, cands, fn)
        row = {}
        for tag, b in (("B", bB), ("D", bD)):
            _, ap, bl, rv = triage_cost(p_test, u_test, y_te, b["low"], b["high"], b["unc"], fn)
            row[tag] = {
                "cost": float(per_transaction_cost(ap, bl, rv, y_te, fn).sum()),
                "missed": int(np.sum(ap & (y_te == 1))),
                "review": float(100 * rv.mean()),
            }
        out["regimes"][str(fn)] = row

    for budget in BUDGETS:
        row = {}
        for tag, c in (("B", [np.inf]), ("D", cands)):
            b = search_budget(p_oof, u_oof, y_oof, c, budget)
            if b is None:
                row[tag] = None
                continue
            _, ap, bl, rv = triage_cost(p_test, u_test, y_te, b["low"], b["high"], b["unc"], 1)
            row[tag] = {"missed": int(np.sum(ap & (y_te == 1))), "review": float(100 * rv.mean())}
        out["budgets"][str(budget)] = row
    return out


def aggregate(results):
    print(f"\n=== {len(results)} seeds: {[r['seed'] for r in results]} ===")

    print("\nDetector on test, mean +/- std over seeds:")
    for k in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        v = np.array([r["metrics"][k] for r in results])
        print(f"  {k:<10}{v.mean():.4f} +/- {v.std(ddof=1):.4f}")

    print(f"\n{'FN cost':>8}{'B cost':>10}{'D cost':>10}{'mean D-B':>11}"
          f"{'D better':>10}{'D worse':>9}{'missed B':>10}{'missed D':>10}{'review% D':>11}")
    for fn in FN_COSTS:
        rows = [r["regimes"][str(fn)] for r in results]
        b = np.array([x["B"]["cost"] for x in rows])
        d = np.array([x["D"]["cost"] for x in rows])
        diff = d - b
        print(f"{fn:>8}{b.mean():>10.0f}{d.mean():>10.0f}{diff.mean():>11.0f}"
              f"{int((diff < 0).sum()):>10}{int((diff > 0).sum()):>9}"
              f"{np.mean([x['B']['missed'] for x in rows]):>10.1f}"
              f"{np.mean([x['D']['missed'] for x in rows]):>10.1f}"
              f"{np.mean([x['D']['review'] for x in rows]):>11.2f}")

    print(f"\n{'Budget':>8}{'review% B':>11}{'review% D':>11}{'missed B':>10}{'missed D':>10}{'D better':>10}{'D worse':>9}")
    for budget in BUDGETS:
        rows = [r["budgets"][str(budget)] for r in results]
        rows = [x for x in rows if x["B"] and x["D"]]
        if not rows:
            print(f"{budget * 100:>7.2f}%   infeasible under the threshold grid")
            continue
        mb = np.array([x["B"]["missed"] for x in rows])
        md = np.array([x["D"]["missed"] for x in rows])
        print(f"{budget * 100:>7.2f}%{np.mean([x['B']['review'] for x in rows]):>11.2f}"
              f"{np.mean([x['D']['review'] for x in rows]):>11.2f}{mb.mean():>10.1f}{md.mean():>10.1f}"
              f"{int((md < mb).sum()):>10}{int((md > mb).sum()):>9}")


if __name__ == "__main__":
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    results = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = json.load(f)
    done = {r["seed"] for r in results}

    for seed in SEEDS:
        if seed in done:
            continue
        print(f"\n##### seed {seed} #####")
        results.append(run_seed(seed))
        with open(RESULTS, "w") as f:
            json.dump(results, f, indent=2)

    aggregate(results)