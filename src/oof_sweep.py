"""
oof_sweep.py
------------
Tunes thresholds on OUT-OF-FOLD predictions over train+val (many more
frauds to tune on than the validation set alone), then evaluates once
on the untouched test set.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config
from src.imbalance import apply_strategy
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.preprocess import run_preprocessing_pipeline
from src.uncertainty import get_ensemble_disagreement
from src.utils import get_logger

logger = get_logger(__name__)

FN_COSTS = [100, 200, 500, 1000, 2000, 5000]
N_FOLDS = 5
LOW_GRID = np.concatenate(([0.0005, 0.001, 0.002, 0.005, 0.01], np.arange(0.02, 0.40, 0.02)))
HIGH_GRID = np.arange(0.60, 0.995, 0.02)
LOW_RISK_CUTOFF = 0.15


def triage_cost(p, u, y, low, high, unc, fn_cost):
    """Same logic as assign_risk_tier + compute_total_cost, vectorized."""
    unc_flag = u >= unc
    approve = ~unc_flag & (p < low)
    block = ~unc_flag & (p >= high)
    review = ~(approve | block)
    cost = (fn_cost * np.sum(approve & (y == 1))
            + config.COST_FALSE_POSITIVE * np.sum(block & (y == 0))
            + config.COST_ESCALATION * np.sum(review))
    return cost, approve, block, review


def search(p, u, y, unc_candidates, fn_cost):
    """Ties keep the earlier candidate, so put np.inf (uncertainty OFF) first."""
    best = None
    for unc in unc_candidates:
        for low in LOW_GRID:
            for high in HIGH_GRID:
                if low >= high:
                    continue
                cost = triage_cost(p, u, y, low, high, unc, fn_cost)[0]
                if best is None or cost < best["cost"]:
                    best = {"cost": cost, "low": low, "high": high, "unc": unc}
    return best


def out_of_fold(X, y):
    p_oof, u_oof = np.zeros(len(X)), np.zeros(len(X))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X.iloc[tr], y.iloc[tr])
        xgb = train_xgboost(X_res, y_res)
        lgbm = train_lightgbm(X_res, y_res)
        cat = train_catboost(X_res, y_res)
        p_oof[te], u_oof[te] = get_ensemble_disagreement(xgb, lgbm, cat, X.iloc[te])
        logger.info(f"fold {fold}/{N_FOLDS} done")
    return p_oof, u_oof


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    p_oof, u_oof = out_of_fold(X_all, y_all)
    y_oof = y_all.values

    # final models on all of train+val, scored on the untouched test set
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_all, y_all)
    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)
    p_test, u_test = get_ensemble_disagreement(xgb, lgbm, cat, X_test)
    y_test_arr = y_test.values

    # uncertainty quality on far more frauds than validation had
    mask = p_oof < LOW_RISK_CUTOFF
    n_pos = int(y_oof[mask].sum())
    auc = roc_auc_score(y_oof[mask], u_oof[mask])
    print(f"\nOOF uncertainty quality: {n_pos} frauds among {int(mask.sum())} "
          f"low-risk cases -> AUROC {auc:.3f}")

    quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
    cands = np.concatenate(([np.inf], np.unique(np.quantile(u_oof, quantiles))))

    print(f"\n{'FN cost':>8}{'B test':>10}{'D test':>10}{'D unc cutoff':>14}{'D review%':>11}"
          f"{'missed B':>10}{'missed D':>10}{'OOF D vs B':>12}{'test D vs B':>13}")

    for fn in FN_COSTS:
        bB = search(p_oof, u_oof, y_oof, [np.inf], fn)
        bD = search(p_oof, u_oof, y_oof, cands, fn)

        res = {}
        for tag, b in (("B", bB), ("D", bD)):
            cost, approve, block, review = triage_cost(
                p_test, u_test, y_test_arr, b["low"], b["high"], b["unc"], fn)
            res[tag] = (cost, int(np.sum(approve & (y_test_arr == 1))), 100 * review.mean())

        oof_delta = 100 * (bD["cost"] - bB["cost"]) / bB["cost"]
        test_delta = 100 * (res["D"][0] - res["B"][0]) / res["B"][0]
        print(f"{fn:>8}{res['B'][0]:>10.0f}{res['D'][0]:>10.0f}{bD['unc']:>14.5f}"
              f"{res['D'][2]:>10.2f}%{res['B'][1]:>10}{res['D'][1]:>10}"
              f"{oof_delta:>11.1f}%{test_delta:>12.1f}%")