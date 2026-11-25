"""
cost_sweep.py
-------------
1) Does the uncertainty signal carry ANY information about the frauds the
   model misses? (AUROC inside the auto-approve region; 0.5 = no signal)
2) At what missed-fraud cost does the validation search start choosing
   uncertainty-based escalation?
"""

from sklearn.metrics import roc_auc_score

from src import config
from src.ablation import FIXED_LOW, build_uncertainty_candidates
from src.imbalance import apply_strategy
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.preprocess import run_preprocessing_pipeline
from src.risk_triage import apply_triage, compute_total_cost, optimize_thresholds
from src.uncertainty import get_ensemble_disagreement

FN_COSTS = [100, 200, 500, 1000, 2000, 5000]

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)

    p_val, u_val = get_ensemble_disagreement(xgb, lgbm, cat, X_val)
    p_test, u_test = get_ensemble_disagreement(xgb, lgbm, cat, X_test)
    y_val_arr, y_test_arr = y_val.values, y_test.values

    # ---- 1) uncertainty quality inside the auto-approve region ----
    print("\nUncertainty quality (AUROC, 0.5 = no signal):")
    for name, p, u, y in [("validation", p_val, u_val, y_val_arr),
                          ("test", p_test, u_test, y_test_arr)]:
        mask = p < FIXED_LOW
        n_pos = int(y[mask].sum())
        if n_pos:
            auc = roc_auc_score(y[mask], u[mask])
            print(f"  {name}: {n_pos} frauds among {int(mask.sum())} low-risk cases -> AUROC {auc:.3f}")
        else:
            print(f"  {name}: no frauds in the low-risk region")

    # ---- 2) cost sweep ----
    candidates = build_uncertainty_candidates(u_val)
    print(f"\n{'FN cost':>8}{'B cost':>10}{'D cost':>10}{'D unc cutoff':>15}{'D review%':>11}{'missed B':>10}{'missed D':>10}")

    for fn in FN_COSTS:
        config.COST_FALSE_NEGATIVE = float(fn)  # compute_total_cost reads this at call time

        out = {}
        for tag, kwargs in [("B", {"uncertainty_threshold": float("inf")}),
                            ("D", {"uncertainty_candidates": candidates})]:
            best, _ = optimize_thresholds(p_val, u_val, y_val_arr, **kwargs)
            dec = apply_triage(p_test, u_test, low_threshold=best["low"],
                               high_threshold=best["high"],
                               uncertainty_threshold=best["uncertainty"])
            out[tag] = {
                "cost": compute_total_cost(dec, y_test_arr),
                "unc": best["uncertainty"],
                "review": 100 * dec.count("Review") / len(dec),
                "missed": sum(1 for d, y in zip(dec, y_test_arr) if d == "Approve" and y == 1),
            }

        print(f"{fn:>8}{out['B']['cost']:>10.0f}{out['D']['cost']:>10.0f}"
              f"{out['D']['unc']:>15.5f}{out['D']['review']:>10.2f}%"
              f"{out['B']['missed']:>10}{out['D']['missed']:>10}")