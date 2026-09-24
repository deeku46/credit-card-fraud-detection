"""
ablation.py
-----------
Isolates the contribution of the two novelty components (uncertainty-based
escalation, cost-optimized thresholds) by testing four conditions:

  A. Probability only,  fixed thresholds   (no uncertainty, no optimization)
  B. Probability only,  optimized thresholds (no uncertainty)
  C. Probability + uncertainty, fixed thresholds
  D. Probability + uncertainty, optimized thresholds  <- full proposed method

All four are TUNED on the validation set (where applicable) and REPORTED
on the held-out test set, which was never touched during model selection,
strategy selection, or threshold search. This is the unbiased, paper-ready
number.
"""

import numpy as np

from src import config
from src.imbalance import apply_strategy
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.preprocess import run_preprocessing_pipeline
from src.risk_triage import apply_triage, compute_total_cost, optimize_thresholds, summarize_decisions
from src.uncertainty import get_ensemble_disagreement
from src.utils import get_logger

logger = get_logger(__name__)

FIXED_LOW, FIXED_HIGH = 0.15, 0.85
UNCERTAINTY_ON = 0.05
UNCERTAINTY_OFF = float("inf")  # never escalate on disagreement alone


def frauds_auto_approved(decisions: list, y_true: np.ndarray) -> int:
    """Count of actual fraud cases that were auto-approved -- the most
    dangerous failure mode: fraud that slipped through with no human
    ever seeing it.
    """
    y_true = np.asarray(y_true)
    return sum(1 for d, y in zip(decisions, y_true) if d == "Approve" and y == 1)


def run_condition(name, mean_proba_val, uncertainty_val, y_val,
                   mean_proba_test, uncertainty_test, y_test,
                   use_uncertainty: bool, optimize: bool):
    unc_threshold = UNCERTAINTY_ON if use_uncertainty else UNCERTAINTY_OFF

    if optimize:
        best, _ = optimize_thresholds(mean_proba_val, uncertainty_val, y_val,
                                       uncertainty_threshold=unc_threshold)
        low, high = best["low"], best["high"]
    else:
        low, high = FIXED_LOW, FIXED_HIGH

    # Apply the (validation-chosen) thresholds to the TEST set -- unbiased.
    decisions = apply_triage(mean_proba_test, uncertainty_test,
                              low_threshold=low, high_threshold=high,
                              uncertainty_threshold=unc_threshold)
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test
    cost = compute_total_cost(decisions, y_test_arr)
    summary = summarize_decisions(decisions)
    missed = frauds_auto_approved(decisions, y_test_arr)

    logger.info(
        f"[{name}] low={low:.2f} high={high:.2f}  test_cost={cost:.1f}  "
        f"review%={summary['percentages']['Review']:.2f}  "
        f"fraud_auto_approved={missed}"
    )
    return {
        "condition": name, "low": low, "high": high, "test_cost": cost,
        "counts": summary["counts"], "percentages": summary["percentages"],
        "fraud_auto_approved": missed,
    }


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)

    mean_proba_val, uncertainty_val = get_ensemble_disagreement(xgb, lgbm, cat, X_val)
    mean_proba_test, uncertainty_test = get_ensemble_disagreement(xgb, lgbm, cat, X_test)

    conditions = [
        ("A: probability-only, fixed",        False, False),
        ("B: probability-only, optimized",    False, True),
        ("C: +uncertainty, fixed",             True,  False),
        ("D: +uncertainty, optimized (ours)",  True,  True),
    ]

    results = []
    for name, use_unc, opt in conditions:
        results.append(run_condition(
            name, mean_proba_val, uncertainty_val, y_val,
            mean_proba_test, uncertainty_test, y_test,
            use_uncertainty=use_unc, optimize=opt,
        ))

    print(f"\n{'='*95}")
    print(f"{'Condition':<38}{'Test Cost':>12}{'Review %':>12}{'Fraud Auto-Approved':>22}")
    print("=" * 95)
    for r in results:
        print(f"{r['condition']:<38}{r['test_cost']:>12.1f}{r['percentages']['Review']:>11.2f}%{r['fraud_auto_approved']:>22}")
    print("=" * 95)
    print("\nD should beat A/B/C on both cost AND fraud-auto-approved to support the novelty claim.")