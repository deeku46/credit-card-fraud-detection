"""
ablation.py
-----------
Ablation study for the uncertainty-aware, cost-calibrated fraud
triage framework.

Conditions:

  A. Probability only, fixed thresholds
  B. Probability only, optimized probability thresholds
  C. Probability + uncertainty, fixed thresholds
  D. Probability + uncertainty, jointly optimized thresholds

Threshold optimization is performed ONLY on the validation set.
The selected thresholds are then frozen and evaluated on the untouched
test set.
"""

import numpy as np

from src import config
from src.imbalance import apply_strategy
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.preprocess import run_preprocessing_pipeline
from src.risk_triage import (
    apply_triage,
    compute_total_cost,
    optimize_thresholds,
    summarize_decisions,
)
from src.uncertainty import get_ensemble_disagreement
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Fixed baseline thresholds
# ---------------------------------------------------------------------

FIXED_LOW = 0.15
FIXED_HIGH = 0.85

# Fixed uncertainty threshold for condition C.
# This remains the original threshold so that C represents
# "uncertainty added without threshold optimization".
FIXED_UNCERTAINTY = 0.05

# Disable uncertainty completely for A and B.
UNCERTAINTY_OFF = float("inf")


def frauds_auto_approved(decisions: list, y_true: np.ndarray) -> int:
    """Count actual fraud transactions that were auto-approved."""
    y_true = np.asarray(y_true)

    return sum(
        1
        for decision, label in zip(decisions, y_true)
        if decision == "Approve" and label == 1
    )


def build_uncertainty_candidates(uncertainty_val: np.ndarray):
    """Create candidate uncertainty thresholds from validation data.

    Using validation quantiles keeps the search focused on the actual
    uncertainty distribution instead of arbitrary values such as 0.05.
    """

    quantiles = np.array([
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.92,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
        0.995,
    ])

    candidates = np.unique(np.quantile(uncertainty_val, quantiles))

    # float('inf') = uncertainty OFF, listed first so ties favor the simpler
    # model. The old 0.0 candidate is dropped: it sends everything to Review
    # and can never win.
    return np.concatenate(([np.inf], candidates))


def run_condition(
    name,
    mean_proba_val,
    uncertainty_val,
    y_val,
    mean_proba_test,
    uncertainty_test,
    y_test,
    use_uncertainty: bool,
    optimize: bool,
):
    """Tune on validation and evaluate once on test."""

    y_val_arr = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    # ---------------------------------------------------------------
    # Probability-only conditions
    # ---------------------------------------------------------------

    if not use_uncertainty:

        if optimize:
            best, _ = optimize_thresholds(
                mean_proba_val,
                uncertainty_val,
                y_val_arr,
                uncertainty_threshold=UNCERTAINTY_OFF,
            )

            low = best["low"]
            high = best["high"]

        else:
            low = FIXED_LOW
            high = FIXED_HIGH

        uncertainty_threshold = UNCERTAINTY_OFF

    # ---------------------------------------------------------------
    # Uncertainty-aware conditions
    # ---------------------------------------------------------------

    else:

        if optimize:

            uncertainty_candidates = build_uncertainty_candidates(
                uncertainty_val
            )

            best, _ = optimize_thresholds(
                mean_proba_val,
                uncertainty_val,
                y_val_arr,
                uncertainty_candidates=uncertainty_candidates,
            )

            low = best["low"]
            high = best["high"]
            uncertainty_threshold = best["uncertainty"]

        else:

            low = FIXED_LOW
            high = FIXED_HIGH
            uncertainty_threshold = FIXED_UNCERTAINTY

    # ---------------------------------------------------------------
    # Apply frozen validation-selected parameters to TEST
    # ---------------------------------------------------------------

    decisions = apply_triage(
        mean_proba_test,
        uncertainty_test,
        low_threshold=low,
        high_threshold=high,
        uncertainty_threshold=uncertainty_threshold,
    )

    cost = compute_total_cost(
        decisions,
        y_test_arr,
    )

    summary = summarize_decisions(decisions)

    missed = frauds_auto_approved(
        decisions,
        y_test_arr,
    )

    logger.info(
        f"[{name}] "
        f"low={low:.4f} "
        f"high={high:.4f} "
        f"uncertainty={uncertainty_threshold:.6f} "
        f"test_cost={cost:.1f} "
        f"review%={summary['percentages']['Review']:.2f} "
        f"fraud_auto_approved={missed}"
    )

    return {
        "condition": name,
        "low": low,
        "high": high,
        "uncertainty": uncertainty_threshold,
        "test_cost": cost,
        "counts": summary["counts"],
        "percentages": summary["percentages"],
        "fraud_auto_approved": missed,
    }


if __name__ == "__main__":

    # ================================================================
    # 1. PREPROCESS DATA
    # ================================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = run_preprocessing_pipeline()

    # ================================================================
    # 2. HANDLE CLASS IMBALANCE ON TRAINING DATA ONLY
    # ================================================================

    X_res, y_res = apply_strategy(
        config.DEFAULT_IMBALANCE_STRATEGY,
        X_train,
        y_train,
    )

    # ================================================================
    # 3. TRAIN BASE MODELS
    # ================================================================

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)

    # ================================================================
    # 4. GENERATE VALIDATION + TEST PREDICTIONS
    # ================================================================

    mean_proba_val, uncertainty_val = get_ensemble_disagreement(
        xgb,
        lgbm,
        cat,
        X_val,
    )

    mean_proba_test, uncertainty_test = get_ensemble_disagreement(
        xgb,
        lgbm,
        cat,
        X_test,
    )

    # ================================================================
    # 5. RUN A/B/C/D ABLATION
    # ================================================================

    conditions = [

        (
            "A: probability-only, fixed",
            False,
            False,
        ),

        (
            "B: probability-only, optimized",
            False,
            True,
        ),

        (
            "C: +uncertainty, fixed",
            True,
            False,
        ),

        (
            "D: +uncertainty, optimized (ours)",
            True,
            True,
        ),
    ]

    results = []

    for (
        name,
        use_uncertainty,
        optimize,
    ) in conditions:

        result = run_condition(
            name,
            mean_proba_val,
            uncertainty_val,
            y_val,
            mean_proba_test,
            uncertainty_test,
            y_test,
            use_uncertainty=use_uncertainty,
            optimize=optimize,
        )

        results.append(result)

    # ================================================================
    # 6. PRINT FINAL TEST RESULTS
    # ================================================================

    print("\n" + "=" * 115)

    print(
        f"{'Condition':<38}"
        f"{'Test Cost':>12}"
        f"{'Review %':>12}"
        f"{'Fraud Auto-Approved':>22}"
        f"{'Uncertainty':>18}"
    )

    print("=" * 115)

    for result in results:

        print(
            f"{result['condition']:<38}"
            f"{result['test_cost']:>12.1f}"
            f"{result['percentages']['Review']:>11.2f}%"
            f"{result['fraud_auto_approved']:>22}"
            f"{result['uncertainty']:>18.6f}"
        )

    print("=" * 115)

    print(
        "\nThresholds were selected using the validation set only."
        "\nThe test set was used only for final evaluation."
    )