"""
diagnose_uncertainty_threshold.py
---------------------------------
Evaluates how the uncertainty threshold affects validation-set
fraud detection, review rate, and total cost.

This is a diagnostic only. The held-out test set is NOT used.
"""

import numpy as np

from src import config
from src.imbalance import apply_strategy
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.preprocess import run_preprocessing_pipeline
from src.risk_triage import (
    apply_triage,
    compute_total_cost,
    summarize_decisions,
)
from src.uncertainty import get_ensemble_disagreement


if __name__ == "__main__":

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = run_preprocessing_pipeline()

    X_res, y_res = apply_strategy(
        config.DEFAULT_IMBALANCE_STRATEGY,
        X_train,
        y_train,
    )

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)

    mean_proba, uncertainty = get_ensemble_disagreement(
        xgb,
        lgbm,
        cat,
        X_val,
    )

    y_val_arr = y_val.values

    # Candidate thresholds concentrated around the observed uncertainty range.
    thresholds = [
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.010,
        0.012,
        0.015,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
        0.045,
        0.050,
    ]

    print("\nValidation uncertainty-threshold sensitivity")
    print("=" * 100)

    print(
        f"{'Threshold':>12}"
        f"{'Cost':>12}"
        f"{'Review %':>12}"
        f"{'Fraud Approved':>18}"
        f"{'Fraud Caught':>15}"
    )

    print("-" * 100)

    total_fraud = int(np.sum(y_val_arr == 1))

    for threshold in thresholds:

        decisions = apply_triage(
            mean_proba,
            uncertainty,
            low_threshold=0.15,
            high_threshold=0.85,
            uncertainty_threshold=threshold,
        )

        cost = compute_total_cost(
            decisions,
            y_val_arr,
        )

        summary = summarize_decisions(decisions)

        fraud_auto_approved = sum(
            1
            for decision, label in zip(decisions, y_val_arr)
            if decision == "Approve" and label == 1
        )

        fraud_caught = total_fraud - fraud_auto_approved

        print(
            f"{threshold:>12.4f}"
            f"{cost:>12.1f}"
            f"{summary['percentages']['Review']:>11.2f}%"
            f"{fraud_auto_approved:>18}"
            f"{fraud_caught:>15}"
        )

    print("=" * 100)

    print(f"\nTotal validation fraud cases: {total_fraud}")