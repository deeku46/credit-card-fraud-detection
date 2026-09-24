"""
risk_triage.py
--------------
Combines predicted fraud probability with model-disagreement uncertainty
to make a 3-way decision: Approve / Review / Block.

NOVELTY: instead of fixed thresholds, optimize_thresholds() searches for
the threshold pair that MINIMIZES TOTAL COST (false negatives, false
positives, and escalation overhead), rather than using arbitrary cutoffs
like 0.15/0.85. This is the project's core research contribution --
uncertainty-aware, cost-calibrated triage rather than static rules.
"""

import itertools

import numpy as np

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

APPROVE, REVIEW, BLOCK = "Approve", "Review", "Block"


def assign_risk_tier(proba: float, uncertainty: float,
                      low_threshold: float = config.DEFAULT_LOW_RISK_THRESHOLD,
                      high_threshold: float = config.DEFAULT_HIGH_RISK_THRESHOLD,
                      uncertainty_threshold: float = 0.05) -> str:
    """Decide Approve / Review / Block for a single transaction.

    Logic:
      - High uncertainty (models disagree) -> always escalate to Review,
        regardless of probability -- we don't trust the score itself.
      - Otherwise, use probability thresholds: low -> Approve,
        high -> Block, middle -> Review.
    """
    if uncertainty >= uncertainty_threshold:
        return REVIEW
    if proba < low_threshold:
        return APPROVE
    if proba >= high_threshold:
        return BLOCK
    return REVIEW


def apply_triage(mean_proba: np.ndarray, uncertainty: np.ndarray,
                  low_threshold: float = config.DEFAULT_LOW_RISK_THRESHOLD,
                  high_threshold: float = config.DEFAULT_HIGH_RISK_THRESHOLD,
                  uncertainty_threshold: float = 0.05) -> list:
    """Vectorized version: apply assign_risk_tier across an array of predictions."""
    return [
        assign_risk_tier(p, u, low_threshold, high_threshold, uncertainty_threshold)
        for p, u in zip(mean_proba, uncertainty)
    ]


def compute_total_cost(decisions: list, y_true: np.ndarray) -> float:
    """Total operational cost of a set of triage decisions, using the
    cost weights defined in config.py.

    - BLOCK on a legit transaction (y_true=0)  -> false positive cost
    - APPROVE on a fraud transaction (y_true=1) -> false negative cost
    - REVIEW on anything                        -> escalation cost
    - BLOCK on fraud / APPROVE on legit          -> correct, no cost
    """
    total = 0.0
    y_true = np.asarray(y_true)
    for decision, label in zip(decisions, y_true):
        if decision == REVIEW:
            total += config.COST_ESCALATION
        elif decision == BLOCK and label == 0:
            total += config.COST_FALSE_POSITIVE
        elif decision == APPROVE and label == 1:
            total += config.COST_FALSE_NEGATIVE
    return total


def optimize_thresholds(mean_proba: np.ndarray, uncertainty: np.ndarray, y_true: np.ndarray,
                         low_candidates=None, high_candidates=None,
                         uncertainty_threshold: float = 0.05) -> dict:
    """Grid-search low/high probability thresholds to MINIMIZE total cost.

    uncertainty_threshold: pass float('inf') to disable the uncertainty
    signal entirely (used by the ablation study to isolate its effect).
    """
    if low_candidates is None:
        low_candidates = np.arange(0.02, 0.40, 0.02)
    if high_candidates is None:
        high_candidates = np.arange(0.60, 0.995, 0.02)

    best = {"cost": float("inf"), "low": None, "high": None}
    results = []

    for low, high in itertools.product(low_candidates, high_candidates):
        if low >= high:
            continue
        decisions = apply_triage(mean_proba, uncertainty, low_threshold=low, high_threshold=high,
                                  uncertainty_threshold=uncertainty_threshold)
        cost = compute_total_cost(decisions, y_true)
        results.append({"low": round(low, 2), "high": round(high, 2), "cost": cost})
        if cost < best["cost"]:
            best = {"cost": cost, "low": low, "high": high}

    logger.info(
        f"Best thresholds -- low={best['low']:.2f}, high={best['high']:.2f}, "
        f"total cost={best['cost']:.1f}"
    )
    return best, results


def summarize_decisions(decisions: list) -> dict:
    """% of cases in each tier -- one of the operational metrics required."""
    total = len(decisions)
    counts = {tier: decisions.count(tier) for tier in [APPROVE, REVIEW, BLOCK]}
    percentages = {tier: 100 * count / total for tier, count in counts.items()}
    return {"counts": counts, "percentages": percentages}


if __name__ == "__main__":
    from src.imbalance import apply_strategy
    from src.models import train_catboost, train_lightgbm, train_xgboost
    from src.preprocess import run_preprocessing_pipeline
    from src.uncertainty import get_ensemble_disagreement

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)

    mean_proba, uncertainty = get_ensemble_disagreement(xgb, lgbm, cat, X_val)

    # Default fixed thresholds
    default_decisions = apply_triage(mean_proba, uncertainty)
    default_cost = compute_total_cost(default_decisions, y_val.values)
    print(f"\nDefault thresholds (0.15 / 0.85) -- total cost: {default_cost:.1f}")
    print(summarize_decisions(default_decisions))

    # Cost-optimized thresholds
    best, _ = optimize_thresholds(mean_proba, uncertainty, y_val.values)
    optimized_decisions = apply_triage(mean_proba, uncertainty, low_threshold=best["low"], high_threshold=best["high"])
    print(f"\nOptimized thresholds (low={best['low']:.2f} / high={best['high']:.2f}) -- total cost: {best['cost']:.1f}")
    print(summarize_decisions(optimized_decisions))