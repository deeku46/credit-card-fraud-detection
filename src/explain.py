"""
explain.py
----------
Generates SHAP explanations for individual transactions -- specifically
for cases the risk triage sends to Review, so a human analyst gets
"why did the model flag this" evidence, not just a raw probability.

We explain the tree models (XGBoost/LightGBM/CatBoost) directly using
TreeExplainer, which is fast and exact for tree-based models (no
approximation needed, unlike KernelExplainer for arbitrary models).
"""

import numpy as np
import shap

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def build_explainer(model, background_X):
    """Build a SHAP TreeExplainer for a single tree-based model.

    Using feature_perturbation='tree_path_dependent' avoids needing a
    background dataset entirely -- it uses the tree structure itself to
    estimate feature contributions, which also sidesteps XGBoost's
    categorical-split limitation with the default interventional mode.
    """
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    logger.info("Built SHAP explainer (tree_path_dependent mode)")
    return explainer


def explain_instance(explainer, X_row) -> dict:
    """Explain a single transaction. Returns feature -> SHAP value dict,
    sorted by absolute impact (most influential features first).
    """
    shap_values = explainer.shap_values(X_row)

    # TreeExplainer on binary classifiers can return a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # take the "fraud" class contribution

    values = shap_values[0] if shap_values.ndim > 1 else shap_values
    feature_names = X_row.columns.tolist()

    contributions = dict(zip(feature_names, values))
    sorted_contributions = dict(
        sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )
    return sorted_contributions


def top_reasons(contributions: dict, top_n: int = 5) -> list:
    """Return the top_n most influential features as (feature, value, direction) tuples."""
    items = list(contributions.items())[:top_n]
    return [
        (feature, round(value, 4), "increases fraud risk" if value > 0 else "decreases fraud risk")
        for feature, value in items
    ]


if __name__ == "__main__":
    from src.imbalance import apply_strategy
    from src.models import train_xgboost
    from src.preprocess import run_preprocessing_pipeline

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    model = train_xgboost(X_res, y_res)
    explainer = build_explainer(model, X_train)

    # Explain the first validation transaction as a test
    sample_row = X_val.iloc[[0]]
    contributions = explain_instance(explainer, sample_row)
    reasons = top_reasons(contributions, top_n=5)

    print(f"\nExplaining transaction (true label: {y_val.iloc[0]}):")
    for feature, value, direction in reasons:
        print(f"  {feature}: SHAP={value}  ({direction})")

from scipy.stats import norm

FEATURE_LABELS = {
    "Amount": "Transaction amount",
    "Amount_log": "Transaction amount (log-scaled)",
    "Hour": "Time of day",
    "Time": "Elapsed time since first transaction",
}


def describe_feature(name: str, z_value: float) -> str:
    """Human-readable label + how unusual this feature's value is,
    treating the standardized value as an approximate z-score.
    """
    label = FEATURE_LABELS.get(name, f"Encoded transaction pattern feature ({name})")
    percentile = norm.cdf(z_value) * 100
    if abs(z_value) < 1:
        extremity = "within the normal range"
    elif abs(z_value) < 2:
        extremity = "somewhat unusual"
    elif abs(z_value) < 3:
        extremity = "highly unusual"
    else:
        extremity = "extremely unusual"
    return f"{label} -- {extremity} (higher than ~{percentile:.0f}% of typical transactions)"