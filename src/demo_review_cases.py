"""
demo_review_cases.py
---------------------
Runs the full pipeline once, then pulls a handful of actual Review-tier
transactions from the validation set and prints their full investigation
summaries -- this is the concrete "here's why this needs a human" output
your project produces.
"""

from src import config
from src.explain import build_explainer, explain_instance
from src.imbalance import apply_strategy
from src.investigate import generate_investigation_summary
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.preprocess import run_preprocessing_pipeline
from src.risk_triage import assign_risk_tier
from src.uncertainty import get_ensemble_disagreement
from src.utils import get_logger

logger = get_logger(__name__)

N_CASES_TO_SHOW = 3

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)

    mean_proba, uncertainty = get_ensemble_disagreement(xgb, lgbm, cat, X_val)

    # Assign a tier to every validation transaction
    tiers = [
        assign_risk_tier(p, u) for p, u in zip(mean_proba, uncertainty)
    ]

    # Find indices for each tier separately
    from collections import defaultdict
    indices_by_tier = defaultdict(list)
    for i, t in enumerate(tiers):
        indices_by_tier[t].append(i)

    for tier_name in ["Approve", "Review", "Block"]:
        logger.info(f"{tier_name}: {len(indices_by_tier[tier_name])} cases in validation set")

    # Build a SHAP explainer once (using the XGBoost model as the explained model)
    explainer = build_explainer(xgb, X_train)

    y_val_arr = y_val.values

    for tier_name in ["Approve", "Review", "Block"]:
        print(f"\n\n{'#'*60}")
        print(f"# TIER: {tier_name.upper()}")
        print(f"{'#'*60}")

        shown = 0
        for idx in indices_by_tier[tier_name]:
            if shown >= N_CASES_TO_SHOW:
                break
            row = X_val.iloc[[idx]]
            contributions = explain_instance(explainer, row)
            summary = generate_investigation_summary(
                proba=mean_proba[idx],
                uncertainty=uncertainty[idx],
                tier=tiers[idx],
                contributions=contributions,
            )
            print(f"\n{'='*60}")
            print(f"Validation row index: {idx}  |  True label: {y_val_arr[idx]} (0=legit, 1=fraud)")
            print(f"{'='*60}")
            print(summary)
            shown += 1