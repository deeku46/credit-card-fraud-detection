"""
main.py
-------
Runs the full credit card fraud detection pipeline end-to-end:
preprocessing -> imbalance handling -> model training -> uncertainty
estimation -> cost-optimized risk triage -> SHAP explanations ->
figure generation.

Usage:
    python main.py
"""

from src import config
from src.evaluate import compare_strategies, evaluate_model
from src.explain import build_explainer, explain_instance, top_reasons
from src.imbalance import apply_strategy
from src.investigate import generate_investigation_summary
from src.models import save_model, train_catboost, train_ensemble, train_lightgbm, train_xgboost
from src.preprocess import run_preprocessing_pipeline
from src.risk_triage import apply_triage, optimize_thresholds, summarize_decisions
from src.uncertainty import get_ensemble_disagreement
from src.utils import get_logger, timer
from src.visualize import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_risk_tier_breakdown,
    plot_roc_curve,
    plot_strategy_comparison,
    plot_uncertainty_distribution,
)

logger = get_logger("main")


def main():
    logger.info("=" * 60)
    logger.info("STEP 1: Preprocessing")
    logger.info("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()

    logger.info("=" * 60)
    logger.info("STEP 2: Imbalance strategy comparison")
    logger.info("=" * 60)
    strategy_results = compare_strategies(X_train, y_train, X_val, y_val)
    plot_strategy_comparison(strategy_results)

    logger.info("=" * 60)
    logger.info(f"STEP 3: Training models on '{config.DEFAULT_IMBALANCE_STRATEGY}'")
    logger.info("=" * 60)
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)
    ensemble = train_ensemble(X_res, y_res)
    save_model(ensemble, "ensemble_final")

    logger.info("=" * 60)
    logger.info("STEP 4: Evaluation")
    logger.info("=" * 60)
    with timer("inference on validation set", logger):
        y_proba = ensemble.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    evaluate_model(ensemble, X_val, y_val, name="final_ensemble")
    plot_roc_curve(y_val, y_proba)
    plot_pr_curve(y_val, y_proba)
    plot_confusion_matrix(y_val, y_pred)

    logger.info("=" * 60)
    logger.info("STEP 5: Uncertainty estimation")
    logger.info("=" * 60)
    mean_proba, uncertainty = get_ensemble_disagreement(xgb, lgbm, cat, X_val)
    plot_uncertainty_distribution(uncertainty)

    logger.info("=" * 60)
    logger.info("STEP 6: Cost-optimized risk triage")
    logger.info("=" * 60)
    best, _ = optimize_thresholds(mean_proba, uncertainty, y_val.values)
    decisions = apply_triage(mean_proba, uncertainty, low_threshold=best["low"], high_threshold=best["high"])
    summary = summarize_decisions(decisions)
    logger.info(f"Risk tier breakdown: {summary['counts']}")
    plot_risk_tier_breakdown(summary)

    logger.info("=" * 60)
    logger.info("STEP 7: Sample SHAP explanation + investigation summary")
    logger.info("=" * 60)
    explainer = build_explainer(xgb, X_train)
    sample_row = X_val.iloc[[0]]
    contributions = explain_instance(explainer, sample_row)
    summary_text = generate_investigation_summary(
        proba=mean_proba[0], uncertainty=uncertainty[0], tier=decisions[0], contributions=contributions
    )
    print("\n" + summary_text)

    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE. Figures saved to {config.OUTPUTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()