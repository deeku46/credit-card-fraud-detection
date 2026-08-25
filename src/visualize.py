"""
visualize.py
------------
Generates and saves the core figures needed for the paper/PPT:
ROC curve, PR curve, confusion matrix, imbalance strategy comparison,
risk-tier breakdown, and uncertainty distribution.

All figures are saved as PNGs into outputs/ so they can be dropped
straight into the paper or slides.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")


def _save(fig, filename: str):
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


def plot_roc_curve(y_true, y_proba, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="Ensemble", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    _save(fig, filename)


def plot_pr_curve(y_true, y_proba, filename="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    _save(fig, filename)


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    _save(fig, filename)


def plot_strategy_comparison(results: list, filename="strategy_comparison.png"):
    """results: list of dicts from evaluate.compare_strategies()."""
    metrics = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    names = [r["name"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.15
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, label=metric.upper())

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Imbalance Strategy Comparison")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, filename)


def plot_risk_tier_breakdown(summary: dict, filename="risk_tier_breakdown.png"):
    """summary: output of risk_triage.summarize_decisions()."""
    tiers = list(summary["counts"].keys())
    counts = list(summary["counts"].values())

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4CAF50", "#FFC107", "#F44336"]  # green/amber/red
    ax.bar(tiers, counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    ax.set_ylabel("Number of transactions")
    ax.set_title("Risk Tier Breakdown (Validation Set)")
    _save(fig, filename)


def plot_uncertainty_distribution(uncertainty: np.ndarray, filename="uncertainty_distribution.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(uncertainty, bins=50, color="steelblue", edgecolor="black")
    ax.set_xlabel("Model disagreement (std of predicted probabilities)")
    ax.set_ylabel("Number of transactions")
    ax.set_title("Uncertainty Distribution")
    ax.set_yscale("log")  # most values cluster near zero -- log scale shows the tail
    _save(fig, filename)


if __name__ == "__main__":
    from src.evaluate import compare_strategies, evaluate_model
    from src.imbalance import apply_strategy
    from src.models import train_catboost, train_ensemble, train_lightgbm, train_xgboost
    from src.preprocess import run_preprocessing_pipeline
    from src.risk_triage import apply_triage, assign_risk_tier, optimize_thresholds, summarize_decisions
    from src.uncertainty import get_ensemble_disagreement

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()

    # --- Strategy comparison figure ---
    results = compare_strategies(X_train, y_train, X_val, y_val)
    plot_strategy_comparison(results)

    # --- Train final models on chosen strategy for the rest of the figures ---
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)
    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)
    ensemble = train_ensemble(X_res, y_res)

    y_proba = ensemble.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    plot_roc_curve(y_val, y_proba)
    plot_pr_curve(y_val, y_proba)
    plot_confusion_matrix(y_val, y_pred)

    # --- Uncertainty + risk tier figures ---
    mean_proba, uncertainty = get_ensemble_disagreement(xgb, lgbm, cat, X_val)
    plot_uncertainty_distribution(uncertainty)

    best, _ = optimize_thresholds(mean_proba, uncertainty, y_val.values)
    decisions = apply_triage(mean_proba, uncertainty, low_threshold=best["low"], high_threshold=best["high"])
    summary = summarize_decisions(decisions)
    plot_risk_tier_breakdown(summary)

    print(f"\nAll figures saved to: {config.OUTPUTS_DIR}")