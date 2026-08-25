"""
evaluate.py
-----------
Proper evaluation beyond classification_report: ROC-AUC, PR-AUC,
confusion matrix, and a helper to compare multiple imbalance
strategies on the same held-out validation set.
"""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_val, y_val, name: str = "model") -> dict:
    """Compute the core metrics your project spec calls for."""
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    fpr = fp / (fp + tn)

    metrics = {
        "name": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    logger.info(
        f"[{name}] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  "
        f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  FPR={fpr:.5f}"
    )
    return metrics


def compare_strategies(X_train, y_train, X_val, y_val):
    """Train an ensemble under each imbalance strategy and compare on val set."""
    from src import config
    from src.imbalance import apply_strategy
    from src.models import train_ensemble

    results = []
    for strategy in config.IMBALANCE_STRATEGIES:
        logger.info(f"--- Strategy: {strategy} ---")
        X_res, y_res = apply_strategy(strategy, X_train, y_train)
        model = train_ensemble(X_res, y_res)
        metrics = evaluate_model(model, X_val, y_val, name=strategy)
        results.append(metrics)
    return results


if __name__ == "__main__":
    from src.preprocess import run_preprocessing_pipeline

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    results = compare_strategies(X_train, y_train, X_val, y_val)

    print("\n=== Strategy comparison (validation set) ===")
    for r in results:
        print(f"{r['name']:12s} ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
              f"Precision={r['precision']:.4f}  Recall={r['recall']:.4f}  F1={r['f1']:.4f}")