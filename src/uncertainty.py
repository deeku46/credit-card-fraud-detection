"""
uncertainty.py
--------------
Calibrates model probabilities and computes an uncertainty score based
on disagreement between the three base models (XGBoost, LightGBM,
CatBoost). This uncertainty score feeds into risk_triage.py next.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from src.utils import get_logger

logger = get_logger(__name__)


def calibrate_model(base_model, X_train, y_train, method: str = "isotonic"):
    """Wrap a trained model with probability calibration.
    method: 'isotonic' (flexible, needs more data) or 'sigmoid' (Platt scaling).
    """
    calibrated = CalibratedClassifierCV(base_model, method=method, cv="prefit")
    calibrated.fit(X_train, y_train)
    logger.info(f"Calibrated model using method='{method}'")
    return calibrated


def get_ensemble_disagreement(xgb_model, lgbm_model, cat_model, X) -> np.ndarray:
    """Uncertainty signal: standard deviation of the three base models'
    fraud-probability predictions for each row. Low std = models agree
    = confident. High std = models disagree = uncertain, flag for review.
    """
    p_xgb = xgb_model.predict_proba(X)[:, 1]
    p_lgbm = lgbm_model.predict_proba(X)[:, 1]
    p_cat = cat_model.predict_proba(X)[:, 1]

    stacked = np.vstack([p_xgb, p_lgbm, p_cat])
    mean_proba = stacked.mean(axis=0)
    uncertainty = stacked.std(axis=0)

    return mean_proba, uncertainty


if __name__ == "__main__":
    from src import config
    from src.imbalance import apply_strategy
    from src.models import train_catboost, train_lightgbm, train_xgboost
    from src.preprocess import run_preprocessing_pipeline

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)

    mean_proba, uncertainty = get_ensemble_disagreement(xgb, lgbm, cat, X_val)

    print(f"\nMean predicted probability range: {mean_proba.min():.4f} to {mean_proba.max():.4f}")
    print(f"Uncertainty (std) range: {uncertainty.min():.4f} to {uncertainty.max():.4f}")
    print(f"Mean uncertainty: {uncertainty.mean():.4f}")

    # show the 5 most uncertain predictions
    top_uncertain_idx = np.argsort(uncertainty)[-5:]
    print("\nTop 5 most uncertain cases (index, mean_proba, uncertainty, true_label):")
    y_val_arr = y_val.values
    for idx in top_uncertain_idx:
        print(f"  idx={idx}  proba={mean_proba[idx]:.4f}  uncertainty={uncertainty[idx]:.4f}  true_label={y_val_arr[idx]}")