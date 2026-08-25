"""
models.py
---------
Trains XGBoost, LightGBM, and CatBoost on a given (resampled) training
set, and combines them into a soft-voting ensemble.

Models are trained on RESAMPLED data (from imbalance.py) but always
evaluated on the REAL, untouched val/test sets -- otherwise metrics
would be meaningless.
"""

import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def train_xgboost(X_train, y_train):
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_train, y_train)
    logger.info("XGBoost trained")
    return model


def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(**config.LGBM_PARAMS)
    model.fit(X_train, y_train)
    logger.info("LightGBM trained")
    return model


def train_catboost(X_train, y_train):
    model = CatBoostClassifier(**config.CATBOOST_PARAMS)
    model.fit(X_train, y_train)
    logger.info("CatBoost trained")
    return model


def train_ensemble(X_train, y_train):
    """Soft-voting ensemble of all three -- averages predicted probabilities."""
    xgb = XGBClassifier(**config.XGB_PARAMS)
    lgbm = LGBMClassifier(**config.LGBM_PARAMS)
    cat = CatBoostClassifier(**config.CATBOOST_PARAMS)

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("cat", cat)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    logger.info("Voting ensemble trained")
    return ensemble


def save_model(model, name: str):
    import os
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")


def load_model(name: str):
    import os
    path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)


if __name__ == "__main__":
    from src.imbalance import apply_strategy
    from src.preprocess import run_preprocessing_pipeline

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    ensemble = train_ensemble(X_res, y_res)
    save_model(ensemble, "ensemble_smote_enn")

    val_preds = ensemble.predict(X_val)
    from sklearn.metrics import classification_report
    print(classification_report(y_val, val_preds, target_names=["legit", "fraud"]))