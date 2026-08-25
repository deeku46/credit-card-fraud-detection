"""
imbalance.py
------------
Wraps SMOTE, ADASYN, and SMOTE-ENN so we can apply and compare
different class-imbalance strategies on the training set only.

IMPORTANT: resampling is applied to the TRAINING set only, never to
val/test -- those must reflect the real-world fraud rate so evaluation
metrics stay honest.
"""

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def apply_smote(X_train, y_train):
    sampler = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    logger.info(f"SMOTE: {len(X_train)} -> {len(X_res)} rows, fraud rate now {y_res.mean():.4f}")
    return X_res, y_res


def apply_adasyn(X_train, y_train):
    sampler = ADASYN(random_state=config.RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    logger.info(f"ADASYN: {len(X_train)} -> {len(X_res)} rows, fraud rate now {y_res.mean():.4f}")
    return X_res, y_res


def apply_smote_enn(X_train, y_train):
    sampler = SMOTEENN(random_state=config.RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    logger.info(f"SMOTE-ENN: {len(X_train)} -> {len(X_res)} rows, fraud rate now {y_res.mean():.4f}")
    return X_res, y_res


STRATEGY_MAP = {
    "smote": apply_smote,
    "adasyn": apply_adasyn,
    "smote_enn": apply_smote_enn,
}


def apply_strategy(strategy: str, X_train, y_train):
    if strategy not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGY_MAP)}")
    return STRATEGY_MAP[strategy](X_train, y_train)


if __name__ == "__main__":
    from src.preprocess import run_preprocessing_pipeline

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()

    for strategy in config.IMBALANCE_STRATEGIES:
        apply_strategy(strategy, X_train, y_train)