"""
preprocess.py
-------------
Loads the raw transaction data, cleans it, and produces train/val/test
splits ready for imbalance handling and modeling.

Expected input (European Credit Card Fraud Detection dataset):
    - Time    : seconds elapsed between this transaction and the first
    - V1..V28 : PCA-anonymized features
    - Amount  : transaction amount
    - Class   : target, 1 = fraud, 0 = legitimate
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. Raises a clear error if the file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find dataset at {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df
def make_synthetic_dataset(n_samples: int = 2000, fraud_ratio: float = 0.02,
                            n_features: int = 28, seed: int = config.RANDOM_STATE
                            ) -> pd.DataFrame:
    """Generate a small synthetic dataset shaped like the real one, for
    fast tests that don't need the full 284K-row CSV.
    """
    rng = np.random.RandomState(seed)
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))
    fraud = rng.normal(loc=1.2, scale=1.6, size=(n_fraud, n_features))

    X = np.vstack([legit, fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    amount = rng.exponential(scale=88.0, size=n_samples)
    time_col = np.sort(rng.uniform(0, 172792, size=n_samples))

    cols = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "Time", time_col)
    df["Amount"] = amount
    df[config.TARGET_COL] = y

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicates and missing values."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info(f"Dropped {removed} duplicate rows")

    n_missing = df.isnull().sum().sum()
    if n_missing:
        logger.info(f"Found {n_missing} missing values -- imputing with median")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Light, defensible feature engineering."""
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    seconds_in_day = 24 * 3600
    df["Hour"] = (df["Time"] % seconds_in_day) / 3600.0
    return df


def split_and_scale(df: pd.DataFrame):
    """Split into train/val/test, then fit scaler on train only."""
    y = df[config.TARGET_COL]
    X = df.drop(columns=[config.TARGET_COL])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_fraction,
        stratify=y_train_full,
        random_state=config.RANDOM_STATE,
    )

    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    logger.info(f"Split sizes -- train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    logger.info(
        f"Fraud rate -- train: {y_train.mean():.5f}, val: {y_val.mean():.5f}, test: {y_test.mean():.5f}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def run_preprocessing_pipeline():
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    return split_and_scale(df)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    print(X_train.head())