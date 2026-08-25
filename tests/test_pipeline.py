"""
test_pipeline.py
-----------------
Basic smoke tests for the core pipeline stages. These aren't exhaustive
unit tests -- they just confirm each stage runs and produces
sane-shaped output, so a broken import or logic error gets caught
before you're mid-demo in front of a reviewer.
"""

import numpy as np
import pandas as pd

from src import config
from src.imbalance import apply_smote
from src.preprocess import make_synthetic_dataset, split_and_scale
from src.risk_triage import apply_triage, assign_risk_tier, compute_total_cost


def test_synthetic_dataset_shape():
    df = make_synthetic_dataset(n_samples=2000)
    assert config.TARGET_COL in df.columns
    assert df[config.TARGET_COL].nunique() == 2
    assert len(df) == 2000


def test_split_and_scale_produces_correct_sizes():
    df = make_synthetic_dataset(n_samples=2000)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == 2000
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_smote_balances_classes():
    df = make_synthetic_dataset(n_samples=2000)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)
    X_res, y_res = apply_smote(X_train, y_train)
    assert abs(y_res.mean() - 0.5) < 0.01  # should be ~balanced after SMOTE
    assert len(X_res) == len(y_res)


def test_assign_risk_tier_logic():
    # low probability, low uncertainty -> Approve
    assert assign_risk_tier(proba=0.05, uncertainty=0.01) == "Approve"
    # high probability, low uncertainty -> Block
    assert assign_risk_tier(proba=0.95, uncertainty=0.01) == "Block"
    # mid probability -> Review
    assert assign_risk_tier(proba=0.5, uncertainty=0.01) == "Review"
    # high uncertainty -> always Review, regardless of probability
    assert assign_risk_tier(proba=0.02, uncertainty=0.5) == "Review"


def test_compute_total_cost_no_errors_gives_zero():
    decisions = ["Approve", "Block"]
    y_true = np.array([0, 1])  # correct approve on legit, correct block on fraud
    cost = compute_total_cost(decisions, y_true)
    assert cost == 0.0


def test_compute_total_cost_penalizes_missed_fraud():
    decisions = ["Approve"]
    y_true = np.array([1])  # approved a fraud case -> false negative
    cost = compute_total_cost(decisions, y_true)
    assert cost == config.COST_FALSE_NEGATIVE