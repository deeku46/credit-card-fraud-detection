"""
config.py
---------
Central configuration for the credit card fraud detection pipeline.
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------
TEST_SIZE = 0.2
VAL_SIZE = 0.1          # taken out of the remaining train portion
TARGET_COL = "Class"    # 1 = fraud, 0 = legitimate
# ---------------------------------------------------------------------
# Class imbalance handling
# ---------------------------------------------------------------------
IMBALANCE_STRATEGIES = ["smote", "adasyn", "smote_enn"]
DEFAULT_IMBALANCE_STRATEGY = "smote_enn"
# ---------------------------------------------------------------------
# Model hyperparameters (starting points -- tune later based on val results)
# ---------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": RANDOM_STATE,
    "verbose": False,
}
DEFAULT_IMBALANCE_STRATEGY = "smote"

# ---------------------------------------------------------------------
# Risk-tier / triage thresholds (starting points -- see risk_triage.py
# for the cost-optimized version, which is this project's novelty)
# ---------------------------------------------------------------------
DEFAULT_LOW_RISK_THRESHOLD = 0.15   # below this -> auto-approve
DEFAULT_HIGH_RISK_THRESHOLD = 0.85  # above this -> auto-block
# everything in between -> escalate to human review

# ---------------------------------------------------------------------
# Cost-sensitive evaluation (tune these to your own assumptions,
# then justify the numbers in the paper)
# ---------------------------------------------------------------------
COST_FALSE_NEGATIVE = 100.0   # missed fraud -> full transaction loss (relative unit)
COST_FALSE_POSITIVE = 5.0     # legitimate transaction wrongly blocked -> customer friction
COST_ESCALATION = 1.0         # analyst review time, applied to every escalated case
# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------
SHAP_SAMPLE_SIZE = 200  # background sample size for SHAP explainer (speed vs fidelity)
# ---------------------------------------------------------------------
# LLM-based investigation summaries
# ---------------------------------------------------------------------
LLM_PROVIDER = "template"   # "template" (offline, free) or "anthropic" (needs API key)
LLM_MODEL = "claude-sonnet-4-5-20250929"  # only used if LLM_PROVIDER = "anthropic"