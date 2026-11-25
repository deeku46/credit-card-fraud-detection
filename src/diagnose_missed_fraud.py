"""
diagnose_missed_fraud.py
-------------------------
Checks whether the fraud cases that get auto-approved have low model
disagreement (a shared blind spot across all three models) or whether
something else is going on.
"""

from src import config
from src.imbalance import apply_strategy
from src.models import train_catboost, train_lightgbm, train_xgboost
from src.preprocess import run_preprocessing_pipeline
from src.uncertainty import get_ensemble_disagreement

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    xgb = train_xgboost(X_res, y_res)
    lgbm = train_lightgbm(X_res, y_res)
    cat = train_catboost(X_res, y_res)

    mean_proba, uncertainty = get_ensemble_disagreement(xgb, lgbm, cat, X_test)
    y_test_arr = y_test.values

    # actual fraud cases with a low predicted probability (below 0.18),
    # i.e. the ones getting auto-approved
    missed_mask = (y_test_arr == 1) & (mean_proba < 0.18)
    missed_idx = [i for i, m in enumerate(missed_mask) if m]

    print(f"\nFound {len(missed_idx)} fraud cases predicted as low-risk (auto-approved)\n")
    print(f"{'idx':<8}{'proba':>10}{'uncertainty':>14}")
    for i in missed_idx:
        print(f"{i:<8}{mean_proba[i]:>10.4f}{uncertainty[i]:>14.5f}")

    print(f"\nMean uncertainty on these missed cases: {uncertainty[missed_idx].mean():.5f}")
    print(f"Mean uncertainty overall (test set): {uncertainty.mean():.5f}")