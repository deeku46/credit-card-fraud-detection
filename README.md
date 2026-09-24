# Credit Card Fraud Detection under Class Imbalance

## Project Overview

This project focuses on developing a machine learning-based system for
real-time credit card fraud detection under severe class imbalance.

The project investigates improved fraud detection techniques while
exploring uncertainty-aware risk assessment and AI-assisted fraud
investigation. Rather than optimizing for detection accuracy alone, the
core question this project asks is: **when should a fraud-detection
system trust its own prediction enough to act automatically, and when
should it defer to a human?**

## Problem Statement

**PSAIAC_324 — ML Models for Real-Time Credit Fraud Detection in Digital Payments**

Credit card fraud detection is challenging because fraudulent
transactions represent a very small proportion of total transactions
(~0.17% in this dataset). This class imbalance can cause machine
learning models to favor legitimate transactions and miss fraudulent
activity. Beyond the imbalance problem, most existing systems treat
every prediction identically — a transaction scored 0.51 is handled the
same way as one scored 0.99, even though the model's confidence in
those two cases is very different.

## Research Direction

The proposed system combines:

- Imbalanced-data handling, compared experimentally rather than assumed
- Ensemble machine learning across three gradient-boosted tree models
- Uncertainty estimation derived from inter-model disagreement
- Cost-optimized, risk-aware fraud triage (Approve / Review / Block)
- Explainable AI (SHAP) with human-readable, case-specific explanations
  and tailored next-step recommendations
- An ablation study that isolates and validates each novel component
  on a held-out test set

## Research Objective

To investigate whether uncertainty-aware risk triage can determine
when a transaction can be handled automatically and when it should
be escalated for further human investigation, while maintaining strong
fraud-detection performance.

## Research Novelty

A large share of 2024–2026 literature on this dataset already combines
ensemble ML, SMOTE-family resampling, SHAP, and even LLM-based
investigation support (see Literature Survey below). This project does
**not** claim novelty in that combination alone. The contribution here
is a **cost-optimized decision layer** sitting on top of a standard
ensemble detector:

1. **Uncertainty as a first-class signal.** Instead of relying solely
   on the averaged fraud probability, the system measures how much the
   three base models (XGBoost, LightGBM, CatBoost) *disagree* with each
   other. High disagreement overrides a "confident-looking" probability
   and forces escalation to a human — because an average of disagreeing
   models is not a trustworthy number.
2. **Cost-optimized thresholds, not fixed cutoffs.** Rather than
   assuming arbitrary probability thresholds (e.g. 0.15 / 0.85), the
   system grid-searches for the threshold pair that minimizes total
   operational cost — weighing missed fraud, false alarms, and analyst
   review time against each other.
3. **An ablation study isolates whether each piece actually matters.**
   Four conditions (probability-only/fixed, probability-only/optimized,
   +uncertainty/fixed, +uncertainty/optimized) are evaluated on a test
   set that was never touched during model, strategy, or threshold
   selection — so the reported contribution of each component is not
   an artifact of tuning on the same data being reported.

## Base Paper

Chhabra, R., Goswami, S., & Ranjan, R. K. (2024).

*A voting ensemble machine learning based credit card fraud detection
using highly imbalance data.*

Multimedia Tools and Applications, Springer.

DOI: 10.1007/s11042-023-17766-9

## Dataset

European Credit Card Fraud Detection Dataset (via Kaggle, ULB Machine
Learning Group) — 284,807 transactions over two days, with 492 fraud
cases (~0.17% fraud rate). Features `V1`–`V28` are PCA-transformed
components of the original transaction data; `Amount` and `Time` are
the only untransformed fields, since the raw features could not be
released for confidentiality reasons. Not included in this repo due to
size — download from Kaggle and place at `data/creditcard.csv`.

## Pipeline

```text
Preprocessing → Imbalance Handling (SMOTE/ADASYN/SMOTE-ENN)
→ Ensemble Fraud Detection (XGBoost + LightGBM + CatBoost)
→ Uncertainty Estimation (model disagreement)
→ Cost-Optimized Risk Triage (Approve / Review / Block)
→ SHAP Explanations + Feature-Extremity-Based Investigation Summaries
```

**Methodology, stage by stage:**

- **Preprocessing** — duplicate removal, median imputation for missing
  values, log-transformed `Amount`, `Hour`-of-day extracted from `Time`.
  Stratified 70/10/20 train/val/test split; scaler fit on train only to
  avoid leakage.
- **Imbalance handling** — SMOTE, ADASYN, and SMOTE-ENN applied to the
  *training set only* (val/test kept at the real fraud rate so metrics
  stay honest). Compared experimentally; SMOTE selected as the working
  default based on validation PR-AUC/F1.
- **Model training** — XGBoost, LightGBM, and CatBoost trained
  independently, then combined into a soft-voting ensemble.
- **Uncertainty estimation** — standard deviation of the three base
  models' predicted probabilities per transaction; high disagreement
  signals an unreliable average score.
- **Risk triage** — a transaction is escalated to Review if model
  disagreement exceeds a threshold, or if its probability falls between
  the low/high cutoffs. Cutoffs are chosen by minimizing total cost on
  the validation set, then evaluated on the untouched test set.
- **Explainability** — SHAP TreeExplainer identifies the top
  contributing features per transaction. Each is translated into a
  plain-language statement of how unusual that value is (approximate
  percentile, treating the standardized feature value as a z-score),
  rather than a raw SHAP number.
- **Investigation summaries** — for Review-tier cases, tailored
  next-step recommendations are generated based on *which kind* of
  feature is driving the flag (amount-related, time-related, or
  anonymized behavioral pattern).

## Evaluation Metrics

| Metric | What it tells you |
|---|---|
| ROC-AUC | Overall ability to rank fraud above legitimate transactions |
| PR-AUC | More informative than ROC-AUC under extreme imbalance; compare against the no-skill baseline (~0.0017, the fraud rate) |
| Precision | Of flagged transactions, how many are actually fraud |
| Recall | Of actual fraud, how much is caught |
| F1 | Harmonic balance of precision and recall |
| False Positive Rate | Legitimate transactions wrongly flagged |
| % Auto-handled vs. Escalated | Operational metric — how much human review is actually needed |
| Total operational cost | Cost-weighted combination of missed fraud, false alarms, and escalations |
| Latency | Inference time per transaction (fast path vs. Review path with SHAP) |

## Project Status

✅ Core pipeline implemented and running end-to-end.
✅ Ablation study isolates the contribution of uncertainty-based
escalation and cost-optimized thresholds against a held-out test set.
✅ Investigation summaries explain *how unusual* each contributing
feature is (percentile-based) and give tailored next-step
recommendations, instead of generic boilerplate.
🚧 Statistical confidence intervals (bootstrap), cost-sensitivity
sweep, and a direct comparison table against the base paper's reported
numbers are in progress. Paper drafting ongoing.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and place it in `data/`.

## Running

```bash
python main.py
```

Runs the full pipeline end-to-end and saves all figures to `outputs/`.

Individual stages can also be run separately:
```bash
python -m src.preprocess           # data loading, cleaning, splitting
python -m src.evaluate              # strategy comparison (SMOTE/ADASYN/SMOTE-ENN)
python -m src.risk_triage           # cost-optimized threshold search
python -m src.ablation              # ablation study, evaluated on test set
python -m src.demo_review_cases     # sample Approve/Review/Block cases with explanations
pytest tests/ -v                     # smoke tests (synthetic data, fast)
```

## Results (validation set, SMOTE strategy)

| Metric | Score |
|---|---|
| ROC-AUC | 0.999 |
| PR-AUC | 0.897 |
| Precision | 0.719 |
| Recall | 0.872 |
| F1 | 0.789 |

**Risk tier breakdown** (cost-optimized thresholds, low=0.18 / high=0.92):
- Approve (automated): 28,107 (99.06%)
- Review (human escalation): 226 (0.80%)
- Block (automated): 40 (0.14%)

*Ablation study results (test set, four conditions isolating uncertainty
and threshold optimization) — see `src/ablation.py` output.*

## Limitations

- Validation set fraud cases are relatively few (~47), so reported
  point-estimate metrics carry meaningful variance; confidence
  intervals are being added via bootstrap resampling.
- Cost weights (false negative / false positive / escalation) are
  currently assumed rather than empirically derived; a sensitivity
  analysis is planned to show how thresholds shift under different
  cost assumptions.
- `V1`–`V28` are PCA-anonymized, so feature-level explanations describe
  statistical extremity rather than real-world semantic meaning.

## Repository Structure

```text
Mini_Project_7th_Sem/
├── data/                  # dataset (gitignored)
├── src/
│   ├── config.py          # paths, seeds, hyperparameters, thresholds
│   ├── utils.py             # logging, timing helpers
│   ├── preprocess.py        # data loading, cleaning, feature engineering
│   ├── imbalance.py         # SMOTE / ADASYN / SMOTE-ENN
│   ├── models.py             # XGBoost / LightGBM / CatBoost + ensemble
│   ├── evaluate.py           # ROC-AUC, PR-AUC, strategy comparison
│   ├── uncertainty.py        # ensemble disagreement / calibration
│   ├── risk_triage.py        # cost-optimized Approve/Review/Block logic
│   ├── explain.py            # SHAP explanations + feature-extremity scoring
│   ├── investigate.py        # case-specific investigation summaries
│   ├── ablation.py           # isolates uncertainty & threshold-optimization contribution
│   ├── demo_review_cases.py  # sample Approve/Review/Block cases, end to end
│   └── visualize.py          # figure generation
├── models/                  # saved trained models (gitignored)
├── outputs/                  # generated figures
├── tests/
├── Research papers/          # local research papers (not uploaded)
├── requirements.txt
├── main.py                   # runs the full pipeline end-to-end
└── README.md
```

## Team
Team Members: Deekshith P, Sakshi Rohilla, Mohammed Yusuf Maaz, Udith Simha
Research supervisor: Dr. Sameena Banu
