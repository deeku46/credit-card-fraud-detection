# Credit Card Fraud Detection under Class Imbalance

## Project Overview

This project focuses on developing a machine learning-based system for
real-time credit card fraud detection under severe class imbalance.

The project investigates improved fraud detection techniques while
exploring uncertainty-aware risk assessment and AI-assisted fraud
investigation.

## Problem Statement

**PSAIAC_324 — ML Models for Real-Time Credit Fraud Detection in Digital Payments**

Credit card fraud detection is challenging because fraudulent
transactions represent a very small proportion of total transactions.
This class imbalance can cause machine learning models to favor
legitimate transactions and miss fraudulent activity.

## Research Direction

The proposed system combines:

- Imbalanced-data handling
- Ensemble machine learning
- Uncertainty estimation
- Risk-aware fraud triage
- Explainable AI (SHAP)
- AI-assisted fraud investigation
- Real-time decision support

## Research Objective

To investigate whether uncertainty-aware risk triage can determine
when a transaction can be handled automatically and when it should
be escalated for further human investigation.

## Base Paper

Chhabra, R., Goswami, S., & Ranjan, R. K. (2024).

*A voting ensemble machine learning based credit card fraud detection
using highly imbalance data.*

Multimedia Tools and Applications, Springer.

DOI: 10.1007/s11042-023-17766-9

## Dataset

European Credit Card Fraud Detection Dataset (via Kaggle, ULB Machine
Learning Group) — 284,807 transactions with a highly imbalanced fraud
class (~0.17% fraud rate). Not included in this repo due to size —
download and place at `data/creditcard.csv`.

## Pipeline

```text
Preprocessing → Imbalance Handling (SMOTE/ADASYN/SMOTE-ENN)
→ Ensemble Fraud Detection (XGBoost + LightGBM + CatBoost)
→ Uncertainty Estimation (model disagreement)
→ Cost-Optimized Risk Triage (Approve / Review / Block)
→ SHAP Explanations + Investigation Summaries
```

## Project Status

✅ Core pipeline implemented and running end-to-end (preprocessing
through risk triage, SHAP explanations, and investigation summaries).
🚧 Real-time integration, dashboard, and paper drafting in progress.

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

Individual stages can also be run separately, e.g.:
```bash
python -m src.preprocess
python -m src.evaluate
python -m src.risk_triage
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

## Research Novelty

Rather than another fraud classifier (already well-covered by 2024-2026
literature combining ML + SHAP + LLMs), this project's contribution is a
**cost-optimized decision layer**: using ensemble disagreement as an
uncertainty signal, and searching for risk-tier thresholds that minimize
total operational cost (missed fraud, false alarms, and analyst review
time) rather than relying on fixed, arbitrary cutoffs.

## Repository Structure

```text
Mini_Project_7th_Sem/
├── data/                  # dataset (gitignored)
├── src/
│   ├── config.py          # paths, seeds, hyperparameters, thresholds
│   ├── utils.py            # logging, timing helpers
│   ├── preprocess.py       # data loading, cleaning, feature engineering
│   ├── imbalance.py        # SMOTE / ADASYN / SMOTE-ENN
│   ├── models.py            # XGBoost / LightGBM / CatBoost + ensemble
│   ├── evaluate.py          # ROC-AUC, PR-AUC, strategy comparison
│   ├── uncertainty.py       # ensemble disagreement / calibration
│   ├── risk_triage.py       # cost-optimized Approve/Review/Block logic
│   ├── explain.py           # SHAP explanations
│   ├── investigate.py       # investigation summaries (template or LLM)
│   └── visualize.py         # figure generation
├── models/                 # saved trained models (gitignored)
├── outputs/                 # generated figures
├── tests/
├── Research papers/         # local research papers (not uploaded)
├── requirements.txt
├── main.py                  # runs the full pipeline end-to-end
└── README.md
```

## Team

Research supervisor: Dr. Sameena Banu