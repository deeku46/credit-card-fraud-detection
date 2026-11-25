"""
app.py  -  Review GUI for the fraud-triage project.

Run from the project root (venv active):
    pip install streamlit
    streamlit run app.py

Needs (already produced by your earlier runs):
    outputs/predictions.npz          (python -m src.final_eval)
    outputs/multiseed_results.json   (python -m src.multi_seed)
    outputs/paper/*.png              (python -m src.visualize, src.pipeline_figure)
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

OUT = "outputs"
st.set_page_config(page_title="Fraud Triage: Does Disagreement Help?", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load_preds():
    d = np.load(os.path.join(OUT, "predictions.npz"))
    return {k: d[k] for k in d.files}


@st.cache_data
def load_multiseed():
    p = os.path.join(OUT, "multiseed_results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def triage(p, u, low, high, tau):
    review = (u >= tau) | ((p >= low) & (p < high))
    approve = (~review) & (p < low)
    block = (~review) & (p >= high)
    return approve, review, block


def evaluate(p, u, y, low, high, tau, fn, fp=5, rv=1):
    a, r, b = triage(p, u, low, high, tau)
    missed = int(np.sum(a & (y == 1)))
    false_blocks = int(np.sum(b & (y == 0)))
    cost = fn * missed + fp * false_blocks + rv * int(r.sum())
    return {"Approve": int(a.sum()), "Review": int(r.sum()), "Block": int(b.sum()),
            "Frauds missed": missed, "False blocks": false_blocks,
            "Review %": 100 * r.mean(), "Cost": cost}


try:
    P = load_preds()
except Exception as e:
    st.error(f"Could not load outputs/predictions.npz ({e}). Run `python -m src.final_eval` first.")
    st.stop()

p_te, u_te, y_te = P["p_test"], P["u_test"], P["y_test"]

# ---------------------------------------------------------------- header
st.title("Does Ensemble Disagreement Help? Fraud Triage Demo")
st.caption("Credit-card fraud detection with a human-in-the-loop Approve / Review / Block decision. "
           "All numbers on the held-out test set.")

tab1, tab2, tab3, tab4 = st.tabs(["1. Overview", "2. Live triage", "3. Inspect a transaction",
                                  "4. Results"])

# ---------------------------------------------------------------- tab 1
with tab1:
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    pred = (p_te >= 0.5).astype(int)
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{roc_auc_score(y_te, p_te):.3f}")
    c[1].metric("PR-AUC", f"{average_precision_score(y_te, p_te):.3f}")
    c[2].metric("Precision", f"{precision_score(y_te, pred, zero_division=0):.3f}")
    c[3].metric("Recall", f"{recall_score(y_te, pred):.3f}")
    c[4].metric("F1", f"{f1_score(y_te, pred):.3f}")
    st.write(f"Test set: **{len(y_te):,}** transactions, **{int(y_te.sum())}** fraudulent "
             f"({100 * y_te.mean():.2f}%). Classification metrics at probability threshold 0.5.")
    left, right = st.columns([3, 2])
    with left:
        img = os.path.join(OUT, "paper", "pipeline.png")
        if os.path.exists(img):
            st.image(img, caption="Experimental pipeline")
    with right:
        st.subheader("Main goal")
        st.write("Test, fairly and carefully, whether **ensemble disagreement** helps decide "
                 "which fraud cases a human should review, compared with the fraud probability alone.")
        st.subheader("Finding")
        st.write("Disagreement carries some signal but is **largely redundant** with the "
                 "probability, and gives **no consistent cost saving**.")
        st.subheader("Pieces")
        st.markdown("- Three models: XGBoost, LightGBM, CatBoost\n"
                    "- SMOTE inside training folds only\n"
                    "- Out-of-fold threshold tuning, test set used once\n"
                    "- SHAP explanations for reviewed cases")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Move the thresholds and see the decisions change")
    ms = load_multiseed()
    colA, colB = st.columns([1, 2])
    with colA:
        fn = st.select_slider("Cost of a missed fraud", [100, 200, 500, 1000, 2000, 5000], value=1000)
        low = st.slider("Approve below probability", 0.0005, 0.40, 0.02, step=0.0005, format="%.4f")
        high = st.slider("Block at or above probability", 0.60, 0.98, 0.98, step=0.02)
        use_u = st.checkbox("Also send to review when models disagree (method D)", value=False)
        tau = st.slider("Disagreement threshold", 0.0, 0.40, 0.05, step=0.005) if use_u else float("inf")
        if st.button("Load tuned thresholds (from out-of-fold data)"):
            try:
                from src.multi_seed import QUANTILES
                from src.oof_sweep import search
                cands = np.concatenate(([np.inf], np.unique(np.quantile(P["u_oof"], QUANTILES))))
                best = search(P["p_oof"], P["u_oof"], P["y_oof"], [np.inf], fn)
                st.session_state["tuned_B"] = best
                bestD = search(P["p_oof"], P["u_oof"], P["y_oof"], cands, fn)
                st.session_state["tuned_D"] = bestD
            except Exception as e:
                st.warning(f"Could not run the tuner ({e}).")
        for tag in ("B", "D"):
            t = st.session_state.get(f"tuned_{tag}")
            if t:
                name = "probability only (B)" if tag == "B" else "with disagreement (D)"
                st.info(f"Tuned for cost {fn}, {name}: approve below {t['low']}, "
                        f"block at or above {t['high']}, disagreement threshold {t['unc']}")
    with colB:
        resB = evaluate(p_te, u_te, y_te, low, high, float("inf"), fn)
        resD = evaluate(p_te, u_te, y_te, low, high, tau, fn)
        df = pd.DataFrame({"Probability only (B)": resB, "With disagreement (D)": resD}).T
        df["Review %"] = df["Review %"].round(2)
        st.dataframe(df)
        chart = pd.DataFrame({"B": [resB["Approve"], resB["Review"], resB["Block"]],
                              "D": [resD["Approve"], resD["Review"], resD["Block"]]},
                             index=["Approve", "Review", "Block"])
        st.bar_chart(chart)
        st.caption("Bars show how many of the test transactions go to each action. "
                   "Cost = missed-fraud cost x frauds approved + 5 x legitimate blocked + 1 x reviews.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Pick a transaction and see why it was flagged")
    t = st.session_state.get("tuned_D") or {"low": 0.02, "high": 0.98, "unc": 0.05}
    a, r, b = triage(p_te, u_te, float(t["low"]), float(t["high"]), float(t["unc"]))
    tier = np.where(a, "Approve", np.where(b, "Block", "Review"))
    choice = st.radio("Show", ["A reviewed case", "A blocked case", "An approved case", "A missed fraud"],
                      horizontal=True)
    pool = {"A reviewed case": np.where(tier == "Review")[0],
            "A blocked case": np.where(tier == "Block")[0],
            "An approved case": np.where(tier == "Approve")[0],
            "A missed fraud": np.where((tier == "Approve") & (y_te == 1))[0]}[choice]
    if len(pool) == 0:
        st.warning("No transactions in this group for the current thresholds.")
    else:
        pos = st.slider("Case number", 1, len(pool), 1) - 1
        i = int(pool[pos])
        c = st.columns(4)
        c[0].metric("Action", tier[i])
        c[1].metric("Fraud probability", f"{p_te[i]:.1%}")
        c[2].metric("Model disagreement", f"{u_te[i]:.3f}")
        c[3].metric("True label", "Fraud" if y_te[i] == 1 else "Legitimate")
        st.caption("Thresholds used: the tuned method-D thresholds if you loaded them in tab 2, "
                   "otherwise approve below 0.02, block at 0.98, disagreement 0.05.")

        if st.checkbox("Explain with SHAP (trains one model, about 30 seconds the first time)"):
            @st.cache_resource(show_spinner="Training the explanation model...")
            def get_explainer():
                import pandas as _pd
                from src import config
                from src.explain import build_explainer
                from src.imbalance import apply_strategy
                from src.models import train_xgboost
                from src.preprocess import run_preprocessing_pipeline
                Xtr, Xva, Xte, ytr, yva, yte, _ = run_preprocessing_pipeline()
                Xall, yall = _pd.concat([Xtr, Xva]), _pd.concat([ytr, yva])
                Xres, yres = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, Xall, yall)
                model = train_xgboost(Xres, yres)
                return build_explainer(model, Xall), Xte
            try:
                from src.explain import explain_instance, top_reasons
                from src.investigate import generate_investigation_summary
                explainer, Xte = get_explainer()
                contrib = explain_instance(explainer, Xte.iloc[[i]])
                rows = top_reasons(contrib, top_n=5)
                st.write("**Top factors (SHAP)**")
                st.dataframe(pd.DataFrame(rows, columns=["Feature", "SHAP value", "Effect"]),
                             hide_index=True)
                st.write("**Analyst summary**")
                st.code(generate_investigation_summary(float(p_te[i]), float(u_te[i]), str(tier[i]),
                                                       contrib), language=None)
                st.caption("SHAP explains the XGBoost model only. Features V1 to V28 are anonymised, "
                           "and summaries are template-based (no LLM).")
            except Exception as e:
                st.error(f"Could not build the explanation: {e}")

# ---------------------------------------------------------------- tab 4
with tab4:
    st.subheader("Final results")
    ms = load_multiseed()
    if ms:
        fns = sorted(ms[0]["regimes"].keys(), key=int)
        rows = []
        for f in fns:
            B = [r["regimes"][f]["B"]["cost"] for r in ms]
            D = [r["regimes"][f]["D"]["cost"] for r in ms]
            diff = [d - b for b, d in zip(B, D)]
            rows.append({"Missed-fraud cost": int(f), "Mean cost B": round(np.mean(B)),
                         "Mean cost D": round(np.mean(D)), "Mean D minus B": round(np.mean(diff)),
                         "Seeds D cheaper": sum(x < 0 for x in diff),
                         "Seeds D costlier": sum(x > 0 for x in diff)})
        st.write("Probability only (B) vs. with disagreement (D), five seeds:")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Negative = disagreement cheaper. There is no consistent saving.")
    else:
        st.info("Run `python -m src.multi_seed` to show the five-seed table.")
    figs = [("roc_pr_test", "ROC and precision-recall"), ("confusion_matrix_test", "Confusion matrix"),
            ("cost_vs_fn", "Cost change, D vs B, by seed"), ("review_budget_curve", "Missed frauds vs review rate"),
            ("uncertainty_distribution_test", "Disagreement: legitimate vs fraud"),
            ("triage_tiers_test", "Triage outcome"), ("detector_metrics_seeds", "Detector metrics, 5 seeds")]
    cols = st.columns(2)
    k = 0
    for name, cap in figs:
        path = os.path.join(OUT, "paper", f"{name}.png")
        if os.path.exists(path):
            cols[k % 2].image(path, caption=cap)
            k += 1