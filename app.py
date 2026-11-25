"""
app.py  -  Review GUI for the fraud-triage project (redesigned).

Run from the project root (venv active):
    pip install streamlit altair
    streamlit run app.py

Needs (already produced by your earlier runs):
    outputs/predictions.npz          (python -m src.final_eval)
    outputs/multiseed_results.json   (python -m src.multi_seed)
    outputs/paper/*.png              (python -m src.visualize, src.pipeline_figure)
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

OUT = "outputs"
NAVY, TEAL, INK, MUTED = "#1E3A5F", "#0F766E", "#0F172A", "#64748B"
GREEN, AMBER, RED = "#16A34A", "#F59E0B", "#DC2626"
TIER_COLOR = {"Approve": GREEN, "Review": AMBER, "Block": RED}

st.set_page_config(page_title="Fraud Triage Demo", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- style
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, {TEAL} 100%); color: white;
         padding: 28px 34px; border-radius: 18px; margin-bottom: 18px; }}
.hero h1 {{ color: white; font-size: 2.0rem; margin: 0 0 6px 0; }}
.hero p {{ color: #E2F1EE; font-size: 1.05rem; margin: 0; }}
.kpi {{ background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(15,23,42,.06); height: 100%; }}
.kpi .v {{ font-size: 1.9rem; font-weight: 700; color: {NAVY}; line-height: 1.1; }}
.kpi .l {{ font-size: .78rem; font-weight: 700; color: {TEAL}; text-transform: uppercase; letter-spacing: .05em; }}
.kpi .d {{ font-size: .85rem; color: {MUTED}; margin-top: 4px; }}
.tile {{ border-radius: 14px; padding: 14px 10px; color: white; text-align: center; min-height: 112px; display:flex; flex-direction:column; justify-content:center; }}
.tile .v {{ font-size: 1.8rem; font-weight: 700; line-height: 1.1; }}
.tile .l {{ font-size: .85rem; font-weight: 600; opacity: .95; }}
.tile .s {{ font-size: .75rem; opacity: .9; }}
.step {{ background: #F1F5F9; border-radius: 14px; padding: 16px 18px; height: 100%; }}
.step b {{ color: {NAVY}; }}
.step .n {{ display:inline-block; background:{TEAL}; color:white; border-radius:50%; width:26px; height:26px;
           text-align:center; line-height:26px; font-weight:700; margin-right:8px; }}
.callout {{ border-left: 6px solid {TEAL}; background: #ECFDF5; padding: 14px 18px; border-radius: 8px;
           color: {INK}; font-size: 1.02rem; }}
.callout.warn {{ border-left-color: {AMBER}; background: #FFFBEB; }}
.pill {{ display:inline-block; padding: 4px 14px; border-radius: 999px; color: white; font-weight: 700; }}
.bar {{ background: #E2E8F0; border-radius: 999px; height: 12px; overflow: hidden; margin: 4px 0 2px 0; }}
.gap {{ height: 14px; }}
.bar > div {{ height: 100%; border-radius: 999px; }}
.small {{ color: {MUTED}; font-size: .85rem; }}
.stTabs [data-baseweb="tab"] {{ font-size: 1.02rem; font-weight: 600; padding: 10px 18px; }}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, desc):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)


def tile(col, label, value, color, sub=""):
    col.markdown(f'<div class="tile" style="background:{color}"><div class="v">{value}</div>'
                 f'<div class="l">{label}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def callout(text, warn=False):
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


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
st.markdown("""
<div class="hero">
  <h1>🛡️ Does Ensemble Disagreement Help?</h1>
  <p>Credit-card fraud detection where every transaction is sent to <b>Approve</b>, <b>Review</b> or
  <b>Block</b>. Explore the results, change the rules, and see why each case was flagged.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("New here? Quick glossary (30 seconds)"):
    g1, g2 = st.columns(2)
    g1.markdown("**Fraud probability (p)**: how likely a transaction is fraud, averaged over 3 models.\n\n"
                "**Disagreement (u)**: how much the 3 models differ on it. High means they are unsure.")
    g2.markdown("**Approve / Review / Block**: let it through, send to a human, or stop it.\n\n"
                "**Cost**: a missed fraud costs a lot, a false block costs 5, a review costs 1.")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "🎛️ Live triage", "🔍 Inspect a transaction", "📊 Results"])

# ---------------------------------------------------------------- tab 1
with tab1:
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    pred = (p_te >= 0.5).astype(int)
    st.markdown("#### How well does the fraud detector work?")
    c = st.columns(5)
    kpi(c[0], "ROC-AUC", f"{roc_auc_score(y_te, p_te):.3f}", "Ranks fraud above legitimate. 1.0 is perfect.")
    kpi(c[1], "PR-AUC", f"{average_precision_score(y_te, p_te):.3f}", "Quality on the rare fraud class.")
    kpi(c[2], "Precision", f"{precision_score(y_te, pred, zero_division=0):.3f}", "Of flagged cases, how many are real fraud.")
    kpi(c[3], "Recall", f"{recall_score(y_te, pred):.3f}", "Of all real frauds, how many we catch.")
    kpi(c[4], "F1", f"{f1_score(y_te, pred):.3f}", "Balance of precision and recall.")
    st.caption(f"Test set: {len(y_te):,} transactions, {int(y_te.sum())} fraudulent ({100 * y_te.mean():.2f}%). "
               "Metrics at probability threshold 0.5.")

    st.markdown("#### How it works, in 3 steps")
    s = st.columns(3)
    s[0].markdown('<div class="step"><span class="n">1</span><b>Score</b><br>Three models (XGBoost, LightGBM, '
                  'CatBoost) each give a fraud probability. We average them (p) and measure how much they '
                  'differ (u).</div>', unsafe_allow_html=True)
    s[1].markdown('<div class="step"><span class="n">2</span><b>Triage</b><br>Low p is approved, high p is '
                  'blocked, the uncertain middle goes to a human reviewer. Thresholds are tuned to minimise '
                  'total cost.</div>', unsafe_allow_html=True)
    s[2].markdown('<div class="step"><span class="n">3</span><b>Explain</b><br>SHAP shows which features '
                  'pushed a transaction towards fraud, so the reviewer sees reasons, not just a score.</div>',
                  unsafe_allow_html=True)

    st.write("")
    callout("<b>Our question:</b> does sending cases to review when the models <i>disagree</i> beat using the "
            "probability alone?<br><b>Our answer:</b> mostly no. Disagreement carries some signal but is "
            "largely redundant with the probability, and it gave <b>no consistent cost saving</b>.")
    img = os.path.join(OUT, "paper", "pipeline.png")
    if os.path.exists(img):
        with st.expander("Show the full pipeline diagram"):
            st.image(img)

# ---------------------------------------------------------------- tab 2
with tab2:
    st.markdown("#### Move the sliders and watch the decisions change")
    st.caption("Compare two rules on the held-out test set: **B** uses the probability only; "
               "**D** also sends a case to review when the models disagree.")
    colA, colB = st.columns([1, 2], gap="large")
    with colA:
        with st.container(border=True):
            st.markdown("**Rule settings**")
            fn = st.select_slider("Cost of a missed fraud", [100, 200, 500, 1000, 2000, 5000], value=1000,
                                  help="How many times worse a missed fraud is than a review.")
            low = st.slider("Approve below probability", 0.0005, 0.40, 0.02, step=0.0005, format="%.4f",
                            help="Cases scored below this are let through.")
            high = st.slider("Block at or above probability", 0.60, 0.98, 0.98, step=0.02,
                             help="Cases scored at or above this are blocked.")
            use_u = st.toggle("Also review when models disagree (rule D)", value=False)
            tau = (st.slider("Disagreement threshold", 0.0, 0.40, 0.02, step=0.005,
                             help="Cases where the models differ by at least this go to review.")
                   if use_u else float("inf"))
            if st.button("Load tuned thresholds"):
                try:
                    from src.multi_seed import QUANTILES
                    from src.oof_sweep import search
                    cands = np.concatenate(([np.inf], np.unique(np.quantile(P["u_oof"], QUANTILES))))
                    st.session_state["tuned_B"] = search(P["p_oof"], P["u_oof"], P["y_oof"], [np.inf], fn)
                    st.session_state["tuned_D"] = search(P["p_oof"], P["u_oof"], P["y_oof"], cands, fn)
                except ModuleNotFoundError as e:
                    st.warning(f"Missing package **{e.name}**. Start the app with "
                               "`python -m streamlit run app.py` from inside your project venv.")
                except Exception as e:
                    st.warning(f"Could not run the tuner ({e}).")
        for tag in ("B", "D"):
            t = st.session_state.get(f"tuned_{tag}")
            if t:
                name = "probability only (B)" if tag == "B" else "with disagreement (D)"
                st.info(f"**Tuned for cost {fn}, {name}:** approve below {t['low']}, "
                        f"block at or above {t['high']}, disagreement threshold {t['unc']}. "
                        "Set the sliders to these values to reproduce it.")
    with colB:
        resB = evaluate(p_te, u_te, y_te, low, high, float("inf"), fn)
        resD = evaluate(p_te, u_te, y_te, low, high, tau, fn)
        if not use_u:
            callout("Rule D is switched off, so B and D are identical. Turn on <b>Also review when models "
                    "disagree</b> to compare.", warn=True)
        else:
            d = resD["Cost"] - resB["Cost"]
            pct = 100 * d / resB["Cost"] if resB["Cost"] else 0
            if d < 0:
                callout(f"With these settings, rule D is <b>cheaper by {-d:,}</b> ({pct:+.1f}%).")
            elif d > 0:
                callout(f"With these settings, rule D is <b>more expensive by {d:,}</b> ({pct:+.1f}%).",
                        warn=True)
            else:
                callout("With these settings, rule D and rule B cost the same.")
        st.write("")
        for tag, res in (("B", resB), ("D", resD)):
            st.markdown(f"**{'Probability only (B)' if tag == 'B' else 'With disagreement (D)'}**"
                        f" &nbsp; <span class='small'>cost {res['Cost']:,}</span>", unsafe_allow_html=True)
            c = st.columns(5)
            tile(c[0], "Approve", f"{res['Approve']:,}", GREEN)
            tile(c[1], "Review", f"{res['Review']:,}", AMBER, f"{res['Review %']:.1f}% of all")
            tile(c[2], "Block", f"{res['Block']:,}", RED)
            tile(c[3], "Frauds missed", f"{res['Frauds missed']}", NAVY, "approved but fraud")
            tile(c[4], "False blocks", f"{res['False blocks']}", MUTED, "legit but blocked")
            st.write("")
        long = pd.DataFrame([{"Rule": tg, "Action": a, "Count": r[a]}
                             for tg, r in (("B", resB), ("D", resD)) for a in ("Approve", "Review", "Block")])
        ch = (alt.Chart(long).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
              .encode(x=alt.X("Action:N", sort=["Approve", "Review", "Block"], axis=alt.Axis(labelAngle=0, title=None)),
                      xOffset="Rule:N", y=alt.Y("Count:Q", title="Transactions"),
                      color=alt.Color("Rule:N", scale=alt.Scale(domain=["B", "D"], range=[NAVY, TEAL]),
                                      legend=alt.Legend(title="Rule", orient="top")),
                      tooltip=["Rule", "Action", "Count"]).properties(height=240))
        st.altair_chart(ch)
        st.caption("Cost = missed-fraud cost × frauds approved + 5 × legitimate blocked + 1 × reviews.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown("#### Pick a transaction and see why it was flagged")
    t = st.session_state.get("tuned_D") or {"low": 0.02, "high": 0.98, "unc": 0.02}
    a, r, b = triage(p_te, u_te, float(t["low"]), float(t["high"]), float(t["unc"]))
    tier = np.where(a, "Approve", np.where(b, "Block", "Review"))
    choice = st.radio("Show me", ["A reviewed case", "A blocked case", "An approved case", "A missed fraud"],
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
        col = TIER_COLOR[tier[i]]
        meaning = {"Approve": "Probability is low and the models agree, so it is let through.",
                   "Review": "The probability is in the uncertain middle, or the models disagree, "
                             "so a human should look at it.",
                   "Block": "Probability is very high, so it is stopped."}[tier[i]]
        left, right = st.columns([1, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown(f'<span class="pill" style="background:{col}">{tier[i]}</span> &nbsp; '
                            f'<span class="small">{meaning}</span>', unsafe_allow_html=True)
                st.write("")
                st.markdown("**Fraud probability**")
                st.markdown(f'<div class="bar"><div style="width:{min(100, p_te[i] * 100):.1f}%;background:{RED}">'
                            f'</div></div><div class="small">{p_te[i]:.1%}</div><div class="gap"></div>', unsafe_allow_html=True)
                st.markdown("**Model disagreement**")
                st.markdown(f'<div class="bar"><div style="width:{min(100, u_te[i] / 0.4 * 100):.1f}%;'
                            f'background:{AMBER}"></div></div><div class="small">{u_te[i]:.3f} '
                            '(higher means the models differ more)</div><div class="gap"></div>', unsafe_allow_html=True)
                st.markdown("**What it really was**")
                st.markdown(f'<span class="pill" style="background:{RED if y_te[i] == 1 else GREEN}">'
                            f'{"Fraud" if y_te[i] == 1 else "Legitimate"}</span>', unsafe_allow_html=True)
            st.caption("Uses the tuned rule-D thresholds if you loaded them in the Live triage tab, "
                       "otherwise approve below 0.02, block at 0.98, disagreement 0.02.")
        with right:
            with st.container(border=True):
                st.markdown("**Why? (SHAP explanation)**")
                if st.toggle("Explain this case (trains one model, about 30 seconds the first time)"):
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
                        df = pd.DataFrame(rows, columns=["Feature", "SHAP value", "Effect"])
                        ch = (alt.Chart(df).mark_bar(cornerRadiusEnd=4)
                              .encode(y=alt.Y("Feature:N", sort="-x", title=None),
                                      x=alt.X("SHAP value:Q", title="Push towards fraud (right) or legitimate (left)"),
                                      color=alt.condition(alt.datum["SHAP value"] > 0, alt.value(RED), alt.value(GREEN)),
                                      tooltip=["Feature", "SHAP value", "Effect"]).properties(height=200))
                        st.altair_chart(ch)
                        st.markdown("**Analyst summary**")
                        st.info(generate_investigation_summary(float(p_te[i]), float(u_te[i]), str(tier[i]), contrib))
                        st.caption("SHAP explains the XGBoost model only. Features V1 to V28 are anonymised, "
                                   "and summaries are template-based.")
                    except ModuleNotFoundError as e:
                        st.error(f"A package is missing in the Python that is running this app: **{e.name}**.")
                        st.code("venv\\Scripts\\activate\npip install -r requirements.txt\n"
                                "python -m streamlit run app.py", language="bash")
                        st.caption("Start the app from inside your project venv so it can use the same "
                                   "packages as your training code.")
                    except Exception as e:
                        st.error(f"Could not build the explanation: {e}")
                else:
                    st.caption("Switch the toggle on to see which features drove this decision.")

# ---------------------------------------------------------------- tab 4
with tab4:
    st.markdown("#### Final results")
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
        tbl = pd.DataFrame(rows)
        callout("<b>Takeaway:</b> across five seeds and six cost settings, adding disagreement does not save "
                "cost consistently. Negative values below mean D was cheaper; they flip sign between seeds.")
        st.write("")
        st.dataframe(tbl, hide_index=True)
        ch = (alt.Chart(tbl).mark_bar(cornerRadiusEnd=4)
              .encode(x=alt.X("Missed-fraud cost:O", title="Cost of a missed fraud", axis=alt.Axis(labelAngle=0)),
                      y=alt.Y("Mean D minus B:Q", title="Mean cost: D minus B"),
                      color=alt.condition(alt.datum["Mean D minus B"] < 0, alt.value(GREEN), alt.value(RED)),
                      tooltip=list(tbl.columns)).properties(height=240))
        st.altair_chart(ch)
        st.caption("Green = disagreement cheaper, red = costlier. Bars are small and mixed: no consistent saving.")
    else:
        st.info("Run `python -m src.multi_seed` to show the five-seed table.")

    groups = {"Detector": [("roc_pr_test", "ROC and precision-recall"), ("confusion_matrix_test", "Confusion matrix"),
                           ("detector_metrics_seeds", "Detector metrics, 5 seeds")],
              "Cost study": [("cost_vs_fn", "Cost change, D vs B, by seed")],
              "Review budget": [("review_budget_curve", "Missed frauds vs review rate")],
              "Disagreement": [("uncertainty_distribution_test", "Disagreement: legitimate vs fraud"),
                               ("triage_tiers_test", "Triage outcome")]}
    ft = st.tabs(list(groups))
    for tab, (gname, figs) in zip(ft, groups.items()):
        with tab:
            cols = st.columns(2)
            k = 0
            for name, cap in figs:
                path = os.path.join(OUT, "paper", f"{name}.png")
                if os.path.exists(path):
                    cols[k % 2].image(path, caption=cap)
                    k += 1
            if k == 0:
                st.caption("Figures not found in outputs/paper.")
