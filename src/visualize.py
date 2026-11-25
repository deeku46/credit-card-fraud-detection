"""
visualize.py
------------
Generates and saves figures.

The plot_* functions at the top are the original exploratory figures
(used by main.py; computed on the VALIDATION set, so do NOT use them in
the paper).

The paper figures (plot_paper_roc_pr, plot_paper_cost_vs_fn) are built
only from final results and saved to outputs/paper/:
  - outputs/predictions.npz        (written by src.final_eval; test set)
  - outputs/multiseed_results.json (written by src.multi_seed; 5 seeds)

Run:  python -m src.visualize
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")


def _save(fig, filename: str):
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


def plot_roc_curve(y_true, y_proba, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="Ensemble", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    _save(fig, filename)


def plot_pr_curve(y_true, y_proba, filename="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    _save(fig, filename)


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    _save(fig, filename)


def plot_strategy_comparison(results: list, filename="strategy_comparison.png"):
    """results: list of dicts from evaluate.compare_strategies()."""
    metrics = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    names = [r["name"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.15
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, label=metric.upper())

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Imbalance Strategy Comparison")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, filename)


def plot_risk_tier_breakdown(summary: dict, filename="risk_tier_breakdown.png"):
    """summary: output of risk_triage.summarize_decisions()."""
    tiers = list(summary["counts"].keys())
    counts = list(summary["counts"].values())

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4CAF50", "#FFC107", "#F44336"]  # green/amber/red
    ax.bar(tiers, counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    ax.set_ylabel("Number of transactions")
    ax.set_title("Risk Tier Breakdown (Validation Set)")
    _save(fig, filename)


def plot_uncertainty_distribution(uncertainty: np.ndarray, filename="uncertainty_distribution.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(uncertainty, bins=50, color="steelblue", edgecolor="black")
    ax.set_xlabel("Model disagreement (std of predicted probabilities)")
    ax.set_ylabel("Number of transactions")
    ax.set_title("Uncertainty Distribution")
    ax.set_yscale("log")  # most values cluster near zero -- log scale shows the tail
    _save(fig, filename)


# =====================================================================
# PAPER FIGURES (final results only)
# =====================================================================

_BLUE, _ORANGE, _GREY = "#1f5fa8", "#d9730d", "#6b6b6b"

_PAPER_RC = {
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": False, "axes.facecolor": "white",
    "lines.linewidth": 1.2, "pdf.fonttype": 42,
}


def _save_paper(fig, name: str):
    outdir = os.path.join(config.OUTPUTS_DIR, "paper")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved paper figure: {os.path.join(outdir, name)}.png/.pdf")


def plot_paper_roc_pr(y_test, p_test, name="roc_pr_test"):
    """Fig. 2: ROC and PR curves on the TEST set, one IEEE column wide."""
    y = np.asarray(y_test)
    p = np.asarray(p_test)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    auc, ap = roc_auc_score(y, p), average_precision_score(y, p)

    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.9))
        ax[0].plot(fpr, tpr, color=_BLUE)
        ax[0].plot([0, 1], [0, 1], color=_GREY, lw=0.7, ls="--")
        ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
                  xlim=(0, 1), ylim=(0, 1.02))
        ax[0].text(0.95, 0.06, f"AUC = {auc:.3f}", ha="right", fontsize=7)
        ax[0].set_title("(a) ROC")

        ax[1].plot(rec, prec, color=_BLUE)
        ax[1].axhline(y.mean(), color=_GREY, lw=0.7, ls="--")
        ax[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
        ax[1].text(0.05, 0.06, f"AP = {ap:.3f}", ha="left", fontsize=7)
        ax[1].set_title("(b) Precision-recall")
        fig.tight_layout(w_pad=1.0)
        _save_paper(fig, name)


def plot_paper_cost_vs_fn(multiseed_path, name="cost_vs_fn"):
    """Fig. 3: per-seed % cost change of D vs B at each FN cost."""
    with open(multiseed_path) as f:
        res = json.load(f)
    fns = sorted(res[0]["regimes"].keys(), key=int)
    x = np.arange(len(fns))

    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        ax.axhline(0, color=_GREY, lw=0.8)
        means = []
        for i, fn in enumerate(fns):
            ch = np.array([
                100 * (r["regimes"][fn]["D"]["cost"] - r["regimes"][fn]["B"]["cost"])
                / r["regimes"][fn]["B"]["cost"]
                for r in res
            ])
            jitter = np.linspace(-0.12, 0.12, len(ch))
            ax.scatter(i + jitter, ch, s=14, color=_BLUE, alpha=0.8,
                       edgecolor="white", linewidth=0.4, zorder=3,
                       label="Single seed" if i == 0 else None)
            means.append(ch.mean())
        ax.plot(x, means, color=_ORANGE, marker="D", ms=4, lw=1.2, zorder=4,
                label="Mean over seeds")
        ax.set_xticks(x)
        ax.set_xticklabels(fns)
        ax.set_xlabel(r"Cost of a missed fraud $c_{FN}$")
        ax.set_ylabel("Cost change, D vs. B (%)\n(below 0: D cheaper)")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        _save_paper(fig, name)




# ---------------------------------------------------------------------
# Report figures (same final data; larger size, more detail)
# ---------------------------------------------------------------------

_REPORT_SIZE = (5.2, 3.4)


def plot_report_confusion_matrix(y_test, p_test, threshold=0.5, name="confusion_matrix_test"):
    """Confusion matrix on the TEST set at the given probability threshold."""
    y = np.asarray(y_test)
    pred = (np.asarray(p_test) >= threshold).astype(int)
    cm = confusion_matrix(y, pred)
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.imshow(np.log10(cm + 1), cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9,
                        color="white" if np.log10(cm[i, j] + 1) > 2.5 else "black")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"Test set, threshold {threshold}")
        fig.tight_layout()
        _save_paper(fig, name)


def plot_report_uncertainty(y_test, u_test, name="uncertainty_distribution_test"):
    """Disagreement score on the TEST set, legitimate vs fraud (log counts)."""
    y = np.asarray(y_test)
    u = np.asarray(u_test)
    bins = np.linspace(0, max(u.max(), 1e-6), 41)
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=_REPORT_SIZE)
        ax.hist(u[y == 0], bins=bins, color=_BLUE, alpha=0.85, label=f"Legitimate (n={int((y == 0).sum()):,})")
        ax.hist(u[y == 1], bins=bins, color=_ORANGE, alpha=0.85, label=f"Fraud (n={int((y == 1).sum())})")
        ax.set_yscale("log")
        ax.set_xlabel("Model disagreement (std of the three predicted probabilities)")
        ax.set_ylabel("Number of transactions (log scale)")
        ax.legend(frameon=False)
        fig.tight_layout()
        _save_paper(fig, name)


def plot_report_tiers(p_oof, u_oof, y_oof, p_test, u_test, y_test, fn_cost=1000,
                      name="triage_tiers_test"):
    """Approve/Review/Block counts on TEST for B and D (thresholds tuned on OOF)."""
    from src.multi_seed import QUANTILES
    from src.oof_sweep import search, triage_cost

    y_oof, y_te = np.asarray(y_oof), np.asarray(y_test)
    cands = np.concatenate(([np.inf], np.unique(np.quantile(u_oof, QUANTILES))))
    rows = {}
    for tag, c in (("B: probability only", [np.inf]), ("D: probability + disagreement", cands)):
        b = search(p_oof, u_oof, y_oof, c, fn_cost)
        _, ap, bl, rv = triage_cost(p_test, u_test, y_te, b["low"], b["high"], b["unc"], fn_cost)
        rows[tag] = [int(ap.sum()), int(rv.sum()), int(bl.sum()), int(np.sum(ap & (y_te == 1)))]

    labels = ["Approve", "Review", "Block"]
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=_REPORT_SIZE)
        w = 0.36
        x = np.arange(3)
        for k, (tag, v) in enumerate(rows.items()):
            bars = ax.bar(x + (k - 0.5) * w, np.maximum(v[:3], 1), w, color=(_BLUE, _ORANGE)[k],
                          label=f"{tag} ({v[3]} frauds approved)")
            for xi, h in zip(x + (k - 0.5) * w, v[:3]):
                ax.text(xi, max(h, 1) * 1.15, f"{h:,}", ha="center", va="bottom", fontsize=6.5)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Transactions (log scale)")
        ax.set_title(f"Test-set triage outcome, $c_{{FN}}$ = {fn_cost}")
        ax.set_ylim(0.7, ax.get_ylim()[1] * 4)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        _save_paper(fig, name)


def plot_report_budget(multiseed_path, name="review_budget_curve"):
    """Missed frauds vs review rate, B and D, mean +/- std over seeds."""
    with open(multiseed_path) as f:
        res = json.load(f)
    budgets = sorted(res[0]["budgets"].keys(), key=float)
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=_REPORT_SIZE)
        for tag, col, mk in (("B", _BLUE, "o"), ("D", _ORANGE, "D")):
            rv, ms, sd = [], [], []
            for bud in budgets:
                rows = [r["budgets"][bud][tag] for r in res if r["budgets"][bud][tag] is not None]
                rv.append(np.mean([r["review"] for r in rows]))
                ms.append(np.mean([r["missed"] for r in rows]))
                sd.append(np.std([r["missed"] for r in rows], ddof=1))
            ax.errorbar(rv, ms, yerr=sd, color=col, marker=mk, ms=4, capsize=2,
                        label="B: probability only" if tag == "B" else "D: probability + disagreement")
        ax.set_xlabel("Review rate (% of transactions)")
        ax.set_ylabel("Frauds auto-approved (mean over seeds)")
        ax.legend(frameon=False)
        fig.tight_layout()
        _save_paper(fig, name)


def plot_report_seed_metrics(multiseed_path, name="detector_metrics_seeds"):
    """Detector metrics over seeds: mean with std error bars, per-seed dots."""
    with open(multiseed_path) as f:
        res = json.load(f)
    keys = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision", "Precision"),
            ("recall", "Recall"), ("f1", "F1")]
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=_REPORT_SIZE)
        for i, (k, lab) in enumerate(keys):
            v = np.array([r["metrics"][k] for r in res])
            ax.bar(i, v.mean(), 0.55, color=_BLUE, alpha=0.85)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color="black", capsize=3, lw=1)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=8,
                       color=_ORANGE, zorder=3)
            ax.text(i, 0.03, f"{v.mean():.3f}", ha="center", color="white", fontsize=7)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([l for _, l in keys])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Test-set score")
        ax.set_title("Mean over 5 seeds (bars), single seeds (dots)")
        fig.tight_layout()
        _save_paper(fig, name)


if __name__ == "__main__":
    # Final-results figures only. Needs outputs/predictions.npz (src.final_eval)
    # and outputs/multiseed_results.json (src.multi_seed).
    npz_path = os.path.join(config.OUTPUTS_DIR, "predictions.npz")
    ms_path = os.path.join(config.OUTPUTS_DIR, "multiseed_results.json")

    d = np.load(npz_path)

    # --- Paper figures ---
    plot_paper_roc_pr(d["y_test"], d["p_test"])
    plot_paper_cost_vs_fn(ms_path)

    # --- Report figures ---
    plot_report_confusion_matrix(d["y_test"], d["p_test"])
    plot_report_uncertainty(d["y_test"], d["u_test"])
    plot_report_tiers(d["p_oof"], d["u_oof"], d["y_oof"], d["p_test"], d["u_test"], d["y_test"])
    plot_report_budget(ms_path)
    plot_report_seed_metrics(ms_path)

    print(f"\nAll figures saved to: {os.path.join(config.OUTPUTS_DIR, 'paper')}")