"""
investigate.py
--------------
Turns SHAP evidence + risk-tier decision into a readable investigation
summary for a human analyst -- the "AI-assisted investigation" stage
of the pipeline.

Two modes, controlled by config.LLM_PROVIDER:
  - "template": builds a readable summary from SHAP values directly,
    no API needed, works immediately, zero cost.
  - "anthropic": sends the same evidence to Claude for a more natural-
    language summary. Requires ANTHROPIC_API_KEY set as an environment
    variable (never hard-code the key in this file).

Start with "template" to get the full pipeline running end-to-end; swap
to "anthropic" later once you have a key, without changing any other
file -- risk_triage.py and explain.py don't need to know which mode
is active.
"""

import os

from src import config
from src.explain import top_reasons
from src.utils import get_logger

logger = get_logger(__name__)


def _recommend_actions(reasons: list) -> list:
    """Tailored next steps for the analyst, based on which KIND of
    feature is driving the flag -- not a single generic line for every case.
    """
    actions = []
    feature_names = [r[0] for r in reasons]

    if any(f in ("Amount", "Amount_log") for f in feature_names):
        actions.append("Verify the transaction amount directly with the cardholder.")
    if "Hour" in feature_names:
        actions.append("Confirm the transaction time aligns with the cardholder's typical activity pattern.")
    if any(f.startswith("V") for f in feature_names):
        actions.append("Cross-check this transaction against the cardholder's recent purchase history and, "
                        "if available, device/location metadata.")
    actions.append("If the cardholder cannot be reached or confirms they did not authorize this, "
                    "escalate per standard fraud protocol.")
    return actions


def _template_summary(proba: float, uncertainty: float, tier: str, reasons: list) -> str:
    """Build a detailed, case-specific summary -- not boilerplate. Every
    section varies based on the actual features and values for THIS
    transaction, not just the tier label.
    """
    from src.explain import describe_feature

    reason_lines = "\n".join(
        f"  - {describe_feature(feature, value)}  "
        f"[{'pushes toward FRAUD' if value > 0 else 'pushes toward LEGITIMATE'}, SHAP={value:+.3f}]"
        for feature, value, _ in reasons
    )

    if uncertainty >= 0.05:
        confidence_note = (
            f"The three base models DISAGREED with each other on this transaction "
            f"(disagreement score {uncertainty:.3f}) -- meaning the averaged probability "
            f"of {proba:.1%} should not be trusted at face value. This is why it was "
            f"escalated regardless of the raw score."
        )
    else:
        confidence_note = (
            f"The three base models agreed closely (disagreement score {uncertainty:.3f}), "
            f"so the {proba:.1%} fraud probability is a reliable estimate."
        )

    summary = (
        f"RISK TIER: {tier}\n"
        f"Estimated fraud probability: {proba:.1%}\n\n"
        f"{confidence_note}\n\n"
        f"Top factors driving this decision:\n{reason_lines}\n"
    )

    if tier == "Review":
        actions = _recommend_actions(reasons)
        action_lines = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions))
        summary += f"\nRecommended next steps:\n{action_lines}\n"

    return summary


def _anthropic_summary(proba: float, uncertainty: float, tier: str, reasons: list) -> str:
    """Send the same evidence to Claude for a natural-language summary.
    Requires: pip install anthropic, and ANTHROPIC_API_KEY set in your
    environment (e.g. `set ANTHROPIC_API_KEY=your-key` on Windows cmd).
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed -- falling back to template mode")
        return _template_summary(proba, uncertainty, tier, reasons)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set -- falling back to template mode")
        return _template_summary(proba, uncertainty, tier, reasons)

    reason_text = "; ".join(
        f"{feature} ({'+' if value > 0 else ''}{value:.3f})" for feature, value, _ in reasons
    )
    prompt = (
        f"You are assisting a fraud analyst. A transaction was assigned risk tier "
        f"'{tier}' with model-estimated fraud probability {proba:.1%} and model "
        f"disagreement score {uncertainty:.3f}. The top SHAP-attributed factors "
        f"(feature name and contribution) are: {reason_text}. "
        f"Write a concise 2-3 sentence investigation summary explaining why this "
        f"transaction was flagged, suitable for a human analyst deciding whether "
        f"to approve or escalate further."
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_investigation_summary(proba: float, uncertainty: float, tier: str,
                                    contributions: dict, top_n: int = 5) -> str:
    """Main entry point -- routes to template or LLM mode based on config."""
    reasons = top_reasons(contributions, top_n=top_n)

    if config.LLM_PROVIDER == "anthropic":
        return _anthropic_summary(proba, uncertainty, tier, reasons)
    return _template_summary(proba, uncertainty, tier, reasons)


if __name__ == "__main__":
    from src.explain import build_explainer, explain_instance
    from src.imbalance import apply_strategy
    from src.models import train_xgboost
    from src.preprocess import run_preprocessing_pipeline
    from src.uncertainty import get_ensemble_disagreement
    from src.risk_triage import assign_risk_tier

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_preprocessing_pipeline()
    X_res, y_res = apply_strategy(config.DEFAULT_IMBALANCE_STRATEGY, X_train, y_train)

    model = train_xgboost(X_res, y_res)
    explainer = build_explainer(model, X_train)

    sample_row = X_val.iloc[[0]]
    proba = model.predict_proba(sample_row)[:, 1][0]
    contributions = explain_instance(explainer, sample_row)

    # fake a small uncertainty value for this single-model test
    uncertainty = 0.02
    tier = assign_risk_tier(proba, uncertainty)

    summary = generate_investigation_summary(proba, uncertainty, tier, contributions)
    print("\n=== Investigation Summary ===")
    print(summary)