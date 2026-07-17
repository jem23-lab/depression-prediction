"""
shap_explainer/shap_explainer.py
────────────────────────────────────────────────────────────────────
Use Case 1: SHAP-only explanation.
Refactored to use shared/depression_model.py and shared/llm_client.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.depression_model import SHAPResult, LABEL_MAP, LABEL_DESCRIPTIONS
from shared.llm_client       import call_gemini

SYSTEM_PROMPT = (
    "You explain AI assessment factors in plain, everyday language. "
    "Focus only on what in the user's text influenced the result. "
    "Avoid jargon, avoid scores/percentages, and do not give advice."
)


def build_shap_prompt(user_query: str, result: SHAPResult) -> str:
    # Risk tokens
    risk_lines = []
    for t in result.risk_tokens:
        risk_lines.append(f"  '{t['token']}'")
    risk_block  = "\n".join(risk_lines) if risk_lines else "  None detected"

    # Protective tokens
    prot_lines  = [f"  '{t['token']}'" for t in result.protective_tokens]
    prot_block  = "\n".join(prot_lines) if prot_lines else "  None detected"

    label_meaning = LABEL_DESCRIPTIONS.get(result.pred_label, "")

    return f"""You are explaining an AI depression screening result to a user.

USER'S MESSAGE:
"{user_query}"

MODEL PREDICTION:
  Result  : {result.pred_label.upper()}
  Meaning : {label_meaning}

KEY WORDS THAT RAISED THE RISK SIGNAL:
{risk_block}

KEY WORDS THAT LOWERED THE RISK SIGNAL:
{prot_block}

YOUR TASK — write a short, user-friendly response that:
1. States the predicted level in plain words.
2. Mentions only 2-3 key words/phrases from the message and why they mattered.
3. Keeps the focus on explanation of factors, not advice.

Constraints:
- Prefer meaningful phrases or content words. Do not explain generic words like "feel",
  "currently", "could", "couldn't", "thing", or "very" as standalone evidence.
- If the explanation feels long, split it into 2 short paragraphs.
- Highlight the 2-3 key words/phrases by wrapping them in **double asterisks**.
- Do NOT mention SHAP, probabilities, scores, or technical terms.
- Do NOT include self-care tips, support suggestions, or disclaimers.
- Length: 90-130 words.
"""


def generate_shap_explanation(user_query: str, result: SHAPResult) -> str:
    """Calls Gemini with the SHAP-augmented prompt."""
    return call_gemini(build_shap_prompt(user_query, result), system=SYSTEM_PROMPT)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from shared.depression_model import explain_with_shap, format_debug

    text  = "I feel empty and tired every day. Nothing makes sense anymore."
    query = "Can you help me understand how I'm feeling?"

    result = explain_with_shap(text)
    print(format_debug(result))
    print("\n── SHAP Explanation ──────────────────────────────────────────")
    print(generate_shap_explanation(query, result))
