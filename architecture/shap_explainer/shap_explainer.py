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
    "You are an empathetic, non-diagnostic mental health support assistant. "
    "You help users understand AI-generated depression assessments using "
    "SHAP token-level explanations. Speak warmly and in plain language. "
    "Never diagnose. Always recommend professional support."
)

CLINICAL_TOKEN_NOTES = {
    "hopeless":    "strongly linked to depressive hopelessness",
    "empty":       "indicates emotional numbness, a hallmark of depression",
    "tired":       "persistent fatigue is a core depression symptom",
    "worthless":   "feelings of worthlessness are a key clinical indicator",
    "nothing":     "anhedonia and nihilism pattern",
    "anymore":     "implies loss of a previous positive state",
    "death":       "may indicate passive suicidal ideation",
    "suicide":     "requires immediate clinical attention",
    "sad":         "direct expression of low mood",
    "lonely":      "social withdrawal and isolation signal",
    "numb":        "emotional blunting associated with depression",
    "concentrate": "concentration difficulty is a cognitive depression symptom",
    "appetite":    "appetite change is a somatic depression symptom",
}


def build_shap_prompt(user_query: str, result: SHAPResult) -> str:
    # Probability block
    prob_lines = []
    for i, label in LABEL_MAP.items():
        bar = "█" * int(result.pred_probs[i] * 20)
        prob_lines.append(f"  {label:<18s} {bar:<20s} {result.pred_probs[i]*100:.1f}%")
    prob_block = "\n".join(prob_lines)

    # Risk tokens
    risk_lines = []
    for t in result.risk_tokens:
        note = f"  <- {t['note']}" if t["note"] else ""
        risk_lines.append(f"  '{t['token']}' (SHAP={t['shap']:+.4f}){note}")
    risk_block  = "\n".join(risk_lines) if risk_lines else "  None detected"

    # Protective tokens
    prot_lines  = [f"  '{t['token']}' (SHAP={t['shap']:+.4f})" for t in result.protective_tokens]
    prot_block  = "\n".join(prot_lines) if prot_lines else "  None detected"

    label_meaning = LABEL_DESCRIPTIONS.get(result.pred_label, "")

    return f"""You are explaining an AI depression screening result to a user.

─────────────────────────────────────────────────
USER'S MESSAGE:
"{user_query}"

─────────────────────────────────────────────────
MODEL PREDICTION:
  Result  : {result.pred_label.upper()}
  Meaning : {label_meaning}

Probability distribution:
{prob_block}

─────────────────────────────────────────────────
SHAP TOKEN ANALYSIS:
SHAP measures which words drove the prediction. Positive = pushes toward depression.

Words that INCREASED the depression signal:
{risk_block}

Words that REDUCED the depression signal:
{prot_block}

─────────────────────────────────────────────────
YOUR TASK — write a response that:

1. ACKNOWLEDGE the user's message warmly.
2. EXPLAIN the prediction: state the level and mention 2-3 specific words
   from the text that influenced it and why they matter clinically.
3. SUGGEST 2-3 concrete, tailored self-care actions based on the risk words.
4. CLOSE with an empathetic reminder this is an AI assessment, not a diagnosis,
   and encourage professional help. Add a crisis line reminder if level is 'severe'.

Tone: warm, non-clinical, empowering. Length: 250-400 words.
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
