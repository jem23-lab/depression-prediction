"""
shap_explainer.py
────────────────────────────────────────────────────────────────────
Use Case 1: SHAP → structured prompt → Gemini → plain-English explanation.

Pipeline (matches notebook cell [29]):
  SHAPResult
      → build_shap_prompt()   builds a rich, structured prompt
      → generate_explanation() calls Gemini Flash
      → returns user-facing explanation string
"""

import google.generativeai as genai
import os
from depression_model import SHAPResult, LABEL_MAP, LABEL_DESCRIPTIONS

# ── Gemini config ───────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3-flash-preview"   # matches your notebook
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

SYSTEM_PROMPT = (
    "You are an empathetic, non-diagnostic mental health support assistant. "
    "You help users understand AI-generated assessments of depression risk in "
    "clear, warm, plain language. Never diagnose. Always recommend professional help."
)


def _init_gemini():
    """Configure Gemini client. Key is read from env GOOGLE_API_KEY."""
    key = os.environ.get("GOOGLE_API_KEY", GOOGLE_API_KEY)
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable not set.\n"
            "Set it with: export GOOGLE_API_KEY='your_key'"
        )
    genai.configure(api_key=key)
    return genai.GenerativeModel(GEMINI_MODEL)


# ── Prompt builder ──────────────────────────────────────────────────
def build_shap_prompt(
    user_query: str,
    shap_result: SHAPResult,
) -> str:
    """
    Constructs the detailed prompt combining:
      - User's original message
      - Model prediction + probabilities
      - SHAP token contributions (risk + protective)
      - Clinical notes for flagged tokens
    
    Mirrors notebook cell [29] but with richer structure.
    """

    # ── Probability block ───────────────────────────────────────────
    prob_lines = []
    for i, label in LABEL_MAP.items():
        bar  = "█" * int(shap_result.pred_probs[i] * 20)
        prob_lines.append(
            f"  {label:<18s} {bar:<20s} {shap_result.pred_probs[i]*100:.1f}%"
        )
    prob_block = "\n".join(prob_lines)

    # ── Risk tokens block ───────────────────────────────────────────
    risk_lines = []
    for t in shap_result.risk_tokens:
        note = f"  ← {t['note']}" if t["note"] else ""
        risk_lines.append(
            f"  '{t['token']}' (SHAP={t['shap']:+.4f}){note}"
        )
    risk_block = "\n".join(risk_lines) if risk_lines else "  None detected"

    # ── Protective tokens block ─────────────────────────────────────
    prot_lines = []
    for t in shap_result.protective_tokens:
        prot_lines.append(
            f"  '{t['token']}' (SHAP={t['shap']:+.4f})"
        )
    prot_block = "\n".join(prot_lines) if prot_lines else "  None detected"

    # ── Label meaning ───────────────────────────────────────────────
    label_meaning = LABEL_DESCRIPTIONS.get(shap_result.pred_label, "")

    prompt = f"""You are an empathetic explainer for an automatic depression screening tool.

─────────────────────────────────────────────────
USER'S ORIGINAL MESSAGE:
"{user_query}"

─────────────────────────────────────────────────
MODEL PREDICTION:
  Result     : {shap_result.pred_label.upper()}
  Meaning    : {label_meaning}

Probability distribution:
{prob_block}

─────────────────────────────────────────────────
SHAP TOKEN ANALYSIS:
SHAP (SHapley Additive exPlanations) measures exactly which words in the
text drove the model's prediction. Positive = pushes toward depression.

Words that INCREASED the depression signal:
{risk_block}

Words that REDUCED the depression signal:
{prot_block}

─────────────────────────────────────────────────
YOUR TASK — write a response that:

1. ACKNOWLEDGE the user's message warmly and without judgment.

2. EXPLAIN the prediction in plain language:
   - State the predicted level ({shap_result.pred_label}) and what it means
   - Mention 2–3 specific words from the text that influenced the prediction
     and WHY they matter clinically (use the notes above)
   - If protective words exist, point them out as positives

3. SUGGEST 2–3 concrete, evidence-based self-care actions tailored to
   the specific risk words identified (e.g., if 'tired' is flagged, suggest
   sleep hygiene; if 'lonely' is flagged, suggest social connection).

4. CLOSE with:
   - An empathetic reminder that this is an AI tool, NOT a clinical diagnosis
   - A gentle but clear encouragement to speak with a mental health professional
   - If the prediction is 'severe', add a crisis line reminder

Tone: warm, non-clinical, empowering. Length: 250–400 words.
Avoid bullet points in the opening paragraph.
"""
    return prompt


# ── LLM call ────────────────────────────────────────────────────────
def generate_explanation(
    user_query: str,
    shap_result: SHAPResult,
) -> str:
    """
    Calls Gemini with the SHAP-augmented prompt.
    Returns the natural-language explanation string.
    """
    gemini = _init_gemini()
    prompt = build_shap_prompt(user_query, shap_result)

    response = gemini.generate_content(
        contents=[
            {
                "role": "user",
                "parts": [f"{SYSTEM_PROMPT}\n\nUser query: {prompt}"],
            }
        ]
    )
    return response.text.strip() if response.text else ""


# ── Standalone test ─────────────────────────────────────────────────
if __name__ == "__main__":
    from depression_model import explain_with_shap, format_debug_summary

    text  = "I feel empty and tired every day. Nothing makes sense anymore."
    query = "I've been feeling really low lately. Can you help me understand what's happening?"

    print("Running SHAP …")
    result = explain_with_shap(text)
    print(format_debug_summary(result))

    print("\nGenerating Gemini explanation …\n")
    explanation = generate_explanation(user_query=query, shap_result=result)
    print("── LLM Explanation ──────────────────────────────────────────")
    print(explanation)
