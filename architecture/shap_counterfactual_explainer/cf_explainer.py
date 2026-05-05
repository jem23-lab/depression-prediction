"""
counterfactual_explainer/cf_explainer.py
────────────────────────────────────────────────────────────────────
Use Case 4: Counterfactual explanation narrator.

Takes a CounterfactualResult and generates a user-facing explanation
that answers:
  "What would need to change in what I described for the AI to see
   me as less depressed?"

This is the key clinical value of CFs per UbiComp '24 (Gyuwon Jung et al.):
counterfactuals provide ACTIONABLE COPING STRATEGIES tied to concrete
language patterns — not just "you scored X", but "if you described X
differently, it would suggest Y changed".

The explanation covers:
  1. What the model predicted and why (brief, referencing SHAP tokens)
  2. The counterfactual — what minimal change shifts the prediction
  3. What that change means clinically (connecting CF words to PHQ-8 symptoms)
  4. Actionable next steps the user can actually take
  5. Empathetic close + professional help reminder
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.llm_client import call_gemini
from shared.depression_model import LABEL_MAP, LABEL_DESCRIPTIONS
from architecture.shap_counterfactual_explainer.cf_generator import CounterfactualResult

import logging
logger = logging.getLogger("cf_explainer")

SYSTEM_PROMPT = (
    "You explain AI assessment factors in plain, everyday language. "
    "Focus only on what in the user's text influenced the result. "
    "Avoid jargon, avoid scores/percentages, and do not give advice."
)


def build_cf_explanation_prompt(user_query: str, result: CounterfactualResult) -> str:
    """
    Builds a structured prompt for the CF explanation.
    Grounds the LLM in the actual counterfactual candidates found.
    """
    # Best CF block
    if result.best_cf and result.best_cf["flip_success"]:
        best = result.best_cf
        cf_block = (
            f"  Original: \"{user_query}\"\n"
            f"  If changed to: \"{best['text']}\"\n"
            f"  New prediction: {best['label'].upper()}"
        )
        flip_achieved = True
    elif result.best_cf:
        best = result.best_cf
        cf_block = (
            f"  Closest change: \"{best['text']}\"\n"
            f"  Prediction still: {best['label'].upper()} (shifted, but did not flip)"
        )
        flip_achieved = False
    else:
        cf_block = "  No valid counterfactual was generated."
        flip_achieved = False
        best = None

    # SHAP-guided tokens used
    token_names = [t["token"] for t in result.shap_guided_tokens[:4]]

    # Label descriptions
    orig_desc   = LABEL_DESCRIPTIONS.get(result.original_label, "")
    target_desc = LABEL_DESCRIPTIONS.get(result.target_label, "")

    return f"""You are explaining an AI depression screening result to a user.

USER'S ORIGINAL MESSAGE:
"{user_query}"

CURRENT MODEL PREDICTION: {result.original_label.upper()}
{orig_desc}

KEY WORDS THAT MOST INFLUENCED THE RESULT:
{', '.join(token_names) if token_names else 'No specific words identified'}

WHAT CHANGE COULD SHIFT THE RESULT:
Target label: {result.target_label.upper()}
{target_desc}

Best example change:
{cf_block}

YOUR TASK — write a short, user-friendly response that:
1. States the predicted level in plain words.
2. Mentions 2-3 key words/phrases that influenced the result and why.
3. Gives one simple “if the message said X instead of Y” example.
4. Keeps the focus on explanation of factors, not advice.

Constraints:
- Write ONE paragraph only (no lists or bullet points).
- Highlight the 2-3 key words/phrases by wrapping them in double quotes.
- Do NOT mention "counterfactual", SHAP, scores, probabilities, or technical terms.
- Do NOT include self-care tips, support suggestions, or disclaimers.
- Length: 110-160 words.
"""


def generate_cf_explanation(user_query: str, result: CounterfactualResult) -> str:
    """Calls Gemini with the CF-augmented prompt."""
    prompt = build_cf_explanation_prompt(user_query, result)
    return call_gemini(prompt, system=SYSTEM_PROMPT)


def format_cf_telegram_preview(result: CounterfactualResult) -> str:
    """
    Short preview message sent to Telegram while Gemini processes.
    """
    label      = result.original_label
    target     = result.target_label
    confidence = result.original_probs[result.original_probs.argmax()] * 100
    n_valid    = sum(1 for c in result.candidates if c["flip_success"])
    n_total    = len(result.candidates)

    if result.best_cf and result.best_cf["flip_success"]:
        flip_status = f"Label flip achieved in {n_valid}/{n_total} candidates"
    else:
        flip_status = f"No label flip (strong decision boundary) — {n_total} candidates analysed"

    token_str = ", ".join(f"'{t['token']}'" for t in result.shap_guided_tokens[:3])

    return (
        f"Counterfactual Analysis\n"
        f"  Current level    : {label} ({confidence:.1f}%)\n"
        f"  Target level     : {target}\n"
        f"  Key SHAP tokens  : {token_str}\n"
        f"  {flip_status}\n\n"
        "Generating full explanation..."
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from cf_generator import generate_counterfactuals, format_cf_debug

    text  = "I feel empty and tired every day. Nothing makes sense anymore."
    query = "Can you help me understand what I could do differently?"

    print("Generating counterfactuals...")
    result = generate_counterfactuals(text)
    print(format_cf_debug(result))

    print("\nGenerating Gemini CF explanation...\n")
    explanation = generate_cf_explanation(query, result)
    print("── CF Explanation ───────────────────────────────────────────")
    print(explanation)
