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
    "You are an empathetic, non-diagnostic mental health support assistant. "
    "You specialise in explaining AI depression assessments through counterfactual "
    "reasoning — showing users what small changes could shift their result. "
    "Speak warmly, practically, and in plain language. Never diagnose. "
    "Always recommend professional support."
)


def build_cf_explanation_prompt(user_query: str, result: CounterfactualResult) -> str:
    """
    Builds a structured prompt for the CF explanation.
    Grounds the LLM in the actual counterfactual candidates found.
    """
    # Probability block
    prob_lines = []
    for i, label in LABEL_MAP.items():
        bar = "█" * int(result.original_probs[i] * 20)
        prob_lines.append(f"  {label:<18s} {bar:<20s} {result.original_probs[i]*100:.1f}%")
    prob_block = "\n".join(prob_lines)

    # Best CF block
    if result.best_cf and result.best_cf["flip_success"]:
        best = result.best_cf
        cf_block = (
            f"  Text          : \"{best['text']}\"\n"
            f"  New prediction: {best['label'].upper()} "
            f"({best['probs'][int(list(LABEL_MAP.values()).index(best['label']))] * 100:.1f}% confidence)\n"
            f"  Minimality    : {best['minimality']:.2f} (1.0 = no words changed)\n"
            f"  Meaning kept  : {best['semantic_sim']:.2f} (1.0 = identical meaning)\n"
        )
        flip_achieved = True
    elif result.best_cf:
        best = result.best_cf
        cf_block = (
            f"  Text             : \"{best['text']}\"\n"
            f"  Prediction still : {best['label'].upper()} (label did not flip, but moved)\n"
            f"  Note: The model's decision boundary is strong — larger changes may be needed.\n"
        )
        flip_achieved = False
    else:
        cf_block = "  No valid counterfactual was generated."
        flip_achieved = False
        best = None

    # All candidates (show top 3)
    candidates_block = ""
    if result.candidates:
        lines = []
        for i, c in enumerate(result.candidates[:3], 1):
            status = "FLIP" if c["flip_success"] else "no flip"
            lines.append(
                f"  {i}. [{status}] [{c['label']}] \"{c['text'][:90]}...\"\n"
                f"     minimality={c['minimality']:.2f}  meaning_kept={c['semantic_sim']:.2f}"
            )
        candidates_block = "\n".join(lines)

    # SHAP-guided tokens used
    token_names = [t["token"] for t in result.shap_guided_tokens[:4]]

    # Label descriptions
    orig_desc   = LABEL_DESCRIPTIONS.get(result.original_label, "")
    target_desc = LABEL_DESCRIPTIONS.get(result.target_label, "")

    return f"""You are explaining a counterfactual AI depression assessment to a user.

─────────────────────────────────────────────────
USER'S ORIGINAL MESSAGE:
"{user_query}"

─────────────────────────────────────────────────
CURRENT MODEL PREDICTION: {result.original_label.upper()}
{orig_desc}

Probability distribution:
{prob_block}

WORDS THAT MOST DROVE THIS PREDICTION (from SHAP):
{", ".join(token_names) if token_names else "No specific tokens identified"}

─────────────────────────────────────────────────
COUNTERFACTUAL ANALYSIS (what would change the prediction):
Target label: {result.target_label.upper()}
{target_desc}

Best counterfactual found:
{cf_block}

All candidates generated:
{candidates_block}

Label flip achieved: {"YES" if flip_achieved else "NO — model boundary is strong"}

─────────────────────────────────────────────────
YOUR TASK — write a warm, clear explanation that:

1. ACKNOWLEDGE the user's message gently. Validate that what they're feeling is real.

2. EXPLAIN the prediction in 2-3 sentences:
   - What "{result.original_label}" means in plain words
   - Which specific words (from SHAP tokens) signalled this to the model

3. COUNTERFACTUAL INSIGHT — this is the core of the explanation:
   - Show the best counterfactual: "If you had written X instead of Y,
     the model would have predicted [target]"
   - Explain WHY that small change matters clinically
     (e.g., "changing 'I never go out' to 'I sometimes go out' suggests
     the anhedonia pattern is less severe")
   - {"Mention that the label DID flip, which shows the model's boundary is sensitive to these words." if flip_achieved else "Note that even though the label didn't flip, the wording shift shows movement toward recovery language."}

4. ACTIONABLE BRIDGE — connect the CF to real life:
   - "The counterfactual suggests that if you could [specific behaviour],
     the pattern the AI detected would weaken"
   - Give 2 concrete, evidence-based steps tied directly to the CF change
     (e.g., if CF changed social isolation language, suggest one small social
     activity; if it changed fatigue language, suggest sleep hygiene)

5. CLOSE:
   - Remind them this is an AI tool, not a diagnosis
   - Encourage speaking to a mental health professional
   - {"Add a crisis helpline reminder." if result.original_label == "severe" else "End with an encouraging, empowering note."}

Tone: warm, practical, empowering — not clinical. Length: 300-420 words.
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
