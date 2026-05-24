"""
hybrid_shap_rag_cf/hybrid_explainer.py
────────────────────────────────────────────────────────────────────
Use Case 3: Unified prompt builder that fuses SHAP + RAG + CF signals
into one structured context block and sends it to Gemini in ONE call.

Evidence layers sent to the LLM:

  Layer 1 — SHAP  : which tokens drove the prediction and how strongly
  Layer 2 — RAG   : which PHQ-8 clinical symptoms match the user's text
  Layer 3 — CF    : what minimal text edit would shift the label

The LLM is instructed to SYNTHESISE — not section-by-section, but as one
flowing narrative where each layer informs the others:
  "The word 'empty' [SHAP] aligns with Depressed Mood [RAG].
   Replacing it with 'a bit flat' [CF] would shift the prediction,
   suggesting the intensity marker 'empty' is what the model latches on to."
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.depression_model         import LABEL_MAP, LABEL_DESCRIPTIONS
from shared.llm_client               import call_gemini
from architecture.rag_explainer.rag_retriever     import format_retrieved_for_prompt
# Correct package-qualified import — works whether called from root or subfolder
from architecture.hybrid_shap_rag_counterfactual.hybrid_pipeline import HybridResult

logger = logging.getLogger("hybrid_explainer")

SYSTEM_PROMPT = (
    "You explain AI assessment factors in plain, everyday language. "
    "Focus only on what in the user's text influenced the result. "
    "Avoid jargon, avoid scores/percentages, and do not give advice."
)


# ── Prompt builder ────────────────────────────────────────────────────
def build_hybrid_prompt(user_query: str, result: HybridResult) -> str:
    """
    Assembles the full fused prompt from all three XAI signals.
    Layers that errored are clearly marked [unavailable] so the LLM
    can degrade gracefully rather than hallucinating missing data.
    """

    # ── Probability bar chart ─────────────────────────────────────────
    prob_lines = []
    for i, label in LABEL_MAP.items():
        bar = "█" * int(result.pred_probs[i] * 20)
        prob_lines.append(f"  {label:<18s} {bar:<20s} {result.pred_probs[i]*100:.1f}%")
    prob_block    = "\n".join(prob_lines)
    label_meaning = LABEL_DESCRIPTIONS.get(result.pred_label, "")

    # ── Layer 1: SHAP ─────────────────────────────────────────────────
    if result.shap_result and not result.shap_error:
        sr         = result.shap_result
        risk_lines = []
        for t in sr.risk_tokens[:5]:
            risk_lines.append(
                f"    '{t['token']}' SHAP={t['shap']:+.4f} — {t['direction']}"
            )
        prot_lines = [
            f"    '{t['token']}' SHAP={t['shap']:+.4f} — {t['direction']}"
            for t in sr.protective_tokens[:3]
        ]
        shap_block = (
            "  Risk tokens (increase depression signal):\n"
            + ("\n".join(risk_lines) if risk_lines else "    None")
            + "\n\n  Protective tokens (reduce depression signal):\n"
            + ("\n".join(prot_lines) if prot_lines else "    None")
        )
    else:
        shap_block = f"  [SHAP unavailable: {result.shap_error}]"

    # ── Layer 2: RAG ──────────────────────────────────────────────────
    if result.rag_result and not result.rag_error:
        rr            = result.rag_result
        symptom_names = [d.symptom_name for d in rr.retrieved_docs]
        rag_block     = (
            f"  Matched PHQ-8 symptoms: {', '.join(symptom_names)}\n\n"
            + format_retrieved_for_prompt(rr.retrieved_docs)
        )
    else:
        rag_block = f"  [RAG unavailable: {result.rag_error}]"

    # ── Layer 3: Counterfactual ───────────────────────────────────────
    if result.cf_result and not result.cf_error:
        cr    = result.cf_result
        best  = cr.best_cf
        valid = [c for c in cr.candidates if c["flip_success"]]

        if best and best["flip_success"]:
            label_idx = list(LABEL_MAP.values()).index(best["label"])
            cf_block = (
                f"  Target label  : {cr.target_label.upper()}\n"
                f"  Flip achieved : YES ({len(valid)}/{len(cr.candidates)} candidates)\n\n"
                f"  Best counterfactual:\n"
                f"    Original  : \"{result.text}\"\n"
                f"    Modified  : \"{best['text']}\"\n"
                f"    New label : {best['label'].upper()} "
                f"({best['probs'][label_idx]*100:.1f}%)\n"
                f"    Minimality   : {best['minimality']:.2f}  (1.0 = no words changed)\n"
                f"    Meaning kept : {best['semantic_sim']:.2f}  (1.0 = identical meaning)"
            )
        elif best:
            cf_block = (
                f"  Target label  : {cr.target_label.upper()}\n"
                f"  Flip achieved : NO (strong decision boundary)\n\n"
                f"  Closest candidate (movement toward target):\n"
                f"    Modified  : \"{best['text']}\"\n"
                f"    Label     : {best['label'].upper()}\n"
                f"    Minimality   : {best['minimality']:.2f}\n"
                f"    Meaning kept : {best['semantic_sim']:.2f}"
            )
        else:
            cf_block = "  [No counterfactual candidates generated]"

        # Append all candidates for context
        if cr.candidates:
            cf_block += "\n\n  All candidates:\n"
            for i, c in enumerate(cr.candidates[:3], 1):
                status    = "FLIP" if c["flip_success"] else "no flip"
                cf_block += (
                    f"    {i}. [{status}] [{c['label']}] "
                    f"min={c['minimality']:.2f} sim={c['semantic_sim']:.2f}\n"
                    f"       \"{c['text'][:90]}...\"\n"
                )
    else:
        cf_block = f"  [Counterfactual unavailable: {result.cf_error}]"

    # ── Synthesis instruction ─────────────────────────────────────────
    available = []
    if not result.shap_error:  available.append("SHAP")
    if not result.rag_error:   available.append("RAG")
    if not result.cf_error:    available.append("Counterfactual")

    if len(available) == 3:
        synth = (
            "All three XAI signals are available. Write ONE narrative that weaves them together:\n"
            "  - Connect each SHAP token to a matching RAG symptom by name\n"
            "  - Show the CF as a concrete bridge: 'if you had said X instead of Y, the model "
            "would see less of the [RAG symptom] pattern'\n"
            "  - Do NOT write section headers like 'SHAP says...' or 'RAG shows...'"
        )
    elif len(available) == 2:
        synth = f"Two signals available ({' and '.join(available)}). Connect them into one narrative."
    elif len(available) == 1:
        synth = f"Only {available[0]} is available. Base the explanation solely on it."
    else:
        synth = "All signals failed. Provide a general empathetic response based on the prediction only."

    flip_achieved = (
        result.cf_result is not None
        and result.cf_result.best_cf is not None
        and result.cf_result.best_cf["flip_success"]
    )

    cf_note = (
        "Note that the label DID flip — tell the user this means the model is "
        "sensitive to those exact words, which is an encouraging finding."
        if flip_achieved else
        "Even without a label flip, the counterfactual shows directional movement. "
        "Explain that the boundary is strong but change is detectable."
    )

    crisis_note = (
        "Include a crisis helpline reminder (e.g., 988 Suicide & Crisis Lifeline in the US)."
        if result.pred_label == "severe" else
        "End with an empowering, forward-looking sentence."
    )

    # ── Assemble full prompt ──────────────────────────────────────────
    prompt = f"""You are explaining an AI depression screening result using several sources of evidence.

USER'S MESSAGE:
"{user_query}"

MODEL PREDICTION: {result.pred_label.upper()}
{label_meaning}

EVIDENCE (for you):
- Key words/phrases that influenced the result:
{shap_block}

- Related themes from knowledge base:
{rag_block}

- Example wording change that could shift the result:
{cf_block}

WRITE ONE SHORT, USER-FRIENDLY EXPLANATION that:
1. States the predicted level in plain words.
2. Mentions only 2-3 key words/phrases and explains why they mattered.
3. Includes one simple "if the message said X instead of Y" example.
4. Keeps the focus on explanation of factors, not advice.

Constraints:
- If the explanation feels long, split it into 2 short paragraphs.
- Highlight the 2-3 key words/phrases by wrapping them in **double asterisks**.
- Do NOT mention SHAP, RAG, counterfactual, probabilities, or scores.
- Do NOT include self-care tips, support suggestions, or disclaimers.
- Length: 110-160 words.
"""
    return prompt


# ── Gemini call ───────────────────────────────────────────────────────
def generate_hybrid_explanation(user_query: str, result: HybridResult) -> str:
    """Calls Gemini with the fused three-signal prompt. Returns explanation string."""
    prompt = build_hybrid_prompt(user_query, result)
    logger.info("Sending hybrid prompt to Gemini (%d chars)", len(prompt))
    return call_gemini(prompt, system=SYSTEM_PROMPT)


# ── Standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    from architecture.hybrid_shap_rag_counterfactual.hybrid_pipeline import run_hybrid_pipeline, format_hybrid_debug

    text  = "I feel empty and tired every day. I don't go out much anymore and I've lost interest in my hobbies."
    query = "Can you help me understand what's going on with me?"

    print("Running hybrid pipeline (SHAP + RAG + CF)...")
    result = run_hybrid_pipeline(text, cf_candidates=2, cf_attempts=1)
    print(format_hybrid_debug(result))

    print("\nGenerating unified Gemini explanation...\n")
    explanation = generate_hybrid_explanation(query, result)
    print("── Hybrid Explanation ───────────────────────────────────────")
    print(explanation)
