"""
rag_explainer/rag_explainer.py
────────────────────────────────────────────────────────────────────
Use Case 2: RAG-only explanation.

Pipeline:
  user_text
      → predict_proba()           (deproberta — get label + probs)
      → retrieve()                (FAISS search over PHQ-8 KB)
      → build_rag_prompt()        (assemble structured prompt)
      → call_gemini()             (Gemini Flash)
      → user-facing explanation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.depression_model import predict_proba, reframe_text, LABEL_MAP, LABEL_DESCRIPTIONS
from shared.llm_client       import call_gemini
from architecture.rag_explainer.rag_retriever import retrieve, format_retrieved_for_prompt, RetrievedDoc

import numpy as np
import logging
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger("rag_explainer")

SYSTEM_PROMPT = (
    "You are an empathetic, non-diagnostic mental health support assistant. "
    "You help users understand AI-generated depression assessments using "
    "retrieved clinical knowledge. Speak warmly and in plain language. "
    "Never diagnose. Always recommend professional support."
)


@dataclass
class RAGResult:
    """All artefacts from the RAG pipeline."""
    text:            str
    model_input:     str
    pred_label:      str
    pred_label_idx:  int
    pred_probs:      np.ndarray
    retrieved_docs:  List[RetrievedDoc]  = field(default_factory=list)
    was_reframed:    bool                = False


def run_rag_pipeline(user_text: str, top_k: int = 3) -> RAGResult:
    """
    Runs prediction + RAG retrieval. Returns a RAGResult ready for
    explanation generation.
    """
    model_input  = reframe_text(user_text)
    was_reframed = model_input != user_text

    pred_probs     = predict_proba([model_input])[0]
    pred_label_idx = int(np.argmax(pred_probs))
    pred_label     = LABEL_MAP[pred_label_idx]

    logger.info("RAG prediction: %s (%.1f%%)", pred_label, pred_probs[pred_label_idx]*100)

    # Retrieve using original user text for natural-language matching
    # (not the reframed version, which is more clinical)
    retrieved = retrieve(user_text, top_k=top_k)
    logger.info("Retrieved symptoms: %s", [d.symptom_name for d in retrieved])

    return RAGResult(
        text           = user_text,
        model_input    = model_input,
        was_reframed   = was_reframed,
        pred_label     = pred_label,
        pred_label_idx = pred_label_idx,
        pred_probs     = pred_probs,
        retrieved_docs = retrieved,
    )


def build_rag_prompt(user_query: str, result: RAGResult) -> str:
    """
    Builds a structured prompt combining model prediction, retrieved
    PHQ-8 clinical knowledge, and the original user text.
    """
    # Probability block
    prob_lines = []
    for i, label in LABEL_MAP.items():
        bar = "█" * int(result.pred_probs[i] * 20)
        prob_lines.append(f"  {label:<18s} {bar:<20s} {result.pred_probs[i]*100:.1f}%")
    prob_block = "\n".join(prob_lines)

    # Retrieved knowledge
    knowledge_block = format_retrieved_for_prompt(result.retrieved_docs)

    # Label meaning
    label_meaning = LABEL_DESCRIPTIONS.get(result.pred_label, "")

    # Which symptoms were retrieved
    symptom_names = [d.symptom_name for d in result.retrieved_docs]

    prompt = f"""You are explaining an AI depression screening result to a user.

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
RETRIEVED CLINICAL KNOWLEDGE (from PHQ-8 database):
The following depression symptom entries were retrieved because they most
closely match what the user described. Use this knowledge to ground your explanation.

{knowledge_block}

─────────────────────────────────────────────────
MATCHED SYMPTOMS: {", ".join(symptom_names)}

─────────────────────────────────────────────────
YOUR TASK — write a response that:

1. ACKNOWLEDGE the user's message warmly and without judgment.

2. EXPLAIN the prediction in plain language:
   - State the predicted level ({result.pred_label}) and what it means
   - Connect specific phrases from the user's message to the matching
     PHQ-8 symptoms above (e.g., "When you said '...', this matches the
     clinical pattern of [Symptom Name]")
   - Briefly explain what each matched symptom means clinically
     (use the definitions above, but in plain language)

3. VALIDATE: Acknowledge that what the user is experiencing is real and recognised.

4. SUGGEST 2–3 evidence-based self-care actions specifically tailored to
   the matched symptoms (e.g., for Anhedonia: behavioural activation;
   for Sleep Problems: sleep hygiene; for Fatigue: pacing strategies).

5. CLOSE with:
   - An empathetic reminder this is an AI assessment, not a clinical diagnosis
   - Clear encouragement to speak with a mental health professional
   - If the prediction is 'severe', add a crisis helpline reminder

Tone: warm, validating, non-clinical. Length: 250–400 words.
"""
    return prompt


def generate_rag_explanation(user_query: str, result: RAGResult) -> str:
    """
    Calls Gemini with the RAG-augmented prompt.
    Returns the user-facing explanation string.
    """
    prompt = build_rag_prompt(user_query, result)
    return call_gemini(prompt, system=SYSTEM_PROMPT)


def format_rag_debug(result: RAGResult) -> str:
    probs = ", ".join(f"{LABEL_MAP[i]}={result.pred_probs[i]:.3f}" for i in range(3))
    lines = [
        f"Prediction: {result.pred_label} ({probs})",
        "Retrieved:",
    ]
    for d in result.retrieved_docs:
        lines.append(f"  [{d.distance:.3f}] {d.symptom_name} ({d.symptom_type})")
    return "\n".join(lines)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    text  = "I don't really go out much anymore. I've lost interest in my hobbies, and I find it hard to concentrate at work."
    query = "Can you help me understand what's going on with me?"

    print("Running RAG pipeline…")
    result = run_rag_pipeline(text)
    print(format_rag_debug(result))

    print("\nGenerating Gemini explanation…\n")
    explanation = generate_rag_explanation(query, result)
    print("── RAG Explanation ──────────────────────────────────────────")
    print(explanation)
