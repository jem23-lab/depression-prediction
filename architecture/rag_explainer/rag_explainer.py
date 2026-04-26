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

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List

from shared.depression_model import (
    predict_proba, classify_severity,
    LABEL_MAP, LABEL_DESCRIPTIONS,
)
from shared.llm_client       import call_gemini
from architecture.rag_explainer.rag_retriever           import retrieve, format_retrieved_for_prompt, RetrievedDoc

logger = logging.getLogger("rag_explainer")

SYSTEM_PROMPT = (
    "You are an empathetic, non-diagnostic mental health support assistant. "
    "You help users understand AI-generated depression assessments using "
    "retrieved clinical knowledge. Speak warmly and in plain language. "
    "Never diagnose. Always recommend professional support."
)


@dataclass
class RAGResult:
    text:            str
    pred_label:      str
    pred_label_idx:  int
    pred_probs:      np.ndarray
    severity_score:  float         = 0.0
    severity_reason: str           = ""
    retrieved_docs:  List[RetrievedDoc] = field(default_factory=list)


def run_rag_pipeline(user_text: str, top_k: int = 3) -> RAGResult:
    """Prediction on raw text + PHQ-8 retrieval. No reframing."""
    pred_probs                  = predict_proba([user_text])[0]
    pred_label, score, reason   = classify_severity(pred_probs)
    pred_label_idx              = int(np.argmax(pred_probs))

    logger.info("RAG prediction: %s (score=%.3f) — %s", pred_label, score, reason)

    retrieved = retrieve(user_text, top_k=top_k)
    logger.info("Retrieved: %s", [d.symptom_name for d in retrieved])

    return RAGResult(
        text            = user_text,
        pred_label      = pred_label,
        pred_label_idx  = pred_label_idx,
        pred_probs      = pred_probs,
        severity_score  = score,
        severity_reason = reason,
        retrieved_docs  = retrieved,
    )


def build_rag_prompt(user_query: str, result: RAGResult) -> str:
    prob_lines = []
    for i, label in LABEL_MAP.items():
        bar = "█" * int(result.pred_probs[i] * 20)
        prob_lines.append(f"  {label:<18s} {bar:<20s} {result.pred_probs[i]*100:.1f}%")
    prob_block    = "\n".join(prob_lines)
    label_meaning = LABEL_DESCRIPTIONS.get(result.pred_label, "")
    symptom_names = [d.symptom_name for d in result.retrieved_docs]
    knowledge     = format_retrieved_for_prompt(result.retrieved_docs)

    return f"""You are explaining an AI depression screening result to a user.

USER'S MESSAGE:
"{user_query}"

MODEL PREDICTION: {result.pred_label.upper()}
{label_meaning}
Severity score: {result.severity_score:.2f} / 1.0  ({result.severity_reason})

Probability distribution:
{prob_block}

RETRIEVED CLINICAL KNOWLEDGE (PHQ-8):
Matched symptoms: {', '.join(symptom_names)}

{knowledge}

YOUR TASK:
1. Acknowledge warmly and validate the user's experience.
2. Explain the prediction — connect specific phrases from their message to
   the matched PHQ-8 symptoms by name, in plain language.
3. Validate that what they describe is real and clinically recognised.
4. Give 2-3 concrete, evidence-based suggestions tied to the matched symptoms.
5. Close with a reminder this is an AI tool not a diagnosis, and encourage
   professional support. If prediction is 'severe', add crisis resources.

Tone: warm, validating, non-clinical. Length: 250-400 words.
"""


def generate_rag_explanation(user_query: str, result: RAGResult) -> str:
    return call_gemini(build_rag_prompt(user_query, result), system=SYSTEM_PROMPT)


def format_rag_debug(result: RAGResult) -> str:
    probs = ", ".join(f"{LABEL_MAP[i]}={result.pred_probs[i]:.3f}" for i in range(3))
    lines = [
        f"Prediction : {result.pred_label} (score={result.severity_score:.3f})",
        f"Raw probs  : {probs}",
        f"Reason     : {result.severity_reason}",
        "Retrieved  :",
    ]
    for d in result.retrieved_docs:
        lines.append(f"  [{d.distance:.3f}] {d.symptom_name} ({d.symptom_type})")
    return "\n".join(lines)
