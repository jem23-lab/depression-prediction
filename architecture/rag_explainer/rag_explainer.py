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
    "You explain AI assessment factors in plain, everyday language. "
    "Focus only on what in the user's text influenced the result. "
    "Avoid jargon, avoid scores/percentages, and do not give advice."
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
    label_meaning = LABEL_DESCRIPTIONS.get(result.pred_label, "")
    symptom_names = [d.symptom_name for d in result.retrieved_docs]
    knowledge     = format_retrieved_for_prompt(result.retrieved_docs)

    return f"""You are explaining an AI depression screening result to a user.

USER'S MESSAGE:
"{user_query}"

MODEL PREDICTION: {result.pred_label.upper()}
{label_meaning}

MATCHED CLINICAL THEMES (from PHQ-8):
{', '.join(symptom_names)}

REFERENCE NOTES (for you):
{knowledge}

YOUR TASK — write a short, user-friendly response that:
1. States the predicted level in plain words.
2. Connects 2-3 phrases from the message to simple, everyday descriptions
   of the matched themes (avoid clinical jargon).
3. Keeps the focus on explanation of factors, not advice.

Constraints:
- Write ONE paragraph only (no lists or bullet points).
- Highlight the 2-3 key phrases by wrapping them in double quotes.
- Do NOT mention RAG, PHQ-8, probabilities, or scores.
- Do NOT include self-care tips, support suggestions, or disclaimers.
- Length: 90-130 words.
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
