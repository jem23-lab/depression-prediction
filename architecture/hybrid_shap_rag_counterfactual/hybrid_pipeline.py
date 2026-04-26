"""
hybrid_shap_rag_cf/hybrid_pipeline.py
────────────────────────────────────────────────────────────────────
Use Case 3: Hybrid SHAP + RAG + Counterfactual Explanation

This module runs all three explainability pipelines in parallel,
aggregates their outputs into a single rich context, and sends
EVERYTHING to Gemini in one unified prompt.

The three signals provide complementary perspectives:

  SHAP  → WHAT drove the prediction (token-level feature attribution)
           Answers: "Which words in your text signalled depression?"

  RAG   → WHY it matters clinically (PHQ-8 knowledge retrieval)
           Answers: "Which clinical symptoms match what you described?"

  CF    → HOW it could change (minimal counterfactual edits)
           Answers: "What would need to be different to shift the result?"

Combined, the LLM can produce a response that:
  1. Explains the prediction with word-level evidence (SHAP)
  2. Grounds it in clinical knowledge (RAG)
  3. Shows a concrete path forward (CF)
  4. Gives actionable, personalised suggestions tied to all three

Architecture:
  run_hybrid_pipeline()
      ├── explain_with_shap()          → SHAPResult
      ├── run_rag_pipeline()           → RAGResult
      └── generate_counterfactuals()   → CounterfactualResult
      └── build_hybrid_prompt()        → single fused prompt
      └── call_gemini()                → unified explanation
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.depression_model import (
    explain_with_shap, predict_proba, classify_severity,
    LABEL_MAP, LABEL_DESCRIPTIONS, SHAPResult,
)
from architecture.rag_explainer.rag_explainer    import run_rag_pipeline, RAGResult
from architecture.rag_explainer.rag_retriever    import format_retrieved_for_prompt
from architecture.shap_counterfactual_explainer.cf_generator import (
    generate_counterfactuals, CounterfactualResult,
)

logger = logging.getLogger("hybrid_pipeline")


@dataclass
class HybridResult:
    """Aggregated output from all three pipelines."""
    text:           str
    pred_label:     str
    pred_label_idx: int
    pred_probs:     np.ndarray
    severity_score: float = 0.0
    severity_reason: str  = ""

    shap_result:    Optional[SHAPResult]           = None
    rag_result:     Optional[RAGResult]            = None
    cf_result:      Optional[CounterfactualResult] = None

    shap_error:     str = ""
    rag_error:      str = ""
    cf_error:       str = ""


def run_hybrid_pipeline(
    user_text:     str,
    cf_candidates: int = 2,
    cf_attempts:   int = 1,
    rag_top_k:     int = 3,
) -> HybridResult:
    """
    Runs SHAP, RAG, CF on the raw user text.
    Each pipeline failure is caught independently so the others still run.
    """
    # Single prediction call — shared across all three pipelines
    probs                          = predict_proba([user_text])[0]
    pred_label, score, reason      = classify_severity(probs)
    pred_label_idx                 = int(np.argmax(probs))

    result = HybridResult(
        text            = user_text,
        pred_label      = pred_label,
        pred_label_idx  = pred_label_idx,
        pred_probs      = probs,
        severity_score  = score,
        severity_reason = reason,
    )

    logger.info(
        "Hybrid — prediction: %s (score=%.3f) — %s",
        pred_label, score, reason,
    )

    # SHAP — token-level attribution on raw text
    try:
        logger.info("Running SHAP…")
        result.shap_result = explain_with_shap(user_text)
        logger.info("SHAP done: %d risk tokens", len(result.shap_result.risk_tokens))
    except Exception as e:
        result.shap_error = str(e)
        logger.error("SHAP failed: %s", e)

    # RAG — PHQ-8 clinical knowledge retrieval
    try:
        logger.info("Running RAG…")
        result.rag_result = run_rag_pipeline(user_text, top_k=rag_top_k)
        logger.info("RAG done: %s", [d.symptom_name for d in result.rag_result.retrieved_docs])
    except Exception as e:
        result.rag_error = str(e)
        logger.error("RAG failed: %s", e)

    # Counterfactual — SHAP-guided minimal edits
    try:
        logger.info("Running Counterfactual…")
        result.cf_result = generate_counterfactuals(
            user_text,
            n_candidates=cf_candidates,
            n_attempts=cf_attempts,
        )
        valid = sum(1 for c in result.cf_result.candidates if c["flip_success"])
        logger.info("CF done: %d candidates, %d flips", len(result.cf_result.candidates), valid)
    except Exception as e:
        result.cf_error = str(e)
        logger.error("CF failed: %s", e)

    return result


def format_hybrid_debug(result: HybridResult) -> str:
    probs_str = ", ".join(f"{LABEL_MAP[i]}={result.pred_probs[i]:.3f}" for i in range(3))
    lines = [
        f"Hybrid prediction: {result.pred_label} (score={result.severity_score:.3f})",
        f"Raw probs        : {probs_str}",
        f"Reason           : {result.severity_reason}",
    ]
    if result.shap_result:
        tokens = [t["token"] for t in result.shap_result.risk_tokens[:4]]
        lines.append(f"SHAP risk tokens : {tokens}")
    else:
        lines.append(f"SHAP FAILED      : {result.shap_error}")
    if result.rag_result:
        syms = [d.symptom_name for d in result.rag_result.retrieved_docs]
        lines.append(f"RAG symptoms     : {syms}")
    else:
        lines.append(f"RAG FAILED       : {result.rag_error}")
    if result.cf_result:
        valid = sum(1 for c in result.cf_result.candidates if c["flip_success"])
        best  = result.cf_result.best_cf
        lines.append(f"CF candidates    : {len(result.cf_result.candidates)} ({valid} flips)")
        if best:
            lines.append(f"CF best          : [{best['label']}] '{best['text'][:60]}'")
    else:
        lines.append(f"CF FAILED        : {result.cf_error}")
    return "\n".join(lines)


def format_hybrid_telegram_preview(result: HybridResult) -> str:
    confidence   = result.pred_probs[result.pred_label_idx] * 100
    shap_tokens  = ", ".join(
        f"'{t['token']}'" for t in (result.shap_result.risk_tokens[:3] if result.shap_result else [])
    ) or "unavailable"
    rag_symptoms = ", ".join(
        d.symptom_name for d in (result.rag_result.retrieved_docs[:3] if result.rag_result else [])
    ) or "unavailable"

    if result.cf_result and result.cf_result.best_cf and result.cf_result.best_cf["flip_success"]:
        cf_status = f"label flip found (minimality={result.cf_result.best_cf['minimality']:.2f})"
    elif result.cf_result:
        cf_status = f"{len(result.cf_result.candidates)} candidates (no flip — strong boundary)"
    else:
        cf_status = "unavailable"

    return (
        f"Hybrid Analysis (SHAP + RAG + Counterfactual)\n\n"
        f"  Prediction  : {result.pred_label} ({confidence:.1f}%)\n"
        f"  Severity    : {result.severity_score:.2f}/1.0\n"
        f"  SHAP tokens : {shap_tokens}\n"
        f"  RAG matched : {rag_symptoms}\n"
        f"  CF status   : {cf_status}\n\n"
        "Generating unified explanation from all three signals..."
    )
