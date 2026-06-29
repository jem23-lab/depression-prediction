"""
architecture/mcp_modular_agent/mcp_client.py
------------------------------------------------------------
Use Case 5: MCP modular router client with fallback.

Flow:
  1) Predict severity on the raw text.
  2) Ask Gemini router to rank explanation servers (SHAP/RAG/CF).
  3) Try ranked servers in order until one succeeds.
  4) Return normalized payload for bot.py.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from shared.depression_model import predict_proba, classify_severity
from shared.llm_client import call_gemini

logger = logging.getLogger("mcp_modular_agent")

SYSTEM_ROUTER = (
    "You are an MCP router for depression explanation services. "
    "Choose the best explanation server(s) for the given user text and prediction context. "
    "Available servers: shap, rag, counterfactual, hybrid_shap_rag_counterfactual. "
    "Return JSON only."
)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _router_rank(user_text: str, pred_label: str, severity_score: float) -> Tuple[List[str], str]:
    prompt = f"""
User text:
\"{user_text}\"

Prediction context:
- predicted_label: {pred_label}
- severity_score: {severity_score:.3f}

Route to best explanation server(s) among: shap, rag, counterfactual.

Heuristics:
- Use shap for token-level why explanation.
- Use rag for symptom-grounded explanation with clinical knowledge.
- Use counterfactual for actionable "what could change" guidance.

Return strict JSON with keys:
{{
  "ranked_servers": ["<first>", "<second>", "<third>"],
  "rationale": "one short sentence"
}}
""".strip()

    try:
        raw = call_gemini(prompt, system=SYSTEM_ROUTER)
        parsed = _safe_json_parse(raw)
        ranked = [str(x).strip().lower() for x in parsed.get("ranked_servers", [])]
        ranked = [x for x in ranked if x in {"shap", "rag", "counterfactual", "hybrid_shap_rag_counterfactual"}]
        if not ranked:
            ranked = ["shap", "rag", "counterfactual"]
        rationale = str(parsed.get("rationale", ""))
        return ranked, rationale
    except Exception as exc:
        logger.warning("Router failed; using deterministic fallback ranking. Error: %s", exc)
        return ["shap", "rag", "counterfactual"], "Router fallback: deterministic ranking."


def _run_shap(user_text: str) -> str:
    from shared.depression_model import explain_with_shap
    from architecture.shap_explainer.shap_explainer import generate_shap_explanation

    shap_result = explain_with_shap(user_text)
    return generate_shap_explanation(user_text, shap_result)


def _run_rag(user_text: str) -> str:
    from architecture.rag_explainer.rag_explainer import run_rag_pipeline, generate_rag_explanation

    rag_result = run_rag_pipeline(user_text)
    return generate_rag_explanation(user_text, rag_result)


def _run_counterfactual(user_text: str) -> str:
    from architecture.shap_counterfactual_explainer.cf_generator import generate_counterfactuals
    from architecture.shap_counterfactual_explainer.cf_explainer import generate_cf_explanation

    cf_result = generate_counterfactuals(user_text, n_candidates=3, n_attempts=2)
    return generate_cf_explanation(user_text, cf_result)


def _run_hybrid(user_text: str) -> str:
    from architecture.hybrid_shap_rag_counterfactual.hybrid_pipeline import run_hybrid_pipeline
    from architecture.hybrid_shap_rag_counterfactual.hybrid_explainer import generate_hybrid_explanation

    hybrid_result = run_hybrid_pipeline(user_text)
    return generate_hybrid_explanation(user_text, hybrid_result)


def run_mcp_pipeline(user_text: str, fallback: bool = True, top_k: int = 2) -> Dict[str, Any]:
    probs = predict_proba([user_text])[0]
    pred_label, severity_score, _ = classify_severity(probs)
    confidence = float(max(probs))

    ranked, rationale = _router_rank(user_text, pred_label, severity_score)
    if top_k and top_k > 0:
        ranked = ranked[:top_k]

    runners = {
        "shap": _run_shap,
        "rag": _run_rag,
        "counterfactual": _run_counterfactual,
        "hybrid_shap_rag_counterfactual": _run_hybrid,
    }

    selected = ranked[0] if ranked else "shap"
    errors: List[str] = []

    if not ranked:
        ranked = ["shap"]

    for idx, server in enumerate(ranked):
        try:
            explanation = runners[server](user_text)
            return {
                "selected_server": server,
                "fallback_used": idx > 0,
                "prediction": pred_label,
                "confidence": confidence,
                "explanation": explanation,
                "rationale": rationale,
                "errors": errors,
            }
        except Exception as exc:
            errors.append(f"{server}: {exc}")
            logger.warning("MCP server failed (%s): %s", server, exc)
            if not fallback:
                break

    raise RuntimeError(
        "All selected MCP explanation servers failed. "
        f"Selected order: {ranked}. Errors: {' | '.join(errors)}"
    )
