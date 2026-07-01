"""
architecture/mcp_modular_agent/mcp_client.py
------------------------------------------------------------
Use Case 5: agentic planner for explanation strategy.

Flow:
  1) Predict severity on the raw text.
  2) Ask an LLM planner for strict JSON:
       {"intent": "prediction_reason", "tools": ["shap"]}
  3) Run the selected Python explanation tools to collect evidence.
  4) Ask a final LLM response generator to merge the evidence into one
     coherent user-facing explanation.
"""

import json
import logging
from typing import Any, Dict, List

from shared.depression_model import predict_proba, classify_severity
from shared.llm_client import call_gemini

logger = logging.getLogger("mcp_modular_agent")

VALID_TOOLS = {"shap", "rag", "counterfactual", "hybrid"}
HYBRID_TOOLS = ["shap", "rag", "counterfactual"]

SYSTEM_PLANNER = (
    "You are an explanation strategy planner. "
    "Choose which evidence tools should be used to explain an AI depression "
    "screening prediction. Return JSON only."
)

SYSTEM_RESPONSE_GENERATOR = (
    "You generate clear explanations from provided evidence. "
    "Use only the evidence given. Do not invent information."
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


def _normalize_tools(raw_tools: Any) -> List[str]:
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        return ["shap"]

    aliases = {
        "cf": "counterfactual",
        "counter factual": "counterfactual",
        "counterfactuals": "counterfactual",
        "knowledge": "rag",
        "retrieval": "rag",
        "hybrid_shap_rag_counterfactual": "hybrid",
    }

    tools: List[str] = []
    for item in raw_tools:
        tool = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if tool in VALID_TOOLS and tool not in tools:
            tools.append(tool)

    return tools or ["shap"]


def _plan_strategy(
    user_text: str,
    pred_label: str,
    severity_score: float,
    user_question: str = "",
) -> Dict[str, Any]:
    prompt = f"""
User text:
"{user_text}"

Participant question:
"{user_question or 'Explain the prediction.'}"

Prediction context:
- predicted_label: {pred_label}
- severity_score: {severity_score:.3f}

Decide the participant's explanation intent and select tools.

Intent options:
- prediction_reason: explain which words or phrases drove the prediction
- knowledge: connect the text to symptom knowledge
- actionable: explain what wording changes would shift the prediction
- hybrid: combine word-level, knowledge, and actionable evidence

Tool options:
- shap: word-level prediction evidence
- rag: symptom/knowledge evidence
- counterfactual: actionable wording-change evidence
- hybrid: use shap, rag, and counterfactual together

Return strict JSON only, with this schema:
{{
  "intent": "prediction_reason | knowledge | actionable | hybrid",
  "tools": ["shap | rag | counterfactual | hybrid"],
  "rationale": "one short sentence"
}}

Examples:
{{"intent":"prediction_reason","tools":["shap"],"rationale":"The user asks why the model predicted this."}}
{{"intent":"knowledge","tools":["rag"],"rationale":"The user asks for symptom-grounded context."}}
{{"intent":"actionable","tools":["counterfactual"],"rationale":"The user asks what could change the result."}}
{{"intent":"hybrid","tools":["hybrid"],"rationale":"The user needs a complete explanation using multiple evidence types."}}
""".strip()

    try:
        raw = call_gemini(prompt, system=SYSTEM_PLANNER)
        parsed = _safe_json_parse(raw)
        intent = str(parsed.get("intent", "prediction_reason")).strip().lower()
        if intent not in {"prediction_reason", "knowledge", "actionable", "hybrid"}:
            intent = "prediction_reason"
        tools = _normalize_tools(parsed.get("tools", ["shap"]))
        return {
            "intent": intent,
            "tools": tools,
            "rationale": str(parsed.get("rationale", "")).strip(),
            "raw_plan": parsed,
        }
    except Exception as exc:
        logger.warning("Planner failed; using deterministic fallback. Error: %s", exc)
        return {
            "intent": "prediction_reason",
            "tools": ["shap"],
            "rationale": "Planner fallback: using word-level prediction evidence.",
            "raw_plan": {},
        }


def _format_shap_evidence(user_text: str) -> str:
    from shared.depression_model import explain_with_shap

    result = explain_with_shap(user_text)
    lines = ["SHAP evidence: words/phrases that influenced the prediction."]

    risk = result.risk_tokens[:5]
    protective = result.protective_tokens[:3]

    if risk:
        lines.append("Risk-increasing words:")
        for token in risk:
            lines.append(f"- {token.get('token', '')}: {float(token.get('shap', 0.0)):+.3f}")

    if protective:
        lines.append("Risk-lowering words:")
        for token in protective:
            lines.append(f"- {token.get('token', '')}: {float(token.get('shap', 0.0)):+.3f}")

    if not risk and not protective:
        lines.append("- No strong word-level evidence was detected.")

    return "\n".join(lines)


def _format_rag_evidence(user_text: str) -> str:
    from architecture.rag_explainer.rag_explainer import run_rag_pipeline

    result = run_rag_pipeline(user_text, top_k=3)
    lines = ["RAG evidence: symptom knowledge related to the text."]

    if not result.retrieved_docs:
        lines.append("- No closely related knowledge-base entries were retrieved.")
        return "\n".join(lines)

    for doc in result.retrieved_docs[:3]:
        patterns = "; ".join(doc.natural_language_patterns[:3])
        lines.append(
            f"- {doc.symptom_name}: {doc.clinical_definition} "
            f"Typical expressions include: {patterns}."
        )

    return "\n".join(lines)


def _format_counterfactual_evidence(user_text: str) -> str:
    from architecture.shap_counterfactual_explainer.cf_generator import generate_counterfactuals

    result = generate_counterfactuals(user_text, n_candidates=2, n_attempts=1)
    lines = ["Counterfactual evidence: wording changes that could shift the prediction."]

    if result.shap_guided_tokens:
        tokens = ", ".join(t.get("token", "") for t in result.shap_guided_tokens[:4])
        lines.append(f"Important change targets: {tokens}.")

    if result.best_cf:
        best = result.best_cf
        status = "changed the predicted label" if best.get("flip_success") else "did not change the label"
        lines.append(f"Best candidate ({status}):")
        lines.append(f"- Original label: {result.original_label}")
        lines.append(f"- Target label: {result.target_label}")
        lines.append(f"- Modified text: {best.get('text', '')}")
        lines.append(f"- New label: {best.get('label', '')}")
        lines.append(f"- Minimality: {float(best.get('minimality', 0.0)):.2f}")
        lines.append(f"- Meaning kept: {float(best.get('semantic_sim', 0.0)):.2f}")
        return "\n".join(lines)

    if result.candidates:
        candidate = result.candidates[0]
        lines.append(f"- Candidate text: {candidate.get('text', '')}")
        lines.append(f"- New label: {candidate.get('label', '')}")
    else:
        lines.append("- No counterfactual candidates were generated.")

    return "\n".join(lines)


def _execution_tools(planned_tools: List[str]) -> List[str]:
    if "hybrid" in planned_tools:
        return HYBRID_TOOLS[:]
    return planned_tools


def _collect_evidence(user_text: str, planned_tools: List[str], fallback: bool) -> tuple[List[str], List[str]]:
    evidence_blocks: List[str] = []
    errors: List[str] = []

    runners = {
        "shap": _format_shap_evidence,
        "rag": _format_rag_evidence,
        "counterfactual": _format_counterfactual_evidence,
    }

    for tool in _execution_tools(planned_tools):
        try:
            evidence_blocks.append(runners[tool](user_text))
        except Exception as exc:
            errors.append(f"{tool}: {exc}")
            logger.warning("MCP evidence tool failed (%s): %s", tool, exc)
            if not fallback:
                break

    if not evidence_blocks and fallback:
        try:
            evidence_blocks.append(_format_shap_evidence(user_text))
            errors.append("fallback: used shap evidence")
        except Exception as exc:
            errors.append(f"fallback shap: {exc}")

    return evidence_blocks, errors


def _generate_response(
    user_text: str,
    user_question: str,
    pred_label: str,
    confidence: float,
    plan: Dict[str, Any],
    evidence_blocks: List[str],
) -> str:
    evidence = "\n\n".join(evidence_blocks).strip() or "No evidence was available."
    prompt = f"""
Using the evidence below, generate one coherent explanation.

Do not invent information.
Do not mention internal tool names unless needed for clarity.
Do not provide diagnosis, treatment instructions, or crisis advice.
Keep the focus on explaining the AI prediction.

User text:
"{user_text}"

Participant question:
"{user_question or 'Explain the prediction.'}"

Prediction:
- label: {pred_label}
- confidence: {confidence:.3f}

Planner decision:
- intent: {plan.get("intent", "prediction_reason")}
- tools: {", ".join(plan.get("tools", []))}
- rationale: {plan.get("rationale", "")}

Evidence:
{evidence}

Write a concise, complete explanation in plain language.

Use the evidence according to the planner's intent:
- Cover all evidence that is important for understanding the prediction.
- Merge overlapping evidence instead of repeating the same point.
- If word-level evidence is present, explain how those words affected the prediction.
- If knowledge evidence is present, connect it to the user's wording and the matching theme.
- If counterfactual evidence is present, explain what wording change was tested and what it shows.

Rules:
- Use only the evidence above; do not add outside facts, advice, diagnosis, or reassurance.
- Do not mention SHAP, RAG, counterfactual, planner, tools, scores, or confidence.
- Directly answer the participant's question and keep it understandable for a non-technical user.
- Prefer one or two short paragraphs, not bullet points.
- Be concise, but do not omit evidence that changes the meaning of the explanation.
""".strip()

    return call_gemini(prompt, system=SYSTEM_RESPONSE_GENERATOR)


def run_mcp_pipeline(
    user_text: str,
    fallback: bool = True,
    top_k: int = 2,
    user_question: str = "",
) -> Dict[str, Any]:
    """
    Agentic use case:
    planner JSON -> selected Python evidence tools -> final LLM synthesis.

    top_k is kept for backward compatibility with the previous router API.
    The planner's tool list is not truncated because hybrid plans need all
    selected evidence sources.
    """
    probs = predict_proba([user_text])[0]
    pred_label, severity_score, _ = classify_severity(probs)
    confidence = float(max(probs))

    plan = _plan_strategy(user_text, pred_label, severity_score, user_question=user_question)
    evidence_blocks, errors = _collect_evidence(user_text, plan["tools"], fallback=fallback)

    if not evidence_blocks:
        raise RuntimeError(f"All selected MCP evidence tools failed. Errors: {' | '.join(errors)}")

    explanation = _generate_response(user_text, user_question, pred_label, confidence, plan, evidence_blocks)
    executed_tools = _execution_tools(plan["tools"])

    return {
        "selected_server": "+".join(plan["tools"]),
        "selected_tools": plan["tools"],
        "executed_tools": executed_tools,
        "fallback_used": any(e.startswith("fallback:") for e in errors),
        "prediction": pred_label,
        "confidence": confidence,
        "intent": plan["intent"],
        "planner_json": {
            "intent": plan["intent"],
            "tools": plan["tools"],
        },
        "explanation": explanation,
        "evidence": evidence_blocks,
        "rationale": plan.get("rationale", ""),
        "errors": errors,
    }
