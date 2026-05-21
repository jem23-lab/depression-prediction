"""
bot.py  (root-level, shared by all use cases)
────────────────────────────────────────────────────────────────────
Single Telegram bot routing to all explanation pipelines.

Use cases:
  1 → SHAP only
  2 → RAG only
  3 → Hybrid: SHAP + RAG + Counterfactual  (all three signals → one prompt)
  4 → Counterfactual only
  5 → MCP agent

Run:
  export TELEGRAM_BOT_TOKEN="..."
  export GOOGLE_API_KEY="..."
  python bot.py
"""

import os
import sys
import logging
import random
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ── Path setup ───────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from shared.conversation import process_message
from shared.llm_client import strip_markdown
from shared.depression_model import load as preload_model
from shared.eval_logger import append_evaluation_row
from shared.training_examples import PARAGRAPHS

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("depression_bot")


# ── Lazy pipeline importers ──────────────────────────────────────────
def _get_shap_pipeline():
    from shared.depression_model import explain_with_shap, format_debug
    from architecture.shap_explainer.shap_explainer import generate_shap_explanation
    return explain_with_shap, format_debug, generate_shap_explanation


def _get_rag_pipeline():
    from architecture.rag_explainer.rag_explainer import (
        run_rag_pipeline, generate_rag_explanation, format_rag_debug,
    )
    return run_rag_pipeline, generate_rag_explanation, format_rag_debug


def _get_cf_pipeline():
    from architecture.shap_counterfactual_explainer.cf_generator import generate_counterfactuals, format_cf_debug
    from architecture.shap_counterfactual_explainer.cf_explainer import (
        generate_cf_explanation, format_cf_telegram_preview,
    )
    return generate_counterfactuals, format_cf_debug, generate_cf_explanation, format_cf_telegram_preview


def _get_hybrid_pipeline():
    from architecture.hybrid_shap_rag_counterfactual.hybrid_pipeline import run_hybrid_pipeline, format_hybrid_debug, \
        format_hybrid_telegram_preview
    from architecture.hybrid_shap_rag_counterfactual.hybrid_explainer import generate_hybrid_explanation
    return run_hybrid_pipeline, format_hybrid_debug, format_hybrid_telegram_preview, generate_hybrid_explanation


def _get_mcp_pipeline():
    from architecture.mcp_modular_agent.mcp_client import run_mcp_pipeline
    return run_mcp_pipeline


# ── Helpers ──────────────────────────────────────────────────────────
async def safe_send(update: Update, text: str, chunk_size: int = 4000):
    """Strip Markdown symbols and chunk-send as plain text."""
    text = strip_markdown(text)
    for i in range(0, max(len(text), 1), chunk_size):
        await update.message.reply_text(text[i: i + chunk_size])


async def send_footer(update: Update):
    await update.message.reply_text(
        "─────────────────────────────\n"
        "Type /assess to run another analysis.\n"
        "Type /reset to clear your session."
    )


def _format_box(title: str, body: str, width: int = 48) -> str:
    line = "=" * width
    divider = "-" * width
    return f"{line}\n{title}\n{divider}\n{body}"


def _join_box_lines(lines: list) -> str:
    return "\n".join([ln for ln in lines if ln is not None])


# PARAGRAPHS are defined in shared.training_examples


USE_CASES = {
    "1": "SHAP",
    "2": "RAG",
    "3": "HYBRID",
    "4": "COUNTERFACTUAL",
    # "5": "MCP",
}

EVAL_CRITERIA = [
    ("clarity", "Clarity (is the explanation easy to understand?)"),
    ("correctness", "Correctness (does the explanation logically and factually align with the question?)"),
    ("helpfulness", "Helpfulness (does the explanation address what you actually wanted to know?)"),
]


# ── Central message handler ──────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    logger.info("User %s: %s", user_id, text[:80])

    if await _handle_rating(update, context):
        return

    result = process_message(user_id, text)
    await safe_send(update, result["response"])

    if result["status"] == "ready":
        await _start_evaluation(update, context)


async def _handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    flow = context.user_data.get("eval_flow")
    if not flow:
        return False

    raw = (update.message.text or "").strip()
    try:
        score = int(raw)
    except ValueError:
        await update.message.reply_text("Please enter a valid integer score from 1 to 5.")
        return True

    if score < 1 or score > 5:
        await update.message.reply_text("Score must be between 1 and 5.")
        return True

    step = flow["step"]
    criterion_key, _ = EVAL_CRITERIA[step]
    flow["ratings"][criterion_key] = score
    flow["step"] = step + 1

    updated_text = _evaluation_prompt(
        paragraph_text=flow["paragraph_text"],
        explanation=flow["explanation"],
        ratings=flow["ratings"],
        step=flow["step"],
    )
    try:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=flow["prompt_message_id"],
            text=updated_text,
        )
    except Exception:
        sent = await update.message.reply_text(updated_text)
        flow["prompt_message_id"] = sent.message_id

    if flow["step"] < len(EVAL_CRITERIA):
        return True

    ratings = flow["ratings"]
    avg = round((ratings["clarity"] + ratings["correctness"] + ratings["helpfulness"]) / 3.0, 3)
    csv_path = os.path.join(_ROOT, "logs", "evaluation_records.csv")
    append_evaluation_row(
        csv_path,
        {
            "user_id": str(update.effective_user.id),
            "session_id": flow["session_id"],
            "paragraph_id": flow["paragraph_id"],
            "paragraph_text": flow["paragraph_text"],
            "selected_use_case": flow["selected_use_case"],
            "selected_use_case_name": flow["selected_use_case_name"],
            "prediction_label": flow["prediction_label"],
            "prediction_confidence": flow["prediction_confidence"],
            "explanation_text": flow["explanation"],
            "rating_clarity": ratings["clarity"],
            "rating_correctness": ratings["correctness"],
            "rating_helpfulness": ratings["helpfulness"],
            "rating_overall_avg": avg,
        },
    )

    await update.message.reply_text(
        "Thanks. Your evaluation has been saved.\n"
        f"Scores -> Clarity: {ratings['clarity']}, Correctness: {ratings['correctness']}, Helpfulness: {ratings['helpfulness']}\n"
        f"Average: {avg:.2f}\n\n"
        "Type /assess to run another evaluation."
    )
    context.user_data.pop("eval_flow", None)
    return True


async def _start_evaluation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    selected_paragraph = random.choice(PARAGRAPHS)
    paragraph_id = selected_paragraph["id"]
    paragraph_text = selected_paragraph["text"]
    paragraph_severity = selected_paragraph["severity"]

    await update.message.reply_text(_format_box("Evaluation Paragraph", paragraph_text))
    await update.message.reply_text("Running evaluation pipeline...")

    eval_result = await _run_random_explanation(paragraph_text)

    context.user_data["eval_flow"] = {
        "session_id": f"{user_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "paragraph_id": paragraph_id,
        "paragraph_text": paragraph_text,
        "paragraph_severity": paragraph_severity,
        "selected_use_case": eval_result["use_case"],
        "selected_use_case_name": eval_result["use_case_name"],
        "prediction_label": eval_result["prediction_label"],
        "prediction_confidence": eval_result["prediction_confidence"],
        "explanation": eval_result["explanation"],
        "ratings": {"clarity": None, "correctness": None, "helpfulness": None},
        "step": 0,
        "prompt_message_id": None,
    }

    await safe_send(update, _format_box("Evaluation Explanation", eval_result["explanation"]))

    text = _evaluation_prompt(
        paragraph_text=paragraph_text,
        explanation=eval_result["explanation"],
        ratings=context.user_data["eval_flow"]["ratings"],
        step=0,
    )
    sent = await update.message.reply_text(text)
    context.user_data["eval_flow"]["prompt_message_id"] = sent.message_id


async def _run_random_explanation(user_text: str) -> dict:
    use_case = random.choice(list(USE_CASES.keys()))

    if use_case == "1":
        explain_with_shap, _, generate_shap_explanation = _get_shap_pipeline()
        model_result = explain_with_shap(user_text)
        explanation = generate_shap_explanation(user_text, model_result)
    elif use_case == "2":
        rag_pipeline, generate_rag_explanation, _ = _get_rag_pipeline()
        model_result = rag_pipeline(user_text)
        explanation = generate_rag_explanation(user_text, model_result)
    elif use_case == "3":
        run_hybrid, _, _, generate_explanation = _get_hybrid_pipeline()
        model_result = run_hybrid(user_text)
        explanation = generate_explanation(user_text, model_result)
    elif use_case == "4":
        generate_counterfactuals, _, generate_cf_explanation, _ = _get_cf_pipeline()
        model_result = generate_counterfactuals(user_text, n_candidates=3, n_attempts=2)
        explanation = generate_cf_explanation(user_text, model_result)
    # elif use_case == "5":
    #     run_mcp_pipeline = _get_mcp_pipeline()
    #     model_result = run_mcp_pipeline(user_text)
    #     explanation = model_result.get("explanation", "No explanation returned.")

    pred_label, pred_conf = _extract_prediction_confidence(model_result)
    return {
        "use_case": use_case,
        "use_case_name": USE_CASES[use_case],
        "prediction_label": pred_label,
        "prediction_confidence": pred_conf,
        "explanation": explanation,
    }


def _extract_prediction_confidence(payload):
    label = "unknown"
    confidence = 0.0

    if hasattr(payload, "pred_label"):
        label = getattr(payload, "pred_label", "unknown")
        idx = getattr(payload, "pred_label_idx", None)
        probs = getattr(payload, "pred_probs", None)
        if idx is not None and probs is not None:
            try:
                confidence = float(probs[idx])
            except Exception:
                confidence = 0.0
    elif hasattr(payload, "original_label"):
        label = getattr(payload, "original_label", "unknown")
        probs = getattr(payload, "original_probs", None)
        if probs is not None:
            from shared.depression_model import LABEL_MAP
            label_to_idx = {v: k for k, v in LABEL_MAP.items()}
            idx = label_to_idx.get(label)
            try:
                confidence = float(probs[idx]) if idx is not None else float(max(probs))
            except Exception:
                confidence = 0.0
    elif isinstance(payload, dict):
        label = str(payload.get("prediction", "unknown"))
        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0

    return label, confidence


def _evaluation_prompt(paragraph_text: str, explanation: str, ratings: dict, step: int) -> str:
    criteria_lines = [
        "Rate the assistant's explanation based on:",
        "",
    ]
    for key, label in EVAL_CRITERIA:
        value = ratings.get(key)
        criteria_lines.append(f"- {label}: {'[pending]' if value is None else value}")

    if step < len(EVAL_CRITERIA):
        _, label = EVAL_CRITERIA[step]
        criteria_lines.append("")
        criteria_lines.append(f"Please enter {label} score (1-5).")

    return _format_box("Evaluation", "\n".join(criteria_lines))


# ── Use Case 1: SHAP ─────────────────────────────────────────────────
async def run_shap_pipeline(update: Update, user_id: int, user_text: str):
    explain_with_shap, format_debug, generate_shap_explanation = _get_shap_pipeline()
    logger.info("UC1 SHAP for user %s", user_id)

    paragraph_box = _format_box("1) Paragraph", user_text)
    await update.message.reply_text(paragraph_box)
    await update.message.reply_text("Running SHAP prediction...")

    shap_result = explain_with_shap(user_text)
    logger.info(format_debug(shap_result))

    label = shap_result.pred_label
    confidence = shap_result.pred_probs[shap_result.pred_label_idx]
    top_word = shap_result.top_tokens[0]["token"] if shap_result.top_tokens else "—"

    prediction_box = _format_box(
        "2) Prediction",
        _join_box_lines(
            [
                f"Level: {label}",
                f"Confidence: {confidence * 100:.1f}%",
                f"Key word: {top_word}",
            ]
        ),
    )

    await update.message.reply_text(prediction_box)

    if shap_result.top_tokens:
        tool_lines = ["Top SHAP tokens:"]
        for t in shap_result.top_tokens[:6]:
            arrow = "🔴" if t["shap"] > 0 else "🟢"
            line = f"{arrow} '{t['token']}' SHAP={t['shap']:+.4f} {t['direction']}"
            if t["note"]:
                line += f" ({t['note']})"
            tool_lines.append(line)
        await update.message.reply_text(_format_box("3) Tool Result", "\n".join(tool_lines)))

    await update.message.reply_text("Generating full explanation...")

    explanation = generate_shap_explanation(user_text, shap_result)
    await safe_send(update, _format_box("4) Explanation", explanation))

    await send_footer(update)


# ── Use Case 2: RAG ──────────────────────────────────────────────────
async def run_rag_pipeline(update: Update, user_id: int, user_text: str):
    rag_pipeline, generate_rag_explanation, format_rag_debug = _get_rag_pipeline()
    logger.info("UC2 RAG for user %s", user_id)

    paragraph_box = _format_box("1) Paragraph", user_text)
    await update.message.reply_text(paragraph_box)
    await update.message.reply_text("Running RAG prediction...")

    rag_result = rag_pipeline(user_text)
    logger.info(format_rag_debug(rag_result))

    label = rag_result.pred_label
    confidence = rag_result.pred_probs[rag_result.pred_label_idx]
    symptoms = ", ".join(d.symptom_name for d in rag_result.retrieved_docs) or "n/a"

    prediction_box = _format_box(
        "2) Prediction",
        _join_box_lines(
            [
                f"Level: {label}",
                f"Confidence: {confidence * 100:.1f}%",
                f"Matched symptoms: {symptoms}",
            ]
        ),
    )

    await update.message.reply_text(prediction_box)

    tool_lines = ["Retrieved clinical knowledge:"]
    for i, doc in enumerate(rag_result.retrieved_docs, 1):
        tool_lines.append(
            f"{i}. {doc.symptom_name} ({doc.symptom_type})\n   {doc.clinical_definition[:100]}..."
        )
    await update.message.reply_text(_format_box("3) Tool Result", "\n".join(tool_lines)))

    await update.message.reply_text("Generating full explanation...")

    explanation = generate_rag_explanation(user_text, rag_result)
    await safe_send(update, _format_box("4) Explanation", explanation))

    await send_footer(update)


# ── Use Case 3: Hybrid (SHAP + RAG + CF) ────────────────────────────
async def run_hybrid_pipeline_handler(update: Update, user_id: int, user_text: str):
    run_hybrid, format_debug, format_preview, generate_explanation = _get_hybrid_pipeline()
    logger.info("UC3 Hybrid for user %s", user_id)

    paragraph_box = _format_box("1) Paragraph", user_text)
    await update.message.reply_text(paragraph_box)

    await update.message.reply_text(
        "Running all three XAI pipelines (SHAP + RAG + Counterfactual).\n"
        "This is the most comprehensive analysis — please allow 30-60 seconds..."
    )

    result = run_hybrid(user_text)
    logger.info(format_debug(result))

    prediction_box = _format_box("2) Prediction", format_preview(result))

    await update.message.reply_text(prediction_box)

    detail_lines = ["Detailed evidence from all three signals:"]

    if result.shap_result and result.shap_result.top_tokens:
        detail_lines.append("\nSHAP — Risk tokens:")
        for t in result.shap_result.top_tokens[:5]:
            arrow = "🔴" if t["shap"] > 0 else "🟢"
            detail_lines.append(f"{arrow} '{t['token']}' SHAP={t['shap']:+.4f} {t['direction']}")

    if result.rag_result and result.rag_result.retrieved_docs:
        detail_lines.append("\nRAG — Matched PHQ-8 symptoms:")
        for i, doc in enumerate(result.rag_result.retrieved_docs, 1):
            detail_lines.append(
                f"{i}. {doc.symptom_name} ({doc.symptom_type})\n   {doc.clinical_definition[:90]}..."
            )

    if result.cf_result and result.cf_result.candidates:
        detail_lines.append("\nCounterfactual — Candidates:")
        for i, c in enumerate(result.cf_result.candidates[:3], 1):
            status = "FLIP" if c["flip_success"] else "no flip"
            detail_lines.append(
                f"{i}. [{status}] [{c['label']}] min={c['minimality']:.2f}\n   \"{c['text'][:100]}\""
            )

    if len(detail_lines) > 1:
        await update.message.reply_text(_format_box("3) Tool Result", "\n".join(detail_lines)))

    await update.message.reply_text("Generating full explanation...")

    explanation = generate_explanation(user_text, result)
    await safe_send(update, _format_box("4) Explanation", explanation))

    await send_footer(update)


# ── Use Case 4: Counterfactual ───────────────────────────────────────
async def run_cf_pipeline(update: Update, user_id: int, user_text: str):
    generate_counterfactuals, format_cf_debug, generate_cf_explanation, format_cf_preview = _get_cf_pipeline()
    logger.info("UC4 CF for user %s", user_id)

    paragraph_box = _format_box("1) Paragraph", user_text)
    await update.message.reply_text(paragraph_box)

    await update.message.reply_text(
        "Generating counterfactuals (SHAP + multiple LLM calls).\n"
        "This may take 20-40 seconds..."
    )

    result = generate_counterfactuals(user_text, n_candidates=3, n_attempts=2)
    logger.info(format_cf_debug(result))

    prediction_box = _format_box("2) Prediction", format_cf_preview(result))

    await update.message.reply_text(prediction_box)

    if result.candidates:
        lines = ["Counterfactual candidates:"]
        for i, c in enumerate(result.candidates[:3], 1):
            status = "FLIP" if c["flip_success"] else "no flip"
            lines.append(
                f"{i}. [{status}] Predicted: {c['label']}\n"
                f"   Minimality: {c['minimality']:.2f}  Meaning kept: {c['semantic_sim']:.2f}\n"
                f"   \"{c['text'][:110]}\""
            )
        await update.message.reply_text(_format_box("3) Tool Result", "\n".join(lines)))

    await update.message.reply_text("Generating full explanation...")

    explanation = generate_cf_explanation(user_text, result)
    await safe_send(update, _format_box("4) Explanation", explanation))

    await send_footer(update)


# ── Use Case 5: MCP ─────────────────────────────────────────────────
async def run_mcp_pipeline_handler(update: Update, user_id: int, user_text: str):
    run_mcp_pipeline = _get_mcp_pipeline()
    logger.info("UC5 MCP for user %s", user_id)

    paragraph_box = _format_box("1) Paragraph", user_text)
    await update.message.reply_text(paragraph_box)

    await update.message.reply_text(
        "Running MCP modular pipeline.\n"
        "This may take a few seconds..."
    )

    result = run_mcp_pipeline(user_text)

    label = result.get("prediction", "unknown")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    selected_server = result.get("selected_server", "n/a")
    fallback_used = bool(result.get("fallback_used", False))
    rationale = result.get("rationale", "")
    explanation = result.get("explanation", "No explanation returned.")
    errors = result.get("errors", []) or []

    prediction_box = _format_box(
        "2) Prediction",
        _join_box_lines(
            [
                f"Level: {label}",
                f"Confidence: {confidence * 100:.1f}%",
                f"Selected server: {selected_server}",
                f"Fallback used: {'yes' if fallback_used else 'no'}",
            ]
        ),
    )

    await update.message.reply_text(prediction_box)

    detail_lines = ["MCP decision details:"]
    if rationale:
        detail_lines.append(f"Rationale: {rationale}")

    if errors:
        detail_lines.append("")
        detail_lines.append("Errors:")
        for i, e in enumerate(errors, 1):
            detail_lines.append(f"{i}. {e}")

    if len(detail_lines) > 1:
        await update.message.reply_text(_format_box("3) Tool Result", "\n".join(detail_lines)))

    await update.message.reply_text("Generating full explanation...")

    await safe_send(update, _format_box("4) Explanation", explanation))

    await send_footer(update)


# ── Bot entrypoint ───────────────────────────────────────────────────
async def _on_startup(app: Application):
    try:
        preload_model()
        logger.info("Model preload complete.")
    except Exception as exc:
        logger.exception("Model preload failed: %s", exc)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler(["start", "help", "assess", "reset"], handle_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.post_init = _on_startup
    logger.info("Bot started. Listening for updates...")
    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
