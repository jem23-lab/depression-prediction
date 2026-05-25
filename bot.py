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
import re
import asyncio

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
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
from shared.training_examples import (
    PARAGRAPHS,
    get_cached_prediction,
    save_prediction,
    get_cached_explanation,
    save_explanation,
)
from shared.depression_model import explain_with_shap, predict_proba, classify_severity, LABEL_MAP

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
        await update.message.reply_text(text[i: i + chunk_size], parse_mode="HTML")


async def send_footer(update: Update):
    await update.message.reply_text(
        "================================================\n"
        "Type /begin to start a new session."
    )


def _rating_keyboard() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(str(i), callback_data=f"rate:{i}") for i in range(1, 6)]
    return InlineKeyboardMarkup([row])


def _pause():
    return asyncio.sleep(random.uniform(2.0, 3.0))


async def _send_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None):
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return None
    return await context.bot.send_message(
        chat_id=chat_id,
        text=strip_markdown(text),
        reply_markup=reply_markup,
        parse_mode="HTML",
    )


def _format_box(title: str, body: str = "", width: int = 48) -> str:
    line = "=" * width
    safe_body = body or ""
    title_line = f"{title}\n\n" if title else ""
    return f"{title_line}{safe_body}\n{line}\n"


def _format_title(base: str, label: str = "") -> str:
    return f"{base} {label}".strip()


def _format_for_display(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"\s+", " ", text).strip()

    markers = [
        "To cope",
        "However",
        "Although",
        "Despite",
        "Since",
        "Between",
        "Socially",
        "Physically",
        "Emotionally",
        "Overall",
        "Lately",
        "My sleep",
        "Sleep",
        "Meanwhile",
        "In addition",
    ]
    for marker in markers:
        text = re.sub(rf"([.!?])\s+({re.escape(marker)})", r"\1\n\n\2", text)

    return text


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
    ("trust", "Trust (does the explanation make you more confident in the AI system's output?)"),
]


# ── Central message handler ──────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        if await _handle_rating(update, context):
            return
        return

    if update.message is None:
        return

    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    logger.info("User %s: %s", user_id, text[:80])

    if await _handle_rating(update, context):
        return

    result = process_message(user_id, text)
    await safe_send(update, result["response"])

    if result["status"] == "ready":
        await _begin_session(update, context)


async def _handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    flow = context.user_data.get("eval_flow")
    if not flow:
        return False

    reply_message = update.message

    if update.callback_query:
        query = update.callback_query
        await query.answer()
        raw = (query.data or "").strip()
        if not raw.startswith("rate:"):
            return False
        raw = raw.split(":", 1)[1]
        reply_message = query.message
    else:
        raw = (update.message.text or "").strip()

    if reply_message is None:
        return False

    try:
        score = int(raw)
    except ValueError:
        await reply_message.reply_text("Please enter a valid integer score from 1 to 5.")
        return True

    if score < 1 or score > 5:
        await reply_message.reply_text("Score must be between 1 and 5.")
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
            reply_markup=_rating_keyboard() if flow["step"] < len(EVAL_CRITERIA) else None,
        )
    except Exception:
        sent = await reply_message.reply_text(
            updated_text,
            reply_markup=_rating_keyboard() if flow["step"] < len(EVAL_CRITERIA) else None,
        )
        flow["prompt_message_id"] = sent.message_id

    if flow["step"] < len(EVAL_CRITERIA):
        return True

    ratings = flow["ratings"]
    avg = round((
        ratings["clarity"]
        + ratings["correctness"]
        + ratings["helpfulness"]
        + ratings["trust"]
    ) / 4.0, 3)
    csv_path = os.path.join(_ROOT, "logs", "evaluation_records.csv")
    append_evaluation_row(
        csv_path,
        {
            "user_id": str(update.effective_user.id),
            "session_id": flow["session_id"],
            "paragraph_id": flow["paragraph_id"],
            "paragraph_text": flow["paragraph_text"],
            "selected_use_case_name": flow["selected_use_case_name"],
            "prediction_label": flow["prediction_label"],
            "prediction_confidence": flow["prediction_confidence"],
            "explanation_text": flow["explanation"],
            "rating_clarity": ratings["clarity"],
            "rating_correctness": ratings["correctness"],
            "rating_helpfulness": ratings["helpfulness"],
            "rating_trust": ratings["trust"],
            "rating_overall_avg": avg,
        },
    )

    context.user_data.pop("eval_flow", None)

    pending = context.user_data.get("pending_explanations") or []
    pending_idx = context.user_data.get("pending_index", 0)

    if pending_idx + 1 < len(pending):
        await reply_message.reply_text(
            _format_box(
                "Thanks for your feedback! 😊\n"
                "Your evaluation has been saved successfully.\n"
                "We’re now showing the second explanation.",
            ),
            parse_mode="HTML",
        )
        context.user_data["pending_index"] = pending_idx + 1
        await _pause()
        method, explanation = pending[pending_idx + 1]
        sample_number = context.user_data.get("sample_index", 1)
        await _send_message(
            update,
            context,
            _format_box(
                _format_title(
                    "2nd Explanation for Person",
                    f"{sample_number} (Why AI did this prediction ? )",
                ),
                explanation,
            ),
        )
        await _pause()

        paragraph_id = context.user_data.get("current_paragraph_id", "")
        paragraph_text = context.user_data.get("current_paragraph_text", "")
        paragraph_severity = context.user_data.get("current_paragraph_severity", "")
        label = context.user_data.get("current_prediction_label", "unknown")
        conf = context.user_data.get("current_prediction_confidence", 0.0)

        context.user_data["eval_flow"] = {
            "session_id": context.user_data.get("session_id"),
            "paragraph_id": paragraph_id,
            "paragraph_text": paragraph_text,
            "paragraph_severity": paragraph_severity,
            "selected_use_case": method,
            "selected_use_case_name": method,
            "prediction_label": label,
            "prediction_confidence": conf,
            "explanation": explanation,
            "ratings": {"clarity": None, "correctness": None, "helpfulness": None, "trust": None},
            "step": 0,
            "prompt_message_id": None,
        }

        prompt = _evaluation_prompt(
            paragraph_text=paragraph_text,
            explanation=explanation,
            ratings=context.user_data["eval_flow"]["ratings"],
            step=0,
        )
        sent = await _send_message(update, context, prompt, reply_markup=_rating_keyboard())
        if sent is not None:
            context.user_data["eval_flow"]["prompt_message_id"] = sent.message_id
        return True

    context.user_data.pop("pending_explanations", None)
    context.user_data.pop("pending_index", None)
    context.user_data.pop("current_paragraph_id", None)
    context.user_data.pop("current_paragraph_text", None)
    context.user_data.pop("current_paragraph_severity", None)
    context.user_data.pop("current_prediction_label", None)
    context.user_data.pop("current_prediction_confidence", None)

    if context.user_data.get("sample_queue"):
        await reply_message.reply_text(
            _format_box(
                "✨ Moving to the next text sample...\n"
                "Thank you for taking part in the evaluation."
            ),
            parse_mode="HTML",
        )
        await _run_next_sample(update, context)
        return True

    await reply_message.reply_text(
        _format_box("Study complete. Thank you for participating 🙇🏻. "),
        parse_mode="HTML",
    )
    return True


async def _begin_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    samples = list(PARAGRAPHS)
    random.shuffle(samples)
    total = min(10, len(samples))
    context.user_data["sample_queue"] = samples[:total]
    context.user_data["sample_index"] = 0
    context.user_data["session_id"] = f"{user_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    await _run_next_sample(update, context)


def _pick_explanation_methods() -> list:
    methods = ["SHAP", "RAG", "HYBRID", "COUNTERFACTUAL"]
    return random.sample(methods, 2)


def _prediction_from_text(paragraph_id: str, user_text: str):
    cached_label, cached_conf = get_cached_prediction(paragraph_id)
    if cached_label is not None and cached_conf is not None:
        return cached_label, cached_conf

    probs = predict_proba([user_text])[0]
    label, _, _ = classify_severity(probs)
    idx = int(probs.argmax()) if probs is not None else 0
    conf = float(probs[idx]) if probs is not None else 0.0
    save_prediction(paragraph_id, label, conf)
    return label, conf


def _override_prediction(model_result, label: str, confidence: float):
    if hasattr(model_result, "pred_label"):
        setattr(model_result, "pred_label", label)
    if hasattr(model_result, "pred_label_idx"):
        label_to_idx = {v: k for k, v in LABEL_MAP.items()}
        setattr(model_result, "pred_label_idx", label_to_idx.get(label, 0))
    if hasattr(model_result, "pred_probs"):
        probs = getattr(model_result, "pred_probs")
        if probs is not None and len(probs) > 0:
            return


def _run_explanation_method(
    method: str,
    paragraph_id: str,
    user_text: str,
    forced_label: str,
    forced_conf: float,
) -> tuple:
    cached = get_cached_explanation(paragraph_id, method)
    if cached:
        console_log = f"Using cached explanation for paragraph {paragraph_id} with method {method}"
        logger.info(console_log)
        return None, cached

    if method == "SHAP":
        model_result = explain_with_shap(user_text)
        _override_prediction(model_result, forced_label, forced_conf)
        _, _, generate_explanation = _get_shap_pipeline()
        explanation = generate_explanation(user_text, model_result)
        save_explanation(paragraph_id, method, explanation)
        return model_result, explanation

    if method == "RAG":
        rag_pipeline, generate_explanation, _ = _get_rag_pipeline()
        model_result = rag_pipeline(user_text)
        _override_prediction(model_result, forced_label, forced_conf)
        explanation = generate_explanation(user_text, model_result)
        save_explanation(paragraph_id, method, explanation)
        return model_result, explanation

    if method == "HYBRID":
        run_hybrid, _, _, generate_explanation = _get_hybrid_pipeline()
        model_result = run_hybrid(user_text)
        if hasattr(model_result, "shap_result") and model_result.shap_result:
            _override_prediction(model_result.shap_result, forced_label, forced_conf)
        if hasattr(model_result, "rag_result") and model_result.rag_result:
            _override_prediction(model_result.rag_result, forced_label, forced_conf)
        explanation = generate_explanation(user_text, model_result)
        save_explanation(paragraph_id, method, explanation)
        return model_result, explanation

    generate_counterfactuals, _, generate_explanation, _ = _get_cf_pipeline()
    model_result = generate_counterfactuals(user_text, n_candidates=3, n_attempts=2)
    if hasattr(model_result, "pred_label"):
        _override_prediction(model_result, forced_label, forced_conf)
    explanation = generate_explanation(user_text, model_result)
    save_explanation(paragraph_id, method, explanation)
    return model_result, explanation


async def _run_next_sample(update: Update, context: ContextTypes.DEFAULT_TYPE):
    queue = context.user_data.get("sample_queue") or []
    if not queue:
        await _send_message(update, context, _format_box("Study complete. Thank you for participating."))
        return

    selected_paragraph = queue.pop(0)
    context.user_data["sample_queue"] = queue
    context.user_data["sample_index"] = context.user_data.get("sample_index", 0) + 1

    paragraph_id = selected_paragraph["id"]
    paragraph_text = selected_paragraph["text"]
    paragraph_severity = selected_paragraph["severity"]

    label, conf = _prediction_from_text(paragraph_id, paragraph_text)
    methods = _pick_explanation_methods()

    context.user_data["current_paragraph_id"] = paragraph_id
    context.user_data["current_paragraph_text"] = paragraph_text
    context.user_data["current_paragraph_severity"] = paragraph_severity
    context.user_data["current_prediction_label"] = label
    context.user_data["current_prediction_confidence"] = conf

    sample_number = context.user_data.get("sample_index", 1)
    await _send_message(
        update,
        context,
        _format_box(
            _format_title("📝 Text Sample (Person", f"{sample_number})"),
            _format_for_display(paragraph_text),
        ),
    )
    await _pause()

    label_sentence = (
        f"Person {sample_number} does not show signs of depression."
        if label == "not depression" else
        f"Person {sample_number} shows signs of {label} depression."
    )
    await _send_message(update, context, _format_box("🤖Prediction", label_sentence))
    await _pause()

    _, explanation_1 = _run_explanation_method(methods[0], paragraph_id, paragraph_text, label, conf)
    _, explanation_2 = _run_explanation_method(methods[1], paragraph_id, paragraph_text, label, conf)

    context.user_data["pending_explanations"] = [
        (methods[0], explanation_1),
        (methods[1], explanation_2),
    ]
    context.user_data["pending_index"] = 0

    await _send_message(
        update,
        context,
        _format_box(
            _format_title(f"1st Explanation for Person", f"{sample_number} (Why AI did this prediction ? )"),
            explanation_1,
        ),
    )
    await _pause()

    context.user_data["eval_flow"] = {
        "session_id": context.user_data.get("session_id"),
        "paragraph_id": paragraph_id,
        "paragraph_text": paragraph_text,
        "paragraph_severity": paragraph_severity,
        "selected_use_case": methods[0],
        "selected_use_case_name": methods[0],
        "prediction_label": label,
        "prediction_confidence": conf,
        "explanation": explanation_1,
        "ratings": {"clarity": None, "correctness": None, "helpfulness": None, "trust": None},
        "step": 0,
        "prompt_message_id": None,
    }
    prompt = _evaluation_prompt(
        paragraph_text=paragraph_text,
        explanation=explanation_1,
        ratings=context.user_data["eval_flow"]["ratings"],
        step=0,
    )
    sent = await _send_message(update, context, prompt, reply_markup=_rating_keyboard())
    if sent is not None:
        context.user_data["eval_flow"]["prompt_message_id"] = sent.message_id


def _evaluation_prompt(paragraph_text: str, explanation: str, ratings: dict, step: int) -> str:
    if step >= len(EVAL_CRITERIA):
        summary_lines = [
            f"- {label}: {ratings.get(key)}" for key, label in EVAL_CRITERIA
        ]
        return _format_box("⭐ Please Rate This Explanation", "\n".join(summary_lines))

    _, label = EVAL_CRITERIA[step]
    body_lines = [
        "Please answer the following questions based on your experience:",
        "",
        f"{label}",
        "",
        f"Tap {label} score (1-5).",
    ]
    return _format_box("⭐ Please Rate This Explanation", "\n".join(body_lines))


# ── Use Case 1: SHAP ─────────────────────────────────────────────────
async def run_shap_pipeline(update: Update, user_id: int, user_text: str):
    explain_with_shap, format_debug, generate_shap_explanation = _get_shap_pipeline()
    logger.info("UC1 SHAP for user %s", user_id)

    paragraph_box = _format_box("1) Paragraph", _format_for_display(user_text))
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

    paragraph_box = _format_box("1) Paragraph", _format_for_display(user_text))
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

    paragraph_box = _format_box("1) Paragraph", _format_for_display(user_text))
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

    paragraph_box = _format_box("1) Paragraph", _format_for_display(user_text))
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

    paragraph_box = _format_box("1) Paragraph", _format_for_display(user_text))
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
    application.add_handler(CommandHandler(["start", "help", "begin"], handle_message))
    application.add_handler(CallbackQueryHandler(handle_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.post_init = _on_startup
    logger.info("Bot started. Listening for updates...")
    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
