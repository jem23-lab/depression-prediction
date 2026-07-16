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
import csv
from datetime import datetime, timezone
import re
import asyncio
import torch

# Keep large Hugging Face downloads out of the small /home quota on NTUU.
_MENTALLAMA_DEFAULT_CACHE_DIR = "/scratch/apriyadar/huggingface"
os.environ.setdefault("HF_HOME", _MENTALLAMA_DEFAULT_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_MENTALLAMA_DEFAULT_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_MENTALLAMA_DEFAULT_CACHE_DIR, "transformers"))
os.environ.setdefault("MENTALLAMA_CACHE_DIR", _MENTALLAMA_DEFAULT_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import LlamaTokenizer, LlamaForCausalLM
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
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

# ── Path setup ───────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("depression_bot")

_MENTALLAMA_PIPELINE = None

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


def _env_flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _mentallama_load_kwargs() -> dict:
    cache_dir = os.environ.get("MENTALLAMA_CACHE_DIR", _MENTALLAMA_DEFAULT_CACHE_DIR).strip()
    os.makedirs(cache_dir, exist_ok=True)

    kwargs = {"cache_dir": cache_dir}
    if _env_flag("MENTALLAMA_LOCAL_FILES_ONLY", "true"):
        kwargs["local_files_only"] = True
    return kwargs


def _get_mentallama_pipeline():
    global _MENTALLAMA_PIPELINE
    if _MENTALLAMA_PIPELINE is not None:
        return _MENTALLAMA_PIPELINE

    model_id = os.environ.get("MENTALLAMA_MODEL_ID", "klyang/MentaLLaMA-chat-7B")
    load_kwargs = _mentallama_load_kwargs()

    tokenizer = LlamaTokenizer.from_pretrained(model_id, **load_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(load_kwargs)
    torch_dtype = os.environ.get("MENTALLAMA_TORCH_DTYPE", "").strip()
    if torch_dtype:
        model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

    use_device_map = _env_flag("MENTALLAMA_DEVICE_MAP_AUTO", "true")
    if use_device_map:
        model_kwargs["device_map"] = "auto"

    model = LlamaForCausalLM.from_pretrained(model_id, **model_kwargs)

    device = os.environ.get("MENTALLAMA_DEVICE", "").strip()
    if use_device_map:
        device = next(model.parameters()).device
    elif not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if not use_device_map:
        model.to(device)
    model.eval()

    _MENTALLAMA_PIPELINE = tokenizer, model, device
    return _MENTALLAMA_PIPELINE


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
    "5": "MCP",
}

EVAL_CRITERIA = [
    ("clarity", "Clarity (is the explanation easy to understand?)"),
    ("correctness", "Correctness (does the explanation logically and factually align with the question?)"),
    ("helpfulness", "Helpfulness (does the explanation address what you actually wanted to know?)"),
    ("trust", "Trust (does the explanation make you more confident in the AI system's output?)"),
]

PAIRWISE_CRITERIA = [
    ("clarity", "Which response explained the prediction more clearly and was easier to understand?"),
    ("accuracy", "Which response appeared more accurate and consistent with the prediction?"),
    ("helpfulness", "Which explanation better answered your question?"),
    ("trust", "Which explanation would you prefer to receive from a healthcare assistant?"),
]

PREFERENCE_OPTIONS = [
    ("strong_a", "Strongly prefer A"),
    ("prefer_a", "Prefer A"),
    ("no_pref", "No preference"),
    ("prefer_b", "Prefer B"),
    ("strong_b", "Strongly prefer B"),
]

BASELINE_METHODS = ["SHAP", "RAG", "HYBRID", "COUNTERFACTUAL"]


# ── Central message handler ──────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        if await _handle_pairwise_callback(update, context):
            return
        if await _handle_rating(update, context):
            return
        return

    if update.message is None:
        return

    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    logger.info("User %s: %s", user_id, text[:80])

    if text.lower() in ("/start", "/help", "/begin"):
        context.user_data.clear()
        result = process_message(user_id, text)
        await safe_send(update, result["response"])
        if result["status"] == "ready":
            await _begin_session(update, context)
        return

    if context.user_data.get("awaiting_question"):
        await _handle_participant_question(update, context, text)
        return

    if context.user_data.get("pairwise_flow"):
        await update.message.reply_text("Please use the response buttons above before continuing.")
        return

    if await _handle_rating(update, context):
        return

    result = process_message(user_id, text)
    await safe_send(update, result["response"])

    if result["status"] == "ready":
        await _begin_session(update, context)


def _comparison_keyboard(step: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=f"cmp:{step}:{value}")]
        for value, label in PREFERENCE_OPTIONS
    ])


def _ask_another_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Yes", callback_data="again:yes"),
            InlineKeyboardButton("No", callback_data="again:no"),
        ]
    ])


def _planner_tools_to_methods(planner_result: dict) -> set:
    tools = planner_result.get("selected_tools") or []
    if not tools:
        selected_server = str(planner_result.get("selected_server", ""))
        tools = [part for part in selected_server.split("+") if part]

    mapping = {
        "shap": "SHAP",
        "rag": "RAG",
        "counterfactual": "COUNTERFACTUAL",
        "hybrid": "HYBRID",
        "hybrid_shap_rag_counterfactual": "HYBRID",
    }
    return {mapping.get(str(tool).lower()) for tool in tools if mapping.get(str(tool).lower())}


def _pick_baseline_method(planner_result: dict) -> str:
    planner_methods = _planner_tools_to_methods(planner_result)
    candidates = [method for method in BASELINE_METHODS if method not in planner_methods]
    if not candidates:
        candidates = BASELINE_METHODS[:]
    return random.choice(candidates)


def _question_context(paragraph_text: str, question: str) -> str:
    return (
        f"Text sample:\n\"{paragraph_text}\"\n\n"
        f"Participant question:\n\"{question}\""
    )


def _preference_winner(choice: str, label_a: str, label_b: str) -> str:
    if choice.endswith("_a"):
        return label_a
    if choice.endswith("_b"):
        return label_b
    return "no_preference"


def _append_interactive_evaluation(row: dict):
    csv_path = os.path.join(_ROOT, "logs", "interactive_evaluation_records.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "user_id",
        "session_id",
        "sample_index",
        "question_index",
        "paragraph_id",
        "paragraph_text",
        "participant_question",
        "prediction_label",
        "prediction_confidence",
        "response_a_type",
        "response_a_method",
        "response_a_text",
        "response_b_type",
        "response_b_method",
        "response_b_text",
        "baseline_method",
        "planner_tools",
        "planner_intent",
        "planner_rationale",
        "rating_clarity",
        "rating_clarity_winner",
        "rating_accuracy",
        "rating_accuracy_winner",
        "rating_helpfulness",
        "rating_helpfulness_winner",
        "rating_trust",
        "rating_trust_winner",
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        data = {key: row.get(key, "") for key in fieldnames}
        data["timestamp_utc"] = data["timestamp_utc"] or datetime.now(timezone.utc).isoformat()
        writer.writerow(data)


async def _send_pairwise_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    flow = context.user_data.get("pairwise_flow")
    if not flow:
        return

    step = flow["step"]
    if step >= len(PAIRWISE_CRITERIA):
        await _finish_pairwise_evaluation(update, context)
        return

    _, question = PAIRWISE_CRITERIA[step]
    await _send_message(
        update,
        context,
        _format_box("Please Compare Response A and Response B", question),
        reply_markup=_comparison_keyboard(step),
    )


async def _handle_pairwise_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    query = update.callback_query
    if query is None:
        return False

    raw = (query.data or "").strip()
    if raw.startswith("again:"):
        await query.answer()
        await _handle_ask_another(update, context, raw.split(":", 1)[1])
        return True

    if not raw.startswith("cmp:"):
        return False

    flow = context.user_data.get("pairwise_flow")
    if not flow:
        return False

    await query.answer()
    parts = raw.split(":", 2)
    if len(parts) != 3:
        return True

    try:
        step = int(parts[1])
    except ValueError:
        return True

    if step != flow.get("step", 0):
        await query.message.reply_text("Please answer the current comparison question.")
        return True

    choice = parts[2]
    criterion_key, _ = PAIRWISE_CRITERIA[step]
    flow["ratings"][criterion_key] = choice
    flow["step"] = step + 1

    if flow["step"] < len(PAIRWISE_CRITERIA):
        await _send_pairwise_prompt(update, context)
    else:
        await _finish_pairwise_evaluation(update, context)
    return True


async def _finish_pairwise_evaluation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    flow = context.user_data.get("pairwise_flow")
    if not flow:
        return

    label_a = flow["response_a_type"]
    label_b = flow["response_b_type"]
    ratings = flow["ratings"]

    _append_interactive_evaluation({
        "user_id": str(update.effective_user.id),
        "session_id": flow["session_id"],
        "sample_index": flow["sample_index"],
        "question_index": flow["question_index"],
        "paragraph_id": flow["paragraph_id"],
        "paragraph_text": flow["paragraph_text"],
        "participant_question": flow["participant_question"],
        "prediction_label": flow["prediction_label"],
        "prediction_confidence": flow["prediction_confidence"],
        "response_a_type": flow["response_a_type"],
        "response_a_method": flow["response_a_method"],
        "response_a_text": flow["response_a_text"],
        "response_b_type": flow["response_b_type"],
        "response_b_method": flow["response_b_method"],
        "response_b_text": flow["response_b_text"],
        "baseline_method": flow["baseline_method"],
        "planner_tools": ", ".join(flow.get("planner_tools", [])),
        "planner_intent": flow.get("planner_intent", ""),
        "planner_rationale": flow.get("planner_rationale", ""),
        "rating_clarity": ratings.get("clarity", ""),
        "rating_clarity_winner": _preference_winner(ratings.get("clarity", ""), label_a, label_b),
        "rating_accuracy": ratings.get("accuracy", ""),
        "rating_accuracy_winner": _preference_winner(ratings.get("accuracy", ""), label_a, label_b),
        "rating_helpfulness": ratings.get("helpfulness", ""),
        "rating_helpfulness_winner": _preference_winner(ratings.get("helpfulness", ""), label_a, label_b),
        "rating_trust": ratings.get("trust", ""),
        "rating_trust_winner": _preference_winner(ratings.get("trust", ""), label_a, label_b),
    })

    context.user_data.pop("pairwise_flow", None)
    await _send_message(
        update,
        context,
        _format_box("Question Complete", "Would you like to ask another question about this same person?"),
        reply_markup=_ask_another_keyboard(),
    )


async def _handle_ask_another(update: Update, context: ContextTypes.DEFAULT_TYPE, answer: str):
    if answer == "yes":
        context.user_data["awaiting_question"] = True
        sample_number = context.user_data.get("sample_index", 1)
        await _send_message(
            update,
            context,
            _format_box(
                f"Ask Another Question About Person {sample_number}",
                "Type any question about the text or prediction.",
            ),
        )
        return

    context.user_data["awaiting_question"] = False
    if context.user_data.get("sample_queue"):
        await _send_message(update, context, _format_box("Moving to the next text sample..."))
        await _run_next_sample(update, context)
        return

    await _send_message(update, context, _format_box("Study complete. Thank you for participating."))


async def _handle_participant_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str):
    if not question:
        await update.message.reply_text("Please type a question about the text or prediction.")
        return

    context.user_data["awaiting_question"] = False
    sample_number = context.user_data.get("sample_index", 1)
    question_index = context.user_data.get("question_index", 0) + 1
    context.user_data["question_index"] = question_index

    paragraph_id = context.user_data.get("current_paragraph_id", "")
    paragraph_text = context.user_data.get("current_paragraph_text", "")
    label = context.user_data.get("current_prediction_label", "unknown")
    conf = context.user_data.get("current_prediction_confidence", 0.0)

    await _send_message(update, context, _format_box("Generating Responses", "Please wait while two responses are generated."))

    try:
        planner_result, planner_answer = _run_planner_answer(paragraph_text, question)
        mentallama_answer = _run_mentallama_answer(paragraph_text, question)
    except Exception as exc:
        logger.exception("Failed to generate interactive responses: %s", exc)
        context.user_data["awaiting_question"] = True
        await _send_message(
            update,
            context,
            _format_box(
                "Generation Failed",
                "Something went wrong while generating the responses. Please try another question.",
            ),
        )
        return

    responses = [
        {
            "type": "planner",
            "method": "MCP",
            "text": planner_answer,
        },
        {
            "type": "mentallama",
            "method": "MentalLLaMA",
            "text": mentallama_answer,
        },
    ]
    random.shuffle(responses)

    await _send_message(
        update,
        context,
        _format_box(
            f"Question About Person {sample_number}",
            question,
        ),
    )
    await _pause()
    await _send_message(update, context, _format_box("Response A", responses[0]["text"]))
    await _pause()
    await _send_message(update, context, _format_box("Response B", responses[1]["text"]))
    await _pause()

    context.user_data["pairwise_flow"] = {
        "session_id": context.user_data.get("session_id"),
        "sample_index": sample_number,
        "question_index": question_index,
        "paragraph_id": paragraph_id,
        "paragraph_text": paragraph_text,
        "participant_question": question,
        "prediction_label": label,
        "prediction_confidence": conf,
        "response_a_type": responses[0]["type"],
        "response_a_method": responses[0]["method"],
        "response_a_text": responses[0]["text"],
        "response_b_type": responses[1]["type"],
        "response_b_method": responses[1]["method"],
        "response_b_text": responses[1]["text"],
        "baseline_method": "MentalLLaMA",
        "planner_tools": planner_result.get("selected_tools", []),
        "planner_intent": planner_result.get("intent", ""),
        "planner_rationale": planner_result.get("rationale", ""),
        "ratings": {"clarity": None, "accuracy": None, "helpfulness": None, "trust": None},
        "step": 0,
    }
    await _send_pairwise_prompt(update, context)


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
    total = min(5, len(samples))
    context.user_data.clear()
    context.user_data["sample_queue"] = samples[:total]
    context.user_data["sample_index"] = 0
    context.user_data["question_index"] = 0
    context.user_data["session_id"] = f"{user_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    await _run_next_sample(update, context)


def _pick_explanation_methods() -> list:
    methods = ["SHAP", "RAG", "HYBRID", "COUNTERFACTUAL", "MCP"]
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
    user_question: str = "",
) -> tuple:
    cache_enabled = not user_question
    cached = get_cached_explanation(paragraph_id, method) if cache_enabled else None
    if cached:
        console_log = f"Using cached explanation for paragraph {paragraph_id} with method {method}"
        logger.info(console_log)
        return None, cached

    prompt_text = _question_context(user_text, user_question) if user_question else user_text

    if method == "SHAP":
        model_result = explain_with_shap(user_text)
        _override_prediction(model_result, forced_label, forced_conf)
        _, _, generate_explanation = _get_shap_pipeline()
        explanation = generate_explanation(prompt_text, model_result)
        if cache_enabled:
            save_explanation(paragraph_id, method, explanation)
        return model_result, explanation

    if method == "RAG":
        rag_pipeline, generate_explanation, _ = _get_rag_pipeline()
        model_result = rag_pipeline(user_text)
        _override_prediction(model_result, forced_label, forced_conf)
        explanation = generate_explanation(prompt_text, model_result)
        if cache_enabled:
            save_explanation(paragraph_id, method, explanation)
        return model_result, explanation

    if method == "HYBRID":
        run_hybrid, _, _, generate_explanation = _get_hybrid_pipeline()
        model_result = run_hybrid(user_text)
        if hasattr(model_result, "shap_result") and model_result.shap_result:
            _override_prediction(model_result.shap_result, forced_label, forced_conf)
        if hasattr(model_result, "rag_result") and model_result.rag_result:
            _override_prediction(model_result.rag_result, forced_label, forced_conf)
        explanation = generate_explanation(prompt_text, model_result)
        if cache_enabled:
            save_explanation(paragraph_id, method, explanation)
        return model_result, explanation

    if method == "MCP":
        run_mcp_pipeline = _get_mcp_pipeline()
        result = run_mcp_pipeline(user_text, fallback=True, top_k=2, user_question=user_question)
        explanation = result.get("explanation", "No explanation returned.")
        if cache_enabled:
            save_explanation(paragraph_id, method, explanation)
        return result, explanation

    generate_counterfactuals, _, generate_explanation, _ = _get_cf_pipeline()
    cf_candidates = 2 if user_question else 3
    cf_attempts = 1 if user_question else 2
    model_result = generate_counterfactuals(user_text, n_candidates=cf_candidates, n_attempts=cf_attempts)
    if hasattr(model_result, "pred_label"):
        _override_prediction(model_result, forced_label, forced_conf)
    explanation = generate_explanation(prompt_text, model_result)
    if cache_enabled:
        save_explanation(paragraph_id, method, explanation)
    return model_result, explanation


def _run_planner_answer(paragraph_text: str, question: str) -> tuple:
    run_mcp_pipeline = _get_mcp_pipeline()
    result = run_mcp_pipeline(paragraph_text, fallback=True, top_k=2, user_question=question)
    return result, result.get("explanation", "No explanation returned.")


def _run_mentallama_answer(paragraph_text: str, question: str) -> str:
    tokenizer, model, device = _get_mentallama_pipeline()
    model_input = f"Consider this post: {paragraph_text.strip()} Question: {question.strip()}"

    inputs = tokenizer(
        model_input,
        return_tensors="pt",
        max_length=int(os.environ.get("MENTALLAMA_MAX_INPUT_TOKENS", "2048")),
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    do_sample = os.environ.get("MENTALLAMA_DO_SAMPLE", "false").lower() == "true"
    generation_kwargs = {
        "max_new_tokens": int(os.environ.get("MENTALLAMA_MAX_NEW_TOKENS", "1024")),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(os.environ.get("MENTALLAMA_TEMPERATURE", "0.7"))

    outputs = model.generate(**inputs, **generation_kwargs)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer or "No response returned."


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

    context.user_data["awaiting_question"] = True
    await _send_message(
        update,
        context,
        _format_box(
            f"Ask a Question About Person {sample_number}",
            "You can ask anything about the prediction, for example:\n"
            "- Why was this prediction made?\n"
            "- Which words mattered?\n"
            "- How could the prediction change?\n"
            "- Why is this not a different severity level?",
        ),
    )


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
