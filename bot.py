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

_MENTALLAMA_DEFAULT_CACHE_DIR = "/scratch/apriyadar/huggingface"
os.environ.setdefault("HF_HOME", _MENTALLAMA_DEFAULT_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_MENTALLAMA_DEFAULT_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_MENTALLAMA_DEFAULT_CACHE_DIR, "transformers"))
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
from shared.training_examples import (
    PARAGRAPHS,
    get_cached_prediction,
    save_prediction,
)
from shared.depression_model import predict_proba, classify_severity

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

def _get_mcp_pipeline():
    from architecture.mcp_modular_agent.mcp_client import run_mcp_pipeline
    return run_mcp_pipeline


def _mentallama_load_kwargs() -> dict:
    os.makedirs(_MENTALLAMA_DEFAULT_CACHE_DIR, exist_ok=True)
    return {
        "cache_dir": _MENTALLAMA_DEFAULT_CACHE_DIR,
        "local_files_only": True,
        "use_safetensors": False,
    }


def _get_mentallama_pipeline():
    global _MENTALLAMA_PIPELINE
    if _MENTALLAMA_PIPELINE is not None:
        return _MENTALLAMA_PIPELINE

    model_id = os.environ.get("MENTALLAMA_MODEL_ID", "klyang/MentaLLaMA-chat-7B")
    load_kwargs = _mentallama_load_kwargs()

    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, **load_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = dict(load_kwargs)
        torch_dtype = os.environ.get("MENTALLAMA_TORCH_DTYPE", "").strip()
        if torch_dtype:
            if not hasattr(torch, torch_dtype):
                raise ValueError(f"Unsupported MENTALLAMA_TORCH_DTYPE={torch_dtype!r}")
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        model_kwargs["device_map"] = "auto"
        offload_dir = os.path.join(_MENTALLAMA_DEFAULT_CACHE_DIR, "offload")
        os.makedirs(offload_dir, exist_ok=True)
        model_kwargs["offload_folder"] = offload_dir
        model_kwargs["offload_state_dict"] = True

        model = LlamaForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        logger.exception(
            "Failed to load MentaLLaMA model_id=%s from local cache_dir=%s: %s",
            model_id,
            _MENTALLAMA_DEFAULT_CACHE_DIR,
            exc,
        )
        raise

    device = next(model.parameters()).device
    model.eval()

    _MENTALLAMA_PIPELINE = tokenizer, model, device
    return _MENTALLAMA_PIPELINE


# ── Helpers ──────────────────────────────────────────────────────────
async def safe_send(update: Update, text: str, chunk_size: int = 4000):
    """Strip Markdown symbols and chunk-send as plain text."""
    text = strip_markdown(text)
    for i in range(0, max(len(text), 1), chunk_size):
        await update.message.reply_text(text[i: i + chunk_size], parse_mode="HTML")


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


QUESTION_TYPES = [
    (
        "parts",
        "Which parts of Text Sample (Person {person}) contributed most to this prediction?",
    ),
    (
        "change",
        "What would need to be different in Text Sample (Person {person}) for the prediction to change?",
    ),
    (
        "symptoms",
        "Which symptoms or behaviours described in Text Sample (Person {person}) could have led to this prediction?",
    ),
    (
        "findings",
        "What are the main findings and pieces of evidence in Text Sample (Person {person}) that led to this result?",
    ),
]

SYSTEM_AGENTIC_MCP = "Agentic MCP XAI"
SYSTEM_MENTALLAMA = "MentalLLaMA"
EXPERIMENT_SYSTEMS = (SYSTEM_AGENTIC_MCP, SYSTEM_MENTALLAMA)

EXPERIMENT_RATING_STATEMENTS = [
    ("answered", "The explanation directly answered the question."),
    ("supported", "The explanation was supported by details in the text sample and was consistent with the displayed prediction."),
    ("understandable", "The explanation was easy to understand."),
    ("confidence", "The explanation gave me an appropriate level of confidence in the prediction."),
]

LIKERT_OPTIONS = [
    (1, "Strongly disagree"),
    (2, "Disagree"),
    (3, "Neither agree nor disagree"),
    (4, "Agree"),
    (5, "Strongly agree"),
]

SEVERITY_DISTRIBUTIONS = [
    {"severe": 3, "moderate": 3, "none": 2},
    {"severe": 3, "moderate": 2, "none": 3},
    {"severe": 2, "moderate": 3, "none": 3},
]

EXPERIMENT_LOG_PATH = os.path.join(_ROOT, "logs", "within_participant_experiment_records.csv")


# ── Central message handler ──────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        if await _handle_experiment_callback(update, context):
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
        await update.message.reply_text("Please choose one of the question buttons above.")
        return

    result = process_message(user_id, text)
    await safe_send(update, result["response"])

    if result["status"] == "ready":
        await _begin_session(update, context)

def _question_by_id(question_id: str):
    for qid, template in QUESTION_TYPES:
        if qid == question_id:
            return qid, template
    return QUESTION_TYPES[0]


def _question_text(question_id: str, person: int) -> str:
    _, template = _question_by_id(question_id)
    return template.format(person=person)


def _current_block(sample_index: int) -> int:
    return 1 if sample_index <= 4 else 2


def _question_choice_keyboard(displayed_options: list[dict]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Option {item['position']}", callback_data=f"qsel:{item['id']}")]
        for item in displayed_options
    ])


def _likert_keyboard(step: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=f"exp_rate:{step}:{score}")]
        for score, label in LIKERT_OPTIONS
    ])


def _experiment_rating_prompt(flow: dict) -> str:
    step = flow["step"]
    ratings = flow["ratings"]
    if step >= len(EXPERIMENT_RATING_STATEMENTS):
        lines = [
            f"- {label}: {ratings.get(key)}"
            for key, label in EXPERIMENT_RATING_STATEMENTS
        ]
        return _format_box("Rate This Explanation", "\n".join(lines))

    _, statement = EXPERIMENT_RATING_STATEMENTS[step]
    return _format_box(
        "Rate This Explanation",
        _join_box_lines([
            f"Statement {step + 1} of {len(EXPERIMENT_RATING_STATEMENTS)}",
            "",
            statement,
            "",
            "Choose one response.",
        ]),
    )


def _assign_experiment_system(context: ContextTypes.DEFAULT_TYPE, question_id: str) -> str:
    assignments = context.user_data.get("question_system_assignments") or {}
    systems_by_block = assignments.get(question_id) or {}
    block_index = _current_block(context.user_data.get("sample_index", 1))
    return systems_by_block.get(str(block_index), SYSTEM_AGENTIC_MCP)


def _normalize_severity(value: str) -> str:
    raw = (value or "").strip().lower()
    if "severe" in raw:
        return "severe"
    if "moderate" in raw:
        return "moderate"
    return "none"


def _read_experiment_counts() -> tuple[dict, set]:
    passage_counts: dict[str, int] = {}
    sessions: set[str] = set()
    if not os.path.exists(EXPERIMENT_LOG_PATH):
        return passage_counts, sessions

    with open(EXPERIMENT_LOG_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            session_id = row.get("session_id", "")
            paragraph_id = row.get("paragraph_id", "")
            if session_id:
                sessions.add(session_id)
            if paragraph_id:
                passage_counts[paragraph_id] = passage_counts.get(paragraph_id, 0) + 1
    return passage_counts, sessions


def _pick_severity_distribution(session_count: int) -> dict:
    return SEVERITY_DISTRIBUTIONS[session_count % len(SEVERITY_DISTRIBUTIONS)]


def _select_passages_for_session() -> list[dict]:
    passage_counts, sessions = _read_experiment_counts()
    distribution = _pick_severity_distribution(len(sessions))
    by_severity: dict[str, list[dict]] = {"severe": [], "moderate": [], "none": []}
    for paragraph in PARAGRAPHS:
        by_severity[_normalize_severity(paragraph.get("severity", ""))].append(paragraph)

    selected: list[dict] = []
    for severity, needed in distribution.items():
        candidates = by_severity.get(severity, [])[:]
        random.shuffle(candidates)
        candidates.sort(key=lambda item: passage_counts.get(item["id"], 0))
        selected.extend(candidates[:needed])

    if len(selected) < 8:
        selected_ids = {item["id"] for item in selected}
        fallback = [item for item in PARAGRAPHS if item["id"] not in selected_ids]
        random.shuffle(fallback)
        fallback.sort(key=lambda item: passage_counts.get(item["id"], 0))
        selected.extend(fallback[:8 - len(selected)])

    random.shuffle(selected)
    return selected[:8]


def _build_question_system_assignments() -> dict:
    question_ids = [qid for qid, _ in QUESTION_TYPES]
    block1_systems = [SYSTEM_AGENTIC_MCP, SYSTEM_AGENTIC_MCP, SYSTEM_MENTALLAMA, SYSTEM_MENTALLAMA]
    random.shuffle(block1_systems)
    return {
        qid: {
            "1": system,
            "2": SYSTEM_MENTALLAMA if system == SYSTEM_AGENTIC_MCP else SYSTEM_AGENTIC_MCP,
        }
        for qid, system in zip(question_ids, block1_systems)
    }


def _allowed_question_ids_for_streak(context: ContextTypes.DEFAULT_TYPE, question_ids: list[str]) -> list[str]:
    system_history = context.user_data.get("system_history") or []
    if len(system_history) < 2 or system_history[-1] != system_history[-2]:
        return question_ids

    assignments = context.user_data.get("question_system_assignments") or {}
    block_index = str(_current_block(context.user_data.get("sample_index", 1)))
    required_system = SYSTEM_MENTALLAMA if system_history[-1] == SYSTEM_AGENTIC_MCP else SYSTEM_AGENTIC_MCP
    allowed = [
        qid for qid in question_ids
        if (assignments.get(qid) or {}).get(block_index) == required_system
    ]
    return allowed or question_ids


def _append_experiment_evaluation(row: dict):
    csv_path = os.path.join(_ROOT, "logs", "within_participant_experiment_records.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "user_id",
        "session_id",
        "block_index",
        "trial_index",
        "paragraph_id",
        "paragraph_text",
        "prediction_label",
        "prediction_confidence",
        "question_id",
        "question_text",
        "explanation_system",
        "explanation_text",
        "rating_answered",
        "rating_supported",
        "rating_understandable",
        "rating_confidence",
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        data = {key: row.get(key, "") for key in fieldnames}
        data["timestamp_utc"] = data["timestamp_utc"] or datetime.now(timezone.utc).isoformat()
        writer.writerow(data)


async def _handle_experiment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    query = update.callback_query
    if query is None:
        return False

    raw = (query.data or "").strip()
    if raw.startswith("qsel:"):
        await query.answer()
        await _handle_question_selection(update, context, raw.split(":", 1)[1])
        return True

    if raw.startswith("exp_rate:"):
        await query.answer()
        await _handle_experiment_rating(update, context, raw)
        return True

    return False


async def _handle_question_selection(update: Update, context: ContextTypes.DEFAULT_TYPE, question_id: str):
    if not context.user_data.get("awaiting_question"):
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text("This question selection is no longer active.")
        return

    sample_number = context.user_data.get("sample_index", 1)
    block_index = _current_block(sample_number)
    remaining = context.user_data.get("block_remaining_question_ids") or []
    if question_id not in remaining:
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text("Please choose one of the currently available questions.")
        return
    allowed_ids = _allowed_question_ids_for_streak(context, remaining)
    if question_id not in allowed_ids:
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text("Please choose one of the currently available questions.")
        return

    remaining.remove(question_id)
    context.user_data["block_remaining_question_ids"] = remaining
    context.user_data["awaiting_question"] = False

    paragraph_id = context.user_data.get("current_paragraph_id", "")
    paragraph_text = context.user_data.get("current_paragraph_text", "")
    label = context.user_data.get("current_prediction_label", "unknown")
    conf = context.user_data.get("current_prediction_confidence", 0.0)
    question = _question_text(question_id, sample_number)
    system = _assign_experiment_system(context, question_id)

    await _send_message(update, context, _format_box("Selected Question", question))
    await _send_message(update, context, _format_box("Generating Explanation", "Please wait while the response is generated."))

    try:
        if system == SYSTEM_AGENTIC_MCP:
            _, explanation = _run_planner_answer(paragraph_text, question)
        else:
            explanation = _run_mentallama_answer(paragraph_text, question)
    except Exception as exc:
        logger.exception("Failed to generate %s explanation: %s", system, exc)
        if question_id not in context.user_data.get("block_remaining_question_ids", []):
            context.user_data.setdefault("block_remaining_question_ids", []).append(question_id)
        context.user_data["awaiting_question"] = True
        await _send_message(
            update,
            context,
            _format_box(
                "Generation Failed",
                "Something went wrong while generating the explanation. Please choose a question again.",
            ),
        )
        return

    context.user_data.setdefault("system_history", []).append(system)

    await _pause()
    await _send_message(update, context, _format_box("Explanation", explanation))
    await _pause()

    context.user_data["experiment_rating_flow"] = {
        "session_id": context.user_data.get("session_id"),
        "block_index": block_index,
        "trial_index": sample_number,
        "paragraph_id": paragraph_id,
        "paragraph_text": paragraph_text,
        "prediction_label": label,
        "prediction_confidence": conf,
        "question_id": question_id,
        "question_text": question,
        "explanation_system": system,
        "explanation_text": explanation,
        "ratings": {key: None for key, _ in EXPERIMENT_RATING_STATEMENTS},
        "step": 0,
        "prompt_message_id": None,
    }

    sent = await _send_message(
        update,
        context,
        _experiment_rating_prompt(context.user_data["experiment_rating_flow"]),
        reply_markup=_likert_keyboard(0),
    )
    if sent is not None:
        context.user_data["experiment_rating_flow"]["prompt_message_id"] = sent.message_id


async def _handle_experiment_rating(update: Update, context: ContextTypes.DEFAULT_TYPE, raw: str):
    flow = context.user_data.get("experiment_rating_flow")
    if not flow:
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text("This rating question is no longer active.")
        return

    parts = raw.split(":", 2)
    if len(parts) != 3:
        return

    try:
        step = int(parts[1])
        score = int(parts[2])
    except ValueError:
        return

    if step != flow.get("step", 0):
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text("Please answer the current rating statement.")
        return

    if score < 1 or score > 5:
        return

    criterion_key, _ = EXPERIMENT_RATING_STATEMENTS[step]
    flow["ratings"][criterion_key] = score
    flow["step"] = step + 1

    reply_message = update.callback_query.message if update.callback_query else None
    if reply_message is None:
        return

    prompt = _experiment_rating_prompt(flow)
    reply_markup = _likert_keyboard(flow["step"]) if flow["step"] < len(EXPERIMENT_RATING_STATEMENTS) else None
    try:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=flow["prompt_message_id"],
            text=prompt,
            reply_markup=reply_markup,
        )
    except Exception:
        sent = await reply_message.reply_text(prompt, reply_markup=reply_markup)
        flow["prompt_message_id"] = sent.message_id

    if flow["step"] < len(EXPERIMENT_RATING_STATEMENTS):
        return

    await _finish_experiment_trial(update, context)


async def _finish_experiment_trial(update: Update, context: ContextTypes.DEFAULT_TYPE):
    flow = context.user_data.get("experiment_rating_flow")
    if not flow:
        return

    ratings = flow["ratings"]
    _append_experiment_evaluation({
        "user_id": str(update.effective_user.id),
        "session_id": flow["session_id"],
        "block_index": flow["block_index"],
        "trial_index": flow["trial_index"],
        "paragraph_id": flow["paragraph_id"],
        "paragraph_text": flow["paragraph_text"],
        "prediction_label": flow["prediction_label"],
        "prediction_confidence": flow["prediction_confidence"],
        "question_id": flow["question_id"],
        "question_text": flow["question_text"],
        "explanation_system": flow["explanation_system"],
        "explanation_text": flow["explanation_text"],
        "rating_answered": ratings.get("answered", ""),
        "rating_supported": ratings.get("supported", ""),
        "rating_understandable": ratings.get("understandable", ""),
        "rating_confidence": ratings.get("confidence", ""),
    })

    context.user_data.pop("experiment_rating_flow", None)
    for key in [
        "current_paragraph_id",
        "current_paragraph_text",
        "current_paragraph_severity",
        "current_prediction_label",
        "current_prediction_confidence",
        "current_question_options",
    ]:
        context.user_data.pop(key, None)

    if context.user_data.get("sample_queue"):
        await _send_message(update, context, _format_box("Trial Complete", "Moving to the next text sample."))
        await _pause()
        await _run_next_sample(update, context)
        return

    await _send_message(update, context, _format_box("Study complete. Thank you for participating."))


async def _begin_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    samples = _select_passages_for_session()
    context.user_data.clear()
    context.user_data["sample_queue"] = samples
    context.user_data["sample_index"] = 0
    context.user_data["question_system_assignments"] = _build_question_system_assignments()
    context.user_data["system_history"] = []
    context.user_data["session_id"] = f"{user_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    await _run_next_sample(update, context)


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


def _run_planner_answer(paragraph_text: str, question: str) -> tuple:
    run_mcp_pipeline = _get_mcp_pipeline()
    result = run_mcp_pipeline(paragraph_text, fallback=True, top_k=2, user_question=question)
    return result, result.get("explanation", "No explanation returned.")


def _clean_mentallama_answer(answer: str) -> str:
    answer = (answer or "").strip()
    answer = re.sub(
        r"^\s*(?:[?¿]\s*)?(?:reasoning|reason|answer|response)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    return answer


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
    answer = _clean_mentallama_answer(tokenizer.decode(generated_tokens, skip_special_tokens=True))
    return answer or "No response returned."


async def _run_next_sample(update: Update, context: ContextTypes.DEFAULT_TYPE):
    queue = context.user_data.get("sample_queue") or []
    if not queue:
        await _send_message(update, context, _format_box("Study complete. Thank you for participating."))
        return

    selected_paragraph = queue.pop(0)
    context.user_data["sample_queue"] = queue
    context.user_data["sample_index"] = context.user_data.get("sample_index", 0) + 1
    sample_number = context.user_data.get("sample_index", 1)
    block_index = _current_block(sample_number)

    if context.user_data.get("current_block_index") != block_index:
        context.user_data["current_block_index"] = block_index
        context.user_data["block_remaining_question_ids"] = [qid for qid, _ in QUESTION_TYPES]

    paragraph_id = selected_paragraph["id"]
    paragraph_text = selected_paragraph["text"]
    paragraph_severity = selected_paragraph["severity"]

    label, conf = _prediction_from_text(paragraph_id, paragraph_text)

    context.user_data["current_paragraph_id"] = paragraph_id
    context.user_data["current_paragraph_text"] = paragraph_text
    context.user_data["current_paragraph_severity"] = paragraph_severity
    context.user_data["current_prediction_label"] = label
    context.user_data["current_prediction_confidence"] = conf

    await _send_message(
        update,
        context,
        _format_box(
            _format_title("Text Sample — Person", str(sample_number)),
            _format_for_display(paragraph_text),
        ),
    )
    await _pause()

    label_sentence = (
        f"Person {sample_number} does not show signs of depression."
        if label == "not depression" else
        f"Person {sample_number} shows signs of {label} depression."
    )
    await _send_message(update, context, _format_box("Prediction", label_sentence))
    await _pause()

    context.user_data["awaiting_question"] = True
    remaining_ids = _allowed_question_ids_for_streak(
        context,
        list(context.user_data.get("block_remaining_question_ids") or []),
    )
    displayed_options = [
        {"id": qid, "text": _question_text(qid, sample_number)}
        for qid in remaining_ids
    ]
    random.shuffle(displayed_options)
    for pos, item in enumerate(displayed_options, 1):
        item["position"] = pos
    context.user_data["current_question_options"] = displayed_options

    option_lines = [
        "What would you most like to understand about this prediction?",
        "",
    ]
    for item in displayed_options:
        option_lines.append(f"Option {item['position']}: {item['text']}")
    option_lines.extend(["", "Please select one question:"])

    await _send_message(
        update,
        context,
        _format_box(
            f"Question Selection — Block {block_index}",
            "\n".join(option_lines),
        ),
        reply_markup=_question_choice_keyboard(displayed_options),
    )

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
