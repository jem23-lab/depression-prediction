"""
bot.py  (root-level, shared by all use cases)
────────────────────────────────────────────────────────────────────
Single Telegram bot that lets the user pick their explanation method
and routes to the correct pipeline.

Architecture:
  conversation.py   → FSM: collect use-case choice + user text
  ┌─────────────────────────────────────────┐
  │ use_case == "1"  → SHAP pipeline        │
  │ use_case == "2"  → RAG pipeline         │
  │ use_case == "3"  → (hybrid, coming soon)│
  └─────────────────────────────────────────┘
  shared/llm_client.py  → Gemini (all routes)

Run:
  export TELEGRAM_BOT_TOKEN="..."
  export GOOGLE_API_KEY="..."
  python bot.py
"""

import os
import sys
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ── Path setup so shared/ and explainer subfolders are importable ────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from shared.conversation   import process_message, reset_session
from shared.llm_client     import strip_markdown
from shared.depression_model import load as preload_model

# ── Lazy imports for each use case (only loaded when needed) ─────────
def _get_shap_pipeline():
    from shared.depression_model   import explain_with_shap, format_debug
    from architecture.shap_explainer.shap_explainer import generate_shap_explanation
    return explain_with_shap, format_debug, generate_shap_explanation

def _get_rag_pipeline():
    from architecture.rag_explainer.rag_explainer import (
        run_rag_pipeline, generate_rag_explanation, format_rag_debug
    )
    return run_rag_pipeline, generate_rag_explanation, format_rag_debug


# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("depression_bot")


# ── Safe Telegram send ───────────────────────────────────────────────
async def safe_send(update: Update, text: str, chunk_size: int = 4000):
    """Strip Markdown and chunk-send plain text to avoid parse errors."""
    text = strip_markdown(text)
    for i in range(0, max(len(text), 1), chunk_size):
        await update.message.reply_text(text[i : i + chunk_size])


# ── Handlers ─────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text    = (update.message.text or "").strip()
    logger.info("User %s [%s]: %s", user_id, text[:60])

    result = process_message(user_id, text)

    # Always send the FSM response first
    await safe_send(update, result["response"])

    # When FSM signals ready → dispatch to the chosen pipeline
    if result["status"] == "ready" and result.get("user_text"):
        await dispatch_pipeline(update, user_id, result["use_case"], result["user_text"])


async def dispatch_pipeline(
    update: Update, user_id: int, use_case: str, user_text: str
):
    """Route to the correct explainer pipeline based on use_case."""
    try:
        if use_case == "1":
            await run_shap_pipeline(update, user_id, user_text)
        elif use_case == "2":
            await run_rag_pipeline(update, user_id, user_text)
        else:
            await update.message.reply_text(
                f"Use case {use_case} is not yet implemented. "
                "Try /assess and choose 1 or 2."
            )
    except Exception as exc:
        logger.exception("Pipeline error (UC%s) for user %s: %s", use_case, user_id, exc)
        await update.message.reply_text(
            f"Something went wrong.\n\nError: {str(exc)[:400]}\n\n"
            "Please try /assess again or check the terminal logs."
        )


# ── Use Case 1: SHAP ─────────────────────────────────────────────────
async def run_shap_pipeline(update: Update, user_id: int, user_text: str):
    explain_with_shap, format_debug, generate_shap_explanation = _get_shap_pipeline()

    logger.info("Running SHAP pipeline for user %s", user_id)
    shap_result = explain_with_shap(user_text)
    logger.info(format_debug(shap_result))

    label      = shap_result.pred_label
    confidence = shap_result.pred_probs[shap_result.pred_label_idx]
    top_word   = shap_result.top_tokens[0]["token"] if shap_result.top_tokens else "—"

    # Preview
    await update.message.reply_text(
        f"🔬 Initial result (SHAP)\n"
        f"  Level      : {label}\n"
        f"  Confidence : {confidence*100:.1f}%\n"
        f"  Key word   : {top_word}\n\n"
        "Generating full explanation..."
    )

    # LLM explanation
    explanation = generate_shap_explanation(user_text, shap_result)
    await safe_send(update, "📊 Your Assessment (SHAP)\n" + "─"*35 + "\n\n" + explanation)

    # Token breakdown
    if shap_result.top_tokens:
        lines = ["🔍 SHAP Token Breakdown", "Words that influenced the prediction:\n"]
        for t in shap_result.top_tokens[:6]:
            arrow = "🔴" if t["shap"] > 0 else "🟢"
            line  = f"  {arrow} '{t['token']}'  SHAP={t['shap']:+.4f}  {t['direction']}"
            if t["note"]:
                line += f"\n       ({t['note']})"
            lines.append(line)
        await update.message.reply_text("\n".join(lines))

    await _send_footer(update)


# ── Use Case 2: RAG ──────────────────────────────────────────────────
async def run_rag_pipeline(update: Update, user_id: int, user_text: str):
    rag_pipeline, generate_rag_explanation, format_rag_debug = _get_rag_pipeline()

    logger.info("Running RAG pipeline for user %s", user_id)
    rag_result = rag_pipeline(user_text)
    logger.info(format_rag_debug(rag_result))

    label      = rag_result.pred_label
    confidence = rag_result.pred_probs[rag_result.pred_label_idx]
    symptoms   = ", ".join(d.symptom_name for d in rag_result.retrieved_docs)

    # Preview
    await update.message.reply_text(
        f"📚 Initial result (RAG)\n"
        f"  Level           : {label}\n"
        f"  Confidence      : {confidence*100:.1f}%\n"
        f"  Matched symptoms: {symptoms}\n\n"
        "Generating full explanation..."
    )

    # LLM explanation
    explanation = generate_rag_explanation(user_text, rag_result)
    await safe_send(update, "📊 Your Assessment (RAG)\n" + "─"*35 + "\n\n" + explanation)

    # Retrieved symptom breakdown
    lines = ["📖 Retrieved Clinical Knowledge", "Symptoms matched to your message:\n"]
    for i, doc in enumerate(rag_result.retrieved_docs, 1):
        lines.append(
            f"  {i}. {doc.symptom_name} ({doc.symptom_type})\n"
            f"     {doc.clinical_definition[:100]}..."
        )
    await update.message.reply_text("\n".join(lines))

    await _send_footer(update)


async def _send_footer(update: Update):
    await update.message.reply_text(
        "─────────────────────────────\n"
        "Type /assess to run another analysis.\n"
        "Type /reset to clear your session."
    )


# ── Entry point ──────────────────────────────────────────────────────
def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    # Pre-load the shared depression model at startup
    logger.info("Pre-loading depression model + SHAP explainer…")
    preload_model()
    logger.info("Model ready. Bot starting…")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start",  handle_message))
    app.add_handler(CommandHandler("help",   handle_message))
    app.add_handler(CommandHandler("assess", handle_message))
    app.add_handler(CommandHandler("reset",  handle_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running. Send /start in Telegram.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
