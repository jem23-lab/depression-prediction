"""
bot.py
────────────────────────────────────────────────────────────────────
Use Case 1 — SHAP-Only Depression Explanation Bot

Full pipeline:
  Telegram message (free text)
      → conversation.py     (FSM — collect raw user text)
      → depression_model.py (deproberta → softmax probs + SHAP tokens)
      → shap_explainer.py   (build prompt → Gemini → plain explanation)
      → Telegram reply

Environment variables required:
  TELEGRAM_BOT_TOKEN   your bot token from @BotFather
  GOOGLE_API_KEY       your Gemini API key

Run:
  pip install -r requirements.txt
  export TELEGRAM_BOT_TOKEN="..."
  export GOOGLE_API_KEY="..."
  python bot.py
"""

import os
import re
import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from conversation     import process_message, reset_session
from depression_model import explain_with_shap, format_debug_summary, LABEL_MAP
from shap_explainer   import generate_explanation


# ── Markdown safety helpers ──────────────────────────────────────────
def strip_markdown(text: str) -> str:
    """
    Removes Markdown symbols that Gemini may produce but Telegram's
    legacy Markdown parser chokes on (unmatched *, _, `, [, ]).
    Sends everything as plain text — safe and readable.
    """
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{1,3}(.*?)_{1,3}",   r"\1", text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r"`{1,3}(.*?)`{1,3}",   r"\1", text, flags=re.DOTALL)
    # Remove markdown links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove heading hashes
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.strip()


async def safe_send(update: Update, text: str, chunk_size: int = 4000):
    """
    Sends a plain-text message (no parse_mode), split into chunks if
    needed to stay under Telegram's 4096-char limit.
    """
    text = strip_markdown(text)
    for i in range(0, max(len(text), 1), chunk_size):
        await update.message.reply_text(text[i : i + chunk_size])

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("depression_bot")


# ── Pre-load model at startup (avoids cold-start lag per user) ───────
logger.info("Pre-loading deproberta model + SHAP explainer …")
from depression_model import _load as _preload_model
_preload_model()
logger.info("Model ready.")


# ── Telegram handlers ────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Central handler — drives FSM and runs pipeline when text is ready."""
    user_id = update.effective_user.id
    text    = (update.message.text or "").strip()

    logger.info("User %s: %s", user_id, text[:100])

    # Drive the FSM
    result = process_message(user_id, text)

    # Send immediate response (acknowledgement / question / status)
    await update.message.reply_text(strip_markdown(result["response"]))

    # When FSM says "ready" → run the full SHAP pipeline
    if result["status"] == "ready" and result.get("user_text"):
        await run_pipeline(update, user_id, result["user_text"])


async def run_pipeline(update: Update, user_id: int, user_text: str):
    """
    Runs deproberta + SHAP + Gemini and sends the explanation back.
    All LLM output is sent via safe_send() — no parse_mode — to avoid
    Telegram's Markdown parser choking on Gemini's formatting.
    """
    try:
        # ── Step 1: SHAP analysis ────────────────────────────────────
        logger.info("Running SHAP for user %s …", user_id)
        shap_result = explain_with_shap(user_text)
        logger.info("SHAP done:\n%s", format_debug_summary(shap_result))

        # ── Step 2: Quick preview (plain text, safe) ─────────────────
        label      = shap_result.pred_label
        confidence = shap_result.pred_probs[shap_result.pred_label_idx]
        top_token  = shap_result.top_tokens[0]["token"] if shap_result.top_tokens else "—"

        preview_msg = (
            f"🔬 Initial result\n"
            f"  Level      : {label}\n"
            f"  Confidence : {confidence*100:.1f}%\n"
            f"  Key word   : {top_token}\n\n"
            f"Generating your full explanation via Gemini..."
        )
        await update.message.reply_text(preview_msg)   # plain text, no parse_mode

        # ── Step 3: Gemini explanation ───────────────────────────────
        explanation = generate_explanation(
            user_query=user_text,
            shap_result=shap_result,
        )

        # ── Step 4: Send full explanation (stripped + chunked) ────────
        header = "📊 Your Personalised Assessment\n" + "─" * 35 + "\n\n"
        await safe_send(update, header + explanation)

        # ── Step 5: SHAP token breakdown (plain text) ─────────────────
        if shap_result.top_tokens:
            token_lines = ["🔍 SHAP Token Breakdown",
                           "Which words influenced the prediction:\n"]
            for t in shap_result.top_tokens[:6]:
                arrow = "🔴" if t["shap"] > 0 else "🟢"
                line  = f"  {arrow} '{t['token']}'  SHAP={t['shap']:+.4f}  {t['direction']}"
                if t["note"]:
                    line += f"\n       ({t['note']})"
                token_lines.append(line)
            await update.message.reply_text("\n".join(token_lines))

        # ── Step 6: Footer ────────────────────────────────────────────
        await update.message.reply_text(
            "─────────────────────────────\n"
            "Type /assess to run another analysis.\n"
            "Type /reset to clear your session."
        )

    except Exception as exc:
        logger.exception("Pipeline error for user %s: %s", user_id, exc)
        await update.message.reply_text(
            f"Something went wrong while generating your explanation.\n"
            f"Please try again with /assess.\n\n"
            f"Error: {str(exc)[:200]}"
        )


# ── Entry point ──────────────────────────────────────────────────────
def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "Get one from https://t.me/BotFather"
        )
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set.\n"
            "Get one from https://aistudio.google.com/app/apikey"
        )

    app = Application.builder().token(token).build()

    # Register all handlers
    app.add_handler(CommandHandler("start",  handle_message))
    app.add_handler(CommandHandler("help",   handle_message))
    app.add_handler(CommandHandler("assess", handle_message))
    app.add_handler(CommandHandler("reset",  handle_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running. Send /start in Telegram to begin.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
