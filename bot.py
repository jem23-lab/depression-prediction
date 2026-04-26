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

from shared.conversation     import process_message
from shared.llm_client       import strip_markdown
from shared.depression_model import load as preload_model

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("depression_bot")


# ── Lazy pipeline importers ──────────────────────────────────────────
def _get_shap_pipeline():
    from shared.depression_model       import explain_with_shap, format_debug
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
    from architecture.hybrid_shap_rag_counterfactual.hybrid_pipeline  import run_hybrid_pipeline, format_hybrid_debug, format_hybrid_telegram_preview
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


# ── Central message handler ──────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text    = (update.message.text or "").strip()
    logger.info("User %s: %s", user_id, text[:80])

    result = process_message(user_id, text)
    await safe_send(update, result["response"])

    if result["status"] == "ready" and result.get("user_text"):
        await dispatch_pipeline(update, user_id, result["use_case"], result["user_text"])


async def dispatch_pipeline(update: Update, user_id: int, use_case: str, user_text: str):
    """Route to the correct use-case pipeline."""
    try:
        if use_case == "1":
            await run_shap_pipeline(update, user_id, user_text)
        elif use_case == "2":
            await run_rag_pipeline(update, user_id, user_text)
        elif use_case == "3":
            await run_hybrid_pipeline_handler(update, user_id, user_text)
        elif use_case == "4":
            await run_cf_pipeline(update, user_id, user_text)
        elif use_case == "5":
            await run_mcp_pipeline_handler(update, user_id, user_text)
        else:
            await update.message.reply_text(
                f"Use case {use_case} is not yet implemented.\n"
                "Please choose 1, 2, 3, 4, or 5 from the /assess menu."
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
    logger.info("UC1 SHAP for user %s", user_id)

    shap_result = explain_with_shap(user_text)
    logger.info(format_debug(shap_result))

    label      = shap_result.pred_label
    confidence = shap_result.pred_probs[shap_result.pred_label_idx]
    top_word   = shap_result.top_tokens[0]["token"] if shap_result.top_tokens else "—"

    await update.message.reply_text(
        f"🔬 Initial result (SHAP)\n"
        f"  Level      : {label}\n"
        f"  Confidence : {confidence*100:.1f}%\n"
        f"  Key word   : {top_word}\n\n"
        "Generating full explanation..."
    )

    explanation = generate_shap_explanation(user_text, shap_result)
    await safe_send(update, "Your Assessment (SHAP)\n" + "-"*35 + "\n\n" + explanation)

    if shap_result.top_tokens:
        lines = ["SHAP Token Breakdown\n"]
        for t in shap_result.top_tokens[:6]:
            arrow = "🔴" if t["shap"] > 0 else "🟢"
            line  = f"  {arrow} '{t['token']}'  SHAP={t['shap']:+.4f}  {t['direction']}"
            if t["note"]:
                line += f"\n       ({t['note']})"
            lines.append(line)
        await update.message.reply_text("\n".join(lines))

    await send_footer(update)


# ── Use Case 2: RAG ──────────────────────────────────────────────────
async def run_rag_pipeline(update: Update, user_id: int, user_text: str):
    rag_pipeline, generate_rag_explanation, format_rag_debug = _get_rag_pipeline()
    logger.info("UC2 RAG for user %s", user_id)

    rag_result = rag_pipeline(user_text)
    logger.info(format_rag_debug(rag_result))

    label     = rag_result.pred_label
    confidence = rag_result.pred_probs[rag_result.pred_label_idx]
    symptoms  = ", ".join(d.symptom_name for d in rag_result.retrieved_docs)

    await update.message.reply_text(
        f"📚 Initial result (RAG)\n"
        f"  Level           : {label}\n"
        f"  Confidence      : {confidence*100:.1f}%\n"
        f"  Matched symptoms: {symptoms}\n\n"
        "Generating full explanation..."
    )

    explanation = generate_rag_explanation(user_text, rag_result)
    await safe_send(update, "Your Assessment (RAG)\n" + "-"*35 + "\n\n" + explanation)

    lines = ["Retrieved Clinical Knowledge\n"]
    for i, doc in enumerate(rag_result.retrieved_docs, 1):
        lines.append(
            f"  {i}. {doc.symptom_name} ({doc.symptom_type})\n"
            f"     {doc.clinical_definition[:100]}..."
        )
    await update.message.reply_text("\n".join(lines))

    await send_footer(update)


# ── Use Case 3: Hybrid (SHAP + RAG + CF) ────────────────────────────
async def run_hybrid_pipeline_handler(update: Update, user_id: int, user_text: str):
    run_hybrid, format_debug, format_preview, generate_explanation = _get_hybrid_pipeline()
    logger.info("UC3 Hybrid for user %s", user_id)

    await update.message.reply_text(
        "Running all three XAI pipelines (SHAP + RAG + Counterfactual).\n"
        "This is the most comprehensive analysis — please allow 30-60 seconds..."
    )

    result = run_hybrid(user_text)
    logger.info(format_debug(result))

    # Preview showing what all three found
    await update.message.reply_text(format_preview(result))

    # Single unified LLM explanation synthesising all three
    explanation = generate_explanation(user_text, result)
    await safe_send(update, "Your Hybrid Assessment (SHAP + RAG + Counterfactual)\n" + "-"*50 + "\n\n" + explanation)

    # ── Detail breakdown: all three signals ──────────────────────────
    detail_lines = ["Detailed Evidence from All Three Signals\n"]

    # SHAP tokens
    if result.shap_result and result.shap_result.top_tokens:
        detail_lines.append("SHAP — Risk Tokens:")
        for t in result.shap_result.top_tokens[:5]:
            arrow = "🔴" if t["shap"] > 0 else "🟢"
            detail_lines.append(f"  {arrow} '{t['token']}' SHAP={t['shap']:+.4f}  {t['direction']}")
        detail_lines.append("")

    # RAG symptoms
    if result.rag_result and result.rag_result.retrieved_docs:
        detail_lines.append("RAG — Matched PHQ-8 Symptoms:")
        for i, doc in enumerate(result.rag_result.retrieved_docs, 1):
            detail_lines.append(
                f"  {i}. {doc.symptom_name} ({doc.symptom_type})\n"
                f"     {doc.clinical_definition[:90]}..."
            )
        detail_lines.append("")

    # CF candidates
    if result.cf_result and result.cf_result.candidates:
        detail_lines.append("Counterfactual — Candidates:")
        for i, c in enumerate(result.cf_result.candidates[:3], 1):
            status = "FLIP" if c["flip_success"] else "no flip"
            detail_lines.append(
                f"  {i}. [{status}] [{c['label']}] min={c['minimality']:.2f}\n"
                f"     \"{c['text'][:100]}\""
            )

    if len(detail_lines) > 1:
        await update.message.reply_text("\n".join(detail_lines))

    await send_footer(update)


# ── Use Case 4: Counterfactual ───────────────────────────────────────
async def run_cf_pipeline(update: Update, user_id: int, user_text: str):
    generate_counterfactuals, format_cf_debug, generate_cf_explanation, format_cf_preview = _get_cf_pipeline()
    logger.info("UC4 CF for user %s", user_id)

    await update.message.reply_text(
        "Generating counterfactuals (SHAP + multiple LLM calls).\n"
        "This may take 20-40 seconds..."
    )

    result = generate_counterfactuals(user_text, n_candidates=3, n_attempts=2)
    logger.info(format_cf_debug(result))

    await update.message.reply_text(format_cf_preview(result))

    explanation = generate_cf_explanation(user_text, result)
    await safe_send(update, "Your Counterfactual Assessment\n" + "-"*35 + "\n\n" + explanation)

    if result.candidates:
        lines = ["Counterfactual Candidates\n"]
        for i, c in enumerate(result.candidates[:3], 1):
            status = "FLIP" if c["flip_success"] else "no flip"
            lines.append(
                f"  {i}. [{status}] Predicted: {c['label']}\n"
                f"     Minimality: {c['minimality']:.2f}  Meaning kept: {c['semantic_sim']:.2f}\n"
                f"     \"{c['text'][:110]}\""
            )
        await update.message.reply_text("\n".join(lines))

    await send_footer(update)


# ── Use Case 5: MCP ─────────────────────────────────────────────────
async def run_mcp_pipeline_handler(update: Update, user_id: int, user_text: str):
    run_mcp_pipeline = _get_mcp_pipeline()
    logger.info("UC5 MCP for user %s", user_id)

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
    explanation = result.get("explanation", "")
    errors = result.get("errors", []) or []

    await update.message.reply_text(
        f"🧠 Initial result (MCP)\n"
        f"  Level           : {label}\n"
        f"  Confidence      : {confidence*100:.1f}%\n"
        f"  Selected server : {selected_server}\n"
        f"  Fallback used   : {'yes' if fallback_used else 'no'}\n\n"
        "Generating full explanation..."
    )

    if explanation:
        await safe_send(update, "Your Assessment (MCP)\n" + "-"*35 + "\n\n" + explanation)

    detail_lines = ["MCP Decision Details\n"]
    if rationale:
        detail_lines.append(f"Rationale: {rationale}")

    if errors:
        detail_lines.append("")
        detail_lines.append("Errors:")
        for i, e in enumerate(errors, 1):
            detail_lines.append(f"  {i}. {e}")

    if len(detail_lines) > 1:
        await update.message.reply_text("\n".join(detail_lines))

    await send_footer(update)


# ── Entry point ──────────────────────────────────────────────────────
def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    logger.info("Pre-loading depression model + SHAP explainer...")
    preload_model()
    logger.info("Model ready. Bot starting...")

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
