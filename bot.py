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


PARAGRAPHS = [
    {
        "id": "daic_woz_severe_321",
        "severity": "severe",
        "text": (
            "Lately, I’ve been feeling deeply depressed and weighed down by worries.My life feels"
            "heavy; I had to settle for a retail job because I couldn't find work in my field,"
            "and my daughter is currently battling cancer, leaving me frustrated and anxious. I"
            "haven't had a good night's sleep in a year, waking up every few hours. This severe "
            "fatigue leaves me groggy, lacking energy, and struggling to concentrate, causing me"
            "to forget things and make mistakes at work. I’ve experienced a painful shift in my"
            "social functioning and interests; I haven't gone out to dinner with a close friend "
            "in over a year. My self-worth is low—I don't know my best qualities and regret my "
            "lack of education— and my appetite is disrupted.Though I find moments of pure joy"
            " playing with my granddaughter, I mostly feel trapped by sadness and failure, coping "
            "by keeping quiet when struggling."
        )
    },
    {
        "id": "daic_woz_severe_346",
        "severity": "severe",
        "text": (
            "Lately, I’ve been feeling really sad, stressed, and miserable, carrying a severe "
            "heaviness from my past hardships and trauma. Sleeping is incredibly difficult; I "
            "lie awake dwelling on stressors or wake up shaking from intense nighttime panic "
            "attacks and vivid nightmares. This constant insomnia leaves me completely exhausted,"
            " moody, and struggling with severe fatigue, poor concentration, and a disrupted "
            "appetite. My self-worth is low—I often forget my good qualities, regret past choices"
            " like my last marriage, and feel a deep sense of failure. Socially, my family can be "
            "judgmental, and though I have a loving boyfriend, I struggle with an internal urge "
            "to withdraw. I have to forcefully push myself to attend auditions or make it to "
            "appointments because a part of me just wants to hide away. While I try to survive "
            "and stay busy to avoid feeling that life isn't worth it, everything feels like an "
            "agonizing struggle."
        )
    },
    {
        "id": "daic_woz_severe_348",
        "severity": "severe",
        "text": (
            "Lately, I’ve been feeling so tired, sad, depressed, and blue. I was diagnosed"
            " with depression about a year ago because I just couldn't pull myself out of it,"
            " and right now, I’m simply not happy. My sleep is terrible—I barely sleep at all,"
            " which leaves me feeling crazy, distracted, and struggling to cope with even mundane"
            " daily things. This severe lack of motivation, energy, and low self-worth makes me "
            "feel like I'm not my usual self. Even my appetite has been completely disrupted. "
            "To cope, I’ve started staying to myself and withdrawing, not going out with friends "
            "like I used to. My new relationship has lost its spark and just feels okay now, and "
            "the cold weather easily brings me down. Although going to therapy helps me get "
            "things off my chest, and I still feel proud and fulfilled by my children, everyday "
            "life currently feels like an overwhelming weight."
        )
    },
    {
        "id": "daic_woz_severe_362",
        "severity": "severe",
        "text": (
            "Lately, I’ve been feeling so-so, run down, and a bit stressed, struggling with"
            " ongoing depression, lack of interest, and too many regrets weighing on my mind. "
            "Sleep is never easy and always bad, leaving me tired, lethargic, and without energy."
            " It makes it incredibly hard to keep my thoughts in order or manage the absolute "
            "basics of my day. My appetite has also been severely disrupted. Because of my past"
            " trauma from a near-fatal stalker attack, I find myself battling complex cognitive"
            " and emotional patterns. To cope, I actively use the proactive tools from therapy "
            "and reach out to others rather than isolating myself. While I try to stay rational,"
            " calm, and reliable, I can't even remember the last time I felt genuinely happy. "
            "Despite these severe struggles with self-worth and exhaustion, my children remain "
            "my greatest pride, and working with animals still brings a brief smile to my face."
        )
    },
    {
        "id": "daic_woz_severe_367",
        "severity": "severe",
        "text": (
            "Lately, I feel pensive and down, carrying a heavy sense of hopelessness, severe "
            "depression, and a total loss of interest in life. Since losing my job and moving "
            "to LA, I feel like a complete failure who can't get back on my feet, leading to "
            "estrangement from my family and friends. I constantly dwell on past mistakes, "
            "especially a failed relationship that spiraled out of control, directing my anger"
            " inward with intense guilt. My mind is always overwhelmed; everything triggers "
            "painful memories, forcing me to double-task to keep my brain occupied, though "
            "concentrating remains extremely difficult. Physically, I am plagued by light, "
            "restless sleep, low energy, and appetite issues. To cope with this emotional "
            "instability and stay sober, I attend AA meetings to put my formless pain into "
            "words. I wonder if I'll ever feel like a normal, happy person again, remembering"
            " a Christmas years ago when life felt whole."
        )
    },
    {
        "id": "daic_woz_severe_426",
        "severity": "severe",
        "text": (
            "Lately, I’ve been feeling not good at all, carrying a deep sense of depression "
            "and hopelessness. Since being released from prison, I feel older, overwhelmed by "
            "responsibilities, and like my life isn't where it's supposed to be, which leaves "
            "me struggling with self-worth and feeling sorry for myself. Sleep is never easy, "
            "and I am plagued by severe restlessness, severe low energy, and appetite disruption."
            " My concentration is impaired, making it hard to focus, yet I remain fully "
            "determined not to give up. Because of past traumas and my PTSD diagnosis, "
            "I sometimes experience intense emotional instability and flashbacks of past "
            "violence, fights, and shootouts that inhibit me. To cope, I actively rely on"
            " therapy, which teaches me to step back and rationally assess situations instead "
            "of reacting purely on emotion. Although I still enjoy music and social gatherings, "
            "true happiness feels far away, anchored in memories of mutual love."
        )
    },
    {
        "id": "daic_woz_severe_440",
        "severity": "severe",
        "text": (
            "Lately, I’ve been feeling very stressed, moody, and irritable, with my mind "
            "constantly jumping from one thought to another, making concentration incredibly "
            "difficult. My son’s incarceration has been a devastating stressor, leaving me "
            "feeling deeply depressed and hopeless. Between worrying about my children and "
            "having racing thoughts, getting a good night’s sleep is nearly impossible, leaving"
            " me completely exhausted with very low energy. My appetite is significantly disrupted, "
            "and I struggle with feelings of failure and self-worth regarding marital arguments "
            "and parenting mistakes. Despite being naturally outgoing, I tend to withdraw and avoid "
            "talking entirely when upset, hiding my emotions. To cope and relax, I design jewelry. "
            "Though I heavily relied on therapy, losing my therapist due to funding cuts has left me "
            "to navigate this emotional instability completely alone. I try to remain a compassionate "
            "go-getter for my family, but right now, finding genuine happiness is a real struggle."
        )
    },
    {
        "id": "daic_woz_moderate_319",
        "severity": "moderate",
        "text": (
            "Lately, I’ve been feeling down and not like myself, struggling with a "
            "persistent sense of lethargy and low energy that often leaves me lying "
            "around. Since being diagnosed with depression a year ago, I face a regular "
            "loss of interest in activities, though watching USC football can still lift "
            "my spirits. My sleep is frequently disrupted, leaving me irritable and cranky, "
            "and I notice regular appetite changes. Almost daily, I face severe concentration "
            "difficulties. As an honest, straightforward person, I try to remove myself from "
            "annoying situations, but I carry many regrets about my past and a subtle sense of "
            "failure. Being currently unemployed and facing health and financial limitations has "
            "heavily restricted my life, and after recently losing my father—my last living "
            "parent—it’s been a long time since I felt truly happy. I remain deeply proud of my "
            "four adult children, though I constantly worry about them."
        )
    },
    {
        "id": "daic_woz_moderate_330",
        "severity": "moderate",
        "text": (
            "I describe myself as an introverted, patient, and curious student pursuing "
            "biological sciences. While I often tell others I’m doing well, I frequently feel"
            " down and overwhelmed by the daunting challenges ahead, and I am honestly not sure "
            "when I last felt truly happy. Privately, I struggle almost daily with severe "
            "feelings of failure, self-worth issues, and significant appetite changes. My "
            "concentration remains unaffected, but more than half the time, I experience a "
            "noticeable sense of restlessness or physical slowing. I also cope with occasional "
            "low energy, a mild loss of interest, and difficulty falling asleep, which leaves me "
            "feeling nervous and forgetful. Being an introvert, I handle these burdens by "
            "withdrawing into solitary coping behaviors like reading, listening to music, and "
            "taking night walks. Although I am proud of my academic achievements and have a "
            "supportive mentor, navigating these hidden emotional struggles feels very hard."
        )
    },
    {
        "id": "daic_woz_moderate_335",
        "severity": "moderate",
        "text": (
            "As an extroverted and creative actor, I’m usually an open book, but moving back "
            "to Los Angeles against my will to care for my sick father has been incredibly "
            "stressful, especially living with him and my brother in a hoarding environment. "
            "Lately, this situation has manifested in distressing, out-of-control stress dreams "
            "nearly every day, alongside severe, near-daily appetite disruptions. More than half "
            "the time, I contend with persistent fatigue and low energy, which weighs heavily "
            "on me. On several days, I find myself feeling down and mildly hopeless about my "
            "father's illness, struggling with intermittent concentration difficulties and a "
            "slight loss of interest in my regular activities. I also experience occasional "
            "self-worth issues, particularly with voicing my needs and standing up for myself. "
            "Despite these compounding emotional and environmental burdens, I strive to stay "
            "resilient, drawing strength from my talents, close friendships, and my natural "
            "ability to make the best of bad situations."
        )
    },
    {
        "id": "daic_woz_moderate_372",
        "severity": "moderate",
        "text": (
            "I am currently feeling pretty down and finding "
            "everything bittersweet. Ever since losing custody of my son, I struggle nearly "
            "every day with overwhelming feelings of failure and self-worth issues, feeling "
            "completely invisible and unacknowledged as a mother. I cry constantly, including at "
            "family events and in weekly therapy sessions. My small home environment is "
            "incredibly crowded and stressful, leading to frequent arguments with my husband, who "
            "is disappointed that I’ve lost my usual optimistic, fun spark. More than half the "
            "time, racing thoughts disrupt my sleep, forcing me to rely on medication, and I "
            "contend with a shortened attention span that impairs my concentration and a general "
            "loss of interest in life. I also experience occasional fatigue and appetite changes. "
            "Having no close friends further contributes to my social isolation. However, I "
            "remain dedicated to self-improvement; I take antidepressants, seek employment, and "
            "attend local literary events to fuel my dream of writing."
        )
    },
    {
        "id": "daic_woz_moderate_376",
        "severity": "moderate",
        "text": (
            "I am currently experiencing moderate depression, a condition I’ve managed with "
            "psychiatric medication for years, but I’m deeply struggling after the recent deaths "
            "of my mother and uncle. For more than half the days, I endure a pervasive loss of "
            "interest in activities I once loved, like reading or shopping, and frequently feel "
            "down and hopeless. My mind constantly runs on overtime, making me feel "
            "scatterbrained and causing severe concentration difficulties. This mental strain "
            "disrupts my sleep; I struggle to stay asleep, often getting only three to four hours"
            " a night. Consequently, I face persistent fatigue, low energy, and appetite issues, "
            "leading to emotional overeating. Although insurance barriers prevent me from seeing "
            "a therapist and I get irritated easily, I maintain confidence in my reliability and "
            "strong work ethic. I am actively trying to cope by keeping my mind busy, dieting, "
            "and walking along the beach."
        )
    },
    {
        "id": "daic_woz_moderate_412",
        "severity": "moderate",
        "text": (
            "I am noticing that my thoughts have been just not as positive lately, and I've "
            "been feeling pretty tired. My mind runs on overdrive, causing me to toss and turn "
            "all night while worrying about work, the economy, and what the future holds. "
            "Because of this, it often feels like a struggle just to get through the day. I "
            "find myself overeating quite a bit, which has changed my weight since I moved to "
            "Los Angeles from Texas for work. It is also hard to stay as focused as I used to be, "
            "and I feel a bit scatterbrained. Despite being naturally shy and keeping a low "
            "amount of contact with my remaining family, I actively try to cope by keeping my "
            "mind busy. I find genuine relaxation and a sense of relief by going to meetup groups, "
            "making new friends, and going out Latin dancing several times a week."
        )
    },
    {
        "id": "daic_woz_moderate_422",
        "severity": "moderate",
        "text": (
            "Lately, I feel physically crappy and down a lot of the time, which feels reasonable"
            " given my difficult health problems. Nearly every day, I experience severe fatigue, "
            "feeling as though I haven't slept for two days. Even though I sleep a lot, I never "
            "feel truly rested, and any moments of alertness are short-lived. As a deep thinker, "
            "my mind is constantly active, and I struggle with the realistic limitations my "
            "health places on my dream of becoming a veterinarian. Socially, I am a private "
            "person on the cusp between shy and outgoing. I intentionally withhold my "
            "difficulties from friends so I don't become a killjoy or make them uncomfortable. "
            "Despite these stressors and my regret over not finding a long-term life partner now "
            "that I am forty, I remain highly motivated. I am proud that I have persevered on my "
            "own, returning to school to complete my education and earning straight A’s."
        )
    }
]

USE_CASES = {
    "1": "SHAP",
    "2": "RAG",
    "3": "HYBRID",
    "4": "COUNTERFACTUAL",
    # "5": "MCP",
}

EVAL_CRITERIA = [
    ("clarity", "Clarity"),
    ("correctness", "Correctness (logical and factual alignment with question)"),
    ("helpfulness", "Helpfulness (how well it answers the user's intent)"),
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
        "1. Clarity",
        "2. Correctness (logical and factual alignment with question)",
        "3. Helpfulness (how well it answers the user's intent)",
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
