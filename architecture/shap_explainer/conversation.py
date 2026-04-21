"""
conversation.py
────────────────────────────────────────────────────────────────────
Manages the Telegram conversation flow for Use Case 1.

Unlike a form-based approach, the deproberta model takes RAW TEXT as
input — so we simply ask the user to describe how they're feeling in
their own words, then pass that text directly to the model.

States:
  IDLE       → waiting for /start or /assess
  WAITING    → prompted user to share their feelings; waiting for text
  DONE       → text received, pipeline can run
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Session state ───────────────────────────────────────────────────
@dataclass
class UserSession:
    user_id:    int
    state:      str = "IDLE"       # IDLE | WAITING | PROCESSING | DONE
    user_text:  str = ""           # the raw text the user shared


# ── Session registry ────────────────────────────────────────────────
_sessions: dict[int, UserSession] = {}


def get_session(user_id: int) -> UserSession:
    if user_id not in _sessions:
        _sessions[user_id] = UserSession(user_id=user_id)
    return _sessions[user_id]


def reset_session(user_id: int):
    _sessions[user_id] = UserSession(user_id=user_id)


# ── FSM message processor ───────────────────────────────────────────
def process_message(user_id: int, text: str) -> dict:
    """
    Drives the conversation FSM.

    Returns:
      {
        "response"     : str,        message to send back to the user
        "parse_mode"   : "Markdown",
        "status"       : "idle" | "waiting" | "ready" | "error",
        "user_text"    : str | None, raw text to analyse (when status="ready")
      }
    """
    session = get_session(user_id)
    text    = text.strip()

    # ── /start or /help ─────────────────────────────────────────────
    if text.lower() in ("/start", "/help"):
        reset_session(user_id)
        return _reply(
            "👋 *Welcome to the Depression Screening Assistant*\n\n"
            "This tool uses an AI model (`deproberta-large-depression`) combined "
            "with *SHAP explanations* to help you understand which words in your "
            "message signal emotional distress — and *why*.\n\n"
            "📝 Type */assess* to begin.\n"
            "🔄 Type */reset* to start over.\n\n"
            "⚠️ _This is not a clinical tool. If you are in crisis, please reach "
            "out to a mental health professional or a helpline immediately._",
            status="idle",
        )

    # ── /reset ───────────────────────────────────────────────────────
    if text.lower() == "/reset":
        reset_session(user_id)
        return _reply("🔄 Session reset. Type /assess to start again.", status="idle")

    # ── /assess ──────────────────────────────────────────────────────
    if text.lower() == "/assess":
        reset_session(user_id)
        session = get_session(user_id)
        session.state = "WAITING"
        return _reply(
            "🩺 *Tell me how you're feeling*\n\n"
            "Please write a few sentences describing your mood, energy, or "
            "what has been on your mind recently. Be as honest as you like — "
            "there are no right or wrong answers.\n\n"
            "_Example: \"I've been feeling really tired and empty lately. "
            "I don't enjoy things I used to love and I can't concentrate.\"_",
            status="waiting",
        )

    # ── WAITING: user shares their feelings ──────────────────────────
    if session.state == "WAITING":
        # Minimum length guard
        if len(text.split()) < 4:
            return _reply(
                "✏️ Please share a bit more — at least a sentence or two about "
                "how you've been feeling. This helps the model give a meaningful result.",
                status="waiting",
            )

        session.user_text = text
        session.state     = "DONE"
        return _reply(
            "✅ *Got it.* Analysing your message with the AI model and "
            "SHAP explanations…\n_This may take a few seconds._",
            status="ready",
            user_text=text,
        )

    # ── IDLE: user sends text without /assess ─────────────────────────
    if session.state == "IDLE":
        # Treat any substantive free text as an implicit /assess
        if len(text.split()) >= 5 and not text.startswith("/"):
            session.state     = "DONE"
            session.user_text = text
            return _reply(
                "✅ *Received your message.* Analysing with SHAP…\n"
                "_This may take a few seconds._",
                status="ready",
                user_text=text,
            )

    # ── Fallback ──────────────────────────────────────────────────────
    return _reply(
        "👋 Type */assess* to begin an analysis, or */help* for more information.",
        status="idle",
    )


def _reply(
    text: str,
    status: str,
    user_text: Optional[str] = None,
) -> dict:
    return {
        "response":   text,
        "parse_mode": "Markdown",
        "status":     status,
        "user_text":  user_text,
    }
