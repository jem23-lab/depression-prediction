"""
shared/conversation.py
────────────────────────────────────────────────────────────────────
Shared Telegram conversation FSM for fixed-text evaluation flow.

States:
  IDLE  → waiting for /start, /help, or /begin
  READY → evaluation flow can start
"""

from dataclasses import dataclass
from typing import Optional


# ── Session state ────────────────────────────────────────────────────
@dataclass
class UserSession:
    user_id: int
    state: str = "IDLE"      # IDLE | READY


_sessions: dict[int, UserSession] = {}


def get_session(user_id: int) -> UserSession:
    if user_id not in _sessions:
        _sessions[user_id] = UserSession(user_id=user_id)
    return _sessions[user_id]


def reset_session(user_id: int):
    _sessions[user_id] = UserSession(user_id=user_id)


# ── FSM processor ────────────────────────────────────────────────────
def process_message(user_id: int, text: str) -> dict:
    """
    Returns:
      {
        "response"  : str,
        "status"    : str,          idle | ready
        "user_text" : str | None,   reserved for compatibility
      }
    """
    session = get_session(user_id)
    text = text.strip()
    lower = text.lower()

    welcome = (
        "👋Welcome to our study.\n\n"
        "Our goal is to improve AI tools by making them more transparent for end users.\n"
        "In this study, we focus on predicting depressive states from text using AI.\n\n"
        "In this study, you will go through 10 texts in total (~30 minutes).\n\n"
        "In each case:\n"
        "1️⃣ 📄You will be shown a text sample from an individual.\n"
        "2️⃣ 🤖The AI will provide a prediction of that person's depressive state.\n"
        "3️⃣ 💡You will be presented with two explanation for that prediction.\n"
        "4️⃣ ⭐️You will rate the explanation.\n\n"
        "Type /begin to begin."
    )

    if lower in ("/start", "/help"):
        reset_session(user_id)
        return _reply(welcome, status="idle")

    if lower == "/begin":
        session.state = "READY"
        return _reply("Starting the study...", status="ready")

    return _reply(
        "Type /begin to begin.",
        status="idle",
    )


def _reply(
    text: str,
    status: str,
    user_text: Optional[str] = None,
) -> dict:
    return {
        "response": text,
        "status": status,
        "user_text": user_text,
    }
