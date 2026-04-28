"""
shared/conversation.py
────────────────────────────────────────────────────────────────────
Shared Telegram conversation FSM for fixed-text evaluation flow.

States:
  IDLE  → waiting for /start, /help, or /assess
  READY → hardcoded paragraph can be processed by bot
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

    if lower in ("/start", "/help"):
        reset_session(user_id)
        return _reply(
            "👋 Welcome to the Depression Explanation Evaluation Bot\n\n"
            "This tool shows one fixed participant paragraph and generates an AI explanation "
            "using a randomly selected explanation method.\n\n"
            "After that, you will rate the explanation from 1 to 5 on:\n"
            "1) Clarity\n2) Correctness\n3) Helpfulness\n\n"
            "Type /assess to begin.\n"
            "Type /reset to start over.",
            status="idle",
        )

    if lower == "/reset":
        reset_session(user_id)
        return _reply("Session reset. Type /assess to begin.", status="idle")

    if lower == "/assess":
        session.state = "READY"
        return _reply(
            "Starting a new evaluation. Generating explanation...",
            status="ready",
        )

    return _reply(
        "Type /assess to start an evaluation, or /help for more info.",
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
