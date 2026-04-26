"""
shared/conversation.py
────────────────────────────────────────────────────────────────────
Shared Telegram conversation FSM used by ALL use cases.

States:
  IDLE      → waiting for /start, /help, or /assess
  CHOOSING  → user shown use-case menu, waiting for choice
  WAITING   → prompted for free-text input
  DONE      → text ready, pipeline can run

Use cases available:
  1 -> SHAP-only explanation
  2 -> RAG-only explanation
  3 -> Hybrid explanation
  4 -> Counterfactual explanation
  5 -> MCP modular router explanation
"""

from dataclasses import dataclass, field
from typing import Optional

# ── Use case registry ────────────────────────────────────────────────
USE_CASES = {
    "1": {
        "name":        "SHAP Explanation",
        "description": "Uses token-level SHAP to show which words in your message drove the prediction.",
        "emoji":       "🔬",
        "status":      "available",
    },
    "2": {
        "name":        "RAG Explanation",
        "description": "Retrieves matching clinical symptom knowledge to explain your assessment.",
        "emoji":       "📚",
        "status":      "available",
    },
    "3": {
        "name":        "SHAP + RAG + Counterfactual (Hybrid)",
        "description": "Combines token-level SHAP, clinical knowledge retrieval, and counterfactual reasoning into one unified explanation.",
        "emoji":       "🔀",
        "status":      "available",
    },
    "4": {
        "name":        "Counterfactual",
        "description": "Shows what minimal change in your message would shift the AI prediction — and what that means for you.",
        "emoji":       "🔄",
        "status":      "available",
    },
    "5": {
        "name":        "MCP Agent",
        "description": "Modular agent that routes to the best explainer for your query.",
        "emoji":       "🤖",
        "status":      "available",
    },
}


def build_menu_text() -> str:
    lines = ["Please choose an explanation method:\n"]
    for key, uc in USE_CASES.items():
        status = "" if uc["status"] == "available" else " (coming soon)"
        lines.append(f"  {uc['emoji']} {key}. {uc['name']}{status}")
        lines.append(f"     {uc['description']}\n")
    lines.append("Reply with the number (e.g. 1 or 2).")
    return "\n".join(lines)


# ── Session state ────────────────────────────────────────────────────
@dataclass
class UserSession:
    user_id:     int
    state:       str = "IDLE"      # IDLE | CHOOSING | WAITING | DONE
    use_case:    str = ""          # "1", "2", etc.
    user_text:   str = ""


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
        "response"  : str,           message to send back
        "status"    : str,           idle | choosing | waiting | ready
        "use_case"  : str | None,    "1", "2", etc.
        "user_text" : str | None,    text ready for pipeline
      }
    """
    session = get_session(user_id)
    text    = text.strip()
    lower   = text.lower()

    # ── Global commands (work from any state) ────────────────────────
    if lower in ("/start", "/help"):
        reset_session(user_id)
        return _reply(
            "👋 Welcome to the Depression Assessment Assistant\n\n"
            "This tool uses AI to assess depression signals in your message "
            "and explain the result using different XAI methods.\n\n"
            "Type /assess to begin.\n"
            "Type /reset to start over.\n\n"
            "This is not a clinical tool. If you are in crisis, please "
            "contact a mental health professional immediately.",
            status="idle",
        )

    if lower == "/reset":
        reset_session(user_id)
        return _reply("Session reset. Type /assess to begin.", status="idle")

    # ── /assess → show use-case menu ────────────────────────────────
    if lower == "/assess":
        reset_session(user_id)
        session = get_session(user_id)
        session.state = "CHOOSING"
        return _reply(build_menu_text(), status="choosing")

    # ── CHOOSING state: user picks a use case ────────────────────────
    if session.state == "CHOOSING":
        if text not in USE_CASES:
            return _reply(
                f"Please reply with a number from 1 to {len(USE_CASES)}.\n\n"
                + build_menu_text(),
                status="choosing",
            )
        uc = USE_CASES[text]
        if uc["status"] != "available":
            available_keys = [k for k, v in USE_CASES.items() if v["status"] == "available"]
            return _reply(
                f"{uc['emoji']} {uc['name']} is coming soon.\n\n"
                f"Please choose an available option: {', '.join(available_keys)}.\n\n"
                + build_menu_text(),
                status="choosing",
            )
        session.use_case = text
        session.state    = "WAITING"
        return _reply(
            f"{uc['emoji']} {uc['name']} selected.\n\n"
            "Please describe how you have been feeling recently in a few sentences. "
            "Be as honest as you like — there are no right or wrong answers.\n\n"
            "Example: \"I've been feeling really tired and empty lately. "
            "I don't enjoy things I used to love and I can't concentrate.\"",
            status="waiting",
            use_case=text,
        )

    # ── WAITING state: user provides their text ──────────────────────
    if session.state == "WAITING":
        if len(text.split()) < 4:
            return _reply(
                "Please share a bit more — at least a sentence about "
                "how you have been feeling. This helps the model give a better result.",
                status="waiting",
                use_case=session.use_case,
            )
        session.user_text = text
        session.state     = "DONE"
        return _reply(
            "Got it. Analysing your message...\nThis may take a few seconds.",
            status="ready",
            use_case=session.use_case,
            user_text=text,
        )

    # ── IDLE: treat substantive free text as implicit /assess ────────
    if session.state == "IDLE" and len(text.split()) >= 5 and not lower.startswith("/"):
        session.state    = "CHOOSING"
        return _reply(
            "I see you have shared something. First, please choose "
            "which explanation method to use:\n\n" + build_menu_text(),
            status="choosing",
        )

    # ── Fallback ─────────────────────────────────────────────────────
    return _reply(
        "Type /assess to start an assessment, or /help for more info.",
        status="idle",
    )


def _reply(
    text: str,
    status: str,
    use_case: Optional[str] = None,
    user_text: Optional[str] = None,
) -> dict:
    return {
        "response":  text,
        "status":    status,
        "use_case":  use_case,
        "user_text": user_text,
    }
