"""
shared/llm_client.py
────────────────────────────────────────────────────────────────────
Shared Gemini client used by ALL explainer use cases.

Handles:
  - API key config
  - Model fallback chain (tries models in order until one works)
  - strip_markdown() for safe Telegram output
"""

import os
import re
import logging
import google.generativeai as genai

logger = logging.getLogger("llm_client")

# Fallback chain — tried in order until one succeeds
GEMINI_MODELS = [
    "gemini-3-flash-preview"
]


def call_gemini(prompt: str, system: str = "") -> str:
    """
    Calls Gemini with prompt + optional system prefix.
    Tries each model in GEMINI_MODELS until one succeeds.
    Raises RuntimeError if all fail.
    """
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Export it with: export GOOGLE_API_KEY='your_key'"
        )
    genai.configure(api_key=key)

    full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
    contents    = [{"role": "user", "parts": [full_prompt]}]

    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            logger.info("Trying Gemini model: %s", model_name)
            model    = genai.GenerativeModel(model_name)
            response = model.generate_content(contents=contents)
            text     = (response.text or "").strip()
            if text:
                logger.info("Success with: %s", model_name)
                return text
            logger.warning("%s returned empty response", model_name)
        except Exception as e:
            logger.warning("%s failed: %s", model_name, e)
            last_error = e

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}. "
        f"Models tried: {GEMINI_MODELS}"
    )


def strip_markdown(text: str) -> str:
    """
    Removes Markdown symbols so Telegram doesn't choke on unmatched
    *, _, `, [ characters from Gemini output.
    """
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{1,3}(.*?)_{1,3}",   r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`{1,3}(.*?)`{1,3}",   r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.strip()
