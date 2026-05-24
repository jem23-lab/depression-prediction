"""
shared/llm_client.py
────────────────────────────────────────────────────────────────────
Shared LLM client used by ALL explainer use cases.

Uses a vLLM-compatible OpenAI API endpoint.
"""

import os
import re
import logging
from openai import OpenAI

logger = logging.getLogger("llm_client")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://134.60.124.43:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "token-is-ignored-by-vllm")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "openai/gpt-oss-20b")


def call_gemini(prompt: str, system: str = "") -> str:
    """
    Calls a vLLM-compatible OpenAI endpoint with prompt + optional system prefix.
    Keeps the function name for compatibility with existing pipelines.
    """
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        extra_body={"top_k": 50},
    )

    content = response.choices[0].message.content
    return (content or "").strip()


def strip_markdown(text: str) -> str:
    """
    Removes Markdown symbols so Telegram doesn't choke on unmatched
    *, _, `, [ characters while preserving bold emphasis.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*{2}(.*?)\*{2}", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{1,3}(.*?)_{1,3}",   r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`{1,3}(.*?)`{1,3}",   r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.strip()
