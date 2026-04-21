"""
test_pipeline.py
────────────────────────────────────────────────────────────────────
End-to-end test of Use Case 1 WITHOUT Telegram.
Mirrors the notebook flow: cells [23] → [24] → [25] → [29]

Usage:
    python test_pipeline.py
    python test_pipeline.py --text "I feel hopeless and tired every day."
    python test_pipeline.py --scenario severe
"""

import argparse
import numpy as np
from depression_model import (
    explain_with_shap,
    predict_proba,
    format_debug_summary,
    LABEL_MAP,
)
from shap_explainer import generate_explanation, build_shap_prompt

# ── Test scenarios (matching notebook samples) ──────────────────────
SCENARIOS = {
    "notebook": {
        "text":  "I feel empty and tired every day. Nothing makes sense anymore.",
        "query": "I've been feeling really low lately. Can you help me understand?",
    },
    "severe": {
        "text": (
            "I feel completely worthless and hopeless. I can't sleep, I can't eat. "
            "I've been crying all day. I don't see the point in anything anymore. "
            "Sometimes I think everyone would be better off without me."
        ),
        "query": "I need help understanding why I feel this way.",
    },
    "moderate": {
        "text": (
            "I've been feeling sad and tired a lot lately. I don't really enjoy "
            "things I used to love. Concentrating at work is really hard. "
            "I feel a bit numb most days."
        ),
        "query": "Can you assess how I've been feeling?",
    },
    "mild": {
        "text": (
            "I've been a bit down lately, maybe a little stressed. "
            "I'm sleeping okay but feel a bit flat. Not my best week."
        ),
        "query": "Just checking in on my mental health.",
    },
    "positive": {
        "text": (
            "I'm feeling pretty good today. Had a great workout and met some friends. "
            "Work is going well and I feel optimistic about the future."
        ),
        "query": "How am I doing mentally?",
    },
}


def run_pipeline(text: str, query: str, show_prompt: bool = False):
    """Runs the full UC1 pipeline and prints results."""
    print("\n" + "=" * 65)
    print("  USE CASE 1 — SHAP-ONLY DEPRESSION EXPLANATION")
    print("=" * 65)

    # ── Cell [24] equivalent: predict_proba ─────────────────────────
    print(f"\n📝 Text  : {text}")
    print(f"❓ Query : {query}")

    probs = predict_proba([text])[0]
    print("\n── [Cell 24] Prediction ─────────────────────────────────────")
    for i, label in LABEL_MAP.items():
        bar = "█" * int(probs[i] * 25)
        print(f"  {label:<18s} {bar:<25s} {probs[i]*100:.1f}%")

    # ── Cell [25] equivalent: SHAP ───────────────────────────────────
    print("\n── [Cell 25] SHAP Token Analysis ────────────────────────────")
    shap_result = explain_with_shap(text)
    print(format_debug_summary(shap_result))

    print("\n  Risk tokens (push toward depression):")
    for t in shap_result.risk_tokens:
        note = f"  ← {t['note']}" if t["note"] else ""
        print(f"    '{t['token']}' SHAP={t['shap']:+.4f}{note}")

    if shap_result.protective_tokens:
        print("\n  Protective tokens (reduce depression signal):")
        for t in shap_result.protective_tokens:
            print(f"    '{t['token']}' SHAP={t['shap']:+.4f}")

    # ── Cell [29] equivalent: Gemini prompt + response ───────────────
    if show_prompt:
        print("\n── [Cell 29] LLM Prompt Preview ─────────────────────────────")
        print(build_shap_prompt(query, shap_result))

    print("\n── [Cell 29] Gemini Explanation ─────────────────────────────")
    explanation = generate_explanation(user_query=query, shap_result=shap_result)
    print(explanation)
    print("=" * 65)

    return shap_result, explanation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Use Case 1 — SHAP pipeline")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        default="notebook",
                        help="Pre-defined test scenario (default: notebook)")
    parser.add_argument("--text",  type=str, default=None,
                        help="Custom text to analyse")
    parser.add_argument("--query", type=str,
                        default="Can you help me understand how I'm feeling?",
                        help="User query (used with --text)")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Print the full LLM prompt before the response")
    args = parser.parse_args()

    if args.text:
        run_pipeline(args.text, args.query, show_prompt=args.show_prompt)
    else:
        s = SCENARIOS[args.scenario]
        run_pipeline(s["text"], s["query"], show_prompt=args.show_prompt)
