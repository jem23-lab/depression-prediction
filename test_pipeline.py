"""
test_pipeline.py  (root-level)
────────────────────────────────────────────────────────────────────
CLI test for Use Cases 1 (SHAP) and 2 (RAG) without Telegram.

Usage:
    python test_pipeline.py --uc 1 --scenario moderate
    python test_pipeline.py --uc 2 --scenario indirect
    python test_pipeline.py --uc 1 --text "I feel empty and hopeless."
    python test_pipeline.py --uc 2 --text "I can't sleep and have no appetite."
"""

import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SCENARIOS = {
    "indirect": "I don't really go out much anymore. I've lost interest in my hobbies, and I find it hard to concentrate at work.",
    "moderate": "I've been feeling sad and tired a lot lately. I don't enjoy things I used to love. Concentrating at work is really hard.",
    "severe":   "I feel completely worthless and hopeless. I can't sleep, I can't eat. I've been crying all day and see no point in anything.",
    "positive": "I'm feeling pretty good today. Had a great workout and met some friends. Work is going well.",
}


def test_uc1(text: str):
    from shared.depression_model        import explain_with_shap, format_debug
    from shap_explainer.shap_explainer  import generate_shap_explanation

    print("\n" + "="*65)
    print("  USE CASE 1 — SHAP-ONLY EXPLANATION")
    print("="*65)
    print(f"\nText: {text}\n")

    result = explain_with_shap(text)
    print(format_debug(result))

    print("\n  Risk tokens:")
    for t in result.risk_tokens:
        note = f"  <- {t['note']}" if t["note"] else ""
        print(f"    '{t['token']}' SHAP={t['shap']:+.4f}{note}")

    print("\n── Gemini SHAP Explanation ──────────────────────────────────")
    explanation = generate_shap_explanation(text, result)
    print(explanation)
    print("="*65)


def test_uc2(text: str):
    from rag_explainer.rag_explainer import (
        run_rag_pipeline, generate_rag_explanation, format_rag_debug
    )
    from rag_explainer.rag_retriever import format_retrieved_for_prompt

    print("\n" + "="*65)
    print("  USE CASE 2 — RAG-ONLY EXPLANATION")
    print("="*65)
    print(f"\nText: {text}\n")

    result = run_rag_pipeline(text)
    print(format_rag_debug(result))

    print("\n── Retrieved Knowledge ──────────────────────────────────────")
    print(format_retrieved_for_prompt(result.retrieved_docs))

    print("\n── Gemini RAG Explanation ───────────────────────────────────")
    explanation = generate_rag_explanation(text, result)
    print(explanation)
    print("="*65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uc",       choices=["1", "2"], default="1",
                        help="Use case: 1=SHAP, 2=RAG")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        default="indirect")
    parser.add_argument("--text",     type=str, default=None,
                        help="Custom text (overrides --scenario)")
    args = parser.parse_args()

    text = args.text or SCENARIOS[args.scenario]

    if args.uc == "1":
        test_uc1(text)
    else:
        test_uc2(text)
