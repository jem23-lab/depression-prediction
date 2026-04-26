"""
test_pipeline.py
────────────────────────────────────────────────────────────────────
CLI test for all use cases — no Telegram needed.

Usage:
    python test_pipeline.py --uc 1 --scenario moderate
    python test_pipeline.py --uc 2 --scenario indirect
    python test_pipeline.py --uc 3 --scenario moderate      # HYBRID
    python test_pipeline.py --uc 4 --scenario severe
    python test_pipeline.py --uc 5 --scenario moderate      # MCP modular router
    python test_pipeline.py --uc 3 --text "I feel empty and hopeless every day."
"""

import sys, os, argparse, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

SCENARIOS = {
    "indirect": "I don't really go out much anymore. I've lost interest in my hobbies, and I find it hard to concentrate at work.",
    "moderate": "I've been feeling sad and tired a lot lately. I don't enjoy things I used to love. Concentrating at work is really hard.",
    "severe":   "I feel completely worthless and hopeless. I can't sleep, I can't eat. I've been crying all day and see no point in anything.",
    "positive": "I'm feeling pretty good today. Had a great workout and met some friends. Work is going well.",
}


# ── UC1: SHAP ─────────────────────────────────────────────────────────
def test_uc1(text: str):
    from shared.depression_model import explain_with_shap, format_debug
    from architecture.shap_explainer.shap_explainer import generate_shap_explanation

    print("\n" + "="*65)
    print("  USE CASE 1 — SHAP ONLY")
    print("="*65)
    print(f"\nText: {text}\n")

    result = explain_with_shap(text)
    print(format_debug(result))

    print("\n  Risk tokens:")
    for t in result.risk_tokens:
        note = f"  <- {t['note']}" if t["note"] else ""
        print(f"    '{t['token']}' SHAP={t['shap']:+.4f}{note}")

    print("\n── Gemini Explanation ───────────────────────────────────────")
    print(generate_shap_explanation(text, result))
    print("="*65)


# ── UC2: RAG ──────────────────────────────────────────────────────────
def test_uc2(text: str):
    from architecture.rag_explainer.rag_explainer import (
        run_rag_pipeline, generate_rag_explanation, format_rag_debug,
    )
    from architecture.rag_explainer.rag_retriever import format_retrieved_for_prompt

    print("\n" + "="*65)
    print("  USE CASE 2 — RAG ONLY")
    print("="*65)
    print(f"\nText: {text}\n")

    result = run_rag_pipeline(text)
    print(format_rag_debug(result))

    print("\n── Retrieved Knowledge ──────────────────────────────────────")
    print(format_retrieved_for_prompt(result.retrieved_docs))

    print("\n── Gemini Explanation ───────────────────────────────────────")
    print(generate_rag_explanation(text, result))
    print("="*65)


# ── UC3: HYBRID ───────────────────────────────────────────────────────
def test_uc3(text: str):
    from architecture.hybrid_shap_rag_counterfactual.hybrid_pipeline import (
        run_hybrid_pipeline, format_hybrid_debug,
    )
    from architecture.hybrid_shap_rag_counterfactual.hybrid_explainer import generate_hybrid_explanation
    from architecture.rag_explainer.rag_retriever import format_retrieved_for_prompt

    print("\n" + "="*65)
    print("  USE CASE 3 — HYBRID: SHAP + RAG + COUNTERFACTUAL")
    print("  All three signals fused into one unified LLM prompt")
    print("="*65)
    print(f"\nText: {text}\n")

    print("Running all three pipelines...")
    result = run_hybrid_pipeline(text, cf_candidates=2, cf_attempts=1)
    print(format_hybrid_debug(result))

    # SHAP layer
    if result.shap_result:
        print("\n── Layer 1: SHAP ────────────────────────────────────────────")
        for t in result.shap_result.risk_tokens[:5]:
            note = f"  ({t['note']})" if t["note"] else ""
            print(f"  🔴 '{t['token']}' SHAP={t['shap']:+.4f}{note}")
        for t in result.shap_result.protective_tokens[:2]:
            print(f"  🟢 '{t['token']}' SHAP={t['shap']:+.4f}")
    else:
        print(f"\n  SHAP failed: {result.shap_error}")

    # RAG layer
    if result.rag_result:
        print("\n── Layer 2: RAG ─────────────────────────────────────────────")
        print(format_retrieved_for_prompt(result.rag_result.retrieved_docs))
    else:
        print(f"\n  RAG failed: {result.rag_error}")

    # CF layer
    if result.cf_result:
        print("\n── Layer 3: Counterfactual ──────────────────────────────────")
        valid = sum(1 for c in result.cf_result.candidates if c["flip_success"])
        total = len(result.cf_result.candidates)
        print(f"  Flip rate: {valid}/{total} candidates")
        for i, c in enumerate(result.cf_result.candidates[:3], 1):
            status = "FLIP" if c["flip_success"] else "no flip"
            print(f"  {i}. [{status}] [{c['label']}] min={c['minimality']:.2f} sim={c['semantic_sim']:.2f}")
            print(f"     '{c['text'][:100]}'")
    else:
        print(f"\n  CF failed: {result.cf_error}")

    print("\n── Unified Gemini Explanation (all three signals) ───────────")
    print(generate_hybrid_explanation(text, result))
    print("="*65)


# ── UC4: Counterfactual ───────────────────────────────────────────────
def test_uc4(text: str):
    from architecture.shap_counterfactual_explainer.cf_generator import generate_counterfactuals, format_cf_debug
    from architecture.shap_counterfactual_explainer.cf_explainer import generate_cf_explanation

    print("\n" + "="*65)
    print("  USE CASE 4 — COUNTERFACTUAL ONLY")
    print("  Research: FIZLE (2024) + FitCF CGG (2024) + Wachter et al. (2017)")
    print("="*65)
    print(f"\nText: {text}\n")

    print("Running SHAP + counterfactual generation...")
    result = generate_counterfactuals(text, n_candidates=3, n_attempts=2)
    print(format_cf_debug(result))

    print("\n── Candidate Details ────────────────────────────────────────")
    for i, c in enumerate(result.candidates, 1):
        status = "FLIP" if c["flip_success"] else "no flip"
        print(f"  {i}. [{status}] [{c['label']}]")
        print(f"     Text       : {c['text']}")
        print(f"     Minimality : {c['minimality']:.3f}  Sim: {c['semantic_sim']:.3f}  Score: {c['score']:.4f}")

    valid     = sum(1 for c in result.candidates if c["flip_success"])
    total     = len(result.candidates)
    flip_rate = valid / max(total, 1)
    print(f"\n  Flip rate: {flip_rate*100:.1f}%  ({valid}/{total})")

    print("\n── Gemini Explanation ───────────────────────────────────────")
    print(generate_cf_explanation(text, result))
    print("="*65)


# ── UC5: MCP modular router ─────────────────────────────────────────
def test_uc5(text: str):
    from architecture.mcp_modular_agent.mcp_client import run_mcp_pipeline

    print("\n" + "="*65)
    print("  USE CASE 5 — MCP MODULAR ROUTER")
    print("  LLM ranks SHAP/RAG/Counterfactual, then uses fallback")
    print("="*65)
    print(f"\nText: {text}\n")

    result = run_mcp_pipeline(text, fallback=True, top_k=2)

    required = ["selected_server", "fallback_used", "prediction", "confidence", "explanation"]
    missing = [k for k in required if k not in result]
    if missing:
        raise RuntimeError(f"UC5 result missing keys: {missing}")

    print("\n── MCP Router Result ───────────────────────────────────────")
    print(f"  Selected server : {result.get('selected_server')}")
    print(f"  Fallback used   : {result.get('fallback_used')}")
    print(f"  Prediction      : {result.get('prediction')}")
    conf = result.get("confidence")
    conf_txt = f"{conf*100:.1f}%" if isinstance(conf, (int, float)) else "N/A"
    print(f"  Confidence      : {conf_txt}")

    rationale = (result.get("rationale") or "").strip()
    if rationale:
        print("\n── Router Rationale ────────────────────────────────────────")
        print(rationale)

    print("\n── Explanation Preview ─────────────────────────────────────")
    expl = (result.get("explanation") or "").strip()
    print((expl[:1200] + "...") if len(expl) > 1200 else expl)

    errs = result.get("errors") or []
    if errs:
        print("\n── Fallback Errors Encountered ─────────────────────────────")
        for e in errs:
            print(f"  - {e}")

    print("="*65)


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test XAI depression pipelines (no Telegram)")
    parser.add_argument(
        "--uc",
        choices=["1", "2", "3", "4", "5"],
        default="3",
        help="Use case: 1=SHAP, 2=RAG, 3=Hybrid (SHAP+RAG+CF), 4=Counterfactual, 5=MCP router",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="moderate",
        help="Pre-built test scenario",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom input text (overrides --scenario)",
    )
    args = parser.parse_args()

    text = args.text or SCENARIOS[args.scenario]

    dispatch = {"1": test_uc1, "2": test_uc2, "3": test_uc3, "4": test_uc4, "5": test_uc5}
    dispatch[args.uc](text)
