"""
counterfactual_explainer/cf_generator.py
────────────────────────────────────────────────────────────────────
Use Case 4: Counterfactual-Only Explanation

Research basis:
  - FIZLE (Bhattacharjee et al., 2024): LLM-based CF generation without
    task-specific fine-tuning, using structured system prompts.
  - FitCF (2024): Classifier-Guided Generation (CGG) — uses SHAP feature
    importance to identify high-impact tokens BEFORE prompting the LLM,
    giving it explicit guidance on what to change.
  - Wachter et al. (2017): Counterfactuals must be minimal, plausible,
    and produce a label flip.
  - UbiComp '24 (Gyuwon Jung et al.): counterfactual scenarios for mental
    health provide actionable coping strategies, not just explanations.

Pipeline (CGG approach):
  1. Run deproberta → get predicted label + probabilities
  2. Run SHAP → identify top-K tokens that most drive the prediction
  3. Prompt Gemini with:
       - original text
       - current label
       - target label (one level lower severity)
       - SHAP-identified high-impact tokens to focus changes on
       - constraint: minimal edit, keep meaning, flip label
  4. Verify generated CF with deproberta → confirm label flip
  5. Compute evaluation metrics (flip rate, Levenshtein minimality,
     semantic similarity)
  6. Return top-scoring valid CFs

Evaluation metrics (from CEVAL benchmark, Nguyen et al., 2024):
  - Flip Rate     : % of CFs that actually change the predicted label
  - Minimality    : 1 - (Levenshtein / max_len)  → higher = fewer edits
  - Semantic Sim. : cosine similarity of sentence embeddings (preserves meaning)
  - Fluency       : perplexity proxy via LLM confidence
"""

import sys
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.depression_model import (
    predict_proba, explain_with_shap, reframe_text,
    LABEL_MAP, LABEL_DESCRIPTIONS
)
from shared.llm_client import call_gemini

logger = logging.getLogger("cf_generator")

# ── Target label strategy ────────────────────────────────────────────
# For each predicted label, the "counterfactual goal" is one step better.
# severe → moderate → not depression
CF_TARGET_MAP = {
    "severe":         "moderate",
    "moderate":       "not depression",
    "not depression": "not depression",   # already best; still explain
}

# PHQ-8 symptoms used to guide the LLM on WHAT to change
SYMPTOM_RECOVERY_HINTS = {
    "hopeless":    "Replace hopelessness with signs of purpose or hope",
    "empty":       "Replace emotional emptiness with small moments of connection",
    "tired":       "Replace persistent fatigue with energy recovery language",
    "worthless":   "Replace self-worth deficit with self-compassion phrasing",
    "nothing":     "Replace nihilism with acknowledgment of small positives",
    "anymore":     "Replace 'anymore' patterns with openness to change",
    "sad":         "Replace pervasive sadness with mood variability",
    "lonely":      "Replace isolation with mention of social connection",
    "sleep":       "Replace sleep disruption with stabilised sleep routine",
    "concentrate": "Replace concentration difficulty with focused activity",
    "appetite":    "Replace appetite change with stabilised eating patterns",
    "numb":        "Replace emotional numbing with mild emotional re-engagement",
}


@dataclass
class CounterfactualResult:
    """Full output from the CF pipeline for a single user text."""
    original_text:    str
    original_label:   str
    original_probs:   np.ndarray
    target_label:     str

    # List of generated counterfactuals, sorted by score
    candidates:       List[dict] = field(default_factory=list)

    # Best valid CF (label flipped successfully)
    best_cf:          Optional[dict] = None

    # SHAP tokens that guided generation
    shap_guided_tokens: List[dict] = field(default_factory=list)

    # Whether the original was already not-depression
    already_well:     bool = False


def _levenshtein(s1: str, s2: str) -> int:
    """Word-level Levenshtein distance (edit distance on token lists)."""
    w1, w2 = s1.lower().split(), s2.lower().split()
    m, n   = len(w1), len(w2)
    dp     = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if w1[i-1] == w2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]


def _minimality_score(original: str, cf: str) -> float:
    """1 - normalised edit distance. Higher = fewer changes = better."""
    dist = _levenshtein(original, cf)
    max_len = max(len(original.split()), len(cf.split()), 1)
    return round(1.0 - (dist / max_len), 3)


def _semantic_similarity(original: str, cf: str) -> float:
    """
    Cosine similarity of sentence embeddings.
    Falls back to word-overlap Jaccard if sentence-transformers unavailable.
    Higher = more meaning preserved = better.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        embs   = _model.encode([original, cf])
        cos    = float(np.dot(embs[0], embs[1]) /
                       (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9))
        return round(cos, 3)
    except Exception:
        # Jaccard fallback
        s1 = set(original.lower().split())
        s2 = set(cf.lower().split())
        return round(len(s1 & s2) / max(len(s1 | s2), 1), 3)


def _evaluate_candidate(original: str, original_label: str,
                         candidate_text: str, target_label: str) -> dict:
    """
    Evaluates one CF candidate on all metrics.
    Returns a dict with flip_success, probs, minimality, semantic_sim, score.
    """
    cf_probs      = predict_proba([candidate_text])[0]
    cf_label_idx  = int(np.argmax(cf_probs))
    cf_label      = LABEL_MAP[cf_label_idx]
    flip_success  = (cf_label != original_label)

    minimality    = _minimality_score(original, candidate_text)
    semantic_sim  = _semantic_similarity(original, candidate_text)

    # Composite score: flip is binary gate; then balance minimality + similarity
    if flip_success:
        score = 1.0 + 0.5 * minimality + 0.5 * semantic_sim
    else:
        # Partial credit if it moved toward target even without flip
        label_order = {"severe": 0, "moderate": 1, "not depression": 2}
        movement    = label_order.get(cf_label, 0) - label_order.get(original_label, 0)
        score       = max(0.0, 0.3 * movement / 2 + 0.2 * minimality)

    return {
        "text":         candidate_text,
        "label":        cf_label,
        "probs":        cf_probs,
        "flip_success": flip_success,
        "minimality":   minimality,
        "semantic_sim": semantic_sim,
        "score":        round(score, 4),
    }


def _build_cf_generation_prompt(
    original_text:    str,
    original_label:   str,
    target_label:     str,
    shap_tokens:      List[dict],
    n_candidates:     int = 3,
) -> str:
    """
    Builds the FIZLE/CGG-style prompt that instructs Gemini to generate
    valid, minimal, plausible counterfactuals.

    Based on:
      - FIZLE-guided (Bhattacharjee et al., 2024): structured system prompt
        with classifier integration
      - FitCF CGG (2024): SHAP importance used to specify target tokens
    """
    # Build token guidance block from SHAP
    token_hints = []
    for t in shap_tokens[:5]:
        word  = t["token"].strip(".,!?'\"").lower()
        hint  = SYMPTOM_RECOVERY_HINTS.get(word, f"Soften or replace '{t['token']}'")
        token_hints.append(f"  - Token '{t['token']}' (SHAP={t['shap']:+.4f}): {hint}")
    token_block = "\n".join(token_hints) if token_hints else "  (No specific tokens — make holistic edits)"

    severity_order = "not depression (best) ← moderate ← severe (worst)"

    return f"""You are a Counterfactual Explanation Generator for a depression screening AI.

TASK: Generate {n_candidates} minimally-edited counterfactual versions of the text below
that would make the depression screening model predict "{target_label}" instead of "{original_label}".

ORIGINAL TEXT (current prediction: {original_label.upper()}):
"{original_text}"

SEVERITY SCALE: {severity_order}
TARGET PREDICTION: {target_label.upper()} (one step improvement)

SHAP-IDENTIFIED HIGH-IMPACT TOKENS (focus your edits HERE first):
{token_block}

CONSTRAINTS — each counterfactual MUST:
1. Change as FEW words as possible (minimal edit distance — this is critical)
2. Sound natural and plausible as something a real person would write
3. Preserve the overall topic and personal voice
4. Show genuine (if modest) emotional or behavioural improvement
5. NOT be a completely rewritten text — only targeted word/phrase changes

MENTAL HEALTH REALISM — edits should reflect:
- Small but real behavioural changes (e.g., "I rarely go out" → "I sometimes go out")
- Reduced intensity of negative language (e.g., "I never enjoy anything" → "I rarely enjoy things")
- Partial recovery signals, NOT forced positivity
- PHQ-8 symptoms improving: anhedonia, mood, fatigue, sleep, concentration, appetite

OUTPUT FORMAT — respond with EXACTLY {n_candidates} counterfactuals, one per line,
each on its own line starting with "CF:" and nothing else:
CF: <counterfactual text here>
CF: <counterfactual text here>
CF: <counterfactual text here>

Do not include any explanation, preamble, or extra text. Only the CF: lines."""


def _parse_cf_candidates(llm_response: str) -> List[str]:
    """Extract CF: lines from LLM output."""
    candidates = []
    for line in llm_response.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("cf:"):
            text = line[3:].strip().strip('"\'')
            if len(text.split()) >= 4:
                candidates.append(text)
    return candidates


def generate_counterfactuals(
    user_text:    str,
    n_candidates: int = 3,
    n_attempts:   int = 2,
) -> CounterfactualResult:
    """
    Main CF pipeline. Runs SHAP → LLM generation → verification → ranking.

    Args:
        user_text:    original user message
        n_candidates: number of CFs to generate per LLM call
        n_attempts:   number of LLM generation attempts (for higher flip rate)

    Returns:
        CounterfactualResult with ranked candidates and best valid CF
    """
    # ── Step 1: predict + SHAP ───────────────────────────────────────
    model_input    = reframe_text(user_text)
    probs          = predict_proba([model_input])[0]
    original_idx   = int(np.argmax(probs))
    original_label = LABEL_MAP[original_idx]
    target_label   = CF_TARGET_MAP[original_label]

    logger.info("CF pipeline: %s → target: %s", original_label, target_label)

    result = CounterfactualResult(
        original_text  = user_text,
        original_label = original_label,
        original_probs = probs,
        target_label   = target_label,
        already_well   = (original_label == "not depression"),
    )

    if result.already_well:
        logger.info("Already not depression — generating 'maintain wellness' CFs")

    # ── Step 2: SHAP for token guidance ─────────────────────────────
    shap_result = explain_with_shap(user_text)
    result.shap_guided_tokens = shap_result.risk_tokens[:5]
    logger.info("SHAP risk tokens: %s", [t["token"] for t in result.shap_guided_tokens])

    # ── Step 3: LLM generation (multiple attempts) ────────────────────
    all_candidates = []
    for attempt in range(n_attempts):
        logger.info("LLM generation attempt %d/%d", attempt + 1, n_attempts)
        prompt = _build_cf_generation_prompt(
            original_text  = user_text,
            original_label = original_label,
            target_label   = target_label,
            shap_tokens    = result.shap_guided_tokens,
            n_candidates   = n_candidates,
        )
        try:
            llm_response = call_gemini(prompt)
            parsed       = _parse_cf_candidates(llm_response)
            logger.info("Attempt %d: got %d candidates", attempt + 1, len(parsed))
            all_candidates.extend(parsed)
        except Exception as e:
            logger.warning("LLM attempt %d failed: %s", attempt + 1, e)

    # Deduplicate
    seen = set()
    unique = []
    for c in all_candidates:
        if c not in seen and c != user_text:
            seen.add(c)
            unique.append(c)

    # ── Step 4: evaluate all candidates ─────────────────────────────
    evaluated = []
    for cf_text in unique:
        metrics = _evaluate_candidate(user_text, original_label, cf_text, target_label)
        evaluated.append(metrics)
        logger.info(
            "CF: '%s...' → %s | flip=%s | min=%.3f | sim=%.3f | score=%.4f",
            cf_text[:50], metrics["label"], metrics["flip_success"],
            metrics["minimality"], metrics["semantic_sim"], metrics["score"]
        )

    # Sort: valid flips first, then by composite score
    evaluated.sort(key=lambda x: (x["flip_success"], x["score"]), reverse=True)
    result.candidates = evaluated

    # Best valid CF
    valid = [c for c in evaluated if c["flip_success"]]
    result.best_cf = valid[0] if valid else (evaluated[0] if evaluated else None)

    logger.info(
        "CF pipeline done: %d candidates, %d valid flips",
        len(evaluated), len(valid)
    )
    return result


def format_cf_debug(result: CounterfactualResult) -> str:
    probs = ", ".join(f"{LABEL_MAP[i]}={result.original_probs[i]:.3f}" for i in range(3))
    lines = [
        f"Original: {result.original_label} ({probs})",
        f"Target  : {result.target_label}",
        f"SHAP guided tokens: {[t['token'] for t in result.shap_guided_tokens]}",
        f"Candidates: {len(result.candidates)} ({sum(1 for c in result.candidates if c['flip_success'])} valid flips)",
    ]
    if result.best_cf:
        c = result.best_cf
        lines.append(
            f"Best CF: [{c['label']}] '{c['text'][:80]}...'\n"
            f"  flip={c['flip_success']} min={c['minimality']} sim={c['semantic_sim']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    test_texts = [
        "I feel empty and tired every day. Nothing makes sense anymore.",
        "I don't really go out much anymore. I've lost interest in my hobbies, and I find it hard to concentrate at work.",
        "I feel completely worthless and hopeless. I can't sleep, I can't eat. I see no point in anything.",
    ]

    for text in test_texts:
        print("\n" + "="*65)
        print(f"Text: {text}")
        result = generate_counterfactuals(text)
        print(format_cf_debug(result))
