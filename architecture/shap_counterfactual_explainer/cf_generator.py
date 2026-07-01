"""
counterfactual_explainer/cf_generator.py
────────────────────────────────────────────────────────────────────
Use Case 4: Counterfactual-Only Explanation

Research basis:
  - FIZLE (Bhattacharjee et al., 2024): LLM-based CF generation using
    structured prompts with classifier integration, no fine-tuning needed.
  - FitCF / CGG (2024): Use SHAP to identify high-impact tokens BEFORE
    prompting the LLM, giving it explicit guidance on what to change.
  - Wachter et al. (2017): CFs must be minimal, plausible, label-flipping.
  - UbiComp '24 (Gyuwon Jung et al.): CFs for mental health should produce
    actionable coping strategies tied to concrete language patterns.

Pipeline:
  1. predict_proba(raw_text)        → label + probs (no reframing)
  2. explain_with_shap(raw_text)    → SHAP risk tokens (on raw text)
  3. build_cf_generation_prompt()   → SHAP-guided Gemini prompt
  4. call_gemini()                  → CF candidates
  5. _evaluate_candidate()          → flip rate, minimality, semantic sim
  6. Return ranked CounterfactualResult

Evaluation metrics (CEVAL benchmark, Nguyen et al., 2024):
  - Flip Rate     : % of CFs that change the predicted label
  - Minimality    : 1 - (word Levenshtein / max_len)
  - Semantic Sim. : cosine similarity of sentence embeddings
"""

import sys
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.depression_model import (
    predict_proba, classify_severity, explain_with_shap,
    LABEL_MAP, LABEL_DESCRIPTIONS,
)
from shared.llm_client import call_gemini

logger = logging.getLogger("cf_generator")

_SEMANTIC_EMBEDDER = None
_SEMANTIC_MODE = os.environ.get("CF_SEMANTIC_MODE", "lexical").strip().lower()

# ── Target label strategy ─────────────────────────────────────────────
CF_TARGET_MAP = {
    "severe":         "moderate",
    "moderate":       "not depression",
    "not depression": "moderate",
}

# ── Recovery hints keyed to common depression tokens ─────────────────
SYMPTOM_RECOVERY_HINTS = {
    "hopeless":    "Replace hopelessness with signs of purpose or hope",
    "empty":       "Replace emotional emptiness with small moments of connection",
    "tired":       "Replace persistent fatigue with energy recovery language",
    "worthless":   "Replace self-worth deficit with self-compassion phrasing",
    "nothing":     "Replace nihilism with acknowledgment of small positives",
    "anymore":     "Replace 'anymore' patterns with openness to change",
    "forever":     "Replace permanence language with temporary framing",
    "die":         "Replace ideation language with help-seeking or coping language",
    "disappear":   "Replace disappearance wish with desire for relief from pain",
    "sad":         "Replace pervasive sadness with mood variability",
    "lonely":      "Replace isolation with mention of social connection",
    "sleep":       "Replace sleep disruption with stabilised sleep routine",
    "numb":        "Replace emotional blunting with mild re-engagement",
    "concentrate": "Replace concentration difficulty with focused activity",
    "appetite":    "Replace appetite change with stabilised eating patterns",
}


@dataclass
class CounterfactualResult:
    original_text:      str
    original_label:     str
    original_probs:     np.ndarray
    severity_score:     float
    target_label:       str
    candidates:         List[dict] = field(default_factory=list)
    best_cf:            Optional[dict] = None
    shap_guided_tokens: List[dict] = field(default_factory=list)
    already_well:       bool = False


# ── Evaluation metrics ────────────────────────────────────────────────
def _levenshtein(s1: str, s2: str) -> int:
    """Word-level Levenshtein distance."""
    w1, w2 = s1.lower().split(), s2.lower().split()
    m, n   = len(w1), len(w2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if w1[i-1] == w2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]


def _minimality_score(original: str, cf: str) -> float:
    dist    = _levenshtein(original, cf)
    max_len = max(len(original.split()), len(cf.split()), 1)
    return round(1.0 - (dist / max_len), 3)


def _lexical_similarity(original: str, cf: str) -> float:
    s1 = set(original.lower().split())
    s2 = set(cf.lower().split())
    return round(len(s1 & s2) / max(len(s1 | s2), 1), 3)


def _get_semantic_embedder():
    global _SEMANTIC_EMBEDDER
    if _SEMANTIC_EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_EMBEDDER


def _semantic_similarity_many(original: str, candidates: Sequence[str]) -> List[float]:
    if not candidates:
        return []

    if _SEMANTIC_MODE not in {"transformer", "sentence_transformer"}:
        return [_lexical_similarity(original, cf) for cf in candidates]

    try:
        model = _get_semantic_embedder()
        embs = model.encode([original, *candidates], convert_to_numpy=True)
        original_emb = embs[0]
        scores = []
        for cf_emb in embs[1:]:
            cos = float(np.dot(original_emb, cf_emb) /
                        (np.linalg.norm(original_emb) * np.linalg.norm(cf_emb) + 1e-9))
            scores.append(round(cos, 3))
        return scores
    except Exception as exc:
        logger.warning("Transformer semantic similarity failed; using lexical fallback: %s", exc)
        return [_lexical_similarity(original, cf) for cf in candidates]


def _evaluate_candidate(
    original: str, original_label: str, candidate_text: str, target_label: str
) -> dict:
    """Score one CF candidate on all three metrics."""
    cf_probs     = predict_proba([candidate_text])[0]
    cf_label, _, _ = classify_severity(cf_probs)
    flip_success = cf_label != original_label
    minimality   = _minimality_score(original, candidate_text)
    semantic_sim = _semantic_similarity_many(original, [candidate_text])[0]

    if flip_success:
        score = 1.0 + 0.5 * minimality + 0.5 * semantic_sim
    else:
        label_order = {"severe": 0, "moderate": 1, "not depression": 2}
        orig_pos = label_order.get(original_label, 0)
        cf_pos = label_order.get(cf_label, 0)
        target_pos = label_order.get(target_label, 0)
        # positive when moving toward target, negative when moving away
        movement = (cf_pos - orig_pos) * (1 if target_pos > orig_pos else -1)
        # movement    = label_order.get(cf_label, 0) - label_order.get(original_label, 0)
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


def _evaluate_candidates(
    original: str,
    original_label: str,
    candidate_texts: Sequence[str],
    target_label: str,
) -> List[dict]:
    """Batch-score CF candidates to avoid repeated model/vectorizer calls."""
    if not candidate_texts:
        return []

    cf_probs_batch = predict_proba(list(candidate_texts))
    semantic_scores = _semantic_similarity_many(original, candidate_texts)
    evaluated = []

    for candidate_text, cf_probs, semantic_sim in zip(candidate_texts, cf_probs_batch, semantic_scores):
        cf_label, _, _ = classify_severity(cf_probs)
        flip_success = cf_label != original_label
        minimality = _minimality_score(original, candidate_text)

        if flip_success:
            score = 1.0 + 0.5 * minimality + 0.5 * semantic_sim
        else:
            label_order = {"severe": 0, "moderate": 1, "not depression": 2}
            orig_pos = label_order.get(original_label, 0)
            cf_pos = label_order.get(cf_label, 0)
            target_pos = label_order.get(target_label, 0)
            movement = (cf_pos - orig_pos) * (1 if target_pos > orig_pos else -1)
            score = max(0.0, 0.3 * movement / 2 + 0.2 * minimality)

        evaluated.append({
            "text":         candidate_text,
            "label":        cf_label,
            "probs":        cf_probs,
            "flip_success": flip_success,
            "minimality":   minimality,
            "semantic_sim": semantic_sim,
            "score":        round(score, 4),
        })

    return evaluated


# ── SHAP-guided CF prompt (FitCF / FIZLE approach) ───────────────────
def _build_cf_prompt(
    original_text:  str,
    original_label: str,
    target_label:   str,
    shap_tokens:    List[dict],
    n_candidates:   int = 3,
) -> str:
    token_hints = []
    for t in shap_tokens[:5]:
        word = t["token"].strip(".,!?'\"").lower()
        hint = SYMPTOM_RECOVERY_HINTS.get(word, f"Soften or replace '{t['token']}'")
        token_hints.append(f"  - '{t['token']}' (SHAP={t['shap']:+.4f}): {hint}")
    token_block = "\n".join(token_hints) or "  (Make holistic edits)"
    output_lines = "\n".join(["CF: <counterfactual text>" for _ in range(n_candidates)])

    return f"""You are a Counterfactual Explanation Generator for a depression screening AI.

TASK: Generate {n_candidates} minimally-edited versions of the text below that would
make the model predict "{target_label}" instead of "{original_label}".

ORIGINAL TEXT (current prediction: {original_label.upper()}):
"{original_text}"

SHAP HIGH-IMPACT TOKENS — focus edits here first:
{token_block}

CONSTRAINTS — each counterfactual MUST:
1. Change as FEW words as possible (minimal edit distance)
2. Sound natural — something a real person could write
3. Preserve the overall topic and personal voice
4. Reflect genuine (if modest) emotional or behavioural improvement
5. NOT be a complete rewrite — only targeted changes

OUTPUT FORMAT — exactly {n_candidates} lines, each starting with "CF:":
{output_lines}

No explanation, preamble, or extra text. Only the CF: lines."""


def _parse_candidates(llm_response: str) -> List[str]:
    candidates = []
    for line in llm_response.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("cf:"):
            text = line[3:].strip().strip('"\'')
            if len(text.split()) >= 4:
                candidates.append(text)
    return candidates


# ── Main pipeline ─────────────────────────────────────────────────────
def generate_counterfactuals(
    user_text:    str,
    n_candidates: int = 3,
    n_attempts:   int = 2,
) -> CounterfactualResult:
    """
    Runs on RAW user text — no reframing.
    SHAP explains the user's actual words.
    CFs are edits to the user's actual words.
    """
    # Step 1: predict on raw text
    probs                          = predict_proba([user_text])[0]
    original_label, score, reason  = classify_severity(probs)
    target_label                   = CF_TARGET_MAP[original_label]

    logger.info("CF pipeline: %s (score=%.3f) → target: %s", original_label, score, target_label)

    result = CounterfactualResult(
        original_text  = user_text,
        original_label = original_label,
        original_probs = probs,
        severity_score = score,
        target_label   = target_label,
        already_well   = (original_label == "not depression"),
    )

    # Step 2: SHAP on raw text for token guidance
    shap_result                = explain_with_shap(user_text)
    result.shap_guided_tokens  = shap_result.risk_tokens[:5]
    logger.info("SHAP risk tokens: %s", [t["token"] for t in result.shap_guided_tokens])

    # Step 3: LLM generation. Request the total candidate budget in one call
    # to avoid repeated LLM round trips; retry only if parsing returns nothing.
    all_candidates = []
    requested_candidates = max(1, n_candidates * max(n_attempts, 1))
    max_attempts = 2 if n_attempts > 1 else 1
    for attempt in range(max_attempts):
        prompt = _build_cf_prompt(
            original_text  = user_text,
            original_label = original_label,
            target_label   = target_label,
            shap_tokens    = result.shap_guided_tokens,
            n_candidates   = requested_candidates,
        )
        try:
            parsed = _parse_candidates(call_gemini(prompt))
            logger.info("Attempt %d: %d candidates", attempt + 1, len(parsed))
            all_candidates.extend(parsed)
            if parsed:
                break
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)

    # Deduplicate
    seen, unique = set(), []
    for c in all_candidates:
        if c not in seen and c != user_text:
            seen.add(c)
            unique.append(c)

    # Step 4: evaluate in batch
    evaluated = _evaluate_candidates(user_text, original_label, unique, target_label)
    evaluated.sort(key=lambda x: (x["flip_success"], x["score"]), reverse=True)

    result.candidates = evaluated
    valid             = [c for c in evaluated if c["flip_success"]]
    result.best_cf    = valid[0] if valid else (evaluated[0] if evaluated else None)

    logger.info("CF done: %d candidates, %d valid flips", len(evaluated), len(valid))
    return result


def format_cf_debug(result: CounterfactualResult) -> str:
    probs = ", ".join(f"{LABEL_MAP[i]}={result.original_probs[i]:.3f}" for i in range(3))
    lines = [
        f"Original : {result.original_label} (score={result.severity_score:.3f}, {probs})",
        f"Target   : {result.target_label}",
        f"Tokens   : {[t['token'] for t in result.shap_guided_tokens]}",
        f"Candidates: {len(result.candidates)} ({sum(1 for c in result.candidates if c['flip_success'])} flips)",
    ]
    if result.best_cf:
        c = result.best_cf
        lines.append(
            f"Best CF  : [{c['label']}] flip={c['flip_success']} "
            f"min={c['minimality']} sim={c['semantic_sim']}\n"
            f"           '{c['text'][:80]}'"
        )
    return "\n".join(lines)
