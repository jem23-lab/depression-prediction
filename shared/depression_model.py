"""
shared/depression_model.py
────────────────────────────────────────────────────────────────────
Lightweight model wrapper used by ALL use cases (SHAP, RAG, hybrid, etc.)

Model : TF-IDF centroid similarity trained from local examples
Labels: derived from training examples at load() time (no hardcoded order)

Exposes:
    predict_proba(texts)        → np.ndarray (n, k)
    classify_severity(probs, text=None)    → (label, severity_score, reason)
    explain_with_shap(text)     → SHAPResult
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import hashlib
import json
import os
import re

import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import joblib

from shared.training_examples import PARAGRAPHS as TRAINING_EXAMPLES

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
LABEL_MAP: dict = {}
LABEL_DESCRIPTIONS: dict = {}
_idx_to_label: dict = {}
_label_to_idx: dict = {}

_vectorizer: Optional[TfidfVectorizer] = None
_classifier: Optional[LogisticRegression] = None
_train_matrix: Optional[np.ndarray] = None
_centroids: Optional[np.ndarray] = None
_centroid_diffs: Optional[np.ndarray] = None

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
_MODEL_PATH = os.path.join(_CACHE_DIR, "depression_tfidf_lr.joblib")


def _normalize_label(label: str) -> str:
    raw = (label or "").strip().lower()
    if raw in {"none", "no depression", "not depression", "not depressed"}:
        return "not depression"
    if "moderate" in raw:
        return "moderate"
    if "severe" in raw:
        return "severe"
    return raw or "unknown"


def _severity_rank(label: str) -> int:
    canonical = _normalize_label(label)
    if "severe" in canonical:
        return 2
    if "moderate" in canonical:
        return 1
    return 0


def _severity_weight(label: str) -> float:
    canonical = _normalize_label(label)
    if "severe" in canonical:
        return 1.0
    if "moderate" in canonical:
        return 0.5
    return 0.0


# ── Calibrated severity classifier ────────────────────────────────────
def classify_severity(probs: np.ndarray, text: Optional[str] = None) -> Tuple[str, float, str]:
    """
    Returns (label, severity_score, reason) from raw model probabilities.

    severity_score: 0.0 (none) → 1.0 (severe), continuous.
    reason: human-readable string explaining the classification decision.
    """
    if not _idx_to_label:
        load()

    probs = np.asarray(probs).flatten()
    if probs.size == 0:
        return "unknown", 0.0, "Empty probability vector"

    argmax_idx = int(np.argmax(probs))
    argmax_label = _idx_to_label.get(argmax_idx, "unknown")

    score = 0.0
    for i in range(len(_idx_to_label)):
        score += float(probs[i]) * _severity_weight(_idx_to_label[i])

    parts = ", ".join(
        f"{_idx_to_label[i]}={float(probs[i]):.1%}" for i in range(len(_idx_to_label))
    )
    reason = f"Argmax prediction ({parts})"

    return argmax_label, round(score, 4), reason


def _training_fingerprint(rows: List[dict]) -> str:
    items = []
    for row in rows:
        text = (row.get("text") or "").strip()
        label = _normalize_label(row.get("severity") or row.get("label") or "")
        items.append({"label": label, "text": text})
    payload = json.dumps(items, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_model(fingerprint: str) -> bool:
    global _vectorizer, _classifier
    global LABEL_MAP, _idx_to_label, _label_to_idx
    global _train_matrix, _centroids, _centroid_diffs

    if not os.path.exists(_MODEL_PATH):
        return False

    try:
        data = joblib.load(_MODEL_PATH)
    except Exception as exc:
        logger.warning("Failed to load cached model: %s", exc)
        return False

    if data.get("fingerprint") != fingerprint:
        return False

    _vectorizer = data.get("vectorizer")
    _classifier = data.get("classifier")
    _idx_to_label = data.get("idx_to_label") or {}
    _label_to_idx = data.get("label_to_idx") or {}
    _train_matrix = data.get("train_matrix")
    _centroids = data.get("centroids")
    _centroid_diffs = data.get("centroid_diffs")
    LABEL_MAP = dict(_idx_to_label)

    if _vectorizer is None or not _idx_to_label:
        return False
    if _centroids is None or _centroid_diffs is None:
        return False

    logger.info("Loaded cached model from %s", _MODEL_PATH)
    return True


def _save_model(fingerprint: str) -> None:
    if _vectorizer is None or not _idx_to_label:
        return
    if _centroids is None or _centroid_diffs is None:
        return

    os.makedirs(_CACHE_DIR, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "vectorizer": _vectorizer,
        "classifier": _classifier,
        "idx_to_label": _idx_to_label,
        "label_to_idx": _label_to_idx,
        "train_matrix": _train_matrix,
        "centroids": _centroids,
        "centroid_diffs": _centroid_diffs,
    }
    joblib.dump(payload, _MODEL_PATH)


# ── Singletons ────────────────────────────────────────────────────────
def load():
    global _vectorizer, _classifier
    global LABEL_MAP, _idx_to_label, _label_to_idx
    global _train_matrix, _centroids, _centroid_diffs

    if _vectorizer is not None and _centroids is not None:
        return

    fingerprint = _training_fingerprint(TRAINING_EXAMPLES)
    if _load_cached_model(fingerprint):
        return

    texts: List[str] = []
    labels: List[str] = []
    for row in TRAINING_EXAMPLES:
        text = (row.get("text") or "").strip()
        label = _normalize_label(row.get("severity") or row.get("label") or "")
        if not text or label == "unknown":
            continue
        texts.append(text)
        labels.append(label)

    if not texts:
        raise RuntimeError("No training examples found for the simple model.")

    label_set = sorted(set(labels), key=_severity_rank)
    _idx_to_label = {i: lbl for i, lbl in enumerate(label_set)}
    _label_to_idx = {lbl: i for i, lbl in _idx_to_label.items()}
    LABEL_MAP = dict(_idx_to_label)

    _vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    train_matrix = _vectorizer.fit_transform(texts)
    _train_matrix = train_matrix

    num_labels = len(_idx_to_label)
    centroids = []
    for label_idx in range(num_labels):
        mask = np.array([_label_to_idx[lbl] == label_idx for lbl in labels])
        if not mask.any():
            centroid = np.zeros((train_matrix.shape[1],), dtype=float)
        else:
            centroid = train_matrix[mask].mean(axis=0)
            centroid = np.asarray(centroid).ravel()
        centroids.append(centroid)

    _centroids = np.vstack(centroids)
    overall = _centroids.mean(axis=0)
    _centroid_diffs = _centroids - overall

    _save_model(fingerprint)

    logger.info("Simple model trained on %d examples. Labels: %s", len(texts), _idx_to_label)


def predict_proba(texts: List[str]) -> np.ndarray:
    """
    Returns similarity-based probabilities shape (n, k).
    Column order matches the label set derived in load().
    """
    load()
    vectors = _vectorizer.transform(texts)
    vectors = normalize(vectors, norm="l2", axis=1)
    centroids = normalize(_centroids, norm="l2", axis=1)

    sims = vectors.dot(centroids.T)
    sims = np.asarray(sims)
    if sims.size == 0:
        return np.zeros((len(texts), len(_idx_to_label)), dtype=float)

    sims = np.maximum(sims, 0.0)
    row_sums = sims.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    probs = sims / row_sums
    return np.asarray(probs, dtype=float)


# ── SHAP result dataclass ─────────────────────────────────────────────
@dataclass
class SHAPResult:
    text:              str
    tokens:            List[str]
    shap_matrix:       np.ndarray
    pred_label_idx:    int
    pred_label:        str
    pred_probs:        np.ndarray
    severity_score:    float = 0.0
    severity_reason:   str   = ""
    top_tokens:        List[dict] = field(default_factory=list)
    risk_tokens:       List[dict] = field(default_factory=list)
    protective_tokens: List[dict] = field(default_factory=list)


_SHAP_EXPLAINER = None
_SHAP_MASKER = None
_LOW_CONTEXT_TOKENS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "cant", "can't", "cannot", "could", "couldn", "couldnt", "couldn't",
    "did", "didnt", "didn't", "do", "does", "doesnt", "doesn't", "doing",
    "don", "dont", "don't", "every", "feel", "feeling", "felt", "for",
    "from", "get", "go", "had", "has", "have", "having", "i", "im", "i'm",
    "in", "is", "it", "its", "it's", "just", "kind", "like", "lot", "make",
    "makes", "me", "my", "of", "on", "or", "really", "seem", "seems", "so",
    "something", "stuff", "that", "the", "then", "things", "this", "to",
    "t", "very", "was", "were", "with", "would", "wouldn", "you",
    "currently", "today", "now", "lately", "recently",
}


def _get_shap_explainer():
    global _SHAP_EXPLAINER, _SHAP_MASKER

    if _SHAP_EXPLAINER is not None:
        return _SHAP_EXPLAINER

    token_pattern = r"[A-Za-z]+(?:['’][A-Za-z]+)?"
    _SHAP_MASKER = shap.maskers.Text(token_pattern)

    def _model_fn(texts):
        return predict_proba(list(texts))

    _SHAP_EXPLAINER = shap.Explainer(_model_fn, _SHAP_MASKER)
    return _SHAP_EXPLAINER


def _token_contributions_shap(text: str) -> Tuple[List[str], np.ndarray]:
    explainer = _get_shap_explainer()
    shap_values = explainer([text])

    tokens = list(shap_values.data[0]) if shap_values.data is not None else []
    values = np.asarray(shap_values.values[0])
    if values.ndim == 1:
        values = values[:, None]

    return tokens, values


def _token_contributions(text: str) -> Tuple[List[str], np.ndarray]:
    """Compute SHAP-like per-token contributions from centroid differences."""
    load()

    try:
        return _token_contributions_shap(text)
    except Exception as exc:
        logger.debug("SHAP tokenizer failed, falling back to centroid contributions: %s", exc)

    analyzer = _vectorizer.build_analyzer()
    tokens = analyzer(text)
    num_labels = len(_idx_to_label)
    if not tokens:
        return [], np.zeros((0, num_labels), dtype=float)

    vector = _vectorizer.transform([text])
    token_counts = Counter(tokens)

    shap_matrix = np.zeros((len(tokens), num_labels), dtype=float)

    if vector.nnz == 0:
        return tokens, shap_matrix

    row = vector.tocsr()
    vocab = _vectorizer.vocabulary_
    for i, token in enumerate(tokens):
        idx = vocab.get(token)
        if idx is None:
            continue
        tfidf_val = float(row[0, idx])
        if tfidf_val == 0.0:
            continue
        per_occurrence = tfidf_val / token_counts[token]
        for class_idx in range(num_labels):
            shap_matrix[i, class_idx] = per_occurrence * float(_centroid_diffs[class_idx, idx])

    return tokens, shap_matrix


def _resolve_display_token(token: str, text: str) -> str:
    if not text:
        return token

    lower_text = text.lower()
    token_lower = token.lower()

    if " " in token:
        idx = lower_text.find(token_lower)
        if idx != -1:
            return text[idx:idx + len(token)]
    else:
        pattern = re.compile(r"\b" + re.escape(token_lower) + r"\b", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return text[match.start():match.end()]

        for suffix in ["n't", "’t", "'t", "t"]:
            pattern = re.compile(r"\b" + re.escape(token_lower + suffix) + r"\b", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                return text[match.start():match.end()]

    return token


def _contextual_display_token(token: str, text: str) -> str:
    if not text or " " in token or not _is_meaningful_display_token(token):
        return token

    pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return token

    left_text = text[:match.start()]
    right_text = text[match.end():]
    left_words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", left_text)[-2:]
    right_words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", right_text)[:3]

    phrase_words = [text[match.start():match.end()]]
    if right_words and right_words[0].lower() in {"about", "after", "because", "from", "over", "with", "without"}:
        phrase_words.extend(right_words)
    elif left_words and left_words[-1].lower() in {"avoid", "avoided", "avoiding", "keep", "kept", "staying"}:
        phrase_words = [left_words[-1]] + phrase_words

    phrase = " ".join(phrase_words).strip()
    if phrase and phrase.lower() != token.lower() and _is_meaningful_display_token(phrase):
        return phrase
    return token


def _aggregate_records(records: List[dict]) -> List[dict]:
    merged = {}
    for rec in records:
        key = rec["token"].lower()
        if key not in merged:
            merged[key] = {
                "token": rec["token"],
                "shap": 0.0,
                "abs_shap": 0.0,
                "direction": rec["direction"],
                "note": rec.get("note", ""),
            }
        merged[key]["shap"] += float(rec["shap"])

    for item in merged.values():
        item["abs_shap"] = abs(float(item["shap"]))
        item["direction"] = "↑ increases risk" if item["shap"] > 0 else "↓ reduces risk"

    return list(merged.values())


def _rank_records(records: List[dict]) -> List[dict]:
    return sorted(
        records,
        key=lambda r: (r["abs_shap"], 1 if " " in r["token"] else 0, len(r["token"])),
        reverse=True,
    )


def _token_terms(token: str) -> List[str]:
    return re.findall(r"[a-z]+(?:['’][a-z]+)?", token.lower())


def _is_meaningful_display_token(token: str) -> bool:
    terms = _token_terms(token)
    if not terms:
        return False

    meaningful_terms = [
        term for term in terms
        if term not in _LOW_CONTEXT_TOKENS and term.replace("'", "").replace("’", "") not in _LOW_CONTEXT_TOKENS
    ]
    if " " in token:
        return bool(meaningful_terms)
    return bool(meaningful_terms) and len(meaningful_terms[0]) >= 3


def _filter_display_records(records: List[dict]) -> List[dict]:
    if not records:
        return records

    strongest = max(float(r["abs_shap"]) for r in records)
    min_abs = strongest * 0.20
    filtered = [
        r for r in records
        if _is_meaningful_display_token(r["token"]) and float(r["abs_shap"]) >= min_abs
    ]
    return filtered or [r for r in records if _is_meaningful_display_token(r["token"])] or records[:3]


def explain_with_shap(text: str, top_n: int = 8) -> SHAPResult:
    """
    Runs SHAP-like token-level explanation on the RAW user text.
    No reframing — explanations reflect exactly what the user wrote.
    """
    tokens, shap_matrix = _token_contributions(text)

    pred_probs = predict_proba([text])[0]
    pred_label, score, reason = classify_severity(pred_probs, text=text)
    pred_label_idx = int(np.argmax(pred_probs)) if pred_probs.size else 0

    pred_col_shap = shap_matrix[:, pred_label_idx] if shap_matrix.size else np.array([])

    records = []
    for tok, sv in zip(tokens, pred_col_shap):
        clean = tok.strip()
        if not clean:
            continue
        display = _contextual_display_token(_resolve_display_token(clean, text), text)
        records.append({
            "token": display,
            "shap": float(sv),
            "abs_shap": abs(float(sv)),
            "direction": "↑ increases risk" if sv > 0 else "↓ reduces risk",
            "note": "",
        })
    records = _aggregate_records(records)
    records = _filter_display_records(_rank_records(records))

    return SHAPResult(
        text=text,
        tokens=tokens,
        shap_matrix=shap_matrix,
        pred_label_idx=pred_label_idx,
        pred_label=pred_label,
        pred_probs=pred_probs,
        severity_score=score,
        severity_reason=reason,
        top_tokens=records[:top_n],
        risk_tokens=[r for r in records if r["shap"] > 0][:5],
        protective_tokens=[r for r in records if r["shap"] < 0][:3],
    )


def format_debug(result: SHAPResult) -> str:
    probs = ", ".join(
        f"{_idx_to_label[i]}={result.pred_probs[i]:.3f}" for i in range(len(_idx_to_label))
    )
    lines = [
        f"Prediction : {result.pred_label} (score={result.severity_score:.3f})",
        f"Raw probs  : {probs}",
        f"Reason     : {result.severity_reason}",
        "Top tokens :",
    ]
    for t in result.top_tokens[:5]:
        lines.append(f"  '{t['token']}' SHAP={t['shap']:+.4f}  {t['direction']}")
    return "\n".join(lines)
