"""
shared/depression_model copy.py
────────────────────────────────────────────────────────────────────
Lightweight model wrapper used by ALL use cases (SHAP, RAG, hybrid, etc.)

Model : TF-IDF + LogisticRegression trained from local examples
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

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from shared.training_examples import PARAGRAPHS as TRAINING_EXAMPLES

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
LABEL_MAP: dict = {}
LABEL_DESCRIPTIONS: dict = {}
_idx_to_label: dict = {}
_label_to_idx: dict = {}

_vectorizer: Optional[TfidfVectorizer] = None
_classifier: Optional[LogisticRegression] = None


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


# ── Singletons ────────────────────────────────────────────────────────
def load():
    global _vectorizer, _classifier
    global LABEL_MAP, _idx_to_label, _label_to_idx

    if _classifier is not None:
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
        ngram_range=(1, 1),
    )
    _classifier = LogisticRegression(max_iter=1000)

    x_train = _vectorizer.fit_transform(texts)
    y_train = np.array([_label_to_idx[lbl] for lbl in labels])
    _classifier.fit(x_train, y_train)

    logger.info("Simple model trained on %d examples. Labels: %s", len(texts), _idx_to_label)


def predict_proba(texts: List[str]) -> np.ndarray:
    """
    Returns raw softmax probabilities shape (n, k).
    Column order matches the label set derived in load().
    """
    load()
    vectors = _vectorizer.transform(texts)
    probs = _classifier.predict_proba(vectors)
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


def _token_contributions(text: str) -> Tuple[List[str], np.ndarray]:
    """Compute SHAP-like per-token contributions for a linear model."""
    load()

    analyzer = _vectorizer.build_analyzer()
    tokens = analyzer(text)
    num_labels = len(_idx_to_label)
    if not tokens:
        return [], np.zeros((0, num_labels), dtype=float)

    vector = _vectorizer.transform([text])
    token_counts = Counter(tokens)

    coefs = _classifier.coef_
    if coefs.shape[0] == 1 and num_labels == 2:
        coefs = np.vstack([-coefs[0], coefs[0]])

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
        for class_idx in range(min(num_labels, coefs.shape[0])):
            shap_matrix[i, class_idx] = per_occurrence * float(coefs[class_idx, idx])

    return tokens, shap_matrix


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
        records.append({
            "token": clean,
            "shap": float(sv),
            "abs_shap": abs(float(sv)),
            "direction": "↑ increases risk" if sv > 0 else "↓ reduces risk",
            "note": "",
        })
    records.sort(key=lambda x: x["abs_shap"], reverse=True)

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
