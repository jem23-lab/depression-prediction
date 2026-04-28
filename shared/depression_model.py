"""
shared/depression_model.py
────────────────────────────────────────────────────────────────────
Shared model wrapper used by ALL use cases (SHAP, RAG, hybrid, etc.)

Model : rafalposwiata/deproberta-large-depression
Labels: resolved at load() from _model.config.id2label

CALIBRATED SEVERITY :
  The model outputs soft probabilities for three classes.
  Simple argmax misses clinically important cases where severe probability
  is meaningful but not the top class (e.g. 22% severe / 68% moderate still
  warrants a severe-level response). We use a severity_prob threshold to
  catch these cases, using the actual per-class index resolved from the
  model's id2label at load time.

Exposes:
    predict_proba(texts)        → np.ndarray (n, 3)
    classify_severity(probs)    → (label, severity_score, reason)
    explain_with_shap(text)     → SHAPResult
    SUICIDAL_IDEATION_PATTERNS  → for crisis detection in bot.py
"""

import re
import logging
import numpy as np
import torch
import shap
from dataclasses import dataclass, field
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "rafalposwiata/deproberta-large-depression"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Canonical label map assumed from the model card.
# load() validates this against _model.config.id2label and corrects if needed.
LABEL_MAP = {0: "severe", 1: "moderate", 2: "not depression"}

LABEL_DESCRIPTIONS = {
    "severe":         "Strong indicators of severe depression — persistent hopelessness, emptiness, or inability to function.",
    "moderate":       "Signs of moderate depression — low mood, fatigue, and reduced interest in life.",
    "not depression": "No significant depression signals detected at this time.",
}

# ── Calibrated severity thresholds ────────────────────────────────────
# If severe probability reaches this level even when moderate has higher prob,
# the case is clinically significant enough to escalate to "severe".
# Lowered from 0.30: the model (trained on social-media text) tends to assign
# 20–25% severe probability for formally-worded but clinically severe input.
# A threshold of 0.20 correctly catches those cases without over-triggering on
# genuinely moderate texts (which typically show severe_prob ≤ 0.15).
SEVERE_ESCALATION_THRESHOLD = 0.20
# Canonical class weights for the weighted severity score (not positional indices)
SEVERITY_WEIGHTS = {"severe": 1.0, "moderate": 0.5, "not depression": 0.0}

# ── Per-class probability indices (resolved at load time) ─────────────
# These may differ from LABEL_MAP keys if the model's id2label order differs.
_severe_idx   = 0
_moderate_idx = 1
_nodep_idx    = 2

# ── Crisis / suicidal ideation patterns (used by bot.py) ─────────────
# These patterns detect when the bot must show crisis resources FIRST.
# We keep detection here so it stays in sync with the model wrapper.
SUICIDAL_IDEATION_PATTERNS = [
    r"fall asleep forever",
    r"sleep forever",
    r"never wake up",
    r"don'?t want to (?:be here|exist|live|wake up|go on)",
    r"want(?:ing)? to disappear",
    r"end(?:ing)? it (?:all|myself)",
    r"no reason to (?:live|go on|continue|exist)",
    r"better off (?:dead|without me|gone)",
    r"can'?t (?:go on|take it anymore|do this anymore)",
    r"want(?:ing)? to (?:die|kill myself|end my life)",
    r"thoughts? of (?:suicide|ending|death)",
    r"(?:lost|losing) the will to (?:live|go on)",
]


def is_crisis_text(text: str) -> bool:
    """Returns True if text contains suicidal ideation or crisis signals."""
    tl = text.lower()
    return any(re.search(p, tl) for p in SUICIDAL_IDEATION_PATTERNS)


# ── Calibrated severity classifier ────────────────────────────────────
def classify_severity(probs: np.ndarray) -> Tuple[str, float, str]:
    """
    Returns (label, severity_score, reason) from raw model probabilities.

    Uses _severe_idx / _moderate_idx / _nodep_idx resolved at load() from
    the model's actual id2label, so the correct probability column is always
    used for each class regardless of the model's internal label ordering.

    Escalation logic: if the severe-class probability reaches
    SEVERE_ESCALATION_THRESHOLD even when moderate has the highest raw
    probability, the case is classified as severe.  The threshold (0.20) is
    tuned for the deproberta model which was trained on social-media text and
    tends to assign 20–25% severe probability to formally-worded but
    clinically severe input.

    severity_score: 0.0 (none) → 1.0 (severe), continuous.
    reason: human-readable string explaining the classification decision.
    """
    severe_prob   = float(probs[_severe_idx])
    moderate_prob = float(probs[_moderate_idx])
    nodep_prob    = float(probs[_nodep_idx])

    # Weighted severity score using canonical class weights
    score = (severe_prob   * SEVERITY_WEIGHTS["severe"]
             + moderate_prob * SEVERITY_WEIGHTS["moderate"]
             + nodep_prob    * SEVERITY_WEIGHTS["not depression"])

    argmax_idx   = int(np.argmax(probs))
    argmax_label = LABEL_MAP[argmax_idx]

    # Escalate to severe if severe_prob is clinically meaningful
    if severe_prob >= SEVERE_ESCALATION_THRESHOLD:
        label  = "severe"
        reason = (
            f"Severe probability {severe_prob:.1%} >= escalation threshold "
            f"{SEVERE_ESCALATION_THRESHOLD:.0%} — escalated from '{argmax_label}'"
        )
    else:
        label  = argmax_label
        reason = f"Argmax prediction (severe={severe_prob:.1%}, moderate={moderate_prob:.1%})"

    return label, round(score, 4), reason


# ── Token-level annotations for SHAP ─────────────────────────────────
CLINICAL_TOKEN_NOTES = {
    "hopeless":    "strongly linked to depressive hopelessness",
    "hopelessness":"core depression symptom",
    "empty":       "emotional numbness — a hallmark of depression",
    "tired":       "persistent fatigue is a core depression symptom",
    "exhausted":   "severe fatigue signal",
    "worthless":   "key clinical indicator of low self-worth",
    "nothing":     "anhedonia and nihilism pattern",
    "anymore":     "implies loss of a previous positive state",
    "forever":     "passive suicidal ideation — wanting to cease existing",
    "death":       "may indicate passive suicidal ideation",
    "die":         "active suicidal ideation — requires clinical attention",
    "suicide":     "requires immediate clinical attention",
    "disappear":   "passive suicidal ideation signal",
    "burden":      "perceived burdensomeness — strong predictor of severe depression",
    "sad":         "direct expression of low mood",
    "lonely":      "social withdrawal and isolation signal",
    "cry":         "frequent crying is a depression marker",
    "sleep":       "sleep disturbance is a core symptom",
    "numb":        "emotional blunting associated with depression",
    "concentrate": "concentration difficulty — cognitive depression symptom",
    "appetite":    "appetite change — somatic depression symptom",
    "unbearable":  "severe subjective distress marker",
    "bad":         "subjective suffering — intensity depends on context",
    "worse":       "worsening trajectory — severity escalation signal",
    "meaningless": "loss of meaning — existential symptom of depression",
    "pointless":   "hopelessness and nihilism pattern",
}


# ── Singletons ────────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_explainer = None


def load():
    global _tokenizer, _model, _explainer
    global _severe_idx, _moderate_idx, _nodep_idx
    if _model is not None:
        return
    print(f"Loading {MODEL_NAME} on {DEVICE} …")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(DEVICE)
    _model.eval()
    print(f"Model loaded. Labels: {_model.config.id2label}")

    # ── Resolve per-class probability indices from the model's id2label ──
    # The model's label strings may use different casing or wording than
    # LABEL_MAP.  We normalise both sides and find which output column
    # corresponds to "severe", "moderate", and "not depression".
    def _canonical(s: str) -> str:
        s = s.lower().strip()
        if s.startswith("not ") or s in ("no depression", "not depression", "none"):
            return "not depression"
        if "moderate" in s:
            return "moderate"
        if "severe" in s:
            return "severe"
        return s

    try:
        model_labels = {int(k): _canonical(v)
                        for k, v in _model.config.id2label.items()}
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Could not parse model id2label keys as integers (%s). "
            "Falling back to assumed order: severe=0, moderate=1, not_dep=2.",
            exc,
        )
        model_labels = {}

    resolved_severe   = None
    resolved_moderate = None
    resolved_nodep    = None
    for idx, canon in model_labels.items():
        if canon == "severe":
            resolved_severe = idx
        elif canon == "moderate":
            resolved_moderate = idx
        elif canon == "not depression":
            resolved_nodep = idx

    if None in (resolved_severe, resolved_moderate, resolved_nodep):
        logger.warning(
            "Could not fully resolve label indices from model id2label=%s. "
            "Falling back to assumed order: severe=0, moderate=1, not_dep=2. "
            "Raw model labels: %s",
            dict(_model.config.id2label),
            model_labels,
        )
    else:
        _severe_idx   = resolved_severe
        _moderate_idx = resolved_moderate
        _nodep_idx    = resolved_nodep
        if (_severe_idx, _moderate_idx, _nodep_idx) != (0, 1, 2):
            logger.warning(
                "Model id2label order differs from assumed LABEL_MAP. "
                "Corrected indices — severe=%d, moderate=%d, not_dep=%d",
                _severe_idx, _moderate_idx, _nodep_idx,
            )
        else:
            logger.info(
                "Label indices confirmed: severe=%d, moderate=%d, not_dep=%d",
                _severe_idx, _moderate_idx, _nodep_idx,
            )

    print("Initialising SHAP explainer …")
    _explainer = shap.Explainer(
        lambda texts: predict_proba(list(texts)),
        _tokenizer,
        output_names=list(LABEL_MAP.values()),
    )
    print("SHAP ready.")


def predict_proba(texts: List[str]) -> np.ndarray:
    """
    Returns raw softmax probabilities shape (n, 3).
    Column order matches the model's id2label (resolved at load time).
    Use _severe_idx / _moderate_idx / _nodep_idx to access each class.
    The USER'S TEXT is passed directly — no reframing.
    """
    load()
    with torch.no_grad():
        enc    = _tokenizer(texts, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(DEVICE)
        logits = _model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


# ── SHAP result dataclass ─────────────────────────────────────────────
@dataclass
class SHAPResult:
    text:              str           # original user text, unchanged
    tokens:            List[str]
    shap_matrix:       np.ndarray
    pred_label_idx:    int           # argmax index (0/1/2)
    pred_label:        str           # calibrated label (may differ from argmax)
    pred_probs:        np.ndarray
    severity_score:    float = 0.0   # continuous 0.0–1.0
    severity_reason:   str   = ""    # why this label was chosen
    top_tokens:        List[dict] = field(default_factory=list)
    risk_tokens:       List[dict] = field(default_factory=list)
    protective_tokens: List[dict] = field(default_factory=list)


def explain_with_shap(text: str, top_n: int = 8) -> SHAPResult:
    """
    Runs SHAP token-level explanation on the RAW user text.
    No reframing — SHAP explains exactly what the user wrote.
    Calibrated severity is applied to the raw probabilities.
    """
    load()

    # SHAP on raw text
    shap_vals   = _explainer([text])
    explanation = shap_vals[0]
    tokens      = list(explanation.data)
    shap_matrix = explanation.values

    # Raw probabilities → calibrated label
    pred_probs                 = predict_proba([text])[0]
    pred_label, score, reason  = classify_severity(pred_probs)
    pred_label_idx             = int(np.argmax(pred_probs))  # for SHAP column selection

    # SHAP values for the argmax column (the model's strongest signal)
    pred_col_shap = shap_matrix[:, pred_label_idx]

    records = []
    for tok, sv in zip(tokens, pred_col_shap):
        clean = tok.strip()
        if not clean or clean in ("", "▁"):
            continue
        key  = clean.lower().strip(".,!?'\"")
        note = CLINICAL_TOKEN_NOTES.get(key, "")
        records.append({
            "token":     clean,
            "shap":      float(sv),
            "abs_shap":  abs(float(sv)),
            "direction": "↑ increases risk" if sv > 0 else "↓ reduces risk",
            "note":      note,
        })
    records.sort(key=lambda x: x["abs_shap"], reverse=True)

    return SHAPResult(
        text              = text,
        tokens            = tokens,
        shap_matrix       = shap_matrix,
        pred_label_idx    = pred_label_idx,
        pred_label        = pred_label,
        pred_probs        = pred_probs,
        severity_score    = score,
        severity_reason   = reason,
        top_tokens        = records[:top_n],
        risk_tokens       = [r for r in records if r["shap"] > 0][:5],
        protective_tokens = [r for r in records if r["shap"] < 0][:3],
    )


def format_debug(result: SHAPResult) -> str:
    probs = ", ".join(f"{LABEL_MAP[i]}={result.pred_probs[i]:.3f}" for i in range(3))
    lines = [
        f"Prediction : {result.pred_label} (score={result.severity_score:.3f})",
        f"Raw probs  : {probs}",
        f"Reason     : {result.severity_reason}",
        "Top tokens :",
    ]
    for t in result.top_tokens[:5]:
        lines.append(f"  '{t['token']}' SHAP={t['shap']:+.4f}  {t['direction']}")
    return "\n".join(lines)
