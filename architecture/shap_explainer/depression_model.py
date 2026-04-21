"""
depression_model.py
────────────────────────────────────────────────────────────────────
Wraps the pretrained HuggingFace model:
    rafalposwiata/deproberta-large-depression

Label map (from model.config.id2label):
    0 → severe
    1 → moderate
    2 → not depression

Exposes:
    predict_proba(texts)          → np.ndarray (n, 3)
    explain_with_shap(text)       → SHAPResult dataclass
"""

import numpy as np
import torch
import shap
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ─────────────────────────────────────────────────────────
MODEL_NAME = "rafalposwiata/deproberta-large-depression"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Label map (matches model.config.id2label from notebook) ────────
LABEL_MAP = {0: "severe", 1: "moderate", 2: "not depression"}

# ── Severity descriptions for the LLM prompt ───────────────────────
LABEL_DESCRIPTIONS = {
    "severe":          "The text shows strong indicators of severe depression — persistent hopelessness, emptiness, or inability to function.",
    "moderate":        "The text shows signs of moderate depression — low mood, fatigue, and reduced interest in life.",
    "not depression":  "The text does not show significant depression signals at this time.",
}

# ── Token-importance descriptions ──────────────────────────────────
CLINICAL_TOKEN_NOTES = {
    "hopeless":   "strongly linked to depressive hopelessness",
    "empty":      "indicates emotional numbness, a hallmark of depression",
    "tired":      "persistent fatigue is a core depression symptom",
    "worthless":  "feelings of worthlessness are a key clinical indicator",
    "nothing":    "anhedonia and nihilism pattern",
    "anymore":    "implies loss of a previous positive state",
    "death":      "may indicate passive suicidal ideation",
    "suicide":    "requires immediate clinical attention",
    "sad":        "direct expression of low mood",
    "lonely":     "social withdrawal and isolation signal",
    "cry":        "frequent crying is a depression marker",
    "sleep":      "sleep disturbance is a core symptom",
    "numb":       "emotional blunting associated with depression",
}


# ── Global singletons (loaded once) ────────────────────────────────
_tokenizer = None
_model     = None
_explainer = None


def _load():
    """Lazy-load model, tokenizer, and SHAP explainer."""
    global _tokenizer, _model, _explainer
    if _model is not None:
        return

    print(f"Loading {MODEL_NAME} on {DEVICE} …")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(DEVICE)
    _model.eval()
    print(f"Model loaded. Label map: {_model.config.id2label}")

    print("Initialising SHAP explainer …")
    _explainer = shap.Explainer(
        _shap_predict_fn,
        _tokenizer,
        output_names=list(LABEL_MAP.values()),
    )
    print("SHAP explainer ready.")


def _shap_predict_fn(texts: List[str]) -> np.ndarray:
    """Callable passed to shap.Explainer — returns softmax probabilities."""
    return predict_proba(list(texts))


# ── Public API ──────────────────────────────────────────────────────
def predict_proba(texts: List[str]) -> np.ndarray:
    """
    Returns softmax probabilities of shape (n_texts, 3).
    Column order: [severe, moderate, not_depression]
    """
    _load()
    with torch.no_grad():
        enc     = _tokenizer(texts, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(DEVICE)
        logits  = _model(**enc).logits
        probs   = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


@dataclass
class SHAPResult:
    """All artefacts produced by explain_with_shap()."""
    text:              str
    tokens:            List[str]
    shap_matrix:       np.ndarray          # shape (n_tokens, 3)
    pred_label_idx:    int
    pred_label:        str
    pred_probs:        np.ndarray          # shape (3,)

    # Per-token contributions for the predicted class (sorted desc by |shap|)
    top_tokens:        List[dict] = field(default_factory=list)
    # Top tokens that push toward depression (positive for severe/moderate)
    risk_tokens:       List[dict] = field(default_factory=list)
    # Top tokens that push away from depression (negative for severe/moderate)
    protective_tokens: List[dict] = field(default_factory=list)


def explain_with_shap(text: str, top_n: int = 8) -> SHAPResult:
    """
    Runs SHAP token-level explanation on a single text.
    Returns a SHAPResult with ranked token contributions.
    """
    _load()

    shap_values = _explainer([text])          # list[Explanation]
    explanation  = shap_values[0]

    tokens      = list(explanation.data)      # list of token strings
    shap_matrix = explanation.values          # (n_tokens, 3)

    # Predict class
    pred_probs     = predict_proba([text])[0]
    pred_label_idx = int(np.argmax(pred_probs))
    pred_label     = LABEL_MAP[pred_label_idx]

    # SHAP values for the predicted label column
    pred_col_shap = shap_matrix[:, pred_label_idx]

    # Build ranked token list
    token_records = []
    for tok, shap_val in zip(tokens, pred_col_shap):
        clean = tok.strip()
        if not clean or clean in ("", "▁"):  # skip padding / BOS tokens
            continue
        note = CLINICAL_TOKEN_NOTES.get(clean.lower().strip(".,!?'\""), "")
        token_records.append({
            "token":     clean,
            "shap":      float(shap_val),
            "abs_shap":  abs(float(shap_val)),
            "direction": "↑ increases depression risk" if shap_val > 0 else "↓ reduces depression risk",
            "note":      note,
        })

    token_records.sort(key=lambda x: x["abs_shap"], reverse=True)

    top_tokens        = token_records[:top_n]
    risk_tokens       = [t for t in token_records if t["shap"] > 0][:5]
    protective_tokens = [t for t in token_records if t["shap"] < 0][:3]

    return SHAPResult(
        text              = text,
        tokens            = tokens,
        shap_matrix       = shap_matrix,
        pred_label_idx    = pred_label_idx,
        pred_label        = pred_label,
        pred_probs        = pred_probs,
        top_tokens        = top_tokens,
        risk_tokens       = risk_tokens,
        protective_tokens = protective_tokens,
    )


def format_debug_summary(result: SHAPResult) -> str:
    """Compact debug string for logging."""
    probs_str = ", ".join(
        f"{LABEL_MAP[i]}={result.pred_probs[i]:.3f}" for i in range(3)
    )
    lines = [
        f"Prediction : {result.pred_label} ({probs_str})",
        "Top tokens :",
    ]
    for t in result.top_tokens[:6]:
        lines.append(
            f"  {t['token']:15s}  SHAP={t['shap']:+.4f}  {t['direction']}"
        )
    return "\n".join(lines)


# ── Quick self-test ─────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "I feel empty and tired every day. Nothing makes sense anymore.",
        "I'm doing great today, feeling hopeful and energetic!",
        "Sometimes I feel a bit sad but overall I'm managing okay.",
    ]
    for txt in test_texts:
        result = explain_with_shap(txt)
        print("\n" + "─" * 60)
        print(f"Text  : {txt}")
        print(format_debug_summary(result))
