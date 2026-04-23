"""
shared/depression_model.py
────────────────────────────────────────────────────────────────────
Shared model wrapper used by ALL use cases (SHAP, RAG, hybrid, etc.)

Model : rafalposwiata/deproberta-large-depression
Labels: 0=severe, 1=moderate, 2=not depression

Exposes:
    predict_proba(texts)     → np.ndarray (n, 3)
    reframe_text(text)       → str (expands indirect phrasing)
    explain_with_shap(text)  → SHAPResult
"""

import re
import numpy as np
import torch
import shap
from dataclasses import dataclass, field
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "rafalposwiata/deproberta-large-depression"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {0: "severe", 1: "moderate", 2: "not depression"}

LABEL_DESCRIPTIONS = {
    "severe":         "Strong indicators of severe depression — persistent hopelessness, emptiness, or inability to function.",
    "moderate":       "Signs of moderate depression — low mood, fatigue, and reduced interest in life.",
    "not depression": "No significant depression signals detected at this time.",
}

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
    "concentrate": "concentration difficulty is a cognitive depression symptom",
    "appetite":   "appetite change is a somatic depression symptom",
    "worthless":  "low self-worth is a key affective symptom",
}

# ── Clinical reframing (indirect → explicit clinical language) ──────
INDIRECT_TO_CLINICAL = [
    (r"don'?t (?:really )?go out",              "I have lost interest in going out and socializing"),
    (r"don'?t (?:really )?leave (?:the )?house","I feel too low to leave the house"),
    (r"lost interest in (?:my )?hobbie?s?",     "I no longer enjoy my hobbies, I feel anhedonia"),
    (r"don'?t enjoy (?:things|anything)",       "I feel no pleasure in things I used to enjoy"),
    (r"can'?t be bothered",                     "I feel too fatigued and unmotivated to do anything"),
    (r"don'?t (?:really )?see (?:my )?friends", "I have withdrawn from social contact"),
    (r"hard(?:er)? to concentrate",             "I am struggling to concentrate, cognitive symptom of depression"),
    (r"can'?t (?:focus|concentrate)",           "I cannot concentrate, a symptom of depression"),
    (r"find it hard to (?:focus|think)",        "I experience difficulty thinking clearly"),
    (r"brain (?:fog|feels? slow)",              "I have brain fog and cognitive slowing"),
    (r"can'?t sleep",                           "I have insomnia and sleep disturbance"),
    (r"sleep(?:ing)? too much",                 "I am sleeping excessively, hypersomnia"),
    (r"tired all the time",                     "I feel persistent fatigue every day"),
    (r"always tired",                           "I feel chronic fatigue and low energy"),
    (r"no energy",                              "I have no energy, persistent fatigue"),
    (r"don'?t feel (?:like )?myself",           "I feel like I have lost my sense of self"),
    (r"feel(?:ing)? (?:a bit |pretty )?(?:low|down|flat)", "I feel persistent low mood"),
    (r"feel(?:ing)? (?:really )?sad",           "I feel deep sadness"),
    (r"feel(?:ing)? numb",                      "I feel emotionally numb and empty"),
    (r"don'?t (?:really )?care anymore",        "I feel apathy and loss of motivation"),
    (r"not (?:really )?eating",                 "I have lost my appetite"),
    (r"(?:barely|hardly) eat(?:ing)?",          "I have significant appetite loss"),
    (r"feel(?:ing)? (?:like a )?failure",       "I feel worthless and like a failure"),
    (r"what'?s the point",                      "I feel hopeless and see no purpose"),
    (r"don'?t see the point",                   "I feel hopeless and without purpose"),
    (r"nothing matters",                        "I feel hopeless, nothing matters to me"),
]


def reframe_text(text: str) -> str:
    """Expands indirect phrasing into explicit clinical language."""
    parts = []
    for pattern, phrase in INDIRECT_TO_CLINICAL:
        if re.search(pattern, text.lower()):
            parts.append(phrase)
    return ". ".join(parts) + ". " + text if parts else text


# ── Singletons ───────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_explainer = None


def load():
    global _tokenizer, _model, _explainer
    if _model is not None:
        return
    print(f"Loading {MODEL_NAME} on {DEVICE} …")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(DEVICE)
    _model.eval()
    print(f"Model loaded. Labels: {_model.config.id2label}")
    print("Initialising SHAP explainer …")
    _explainer = shap.Explainer(
        lambda texts: predict_proba(list(texts)),
        _tokenizer,
        output_names=list(LABEL_MAP.values()),
    )
    print("SHAP ready.")


def predict_proba(texts: List[str]) -> np.ndarray:
    """Returns softmax probs shape (n, 3): [severe, moderate, not_depression]"""
    load()
    with torch.no_grad():
        enc    = _tokenizer(texts, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(DEVICE)
        logits = _model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


@dataclass
class SHAPResult:
    text:              str
    model_input:       str
    tokens:            List[str]
    shap_matrix:       np.ndarray
    pred_label_idx:    int
    pred_label:        str
    pred_probs:        np.ndarray
    was_reframed:      bool          = False
    top_tokens:        List[dict]    = field(default_factory=list)
    risk_tokens:       List[dict]    = field(default_factory=list)
    protective_tokens: List[dict]    = field(default_factory=list)


def explain_with_shap(text: str, top_n: int = 8) -> SHAPResult:
    """Run SHAP token-level explanation. Reframes indirect text first."""
    load()
    model_input  = reframe_text(text)
    was_reframed = model_input != text

    shap_vals   = _explainer([model_input])
    explanation  = shap_vals[0]
    tokens       = list(explanation.data)
    shap_matrix  = explanation.values

    pred_probs     = predict_proba([model_input])[0]
    pred_label_idx = int(np.argmax(pred_probs))
    pred_label     = LABEL_MAP[pred_label_idx]
    pred_col_shap  = shap_matrix[:, pred_label_idx]

    records = []
    for tok, sv in zip(tokens, pred_col_shap):
        clean = tok.strip()
        if not clean or clean in ("", "▁"):
            continue
        note = CLINICAL_TOKEN_NOTES.get(clean.lower().strip(".,!?'\""), "")
        records.append({
            "token":     clean,
            "shap":      float(sv),
            "abs_shap":  abs(float(sv)),
            "direction": "↑ increases risk" if sv > 0 else "↓ reduces risk",
            "note":      note,
        })
    records.sort(key=lambda x: x["abs_shap"], reverse=True)

    return SHAPResult(
        text=text, model_input=model_input, was_reframed=was_reframed,
        tokens=tokens, shap_matrix=shap_matrix,
        pred_label_idx=pred_label_idx, pred_label=pred_label, pred_probs=pred_probs,
        top_tokens=records[:top_n],
        risk_tokens=[r for r in records if r["shap"] > 0][:5],
        protective_tokens=[r for r in records if r["shap"] < 0][:3],
    )


def format_debug(result: SHAPResult) -> str:
    probs = ", ".join(f"{LABEL_MAP[i]}={result.pred_probs[i]:.3f}" for i in range(3))
    lines = [f"Prediction: {result.pred_label} ({probs})", "Top tokens:"]
    for t in result.top_tokens[:5]:
        lines.append(f"  '{t['token']}' SHAP={t['shap']:+.4f}  {t['direction']}")
    return "\n".join(lines)
