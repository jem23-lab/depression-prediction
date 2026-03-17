import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastmcp import FastMCP
from typing import List, Dict, Any
import logging

# TextAttack imports for Counterfactuals
try:
    import textattack
    from textattack.attack_recipes import TextFoolerJin2019
    from textattack.models.wrappers import HuggingFaceModelWrapper
except ImportError:
    textattack = None

# ====== Model & Tokenizer Configuration ======

MODEL_NAME = "rafalposwiata/deproberta-large-depression"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize MCP Server
server = FastMCP("ExplainableDepressionServer")

# Setup logger
logger = logging.getLogger("ExplainableDepressionServer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

# Internal lazy-loaded state
_tokenizer = None
_model = None
_explainer = None
_initialized = False


def init_model() -> bool:
    """Lazy initialization of tokenizer, model and SHAP explainer.
    Returns True on success, False on failure (server stays alive).
    """
    global _tokenizer, _model, _explainer, _initialized
    if _initialized:
        return True

    try:
        logger.info("Initializing model and tokenizer: %s (device=%s)", MODEL_NAME, DEVICE)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.to(DEVICE)
        _model.eval()

        # Initialize SHAP Explainer
        try:
            _explainer = shap.Explainer(_model, _tokenizer)
        except Exception:
            # SHAP explainer can be fragile; log but don't crash
            logger.exception("Failed to initialize SHAP explainer; explanations may fail")
            _explainer = None

        # label map
        if hasattr(_model.config, 'id2label'):
            label_map = _model.config.id2label
        else:
            # keep a minimal default if model lacks mapping
            label_map = {0: "severe", 1: "moderate", 2: "not depression"}
        # store on model config for later use
        _model.config.id2label = label_map

        _initialized = True
        logger.info("Model initialization complete")
        return True
    except Exception:
        logger.exception("Model initialization failed")
        _initialized = False
        return False


# Try to initialize but do not crash on errors
# NOTE: removed implicit initialization to avoid heavy downloads at import time; tools will call init_model() lazily.


# ====== Helper Functions ======

def predict_proba(texts: List[str]) -> np.ndarray:
    """Return softmax probabilities for the batch of texts.
    Assumes model/tokenizer are initialized prior to call.
    """
    with torch.no_grad():
        enc = _tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        outputs = _model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


# ====== MCP Tools ======

@server.tool()
def health() -> Dict[str, Any]:
    """Simple health check for orchestrator readiness checks."""
    return {"ok": True, "ready": _initialized, "model": MODEL_NAME, "device": DEVICE}


@server.tool()
def predict_depression(text: str) -> Dict[str, Any]:
    """
    Predicts the level of depression for a given text input.
    Returns the predicted class and the probabilities for all classes.
    """
    if not _initialized and not init_model():
        return {"ok": False, "error": "model not available"}

    try:
        probs = predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        label_map = _model.config.id2label

        out = {
            "ok": True,
            "prediction": label_map.get(pred_idx, str(pred_idx)),
            "confidence": float(probs[pred_idx]),
            "all_probabilities": {label_map.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
        }
        return out
    except Exception:
        logger.exception("predict_depression failed for input (truncated): %s", str(text)[:200])
        return {"ok": False, "error": "internal error during prediction"}


@server.tool()
def shap_text_explain(text: str) -> Dict[str, Any]:
    """
    Explains the prediction for a specific text input using SHAP.
    """
    if not _initialized and not init_model():
        return {"ok": False, "error": "model not available"}

    if _explainer is None:
        return {"ok": False, "error": "SHAP explainer not initialized"}

    try:
        # 1. Get prediction probabilities and predicted class
        probs = predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        label_map = _model.config.id2label

        # 2. Generate SHAP values for the text
        shap_values = _explainer([text])

        # 3. Extract tokens and their specific impact values for the predicted class
        token_impacts = shap_values.values[0, :, pred_idx].tolist()
        tokens = shap_values.data[0].tolist()

        return {
            "ok": True,
            "predicted_label": label_map.get(pred_idx, str(pred_idx)),
            "tokens": tokens,
            "shap_values": token_impacts,
            "base_value": float(shap_values.base_values[0, pred_idx])
        }
    except Exception:
        logger.exception("shap_text_explain failed for input (truncated): %s", str(text)[:200])
        return {"ok": False, "error": "internal error generating SHAP explanation"}


@server.tool()
def get_top_contributing_words(text: str, k: int = 5) -> Dict[str, Any]:
    """
    Returns the top K words that contributed most positively to the prediction.
    """
    try:
        exp = shap_text_explain(text)
        if not exp.get("ok"):
            return exp

        vals = np.array(exp["shap_values"])
        tokens = exp["tokens"]

        # Sort by SHAP value (positive contribution)
        top_indices = np.argsort(vals)[::-1][:k]

        top_words = [
            {"word": tokens[i], "impact": float(vals[i])}
            for i in top_indices if vals[i] > 0
        ]

        return {"ok": True, "prediction": exp["predicted_label"], "top_impact_words": top_words}
    except Exception:
        logger.exception("get_top_contributing_words failed for input (truncated): %s", str(text)[:200])
        return {"ok": False, "error": "internal error extracting top words"}


@server.tool()
def generate_text_counterfactual(text: str) -> Dict[str, Any]:
    """
    Generates a counterfactual version of the input text. If TextAttack isn't
    installed, returns a structured error. The implementation is best-effort.
    """
    if not _initialized and not init_model():
        return {"ok": False, "error": "model not available"}

    if textattack is None:
        return {"ok": False, "error": "TextAttack library not installed. Counterfactuals unavailable."}

    try:
        # 1. Wrap the model for TextAttack
        model_wrapper = HuggingFaceModelWrapper(_model, _tokenizer)

        # 2. Use TextFooler recipe to find a 'successful' perturbation (counterfactual)
        recipe = TextFoolerJin2019.build(model_wrapper)

        # 3. Create the attack input and run (best-effort)
        attack_input = textattack.shared.AttackedText(text)

        # Note: TextAttack API usage varies; wrap in try/except and present best-effort result
        try:
            # attempt to run the attack (may be slow)
            result = recipe.attack(attack_input, model_wrapper.get_outputs([text])[0].argmax())
        except Exception:
            logger.exception("TextAttack run failed or timed out")
            return {"ok": False, "error": "TextAttack failed to produce a counterfactual"}

        # Best-effort extraction from result; fields vary by version
        try:
            original_text = getattr(result, 'original_text', text)
            cf_text = getattr(result, 'perturbed_text', None)
            if cf_text is None:
                # try alternative attribute names
                cf_text = getattr(result, 'perturbed_result', None)
            return {
                "ok": True,
                "original_text": original_text,
                "counterfactual_text": str(cf_text),
                "raw_result": str(result)
            }
        except Exception:
            logger.exception("Failed to extract counterfactual details")
            return {"ok": False, "error": "counterfactual produced but parsing failed"}

    except Exception:
        logger.exception("generate_text_counterfactual failed")
        return {"ok": False, "error": "internal error generating counterfactual"}


if __name__ == "__main__":
    server.run()