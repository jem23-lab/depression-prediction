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

        # label map
        if hasattr(_model.config, 'id2label') and _model.config.id2label:
            label_map = _model.config.id2label
        else:
            # keep a minimal default if model lacks mapping
            label_map = {0: "severe", 1: "moderate", 2: "not depression"}
        # store on model config for later use
        _model.config.id2label = label_map

        # Initialize SHAP Explainer with a tokenizer-aware masker. We use a simple function wrapper
        # so SHAP sees a callable that returns model outputs given raw strings.
        def f_texts(texts: List[str]):
            try:
                # Coerce various input shapes that SHAP may pass (np.ndarray, list of tokens, etc.)
                import numpy as _np

                # If a single string is provided, wrap in list
                if isinstance(texts, (str, bytes)):
                    texts = [texts]
                # If numpy array, convert to list
                elif isinstance(texts, _np.ndarray):
                    texts = texts.tolist()

                # Now ensure each element is a plain string. SHAP may pass lists of tokens;
                # join them with spaces to produce a textual input acceptable to the tokenizer.
                def _ensure_str(x):
                    if isinstance(x, (list, tuple)):
                        return " ".join(map(str, x))
                    if isinstance(x, bytes):
                        try:
                            return x.decode('utf-8')
                        except Exception:
                            return str(x)
                    return str(x)

                texts = [_ensure_str(t) for t in texts]

                enc = _tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                # Move tensors to device if they exist
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                with torch.no_grad():
                    outputs = _model(**enc)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                return probs
            except Exception:
                logger.exception("Model wrapper failed inside SHAP f_texts")
                raise

        # Use shap.Explainer with the callable model and masker='text'
        try:
            masker = shap.maskers.Text(_tokenizer)
            _explainer = shap.Explainer(f_texts, masker)
            logger.info("SHAP explainer initialized with text masker")
        except Exception:
            # As a more robust fallback, create a partition explainer without a special masker
            try:
                _explainer = shap.Explainer(f_texts)
                logger.info("SHAP explainer initialized (fallback) without text masker")
            except Exception:
                logger.exception("Failed to initialize SHAP explainer; explanations may fail")
                _explainer = None

        _initialized = True
        logger.info("Model initialization complete")
        return True
    except Exception:
        logger.exception("Model initialization failed")
        _initialized = False
        return False


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

        # 2. Generate SHAP values for the text. We request explanations for a single sample.
        shap_values = _explainer([text])

        # shap_values.data may be a list of tokens or strings depending on masker
        # Normalize tokens and values robustly
        try:
            tokens = list(shap_values.data[0])
        except Exception:
            # fallback: try converting to numpy/string
            try:
                tokens = shap_values.data
            except Exception:
                tokens = [str(text)]

        try:
            # shap_values.values shape: (n_examples, n_tokens, n_classes) or (n_examples, n_tokens)
            vals = np.array(shap_values.values)
            if vals.ndim == 3:
                token_impacts = vals[0, :, pred_idx].tolist()
            elif vals.ndim == 2:
                token_impacts = vals[0, :].tolist()
            else:
                token_impacts = vals.flatten().tolist()
        except Exception:
            logger.exception("Failed to parse shap values structure; returning empty impacts")
            token_impacts = [0.0] * len(tokens)

        base_val = None
        try:
            bv = shap_values.base_values
            if isinstance(bv, (list, np.ndarray)):
                bv_arr = np.array(bv)
                if bv_arr.ndim == 2:
                    base_val = float(bv_arr[0, pred_idx])
                elif bv_arr.ndim == 1:
                    base_val = float(bv_arr[0])
                else:
                    base_val = float(bv_arr.flatten()[0])
            else:
                base_val = float(bv)
        except Exception:
            base_val = None

        return {
            "ok": True,
            "predicted_label": label_map.get(pred_idx, str(pred_idx)),
            "tokens": tokens,
            "shap_values": token_impacts,
            "base_value": base_val,
            "model_probabilities": {label_map.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
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

        if len(vals) != len(tokens):
            # align lengths conservatively
            n = min(len(vals), len(tokens))
            vals = vals[:n]
            tokens = tokens[:n]

        # Sort by SHAP value (positive contribution)
        top_indices = np.argsort(vals)[::-1][:k]

        top_words = [
            {"word": tokens[i], "impact": float(vals[i])}
            for i in top_indices if float(vals[i]) > 0
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