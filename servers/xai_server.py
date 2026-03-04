import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastmcp import FastMCP
from typing import List, Dict, Any

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

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# Label Map: {0: 'severe', 1: 'moderate', 2: 'not depression'}
label_map = model.config.id2label if hasattr(model.config, 'id2label') else {0: "severe", 1: "moderate",
                                                                             2: "not depression"}

# Initialize SHAP Explainer for Text
# We use a masker to handle text data
explainer = shap.Explainer(model, tokenizer)


# ====== Helper Functions ======

def predict_proba(texts: List[str]) -> np.ndarray:
    """Return softmax probabilities for the batch of texts."""
    with torch.no_grad():
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


# ====== MCP Tools ======

@server.tool()
def predict_depression(text: str) -> Dict[str, Any]:
    """
    Predicts the level of depression for a given text input.
    Returns the predicted class and the probabilities for all classes.
    """
    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))

    return {
        "prediction": label_map[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {label_map[i]: float(probs[i]) for i in range(len(probs))}
    }

@server.tool()
def shap_text_explain(text: str) -> dict:
    """
    Explains the prediction for a specific text input using SHAP.
    """
    # 1. Get prediction probabilities
    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))

    # 2. Generate SHAP values for the text
    # This uses the SHAP Explainer we initialized with the transformer model
    shap_values = explainer([text])

    # 3. Extract tokens and their specific impact values for the predicted class
    # shape: (1, num_tokens, num_classes)
    token_impacts = shap_values.values[0, :, pred_idx].tolist()
    tokens = shap_values.data[0].tolist()

    return {
        "predicted_label": label_map[pred_idx],
        "tokens": tokens,
        "shap_values": token_impacts,
        "base_value": float(shap_values.base_values[0, pred_idx])
    }


@server.tool()
def get_top_contributing_words(text: str, k: int = 5) -> Dict[str, Any]:
    """
    Returns the top K words that contributed most positively to the prediction.
    """
    explanation = shap_text_explain(text)
    vals = np.array(explanation["shap_values"])
    tokens = explanation["tokens"]

    # Sort by SHAP value (positive contribution)
    top_indices = np.argsort(vals)[::-1][:k]

    top_words = [
        {"word": tokens[i], "impact": float(vals[i])}
        for i in top_indices if vals[i] > 0
    ]

    return {
        "prediction": explanation["predicted_class"],
        "top_impact_words": top_words
    }


@server.tool()
def generate_text_counterfactual(text: str) -> Dict[str, Any]:
    """
    Generates a counterfactual version of the input text.
    It finds the minimal word changes required to change the model's
    depression prediction (e.g., from 'severe' to 'not depression').
    """
    if textattack is None:
        return {"error": "TextAttack library not installed. Counterfactuals unavailable."}

    # 1. Wrap the model for TextAttack
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # 2. Use TextFooler recipe to find a 'successful' perturbation (counterfactual)
    recipe = TextFoolerJin2019.build(model_wrapper)

    # 3. Create the attack input
    # We want to change the prediction, so we treat it as an adversarial attack
    attack_input = textattack.shared.AttackedText(text)

    # 4. Perform the search
    # This finds the nearest version of the text that flips the classification
    result = recipe.attack(attack_input, model_wrapper.get_outputs([text])[0].argmax())

    if isinstance(result, textattack.goal_function_results.GoalFunctionResultStatus.Succeeded):
        return {
            "original_text": text,
            "original_class": label_map[result.original_result.ground_truth_output],
            "counterfactual_text": result.perturbed_result.attacked_text.text,
            "new_class": label_map[result.perturbed_result.output],
            "changes_made": result.perturbed_result.attacked_text.diff_with_old(attack_input)
        }
    else:
        return {"error": "Could not find a minimal counterfactual for this input."}

if __name__ == "__main__":
    server.run()