# Explainable Depression Prediction — Multi-Use-Case Bot

## Folder Structure

```
explainableDepressionPrediction/
│
├── bot.py                          ← SINGLE shared Telegram bot (all use cases)
├── test_pipeline.py                ← CLI test runner (no Telegram needed)
├── requirements.txt
│
├── shared/                         ← Reusable code shared by ALL use cases
│   ├── depression_model.py         ← deproberta model + SHAP + reframe_text()
│   ├── llm_client.py               ← Gemini client with fallback chain
│   ├── conversation.py             ← FSM: use-case menu + text collection
│   └── phq8_knowledge_base.csv     ← PHQ-8 symptom knowledge base (RAG source)
│
├── shap_explainer/                 ← Use Case 1
│   └── shap_explainer.py           ← prompt builder + Gemini call
│
├── rag_explainer/                  ← Use Case 2
│   ├── rag_retriever.py            ← FAISS index over PHQ-8 CSV
│   └── rag_explainer.py            ← RAG pipeline + prompt builder
│
├── hybrid_shap_rag/                ← Use Case 3 (coming soon)
├── counterfactual_explainer/       ← Use Case 4 
└── mcp_modular_agent/              ← Use Case 5 (coming soon)
```

## What is shared vs what is use-case specific

| Component | Shared? | Why |
|-----------|---------|-----|
| deproberta model loading | ✅ shared | One model, loaded once |
| `predict_proba()` | ✅ shared | Same for all UCs |
| `reframe_text()` | ✅ shared | Fixes indirect phrasing for all UCs |
| SHAP explainer | ✅ shared | UC1, UC3 use it |
| Gemini client + fallback | ✅ shared | All UCs call Gemini |
| Telegram FSM + menu | ✅ shared | One bot, user picks UC |
| `strip_markdown()` | ✅ shared | Telegram parse safety for all |
| FAISS RAG index | ❌ UC2 only | Only RAG needs it |
| SHAP prompt builder | ❌ UC1 only | Different prompt structure |
| RAG prompt builder | ❌ UC2 only | Different prompt structure |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test without Telegram
export GOOGLE_API_KEY="your_key"

python test_pipeline.py --uc 1 --scenario indirect   # SHAP
python test_pipeline.py --uc 2 --scenario indirect   # RAG
python test_pipeline.py --uc 1 --text "I feel empty and hopeless every day."
python test_pipeline.py --uc 2 --text "I can't sleep and have no appetite."

# 3. Run the bot
export TELEGRAM_BOT_TOKEN="your_token"
export GOOGLE_API_KEY="your_key"
python bot.py
```

## Bot conversation flow

```
User: /assess

Bot:  Please choose an explanation method:
      🔬 1. SHAP Explanation
         Uses token-level SHAP to show which words drove the prediction.
      📚 2. RAG Explanation
         Retrieves matching clinical symptom knowledge.
      🔀 3. SHAP + RAG Combined  (coming soon)
      🔄 4. Counterfactual        (coming soon)
      🤖 5. MCP Agent             (coming soon)

User: 2

Bot:  RAG selected. Please describe how you've been feeling...

User: I don't go out much, lost interest in hobbies, can't concentrate.

Bot:  [Preview: moderate, 71.2%, matched: Anhedonia, Concentration Problems]
      [Full Gemini explanation grounded in PHQ-8 KB]
      [Retrieved symptom breakdown with clinical definitions]
```

## How to add Use Case 3 (SHAP + RAG)

1. Create `hybrid_shap_rag/hybrid_explainer.py`
2. Import from both `shared.depression_model` (for SHAP) and `rag_explainer.rag_retriever` (for RAG)
3. Add `"3"` to `USE_CASES` in `shared/conversation.py` with `"status": "available"`
4. Add `elif use_case == "3": await run_hybrid_pipeline(...)` in `bot.py`

No other files need changing.
