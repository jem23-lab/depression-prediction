# Use Case 1 — SHAP-Only Depression Explanation via Telegram Bot

## Architecture

```
User free-text message (Telegram)
          │
          ▼
┌──────────────────┐
│  conversation.py │  FSM: IDLE → WAITING → DONE
│                  │  Collects raw text (no form — model takes natural language)
└────────┬─────────┘
         │ user_text (raw string)
         ▼
┌──────────────────────────────────────────┐
│  depression_model.py                     │
│  rafalposwiata/deproberta-large-depression│
│  Labels: severe | moderate | not depression│
│  + shap.Explainer (token-level SHAP)     │
└────────┬─────────────────────────────────┘
         │ SHAPResult: pred_label, probs, risk_tokens, protective_tokens
         ▼
┌──────────────────────────────────────────┐
│  shap_explainer.py                       │
│  build_shap_prompt() → Gemini Flash      │
│  → empathetic plain-English explanation  │
└────────┬─────────────────────────────────┘
         │ explanation string
         ▼
  Telegram reply (prediction + token breakdown + explanation)
```

## Files

| File | Maps to Notebook | Purpose |
|------|-----------------|---------|
| `depression_model.py` | Cells [22–25] | Load deproberta, predict_proba, SHAP |
| `shap_explainer.py` | Cell [29] | Build prompt → Gemini |
| `conversation.py` | — | Telegram FSM |
| `bot.py` | — | Telegram bot entry point |
| `test_pipeline.py` | Cells [24–29] | CLI test without Telegram |

## Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Test locally (no Telegram needed)
export GOOGLE_API_KEY="your_key"
python test_pipeline.py --scenario notebook
python test_pipeline.py --scenario severe --show-prompt
python test_pipeline.py --text "I've been feeling really tired and empty."

# 3. Run the Telegram bot
export TELEGRAM_BOT_TOKEN="your_token"
export GOOGLE_API_KEY="your_key"
python bot.py
```

## Bot flow

```
User: /start
Bot:  Welcome message + instructions

User: /assess
Bot:  "Tell me how you're feeling…"

User: "I feel empty and tired every day. Nothing makes sense anymore."
Bot:  "Analysing…"
      → [Quick preview: severe, 72.7%, key word: 'empty']
      → [Full Gemini explanation 250–400 words]
      → [SHAP token breakdown: 🔴 'empty', 🔴 'tired', 🔴 'anymore' ...]
```

## Model

- **Model**: `rafalposwiata/deproberta-large-depression`
- **Labels**: `severe` (0), `moderate` (1), `not depression` (2)
- **SHAP**: `shap.Explainer` with tokenizer — token-level contributions
- **LLM**: Gemini 2.0 Flash

## Extending to UC2–UC5

| Use Case | Change |
|----------|--------|
| UC2 (RAG) | Replace `shap_explainer.py` with `rag_explainer.py` |
| UC3 (SHAP+RAG) | Combine both; use risk_tokens as retrieval query |
| UC4 (Counterfactuals) | Add `counterfactual.py` (mirrors notebook cell [26]) |
| UC5 (MCP agents) | Wrap each explainer as an MCP tool; route by user intent |
