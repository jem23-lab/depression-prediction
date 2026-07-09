# Explainable Depression Prediction — Multi-Use-Case Bot

## Folder Structure

```
explainableDepressionPrediction_mcp_server/
│
├── bot.py                          ← Single Telegram bot (all use cases)
├── test_pipeline.py                ← CLI test runner (no Telegram needed)
├── requirements.txt
│
├── architecture/
│   ├── shap_explainer/             ← Use Case 1 (SHAP factors)
│   ├── rag_explainer/              ← Use Case 2 (RAG factors)
│   ├── hybrid_shap_rag_counterfactual/ ← Use Case 3 (Hybrid)
│   ├── shap_counterfactual_explainer/  ← Use Case 4 (Counterfactual)
│   └── mcp_modular_agent/          ← Use Case 5 (MCP)
│
├── shared/                         ← Reusable code shared by ALL use cases
│   ├── depression_model.py         ← Model + SHAP + helpers
│   ├── llm_client.py               ← Gemini client with fallback chain
│   ├── conversation.py             ← Evaluation FSM
│   ├── eval_logger.py              ← CSV logging
│   └── phq8_knowledge_base.csv     ← PHQ-8 symptom knowledge base (RAG source)
│
└── logs/
    └── evaluation_records.csv
```

## What is shared vs what is use-case specific

| Component | Shared? | Why |
|-----------|---------|-----|
| Model loading | ✅ shared | One model, loaded once |
| `predict_proba()` | ✅ shared | Same for all UCs |
| SHAP token extraction | ✅ shared | UC1, UC3, UC4 use it |
| Gemini client + fallback | ✅ shared | All UCs call Gemini |
| Telegram evaluation FSM | ✅ shared | One flow for all UCs |
| `strip_markdown()` | ✅ shared | Telegram parse safety for all |
| FAISS RAG index | ❌ UC2 only | Only RAG needs it |
| Prompt builders | ❌ per-UC | Different explanation formats |

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

## Simple model demo

The lightweight TF-IDF model trains from `shared/training_examples.py` at runtime and
caches a persisted copy in `shared/model_cache/` for faster startup on subsequent runs.

```bash
python scripts/run_simple_model_demo.py
```

## Bot flow (current)

The bot is evaluation-first (no menu). When the user sends `/begin`:

1. Bot selects 10 participant paragraphs (DAIC-WOZ style).
2. For each sample, it shows:
   - 1) Text Sample
   - 2) Prediction
   - 3) Participant question prompt
   - 4) Two anonymous responses: Planner (MCP) and MentalT5
   - 5) Pairwise comparison prompts
3. Pairwise ratings are stored in `logs/interactive_evaluation_records.csv`.

The MentalT5 response uses `Tianlin668/MentalT5` through Hugging Face Transformers.
No additional system prompt or explanation template is added; the model receives only
the participant text and question in the upstream format:
`Consider this post: <text> Question: <question>`. You can override the model id
with `MENTALT5_MODEL_ID` and generation length with `MENTALT5_MAX_NEW_TOKENS`.

## Explanation style (important)

Explanations are factor-only and user-friendly:
- Focus only on the words/phrases that influenced the result.
- Do not include advice, self-care tips, or support recommendations.
- Avoid tool jargon (no SHAP/RAG/counterfactual terms) and no scores.
- Keep output short and readable.

## Evaluation flow

1. User sends `/begin`.
2. Bot shows each text sample, prediction, and two explanations.
3. Bot sends a single editable Evaluation box for ratings (buttons 1–5).
4. The Evaluation box updates after each score entry.
5. Results are appended to `logs/evaluation_records.csv`.

Saved CSV columns:
- `timestamp_utc`
- `user_id`
- `session_id`
- `paragraph_id`
- `paragraph_text`
- `selected_use_case`
- `selected_use_case_name`
- `prediction_label`
- `prediction_confidence`
- `explanation_text`
- `rating_clarity`
- `rating_correctness`
- `rating_helpfulness`
- `rating_overall_avg`

## Cache explanations

Use this script to precompute and cache missing explanations in `shared/training_examples.py`. It skips any explanation that is already present.

```zsh
python scripts/cache_explanations.py --dry-run
python scripts/cache_explanations.py --methods SHAP,RAG --limit 5
python scripts/cache_explanations.py --only-ids daic_woz_severe_321,daic_woz_no_depression_312
```

Optional flags:
- `--save-pred` to store `prediction_label` and `prediction_confidence` if missing.
- `--dry-run` to print what would be generated without calling any models.

## Run

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN="your_token"
export GOOGLE_API_KEY="your_key"
python bot.py
```

In Telegram, use `/begin` to start each study session.
