import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from fastmcp import FastMCP
from typing import List, Dict, Any
import logging
from pathlib import Path

# Initialize MCP Server (use a consistent variable name `server`)
server = FastMCP("KnowledgeServer")

# Logging for debugging
logger = logging.getLogger("KnowledgeServer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

# ====== Knowledge Base Initialization ======

BASE = Path(__file__).parent
DOMAIN_DIR = BASE.parent / "domain_knowledge"
DATA_PATH = DOMAIN_DIR / "processed_data.csv"

# Internal state (initialized lazily or at import-time if possible)
_df: pd.DataFrame | None = None
_model: SentenceTransformer | None = None
_knowledge_embeddings = None
_initialized = False


def init_knowledge() -> bool:
    """Initialize/load the knowledge base, embedding model, and embeddings.
    This is idempotent and safe to call from tools; failures are logged and
    cause tools to return a structured error instead of crashing the server.
    """
    global _df, _model, _knowledge_embeddings, _initialized
    if _initialized:
        return True

    try:
        if not DATA_PATH.exists():
            logger.error("Knowledge data file not found: %s", DATA_PATH)
            return False

        _df = pd.read_csv(DATA_PATH)
        _df['context'] = "Question: " + _df['question'].astype(str) + "\nAnswer: " + _df['answer'].astype(str)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        # Pre-compute embeddings; keep them as tensors for fast similarity
        _knowledge_embeddings = _model.encode(_df['context'].tolist(), convert_to_tensor=True)

        _initialized = True
        logger.info("Knowledge base initialized: %d rows", len(_df))
        return True
    except Exception:
        logger.exception("Failed to initialize knowledge base")
        _initialized = False
        return False


# Attempt to initialize at import time but don't crash if it fails
try:
    _ = init_knowledge()
except Exception:
    logger.exception("Unexpected error during implicit knowledge init")


# ====== Tools ======

@server.tool()
def query_knowledge(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Searches the clinical knowledge base for information related to the query.
    Returns the most relevant question-answer pairs from the ICD/clinical documentation.
    """
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "empty or invalid query"}

    if not isinstance(top_k, int) or top_k <= 0:
        top_k = 3

    if not _initialized and not init_knowledge():
        return {"ok": False, "error": "knowledge base not available"}

    try:
        # Encode the query
        query_embedding = _model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity between query and knowledge base
        cos_scores = util.cos_sim(query_embedding, _knowledge_embeddings)[0]

        # Get top K indices (use numpy for stable slicing)
        k = min(top_k, len(_df))
        # torch.topk returns values and indices
        top_results = torch.topk(cos_scores, k=k)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            idx = int(idx.item())
            results.append({
                "score": float(score.item()),
                "question": _df.iloc[idx]['question'],
                "answer": _df.iloc[idx]['answer'],
                "full_context": _df.iloc[idx]['context']
            })

        return {"ok": True, "query": query, "results": results}
    except Exception:
        logger.exception("query_knowledge failed for query=%s", query)
        return {"ok": False, "error": "internal error during query"}


@server.tool()
def get_all_topics() -> Dict[str, Any]:
    """
    Returns a list of unique questions/topics covered in the knowledge base.
    Useful for understanding the scope of the available information.
    """
    if not _initialized and not init_knowledge():
        return {"ok": False, "error": "knowledge base not available"}

    try:
        topics = _df['question'].dropna().unique().tolist()
        return {"ok": True, "topics": topics}
    except Exception:
        logger.exception("get_all_topics failed")
        return {"ok": False, "error": "internal error retrieving topics"}


if __name__ == "__main__":
    server.run()
