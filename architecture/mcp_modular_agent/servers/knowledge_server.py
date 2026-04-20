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
# Prefer phq8_enhanced.csv as requested; fallback to processed_data.csv
PREFERRED_PATH = DOMAIN_DIR / "phq8_enhanced.csv"
FALLBACK_PATH = DOMAIN_DIR / "processed_data.csv"

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
        data_path = PREFERRED_PATH if PREFERRED_PATH.exists() else (FALLBACK_PATH if FALLBACK_PATH.exists() else None)
        if data_path is None:
            logger.error("Knowledge data file not found. Expected %s or %s", PREFERRED_PATH, FALLBACK_PATH)
            return False

        _df = pd.read_csv(data_path)

        # Determine which columns to use for the QA context. Support both CSV formats.
        if 'question_text' in _df.columns and 'clinical_definition' in _df.columns:
            q_col = 'question_text'
            a_col = 'clinical_definition'
        elif 'question' in _df.columns and 'answer' in _df.columns:
            q_col = 'question'
            a_col = 'answer'
        else:
            # Attempt to find sensible fallbacks
            cols = _df.columns.tolist()
            logger.warning("Unrecognized knowledge CSV columns: %s", cols)
            # Create synthetic columns if possible
            q_col = cols[0] if len(cols) >= 1 else None
            a_col = cols[1] if len(cols) >= 2 else None
            if q_col is None or a_col is None:
                logger.error("Cannot determine question/answer columns in knowledge CSV")
                return False

        # Build a unified 'context' column used for embedding/search
        _df['context'] = _df[q_col].astype(str) + "\n" + _df[a_col].astype(str)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading sentence-transformer embeddings on device=%s", device)
        _model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        # Pre-compute embeddings; keep them as tensors for fast similarity
        contexts = _df['context'].tolist()
        if not contexts:
            logger.error("Knowledge CSV contains no rows")
            return False

        # Use the model to compute embeddings; wrap in try/except for safety
        try:
            _knowledge_embeddings = _model.encode(contexts, convert_to_tensor=True)
        except Exception:
            logger.exception("Failed to compute embeddings for knowledge base")
            _knowledge_embeddings = None
            return False

        _initialized = True
        logger.info("Knowledge base initialized: %d rows (from %s)", len(_df), data_path.name)
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
    Returns the most relevant question-answer pairs from the knowledge CSV.
    """
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "empty or invalid query"}

    if not isinstance(top_k, int) or top_k <= 0:
        top_k = 3

    if not _initialized and not init_knowledge():
        return {"ok": False, "error": "knowledge base not available"}

    if _knowledge_embeddings is None:
        logger.error("Knowledge embeddings not available")
        return {"ok": False, "error": "knowledge embeddings not available"}

    try:
        # Encode the query
        query_embedding = _model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity between query and knowledge base
        cos_scores = util.cos_sim(query_embedding, _knowledge_embeddings)[0]

        # Get top K indices (use torch.topk if tensor else numpy)
        k = min(top_k, len(_df))
        try:
            import torch as _torch
            topk = _torch.topk(cos_scores, k=k)
            indices = [int(x.item()) for x in topk.indices]
            scores = [float(x.item()) for x in topk.values]
        except Exception:
            # Fallback: convert to numpy
            arr = np.array(cos_scores.tolist())
            idxs = arr.argsort()[::-1][:k]
            indices = idxs.tolist()
            scores = arr[idxs].tolist()

        results = []
        for score, idx in zip(scores, indices):
            results.append({
                "score": float(score),
                # Best-effort expose the original columns if present
                "question": _df.iloc[idx].get('question_text') or _df.iloc[idx].get('question') or str(_df.iloc[idx].iloc[0]),
                "answer": _df.iloc[idx].get('clinical_definition') or _df.iloc[idx].get('answer') or str(_df.iloc[idx].iloc[1] if _df.shape[1] > 1 else ''),
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
        # Prefer labeled question_text if available
        if 'question_text' in _df.columns:
            topics = _df['question_text'].dropna().unique().tolist()
        elif 'question' in _df.columns:
            topics = _df['question'].dropna().unique().tolist()
        else:
            topics = _df.iloc[:, 0].dropna().unique().tolist()
        return {"ok": True, "topics": topics}
    except Exception:
        logger.exception("get_all_topics failed")
        return {"ok": False, "error": "internal error retrieving topics"}


if __name__ == "__main__":
    server.run()
