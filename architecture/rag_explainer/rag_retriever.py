"""
rag_explainer/rag_retriever.py
────────────────────────────────────────────────────────────────────
Builds a FAISS vector index from the PHQ-8 CSV knowledge base and
retrieves the most relevant symptom entries for a given user text.

CSV columns used:
  symptom_name, core_symptom_type, clinical_definition,
  natural_language_patterns, functional_impact_examples,
  severity_indicators, denial_patterns
"""

import os
import csv
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

logger = logging.getLogger("rag_retriever")

# ── Paths ────────────────────────────────────────────────────────────
# Go to project root
ROOT_DIR = Path(__file__).resolve().parents[2]

# Build CSV path
CSV_PATH = ROOT_DIR / "shared" / "phq8_knowledge_base.csv"

# ── Singletons ───────────────────────────────────────────────────────
_embedder = None
_index    = None
_docs     = []          # list of rich document dicts
_chunks   = []          # list of strings actually embedded (one per doc)


@dataclass
class RetrievedDoc:
    """One retrieved PHQ-8 symptom entry."""
    symptom_name:               str
    symptom_type:               str
    clinical_definition:        str
    natural_language_patterns:  List[str]
    functional_impact_examples: List[str]
    severity_indicators:        List[str]
    denial_patterns:            List[str]
    distance:                   float = 0.0   # L2 distance from query (lower = more relevant)


def _split(cell: str) -> List[str]:
    """Split a semicolon-separated CSV cell into a cleaned list."""
    return [s.strip() for s in cell.split(";") if s.strip()]


def _load_csv() -> List[dict]:
    """Parse the PHQ-8 CSV into a list of rich dicts."""
    docs = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append({
                "symptom_name":               row["symptom_name"].strip(),
                "symptom_type":               row["core_symptom_type"].strip(),
                "clinical_definition":        row["clinical_definition"].strip(),
                "natural_language_patterns":  _split(row["natural_language_patterns"]),
                "functional_impact_examples": _split(row["functional_impact_examples"]),
                "severity_indicators":        _split(row["severity_indicators"]),
                "denial_patterns":            _split(row["denial_patterns"]),
            })
    return docs


def _build_embed_chunk(doc: dict) -> str:
    """
    Build the string that gets embedded for a symptom.
    Combines definition + top natural language patterns + severity indicators
    so retrieval picks up both clinical and colloquial matches.
    """
    patterns = " | ".join(doc["natural_language_patterns"][:15])
    severity = ", ".join(doc["severity_indicators"][:5])
    return (
        f"{doc['symptom_name']}: {doc['clinical_definition']}. "
        f"Expressions: {patterns}. "
        f"Severity signals: {severity}."
    )


def _load():
    """Lazy-load sentence-transformers + FAISS and build the index."""
    global _embedder, _index, _docs, _chunks
    if _index is not None:
        return

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        raise ImportError(
            "sentence-transformers and faiss-cpu are required for RAG.\n"
            "Install with: pip install sentence-transformers faiss-cpu"
        )

    logger.info("Loading PHQ-8 knowledge base from %s", CSV_PATH)
    _docs   = _load_csv()
    _chunks = [_build_embed_chunk(d) for d in _docs]

    logger.info("Loading sentence-transformer embedder…")
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Building FAISS index over %d symptom chunks…", len(_chunks))
    embeddings = _embedder.encode(_chunks, convert_to_numpy=True)
    dim        = embeddings.shape[1]
    _index     = faiss.IndexFlatL2(dim)
    _index.add(embeddings.astype(np.float32))
    logger.info("RAG index ready.")


def retrieve(query: str, top_k: int = 3) -> List[RetrievedDoc]:
    """
    Embed the query and return the top_k most relevant PHQ-8 symptom
    entries as RetrievedDoc objects, sorted by relevance (lowest L2 first).
    """
    _load()
    q_emb = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = _index.search(q_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc = _docs[idx]
        results.append(RetrievedDoc(
            symptom_name               = doc["symptom_name"],
            symptom_type               = doc["symptom_type"],
            clinical_definition        = doc["clinical_definition"],
            natural_language_patterns  = doc["natural_language_patterns"],
            functional_impact_examples = doc["functional_impact_examples"],
            severity_indicators        = doc["severity_indicators"],
            denial_patterns            = doc["denial_patterns"],
            distance                   = float(dist),
        ))
    return results


def format_retrieved_for_prompt(docs: List[RetrievedDoc]) -> str:
    """Format retrieved docs as a structured block for the LLM prompt."""
    lines = []
    for i, doc in enumerate(docs, 1):
        patterns_sample = "; ".join(doc.natural_language_patterns[:6])
        impacts_sample  = "; ".join(doc.functional_impact_examples[:3])
        severity_sample = "; ".join(doc.severity_indicators[:4])
        lines.append(
            f"[{i}] {doc.symptom_name} ({doc.symptom_type} symptom)\n"
            f"    Clinical definition : {doc.clinical_definition}\n"
            f"    Typical expressions : {patterns_sample}\n"
            f"    Functional impact   : {impacts_sample}\n"
            f"    Severity signals    : {severity_sample}"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    query = "I don't really go out much anymore. I've lost interest in my hobbies."
    docs  = retrieve(query, top_k=3)
    print(f"\nQuery: {query}\n")
    print("Retrieved symptoms:")
    for d in docs:
        print(f"  [{d.distance:.3f}] {d.symptom_name}: {d.clinical_definition[:80]}")
    print("\nFormatted for LLM prompt:")
    print(format_retrieved_for_prompt(docs))
