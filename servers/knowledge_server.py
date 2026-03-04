import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from fastmcp import FastMCP
from typing import List, Dict, Any

# Initialize MCP Server
mcp = FastMCP("KnowledgeServer")

# ====== Knowledge Base Initialization ======

# Load the processed data
BASE = os.path.dirname(__file__)
DOMAIN_DIR = os.path.abspath(os.path.join(BASE, "..", "domain_knowledge"))
DATA_PATH = os.path.join(DOMAIN_DIR, "processed_data.csv")
df = pd.read_csv(DATA_PATH)

# Combine question and answer for a richer semantic representation
df['context'] = "Question: " + df['question'] + "\nAnswer: " + df['answer']

# Initialize the embedding model
# 'all-MiniLM-L6-v2' is a fast and effective model for RAG
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Pre-compute embeddings for all documents in the knowledge base
# These are stored in memory for fast retrieval
knowledge_embeddings = model.encode(df['context'].tolist(), convert_to_tensor=True)


# ====== Tools ======

@mcp.tool()
def query_knowledge(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Searches the clinical knowledge base for information related to the query.
    Returns the most relevant question-answer pairs from the ICD/clinical documentation.

    Args:
        query: The user's question or symptoms described in text.
        top_k: Number of relevant snippets to return.
    """
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between query and knowledge base
    cos_scores = util.cos_sim(query_embedding, knowledge_embeddings)[0]

    # Get top K indices
    top_results = torch.topk(cos_scores, k=min(top_k, len(df)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        results.append({
            "score": float(score),
            "question": df.iloc[idx]['question'],
            "answer": df.iloc[idx]['answer'],
            "full_context": df.iloc[idx]['context']
        })

    return {
        "query": query,
        "results": results
    }


@mcp.tool()
def get_all_topics() -> List[str]:
    """
    Returns a list of unique questions/topics covered in the knowledge base.
    Useful for understanding the scope of the available information.
    """
    return df['question'].unique().tolist()


if __name__ == "__main__":
    mcp.run()