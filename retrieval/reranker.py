import logging
import math
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading cross-encoder reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker model loaded successfully.")

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved documents based on the query.
        Each document is expected to have a "text" key.
        Appends "rerank_score" (normalized to [0, 1]) to each document dict.
        """
        if not documents:
            return []

        # Construct query-document pairs
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Predict logits
        logger.info(f"Reranking {len(documents)} documents...")
        logits = self.model.predict(pairs)
        
        # If there's only one document, logits is a float instead of a list
        if isinstance(logits, float):
            logits = [logits]

        # Apply sigmoid function to normalize logits to [0, 1] range
        # Sigmoid(x) = 1 / (1 + exp(-x))
        normalized_scores = []
        for logit in logits:
            try:
                # Logits from MS MARCO usually fall between -10 and 10
                sig = 1.0 / (1.0 + math.exp(-float(logit)))
                normalized_scores.append(sig)
            except OverflowError:
                normalized_scores.append(1.0 if logit > 0 else 0.0)

        # Merge scores and sort documents
        for idx, doc in enumerate(documents):
            doc["rerank_score"] = normalized_scores[idx]

        # Sort by rerank score descending
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs
