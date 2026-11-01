"""
Local Dense Vector Retrieval & Evaluation

Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings)
for offline, material-scoped semantic search. Includes Recall@K and MRR@K benchmarks.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class MaterialVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._encoder = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Embed and store document chunks in local memory.
        """
        self.chunks = chunks
        if not chunks:
            self.embeddings = np.empty((0, 384), dtype=np.float32)
            return

        texts = [c["text"] for c in chunks]
        # Normalize embeddings for fast cosine similarity via dot product
        emb = self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self.embeddings = emb.astype(np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_material_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most semantically relevant chunks for a given query.
        Guarantees strict material-level isolation if filter_material_id is provided.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_emb = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        # Dot product of normalized vectors = Cosine similarity
        scores = np.dot(self.embeddings, q_emb)

        ranked_indices = np.argsort(scores)[::-1]
        results = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if filter_material_id and chunk.get("material_id") != filter_material_id:
                continue

            results.append(
                {
                    **chunk,
                    "similarity_score": float(scores[idx]),
                }
            )

            if len(results) >= top_k:
                break

        return results

    def evaluate_retrieval(
        self,
        eval_pairs: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluates retrieval performance using Recall@K and Mean Reciprocal Rank (MRR@K).
        eval_pairs format: [{'query': str, 'target_chunk_id': str, 'material_id': str}]
        """
        hits_at_k = 0
        reciprocal_ranks = []

        for pair in eval_pairs:
            query = pair["query"]
            target_id = pair["target_chunk_id"]
            mat_id = pair.get("material_id")

            retrieved = self.retrieve(query, top_k=top_k, filter_material_id=mat_id)
            retrieved_ids = [r["chunk_id"] for r in retrieved]

            if target_id in retrieved_ids:
                hits_at_k += 1
                rank = retrieved_ids.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        n = len(eval_pairs) if eval_pairs else 1
        return {
            f"recall@{top_k}": round(hits_at_k / n, 4),
            f"mrr@{top_k}": round(float(np.mean(reciprocal_ranks)), 4),
            "num_test_queries": len(eval_pairs),
        }
