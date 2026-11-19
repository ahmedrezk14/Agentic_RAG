import logging
from typing import List

from models.db_schemes import RetrievedDocument


class CrossEncoderReranker:
    """Local Cross-Encoder reranker on CPU backed by sentence-transformers."""

    def __init__(self,model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",device: str = "cpu",) -> None:
        self.model_id = model_id
        self.device = device
        self.logger = logging.getLogger(__name__)
        self._model = None

    def _get_model(self):
        """Lazily load the CrossEncoder model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, device=self.device)
        return self._model

    def rerank(self,query: str,documents: List[RetrievedDocument],top_k: int,) -> List[RetrievedDocument]:
        """Score (query, document) pairs locally and return top_k sorted by relevance."""
        if not documents:
            return []

        model = self._get_model()
        pairs = [(query, doc.text) for doc in documents]
        scores = model.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        reranked: List[RetrievedDocument] = []
        for orig_doc, score in scored_docs[:top_k]:
            reranked.append(
                RetrievedDocument(
                    text=orig_doc.text,
                    score=float(score),
                    chunk_id=orig_doc.chunk_id,
                )
            )
        return reranked
