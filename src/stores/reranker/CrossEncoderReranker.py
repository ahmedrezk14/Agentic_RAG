import logging 
from typing import List, Optional
from models.db_schemes import RetrievedDocument

class CrossEncoderReranker:
    """
    Two-Stage Retrieval Reranker (Cross-Encoder):
    - Stage 1 (Bi-Encoder / Hybrid RRF): Fast broad retrieval by encoding query and chunks separately (High Recall).
    - Stage 2 (Cross-Encoder): Jointly feeds (query, chunk_text) pairs into a Transformer for full cross-attention scoring,
    re-sorting the top RRF candidates to return the highest-precision top_k results.
    """
    def __init__(self, model_id: str, device: str = 'cpu', max_length: int = 384):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self._model = None
        self.logger = logging.getLogger(__name__)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_id,
                device=self.device,
                max_length=self.max_length,
            )
        return self._model


    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedDocument]:
        
        if not documents:
            return []
        
        model = self._get_model()
        pairs=[(query,doc.text) for doc in documents]
        scores=model.predict(pairs)

        sorted_docs=sorted(
            zip(documents,scores),
            key=lambda x:float(x[1]),
            reverse=True
        )

        reranked=[]

        for original_doc,score in sorted_docs:
            fscore = float(score)
            if score_threshold is not None and fscore < score_threshold:
                continue
            reranked.append(
                RetrievedDocument(
                    text=original_doc.text,
                    score=fscore,
                    chunk_id=original_doc.chunk_id
                )
            )
            if len(reranked) >= top_k:
                break
            
        return reranked

