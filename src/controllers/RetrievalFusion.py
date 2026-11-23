from typing import List, Dict
from models.db_schemes import RetrievedDocument

def _doc_key(doc:RetrievedDocument):
    if doc.chunk_id is None:
        raise ValueError("RetrievedDocument must have a valid chunk_id for RRF fusion.")
    return int(doc.chunk_id)

def reciprocal_rank_fusion(
        dense_results: List[RetrievedDocument],
        sparse_results: List[RetrievedDocument],
        k: int = 60,
        limit: int = 5,
        w_dense: float = 1.3,
        w_sparse: float = 0.7,
    ) -> List[RetrievedDocument]:
    """
    Weighted Reciprocal Rank Fusion (RRF):
    - Combines pre-sorted results from Dense (vector) and Sparse (FTS) searches without relying on their incompatible raw score scales.
    - Uses each document's 1-based position (rank) in each list: score += weight / (k + rank), where k=60 smooths rank differences.
    - Deduplicates documents by chunk_id, summing weighted scores when a chunk appears in both lists, then normalizes to (0, 1].
    """
    dense_list = dense_results or []
    sparse_list = sparse_results or []

    if not dense_list and not sparse_list:
        return []

    rrf_scores = {}  # key,score
    docs_by_key = {}  # key,doc

    active_weight = 0.0
    for ranked_list, weight in ((dense_list, w_dense), (sparse_list, w_sparse)):
        if not ranked_list or weight <= 0.0:
            continue
        active_weight += weight
        for rank, doc in enumerate(ranked_list, start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (weight / (rank + k))
            if key not in docs_by_key:
                docs_by_key[key] = doc

    max_possible = active_weight * (1.0 / (1 + k)) if active_weight else 0.0

    sorted_keys = sorted(rrf_scores.keys(), key=lambda item: rrf_scores[item], reverse=True)

    fused=[]
    for key in sorted_keys[:limit]:
        fused.append(
            RetrievedDocument(
                text=docs_by_key[key].text,
                score=rrf_scores[key]/max_possible,
                chunk_id=docs_by_key[key].chunk_id,
            )
        )    

    return fused
