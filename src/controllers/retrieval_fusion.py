"""Reciprocal Rank Fusion (RRF) utilities for combining dense and sparse retrieval results."""

from typing import Dict, List, Tuple
from models.db_schemes import RetrievedDocument


def _doc_key(doc: RetrievedDocument) -> Tuple[str, object]:
    """Build a stable deduplication key preferring chunk_id over normalized text."""
    if doc.chunk_id is not None:
        return ("chunk_id", doc.chunk_id)
    return ("text", " ".join(doc.text.split()))


def reciprocal_rank_fusion(dense_results: List[RetrievedDocument],sparse_results: List[RetrievedDocument],k: int = 60,limit: int = 5,) -> List[RetrievedDocument]:
    """Fuse dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        dense_results: Ranked documents from vector similarity search.
        sparse_results: Ranked documents from full-text keyword search.
        k: Smoothing constant for RRF (default 60).
        limit: Maximum number of fused documents to return.

    Returns:
        Top `limit` RetrievedDocument items ordered by normalized RRF score.
    """
    dense_list = dense_results or []
    sparse_list = sparse_results or []

    if not dense_list and not sparse_list:
        return []

    rrf_scores: Dict[Tuple[str, object], float] = {}
    docs_by_key: Dict[Tuple[str, object], RetrievedDocument] = {}

    active_lists = 0
    for ranked_list in (dense_list, sparse_list):
        if not ranked_list:
            continue
        active_lists += 1
        for rank, doc in enumerate(ranked_list, start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
            if key not in docs_by_key:
                docs_by_key[key] = doc

    # Normalize RRF score to (0, 1] relative to the theoretical maximum for active lists
    max_possible = active_lists * (1.0 / (k + 1)) if active_lists > 0 else 1.0

    sorted_keys = sorted(rrf_scores.keys(), key=lambda item: rrf_scores[item], reverse=True)
    fused: List[RetrievedDocument] = []
    for key in sorted_keys[:limit]:
        base_doc = docs_by_key[key]
        normalized_score = rrf_scores[key] / max_possible
        fused.append(
            RetrievedDocument(
                text=base_doc.text,
                score=round(normalized_score, 6),
                chunk_id=base_doc.chunk_id,
            )
        )
    return fused
