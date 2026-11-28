"""End-to-end 5-mode retrieval evaluation: Recall@5, MRR@5, Latency (mean/p50/p95), and Unanswerable FPR@5."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

MODES = [
    {"label": "dense", "search_mode": "dense", "use_reranker": False, "use_mmr": False, "fp_threshold": 0.70},
    {"label": "sparse", "search_mode": "sparse", "use_reranker": False, "use_mmr": False, "fp_threshold": 0.50},
    {"label": "hybrid_rrf", "search_mode": "hybrid", "use_reranker": False, "use_mmr": False, "fp_threshold": 0.70},
    {"label": "hybrid_mmr", "search_mode": "hybrid", "use_reranker": False, "use_mmr": True, "fp_threshold": 0.70},
    {"label": "hybrid_rerank", "search_mode": "hybrid", "use_reranker": True, "use_mmr": False, "fp_threshold": -1.0, "rerank_score_threshold": -1.0},
]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two L2-normalized or unnormalized vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


async def apply_mmr(
    query: str,
    candidates: list,
    embedding_client,
    vectordb_client,
    collection_name: str,
    top_k: int = 5,
    lambda_mult: float = 0.78,
) -> list:
    """Apply Maximal Marginal Relevance (MMR) using stored pgvector embeddings."""
    if not candidates or len(candidates) <= 1:
        return candidates[:top_k]

    q_vec = embedding_client.embed_text(text=query)[0]
    stored_vecs = await vectordb_client.get_vectors_by_chunk_ids(
        collection_name=collection_name,
        chunk_ids=[doc.chunk_id for doc in candidates if doc.chunk_id is not None],
    )
    doc_vecs = [stored_vecs[doc.chunk_id] for doc in candidates if doc.chunk_id in stored_vecs]
    candidates = [doc for doc in candidates if doc.chunk_id in stored_vecs]
    if not candidates:
        return []

    rel_scores = [_cosine_sim(q_vec, dv) for dv in doc_vecs]
    selected_indices: List[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected_indices) < top_k:
        best_idx = None
        best_mmr = -1e9
        for idx in remaining:
            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(_cosine_sim(doc_vecs[idx], doc_vecs[s]) for s in selected_indices)
            mmr_score = lambda_mult * rel_scores[idx] - (1.0 - lambda_mult) * redundancy
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def percentile(values: List[float], p: float) -> float:
    """Compute linear-interpolated percentile (0..100) on a non-empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


async def setup_eval_project(
    db_client,
    nlp_controller,
    documents: List[str],
    project_id: int,
) -> Dict[int, int]:
    """Reset eval project_id, insert chunks into PostgreSQL, and index into PgVector."""
    from models.AssetModel import AssetModel
    from models.ChunkModel import ChunkModel
    from models.ProjectModel import ProjectModel
    from models.db_schemes import Asset, DataChunk
    from models.enums.AssetTypeEnum import AssetTypeEnum
    from sqlalchemy import delete

    project_model = await ProjectModel.create_instance(db_client=db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    chunk_model = await ChunkModel.create_instance(db_client=db_client)
    asset_model = await AssetModel.create_instance(db_client=db_client)

    await nlp_controller.reset_vector_db_collection(project=project)
    # Drop any prior collection tables for this project (e.g., from a different embedding dimension)
    from sqlalchemy import text as sql_text
    async with db_client() as session:
        async with session.begin():
            res = await session.execute(
                sql_text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE :pat"),
                {"pat": f"collection_{project.project_id}_%"},
            )
            for (tbl,) in res.fetchall():
                await session.execute(sql_text(f"DROP TABLE IF EXISTS {tbl}"))
            await session.commit()
    await chunk_model.delete_chunks_by_project_id(project_id=project.project_id)
    async with db_client() as session:
        async with session.begin():
            await session.execute(delete(Asset).where(Asset.asset_project_id == project.project_id))
            await session.commit()

    asset = await asset_model.create_asset(
        asset=Asset(
            asset_project_id=project.project_id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_name="cuad_eval_corpus.txt",
            asset_size=sum(len(d.encode("utf-8")) for d in documents),
        )
    )

    def _extract_source(doc_text: str) -> str:
        """Extract the leading [ContractName] tag from a CUAD chunk as source."""
        if doc_text.startswith("[") and "]" in doc_text:
            return doc_text[1 : doc_text.index("]")].strip()
        return "cuad_eval_corpus.txt"

    chunk_records = [
        DataChunk(
            chunk_text=text,
            chunk_metadata={
                "doc_index": idx,
                "source": _extract_source(text),
            },
            chunk_order=idx + 1,
            chunk_project_id=project.project_id,
            chunk_asset_id=asset.asset_id,
        )
        for idx, text in enumerate(documents)
    ]
    
    await chunk_model.insert_many_chunks(chunks=chunk_records)

    stored_chunks = await chunk_model.get_project_chunks(
        project_id=project.project_id, page_no=1, page_size=len(documents) + 50
    )
    # Index in batches of 50 to mirror production indexing
    for i in range(0, len(stored_chunks), 50):
        batch = stored_chunks[i : i + 50]
        await nlp_controller.index_into_vector_db(
            project=project,
            chunks=batch,
            chunk_ids=[c.chunk_id for c in batch],
            do_reset=(i == 0),
        )

    return {c.chunk_metadata["doc_index"]: c.chunk_id for c in stored_chunks}


async def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    """Run all 5 retrieval modes and compute Recall@5, MRR@5, Latency, and Unanswerable metrics."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    os.chdir(SRC_DIR)

    from controllers import NLPController
    from helpers.config import get_settings
    from models.ProjectModel import ProjectModel
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from stores.llm.LLMProviderFactory import LLMProviderFactory
    from stores.llm.templates.template_parser import TemplateParser
    from stores.reranker import CrossEncoderReranker
    from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

    with Path(args.dataset).open("r", encoding="utf-8") as f:
        data = json.load(f)

    settings = get_settings()
    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    db_engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    embedding_client = LLMProviderFactory(settings).create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )
    vectordb_client = VectorDBProviderFactory(config=settings, db_client=db_client).create(
        provider=settings.VECTOR_DB_BACKEND
    )
    await vectordb_client.connect()

    reranker_client = CrossEncoderReranker(model_id=settings.RERANKING_MODEL_ID)
    template_parser = TemplateParser(
        language=settings.PRIMARY_LANG, default_language=settings.DEFAULT_LANG
    )
    nlp_controller = NLPController(
        vectordb_client=vectordb_client,
        generation_client=None,
        embedding_client=embedding_client,
        template_parser=template_parser,
        reranker_client=reranker_client,
    )

    try:
        idx_to_chunk_id = await setup_eval_project(
            db_client=db_client,
            nlp_controller=nlp_controller,
            documents=data["documents"],
            project_id=args.project_id,
        )
        project = await (await ProjectModel.create_instance(db_client)).get_project_or_create_one(
            project_id=args.project_id
        )

        mode_summaries: List[Dict[str, Any]] = []
        for mode in MODES:
            ans_recalls, hard_recalls, all_rrs, latencies = [], [], [], []
            unans_false_positives = 0
            unans_total = 0

            for q in data["queries"]:
                cat = q["category"]
                t0 = time.perf_counter()

                if mode["use_mmr"]:
                    candidates = await nlp_controller.search_vector_db_collections(
                        project=project,
                        text=q["question"],
                        limit=args.rerank_top_k,
                        search_mode="hybrid",
                        use_reranker=False,
                    )
                    results = await apply_mmr(
                        query=q["question"],
                        candidates=candidates or [],
                        embedding_client=embedding_client,
                        vectordb_client=vectordb_client,
                        collection_name=nlp_controller.create_collection_name(project_id=project.project_id),
                        top_k=5,
                        lambda_mult=0.78,
                    )
                else:
                    results = await nlp_controller.search_vector_db_collections(
                        project=project,
                        text=q["question"],
                        limit=5,
                        search_mode=mode["search_mode"],
                        use_reranker=mode["use_reranker"],
                        rerank_top_k=args.rerank_top_k,
                        rerank_score_threshold=mode.get("rerank_score_threshold"),
                    )
                latencies.append((time.perf_counter() - t0) * 1000.0)
                docs = (results or [])[:5]

                if cat == "unanswerable":
                    unans_total += 1
                    top_score = float(docs[0].score) if docs else -1e9
                    if top_score >= mode["fp_threshold"]:
                        unans_false_positives += 1
                else:
                    expected_ids = {idx_to_chunk_id[i] for i in q["relevant_doc_indices"]}
                    retrieved_ids = [doc.chunk_id for doc in docs]
                    recall = len(set(retrieved_ids) & expected_ids) / max(len(expected_ids), 1)
                    first_rank = next(
                        (r for r, cid in enumerate(retrieved_ids, start=1) if cid in expected_ids),
                        None,
                    )
                    rr = (1.0 / first_rank) if first_rank is not None else 0.0
                    all_rrs.append(rr)
                    if cat == "answerable":
                        ans_recalls.append(recall)
                    elif cat == "hard_hidden":
                        hard_recalls.append(recall)

            combined_recalls = ans_recalls + hard_recalls
            fpr = (unans_false_positives / unans_total) if unans_total else 0.0
            mode_summaries.append(
                {
                    "mode": mode["label"],
                    "overall_recall_at_5": round(sum(combined_recalls) / len(combined_recalls), 4),
                    "answerable_recall_at_5": round(sum(ans_recalls) / len(ans_recalls), 4),
                    "hard_hidden_recall_at_5": round(sum(hard_recalls) / len(hard_recalls), 4),
                    "mrr_at_5": round(sum(all_rrs) / len(all_rrs), 4),
                    "unanswerable_fpr_at_5": round(fpr, 4),
                    "unanswerable_rejection_acc": round(1.0 - fpr, 4),
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                    "p50_latency_ms": round(percentile(latencies, 50), 2),
                    "p95_latency_ms": round(percentile(latencies, 95), 2),
                }
            )
    finally:
        await vectordb_client.disconnect()
        await db_engine.dispose()

    return {
        "rerank_top_k": args.rerank_top_k,
        "num_documents": len(data["documents"]),
        "num_queries": len(data["queries"]),
        "results": mode_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 5-mode CUAD retrieval evaluation.")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "evals" / "cuad_dataset.json"),
        help="Path to CUAD dataset JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "evals" / "results.json"),
        help="Path to save evaluation results JSON.",
    )
    parser.add_argument("--project-id", type=int, default=99, help="Project ID for eval isolation.")
    parser.add_argument("--rerank-top-k", type=int, default=25, help="Candidates passed to reranker/MMR.")
    args = parser.parse_args()
    args.dataset = str(Path(args.dataset).resolve())
    args.output = str(Path(args.output).resolve())

    report = asyncio.run(run_eval(args))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 108)
    print(f"CUAD RETRIEVAL EVALUATION ({report['num_queries']} queries, {report['num_documents']} chunks)")
    print("=" * 108)
    print(
        f"{'Mode':<15} | {'Recall@5':>8} | {'Ans R@5':>8} | {'Hard R@5':>8} | "
        f"{'MRR@5':>7} | {'Rej Acc':>8} | {'FPR@5':>7} | {'Mean/p50/p95 (ms)':>22}"
    )
    print("-" * 108)
    for row in report["results"]:
        lat_str = f"{row['avg_latency_ms']:.1f}/{row['p50_latency_ms']:.1f}/{row['p95_latency_ms']:.1f}"
        print(
            f"{row['mode']:<15} | "
            f"{row['overall_recall_at_5']:>8.4f} | "
            f"{row['answerable_recall_at_5']:>8.4f} | "
            f"{row['hard_hidden_recall_at_5']:>8.4f} | "
            f"{row['mrr_at_5']:>7.4f} | "
            f"{row['unanswerable_rejection_acc']:>8.4f} | "
            f"{row['unanswerable_fpr_at_5']:>7.4f} | "
            f"{lat_str:>22}"
        )
    print("=" * 108)
    print(f"Saved results to: {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
