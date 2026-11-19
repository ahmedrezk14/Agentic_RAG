"""Retrieval evaluation harness for measuring Dense (PgVector) and future Hybrid Search baselines."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load evaluation Q&A entries from a JSON file."""
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Dataset at {dataset_path} must be a non-empty JSON list.")
    return data


def normalize_text(text: str) -> str:
    """Normalize text for case-insensitive keyword matching."""
    return " ".join(text.lower().split())


def is_chunk_relevant(
    chunk: Dict[str, Any],
    expected_keywords: List[str],
    expected_chunk_ids: List[int],
) -> bool:
    """Determine whether a single retrieved chunk is relevant to the query."""
    chunk_id = chunk.get("chunk_id")
    if expected_chunk_ids and chunk_id is not None and chunk_id in expected_chunk_ids:
        return True

    chunk_text = normalize_text(chunk.get("text", ""))
    for kw in expected_keywords:
        if normalize_text(kw) in chunk_text:
            return True
    return False


def evaluate_query_results(
    entry: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    top_k: int,
    latency_ms: float,
) -> Dict[str, Any]:
    """Compute Hit@k, Reciprocal Rank@k, and Keyword Recall@k for one query."""
    expected_keywords = entry.get("expected_keywords", [])
    expected_chunk_ids = entry.get("expected_chunk_ids", [])

    top_results = retrieved[:top_k]

    first_relevant_rank: Optional[int] = None
    for idx, chunk in enumerate(top_results, start=1):
        if is_chunk_relevant(chunk, expected_keywords, expected_chunk_ids):
            first_relevant_rank = idx
            break

    hit = 1.0 if first_relevant_rank is not None else 0.0
    reciprocal_rank = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

    combined_text = normalize_text(" ".join(c.get("text", "") for c in top_results))
    matched_keywords = [
        kw for kw in expected_keywords if normalize_text(kw) in combined_text
    ]
    keyword_recall = (
        len(matched_keywords) / len(expected_keywords)
        if expected_keywords
        else 0.0
    )

    return {
        "id": entry.get("id"),
        "question": entry.get("question"),
        "is_impossible": bool(entry.get("is_impossible", False)),
        "hit_at_k": hit,
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": first_relevant_rank,
        "keyword_recall": keyword_recall,
        "matched_keywords": matched_keywords,
        "expected_keywords": expected_keywords,
        "latency_ms": round(latency_ms, 2),
        "retrieved_count": len(top_results),
        "retrieved_scores": [round(float(c.get("score", 0.0)), 4) for c in top_results],
    }


def retrieve_via_http(
    base_url: str,
    project_id: int,
    query: str,
    top_k: int,
    search_mode: str = "dense",
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    timeout_sec: float = 30.0,
) -> Tuple[List[Dict[str, Any]], float]:
    """Call the FastAPI search endpoint and return (results, latency_ms)."""
    url = f"{base_url.rstrip('/')}/api/v1/nlp/index/search/{project_id}"
    payload = json.dumps(
        {
            "text": query,
            "limit": top_k,
            "search_mode": search_mode,
            "use_reranker": use_reranker,
            "rerank_top_n": rerank_top_n,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc.reason}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    results = body.get("results", [])
    return results, elapsed_ms


class DirectRetriever:
    """Direct retrieval runner using the project's NLPController and PGVectorProvider."""

    def __init__(self, project_id: int, use_reranker: bool = False) -> None:
        self.project_id = project_id
        self.use_reranker = use_reranker
        self.db_engine = None
        self.vectordb_client = None
        self.nlp_controller = None
        self.project = None

    async def initialize(self) -> None:
        """Initialize database engine, LLM embedding client, and vector DB client."""
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
        from stores.reranker.CrossEncoderReranker import CrossEncoderReranker
        from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

        settings = get_settings()
        postgres_conn = (
            f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
        )
        self.db_engine = create_async_engine(postgres_conn)
        db_client = sessionmaker(
            self.db_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        llm_factory = LLMProviderFactory(settings)
        vectordb_factory = VectorDBProviderFactory(config=settings, db_client=db_client)

        embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
        embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE,
        )

        self.vectordb_client = vectordb_factory.create(provider=settings.VECTOR_DB_BACKEND)
        await self.vectordb_client.connect()

        template_parser = TemplateParser(
            language=settings.PRIMARY_LANG,
            default_language=settings.DEFAULT_LANG,
        )

        project_model = await ProjectModel.create_instance(db_client=db_client)
        self.project = await project_model.get_project_or_create_one(
            project_id=self.project_id
        )

        reranker_client = None
        if self.use_reranker:
            reranker_client = CrossEncoderReranker(model_id=settings.RERANKER_MODEL)

        self.nlp_controller = NLPController(
            vectordb_client=self.vectordb_client,
            generation_client=None,
            embedding_client=embedding_client,
            template_parser=template_parser,
            reranker_client=reranker_client,
        )

    async def search(
        self,
        query: str,
        top_k: int,
        search_mode: str = "dense",
        use_reranker: bool = False,
        rerank_top_n: int = 20,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Execute vector/sparse/hybrid search via NLPController and return (results, latency_ms)."""
        start = time.perf_counter()
        raw_results = await self.nlp_controller.search_vector_db_collections(
            project=self.project,
            text=query,
            limit=top_k,
            search_mode=search_mode,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if not raw_results:
            return [], elapsed_ms

        formatted = [
            doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()
            for doc in raw_results
        ]
        return formatted, elapsed_ms

    async def close(self) -> None:
        """Dispose database engine and disconnect vector DB client."""
        if self.vectordb_client is not None:
            await self.vectordb_client.disconnect()
        if self.db_engine is not None:
            await self.db_engine.dispose()


async def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run retrieval across the evaluation dataset and compute aggregate metrics."""
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)

    per_query_results: List[Dict[str, Any]] = []
    retriever: Optional[DirectRetriever] = None

    if args.mode == "direct":
        retriever = DirectRetriever(
            project_id=args.project_id,
            use_reranker=args.use_reranker,
        )
        await retriever.initialize()

    try:
        total_queries = len(dataset)
        for idx, entry in enumerate(dataset, start=1):
            query = entry["question"]
            if args.mode == "direct":
                assert retriever is not None
                retrieved, latency_ms = await retriever.search(
                    query=query,
                    top_k=args.top_k,
                    search_mode=args.search_mode,
                    use_reranker=args.use_reranker,
                    rerank_top_n=args.rerank_top_n,
                )
            else:
                retrieved, latency_ms = retrieve_via_http(
                    base_url=args.base_url,
                    project_id=args.project_id,
                    query=query,
                    top_k=args.top_k,
                    search_mode=args.search_mode,
                    use_reranker=args.use_reranker,
                    rerank_top_n=args.rerank_top_n,
                )

            res = evaluate_query_results(
                entry=entry,
                retrieved=retrieved,
                top_k=args.top_k,
                latency_ms=latency_ms,
            )
            per_query_results.append(res)
            print(
                f"[{idx}/{total_queries}] {res['id']} | hit={int(res['hit_at_k'])} | "
                f"rr={res['reciprocal_rank']:.3f} | latency={res['latency_ms']:.1f}ms"
            )
    finally:
        if retriever is not None:
            await retriever.close()

    total = len(per_query_results)
    answerable = [r for r in per_query_results if not r.get("is_impossible", False)]
    unanswerable = [r for r in per_query_results if r.get("is_impossible", False)]

    eval_pool = answerable if answerable else per_query_results
    eval_count = len(eval_pool)

    hit_rate = sum(r["hit_at_k"] for r in eval_pool) / eval_count
    mrr = sum(r["reciprocal_rank"] for r in eval_pool) / eval_count
    avg_keyword_recall = sum(r["keyword_recall"] for r in eval_pool) / eval_count
    avg_latency_ms = sum(r["latency_ms"] for r in per_query_results) / total

    unanswerable_top_scores = [
        r["retrieved_scores"][0] for r in unanswerable if r["retrieved_scores"]
    ]
    unanswerable_avg_top_score = (
        sum(unanswerable_top_scores) / len(unanswerable_top_scores)
        if unanswerable_top_scores
        else None
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "strategy": args.strategy,
            "search_mode": args.search_mode,
            "use_reranker": args.use_reranker,
            "rerank_top_n": args.rerank_top_n if args.use_reranker else None,
            "mode": args.mode,
            "project_id": args.project_id,
            "top_k": args.top_k,
            "num_queries": total,
            "num_answerable": len(answerable),
            "num_unanswerable": len(unanswerable),
            "dataset_path": str(dataset_path),
        },
        "summary": {
            f"hit_rate_at_{args.top_k}": round(hit_rate, 4),
            f"mrr_at_{args.top_k}": round(mrr, 4),
            f"keyword_recall_at_{args.top_k}": round(avg_keyword_recall, 4),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "unanswerable_avg_top_score": (
                round(unanswerable_avg_top_score, 4)
                if unanswerable_avg_top_score is not None
                else None
            ),
        },
        "queries": per_query_results,
    }
    return report


def print_summary_table(report: Dict[str, Any]) -> None:
    """Print a clean ASCII summary table and per-query breakdown to stdout."""
    meta = report["metadata"]
    summary = report["summary"]
    top_k = meta["top_k"]

    print("\n" + "=" * 78)
    print(f"RETRIEVAL EVALUATION SUMMARY | Strategy: {meta['strategy']} | Mode: {meta['mode']}")
    print("=" * 78)
    print(f"{'Metric':<35} | {'Value':>15}")
    print("-" * 78)
    print(f"{'Queries Evaluated (Total)':<35} | {meta['num_queries']:>15}")
    print(f"{'Answerable Queries':<35} | {meta.get('num_answerable', meta['num_queries']):>15}")
    print(f"{'Unanswerable Queries':<35} | {meta.get('num_unanswerable', 0):>15}")
    print(f"{f'Hit Rate@{top_k} (Answerable)':<35} | {summary[f'hit_rate_at_{top_k}']:>15.4f}")
    print(f"{f'MRR@{top_k} (Answerable)':<35} | {summary[f'mrr_at_{top_k}']:>15.4f}")
    print(f"{f'Keyword Recall@{top_k} (Answerable)':<35} | {summary[f'keyword_recall_at_{top_k}']:>15.4f}")
    print(f"{'Average Latency (ms)':<35} | {summary['avg_latency_ms']:>15.2f}")
    if summary.get("unanswerable_avg_top_score") is not None:
        print(f"{'Unanswerable Avg Top Score':<35} | {summary['unanswerable_avg_top_score']:>15.4f}")
    print("=" * 78)

    print("\nPER-QUERY BREAKDOWN")
    print("-" * 78)
    print(f"{'ID':<6} | {'Type':<6} | {'Hit':<5} | {'RR':<6} | {'KW Recall':<10} | {'Latency(ms)':<12} | {'Rank'}")
    print("-" * 78)
    for q in report["queries"]:
        q_type = "UNANS" if q.get("is_impossible") else "ANS"
        rank_str = str(q["first_relevant_rank"]) if q["first_relevant_rank"] is not None else "-"
        print(
            f"{str(q['id']):<6} | "
            f"{q_type:<6} | "
            f"{int(q['hit_at_k']):<5} | "
            f"{q['reciprocal_rank']:<6.3f} | "
            f"{q['keyword_recall']:<10.3f} | "
            f"{q['latency_ms']:<12.2f} | "
            f"{rank_str}"
        )
    print("-" * 78 + "\n")


def print_comparison_table(
    current_report: Dict[str, Any],
    baseline_paths: List[Path],
) -> None:
    """Print a multi-way comparison table against one or more prior JSON reports."""
    reports: List[Dict[str, Any]] = []
    for path in baseline_paths:
        with path.open("r", encoding="utf-8") as f:
            reports.append(json.load(f))
    reports.append(current_report)

    top_k = current_report["metadata"]["top_k"]
    rows = [
        (f"Hit Rate@{top_k} (Answerable)", f"hit_rate_at_{top_k}", "{:.4f}"),
        (f"MRR@{top_k} (Answerable)", f"mrr_at_{top_k}", "{:.4f}"),
        (f"Keyword Recall@{top_k} (Answerable)", f"keyword_recall_at_{top_k}", "{:.4f}"),
        ("Average Latency (ms)", "avg_latency_ms", "{:.2f}"),
        ("Unanswerable Avg Top Score", "unanswerable_avg_top_score", "{:.4f}"),
    ]

    headers = [r["metadata"].get("strategy", "run") for r in reports]
    col_width = 18
    total_width = 34 + 3 + len(headers) * (col_width + 3)

    print("=" * total_width)
    print("COMPARISON: " + " vs ".join(headers))
    print("=" * total_width)
    header_line = f"{'Metric':<34}"
    for h in headers:
        header_line += f" | {h[:col_width]:>{col_width}}"
    print(header_line)
    print("-" * total_width)

    for label, key, fmt in rows:
        line = f"{label:<34}"
        for r in reports:
            val = r["summary"].get(key)
            val_str = fmt.format(float(val)) if val is not None else "-"
            line += f" | {val_str:>{col_width}}"
        print(line)
    print("=" * total_width + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation against PgVector/FastAPI search."
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "evals" / "dataset_cuad.json"),
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "evals" / "results" / "baseline_cuad_dense.json"),
        help="Path to write the JSON evaluation report.",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=2,
        help="Target project ID in the database.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top retrieved chunks to evaluate (k).",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "http"],
        default="direct",
        help="Use 'direct' to call NLPController directly or 'http' to call FastAPI endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:5000",
        help="Base URL of the FastAPI server when --mode=http.",
    )
    parser.add_argument(
        "--strategy",
        default="dense_pgvector",
        help="Label for the retrieval strategy being evaluated.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["dense", "sparse", "hybrid"],
        default="dense",
        help="Retrieval mode to evaluate (dense, sparse, hybrid).",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable second-stage reranking on top of retrieval candidates.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=20,
        help="Number of first-stage candidates to pass to the reranker.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Optional path(s) to prior JSON reports for multi-way comparison.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the retrieval evaluation script."""
    args = parse_args()
    args.dataset = str(Path(args.dataset).resolve())
    args.output = str(Path(args.output).resolve())
    if args.compare:
        args.compare = [str(Path(p).resolve()) for p in args.compare]
    try:
        report = asyncio.run(run_evaluation(args))
    except Exception as exc:
        print(f"[ERROR] Retrieval evaluation failed: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary_table(report)
    if args.compare:
        print_comparison_table(report, [Path(p) for p in args.compare])
    print(f"Saved evaluation report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
