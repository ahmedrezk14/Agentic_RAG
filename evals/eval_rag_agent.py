"""End-to-end LangGraph RAG Agent evaluation on the CUAD benchmark dataset."""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CITATION_PATTERN = re.compile(r"[\[【]Chunk\s*\d+[\]】]", re.IGNORECASE)

def _parse_json(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a JSON object from LLM output, stripping optional markdown fences."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else default
    except json.JSONDecodeError:
        return default


def _call_judge_llm(generation_client, system_prompt: str, user_prompt: str) -> str:
    """Invoke the project's generation_client synchronously with a fresh history list."""
    chat_history = [
        generation_client.construct_prompt(
            prompt=system_prompt,
            role=generation_client.enums.SYSTEM.value,
        )
    ]
    response = generation_client.generate_text(
        prompt=user_prompt,
        chat_history=chat_history,
    )
    return (response or "").strip()


async def post_agent_query(
    base_url: str,
    project_id: int,
    question: str,
    thread_id: str,
    timeout: float = 120.0,
) -> Tuple[int, Dict[str, Any], float]:
    """Send a POST request to /api/v1/nlp/agent/answer/{project_id} and measure latency."""
    url = f"{base_url.rstrip('/')}/api/v1/nlp/agent/answer/{project_id}"
    payload = {"text": question, "thread_id": thread_id}

    t0 = time.perf_counter()
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            try:
                body = resp.json()
            except Exception:
                body = {"raw_text": resp.text}
            return resp.status_code, body, elapsed_ms
    except ImportError:
        def _sync_urllib() -> Tuple[int, Dict[str, Any]]:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.status, json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raw_err = exc.read().decode("utf-8", errors="replace")
                try:
                    return exc.code, json.loads(raw_err)
                except json.JSONDecodeError:
                    return exc.code, {"raw_text": raw_err}

        status_code, body = await asyncio.to_thread(_sync_urllib)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return status_code, body, elapsed_ms


async def judge_answerable(
    generation_client,
    question: str,
    reference_docs: List[str],
    agent_answer: str,
) -> Tuple[bool, str]:
    """Verify factual accuracy of an answerable response against ground-truth chunks."""
    system_prompt = (
        "You are a strict legal RAG evaluator. Compare the assistant's answer against the "
        "ground-truth reference excerpt(s) from the contract.\n"
        "Mark 'is_correct' as true ONLY if the answer accurately captures the key facts from "
        "the reference excerpt(s) without contradicting them or fabricating details.\n"
        "Ignore inline citation tags like [Chunk 1] when judging accuracy.\n"
        'Respond ONLY with valid JSON: {"is_correct": true|false, "reason": "brief 1-sentence reason"}'
    )
    refs_block = "\n\n---\n\n".join(reference_docs)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Ground-Truth Reference Excerpt(s):\n{refs_block}\n\n"
        f"Assistant Answer:\n{agent_answer}"
    )

    raw = await asyncio.to_thread(_call_judge_llm, generation_client, system_prompt, user_prompt)
    parsed = _parse_json(raw, {"is_correct": False, "reason": f"Unparseable judge output: {raw[:120]}"})
    return bool(parsed.get("is_correct", False)), str(parsed.get("reason", ""))


async def judge_unanswerable(
    generation_client,
    question: str,
    agent_answer: str,
) -> Tuple[bool, str]:
    """Verify that an unanswerable query was properly refused without hallucination."""
    system_prompt = (
        "You are evaluating a RAG assistant on an out-of-scope question whose answer does NOT exist "
        "in the uploaded contracts.\n"
        "Mark 'properly_refused' as true ONLY if the assistant clearly states that the information "
        "is not found/available in the documents (or asks for clarification without inventing facts), "
        "and does NOT fabricate contract terms, numbers, or policies.\n"
        'Respond ONLY with valid JSON: {"properly_refused": true|false, "reason": "brief 1-sentence reason"}'
    )
    user_prompt = (
        f"Out-of-Scope Question: {question}\n\n"
        f"Assistant Answer:\n{agent_answer}"
    )

    raw = await asyncio.to_thread(_call_judge_llm, generation_client, system_prompt, user_prompt)
    parsed = _parse_json(raw, {"properly_refused": False, "reason": f"Unparseable judge output: {raw[:120]}"})
    return bool(parsed.get("properly_refused", False)), str(parsed.get("reason", ""))


def _pct(numerator: int, denominator: int) -> float:
    """Compute percentage rounded to two decimal places."""
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


async def run_agent_eval(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute end-to-end CUAD evaluation against the FastAPI LangGraph agent endpoint."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    os.chdir(SRC_DIR)

    from helpers.config import get_settings
    from stores.llm.LLMProviderFactory import LLMProviderFactory

    with Path(args.dataset).open("r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[str] = data["documents"]
    queries: List[Dict[str, Any]] = data["queries"]
    if args.limit is not None:
        queries = queries[: args.limit]

    settings = get_settings()
    generation_client = LLMProviderFactory(settings).create(provider=settings.GENERATION_BACKEND)
    generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    per_query_results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    retry_counts: List[int] = []

    ans_total = 0
    ans_correct = 0
    hard_total = 0
    hard_correct = 0
    unans_total = 0
    unans_refused = 0
    citation_eligible = 0
    citation_hits = 0
    route_hits = 0
    error_count = 0

    total_queries = len(queries)
    for idx, q in enumerate(queries, start=1):
        qid = q["id"]
        category = q["category"]
        question = q["question"]
        thread_id = f"{args.thread_prefix}-{qid}"

        status_code, resp_body, latency_ms = await post_agent_query(
            base_url=args.base_url,
            project_id=args.project_id,
            question=question,
            thread_id=thread_id,
            timeout=args.timeout,
        )
        latencies.append(latency_ms)

        answer = str(resp_body.get("answer") or "") if status_code == 200 else ""
        route: Optional[str] = resp_body.get("route") if status_code == 200 else None
        retry_count = int(resp_body.get("retry_count", 0)) if status_code == 200 else 0
        has_citation = bool(CITATION_PATTERN.search(answer)) if answer else False

        if status_code != 200 or not answer:
            error_count += 1
        else:
            retry_counts.append(retry_count)
            if route == "retrieve":
                route_hits += 1

        is_passed = False
        judge_reason = ""

        if status_code != 200 or not answer:
            judge_reason = f"HTTP {status_code}: {resp_body}"
            if category == "answerable":
                ans_total += 1
            elif category == "hard_hidden":
                hard_total += 1
            elif category == "unanswerable":
                unans_total += 1
        elif category == "unanswerable":
            unans_total += 1
            is_passed, judge_reason = await judge_unanswerable(
                generation_client=generation_client,
                question=question,
                agent_answer=answer,
            )
            if is_passed:
                unans_refused += 1
        else:
            citation_eligible += 1
            if has_citation:
                citation_hits += 1

            ref_docs = [documents[doc_idx] for doc_idx in q["relevant_doc_indices"]]
            is_passed, judge_reason = await judge_answerable(
                generation_client=generation_client,
                question=question,
                reference_docs=ref_docs,
                agent_answer=answer,
            )
            if category == "answerable":
                ans_total += 1
                if is_passed:
                    ans_correct += 1
            elif category == "hard_hidden":
                hard_total += 1
                if is_passed:
                    hard_correct += 1

        print(
            f"[{idx}/{total_queries}] {qid} ({category}) | "
            f"status={status_code} | route={route} | retries={retry_count} | "
            f"cite={has_citation} | pass={is_passed} | {latency_ms:.0f}ms"
        )

        per_query_results.append(
            {
                "id": qid,
                "category": category,
                "contract": q.get("contract"),
                "clause_type": q.get("clause_type"),
                "thread_id": thread_id,
                "question": question,
                "relevant_doc_indices": q.get("relevant_doc_indices", []),
                "status_code": status_code,
                "route": route,
                "retry_count": retry_count,
                "has_inline_citation": has_citation,
                "passed": is_passed,
                "judge_reason": judge_reason,
                "latency_ms": round(latency_ms, 2),
                "answer": answer,
            }
        )

        if args.delay > 0 and idx < total_queries:
            await asyncio.sleep(args.delay)

    overall_ans_total = ans_total + hard_total
    overall_ans_correct = ans_correct + hard_correct

    summary = {
        "total_queries": total_queries,
        "http_errors": error_count,
        "overall_answerable_accuracy_pct": _pct(overall_ans_correct, overall_ans_total),
        "answerable_accuracy_pct": _pct(ans_correct, ans_total),
        "hard_hidden_accuracy_pct": _pct(hard_correct, hard_total),
        "unanswerable_refusal_rate_pct": _pct(unans_refused, unans_total),
        "inline_citation_coverage_pct": _pct(citation_hits, citation_eligible),
        "route_retrieve_accuracy_pct": _pct(route_hits, total_queries),
        "avg_retry_count": round(sum(retry_counts) / len(retry_counts), 3) if retry_counts else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "counts": {
            "answerable": {"passed": ans_correct, "total": ans_total},
            "hard_hidden": {"passed": hard_correct, "total": hard_total},
            "unanswerable": {"refused": unans_refused, "total": unans_total},
            "inline_citations": {"cited": citation_hits, "eligible": citation_eligible},
            "route_retrieve": {"hits": route_hits, "total": total_queries},
        },
    }

    return {
        "base_url": args.base_url,
        "project_id": args.project_id,
        "judge_model": settings.GENERATION_MODEL_ID,
        "summary": summary,
        "results": per_query_results,
    }


def main() -> int:
    """Parse CLI arguments, run the agent evaluation, print summary, and save JSON."""
    parser = argparse.ArgumentParser(description="Evaluate the LangGraph RAG Agent endpoint on CUAD.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Running FastAPI server base URL.",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=99,
        help="Target project ID indexed with the CUAD corpus.",
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "evals" / "cuad_dataset.json"),
        help="Path to CUAD dataset JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "evals" / "agent_eval_results.json"),
        help="Path to save agent evaluation results JSON.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help="Delay in seconds between queries to avoid LLM API rate limits.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds per query.",
    )
    parser.add_argument(
        "--thread-prefix",
        default=f"eval-agent-{int(time.time())}",
        help="Prefix used to build an isolated thread_id per question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of queries to run (for smoke testing).",
    )
    args = parser.parse_args()
    args.dataset = str(Path(args.dataset).resolve())
    args.output = str(Path(args.output).resolve())

    report = asyncio.run(run_agent_eval(args))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    s = report["summary"]
    print("\n" + "=" * 72)
    print(f"LANGGRAPH RAG AGENT EVALUATION ({s['total_queries']} queries, project_id={args.project_id})")
    print("=" * 72)
    print(f"Overall Answerable Accuracy : {s['overall_answerable_accuracy_pct']:6.2f}%")
    print(f"Standard Answerable Acc     : {s['answerable_accuracy_pct']:6.2f}%")
    print(f"Hard Hidden Accuracy        : {s['hard_hidden_accuracy_pct']:6.2f}%")
    print(f"Unanswerable Refusal Rate   : {s['unanswerable_refusal_rate_pct']:6.2f}%")
    print(f"Inline Citation Coverage    : {s['inline_citation_coverage_pct']:6.2f}%")
    print(f"Route Accuracy (retrieve)   : {s['route_retrieve_accuracy_pct']:6.2f}%")
    print(f"Average Retry Count         : {s['avg_retry_count']:6.3f}")
    print(f"Average Latency (ms)        : {s['avg_latency_ms']:8.2f}")
    print(f"HTTP / Empty Errors         : {s['http_errors']}")
    print("=" * 72)
    print(f"Saved results to: {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
