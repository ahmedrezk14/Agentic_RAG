"""Deterministic CUAD v1 benchmark preparation, ingestion, and PgVector indexing for project_id=2."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CUAD_DIR = (PROJECT_ROOT.parent / "CUAD_v1").resolve()

# Preferred 10 diverse contract categories present in CUAD_v1/full_contract_pdf
PREFERRED_CATEGORIES = [
    "Hosting",
    "IP",
    "Franchise",
    "Development",
    "License_Agreements",
    "Supply",
    "Maintenance",
    "Outsourcing",
    "Marketing",
    "Service",
    "Distributor",
    "Co_Branding",
    "Manufacturing",
    "Affiliate_Agreements",
    "Reseller",
]

# CUAD clause types that yield concrete entities, dates, laws, durations, and numeric caps
PRIORITY_CLAUSE_TYPES = [
    "Governing Law",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Cap On Liability",
    "Anti-Assignment",
    "Audit Rights",
    "License Grant",
    "Exclusivity",
    "Non-Compete",
    "Termination For Convenience",
    "Change Of Control",
    "Post-Termination Services",
    "Ip Ownership Assignment",
    "Minimum Commitment",
    "Volume Restriction",
    "Warranty Duration",
    "Revenue/Profit Sharing",
    "Insurance",
    "Liquidated Damages",
    "Covenant Not To Sue",
]

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "had",
    "not", "are", "was", "were", "will", "shall", "may", "can", "could",
    "would", "should", "such", "any", "all", "each", "other", "than", "then",
    "into", "upon", "under", "over", "between", "through", "during", "before",
    "after", "above", "below", "agreement", "contract", "party", "parties",
    "section", "article", "clause", "paragraph", "hereto", "hereby", "herein",
    "hereof", "thereof", "therein", "thereto", "pursuant", "including",
    "without", "except", "unless", "whether", "which", "where", "when", "what",
    "there", "their", "they", "them", "these", "those", "been", "being",
    "more", "less", "most", "some", "only", "also", "either", "neither",
    "both", "same", "first", "second", "third", "company", "corporation",
    "inc", "llc", "ltd", "limited", "rights", "right", "terms", "term",
    "period", "days", "year", "years", "month", "months", "date", "time",
    "written", "notice", "prior", "provide", "provided", "mean", "means",
}


def normalize_space(text: str) -> str:
    """Collapse whitespace for clean matching and display."""
    return " ".join(text.split())


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF file using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(pages)


def is_clean_text(text: str, min_chars: int = 3000, max_chars: int = 180000) -> bool:
    """Verify extracted PDF text is non-empty, readable ASCII-dominant, and bounded."""
    if len(text) < min_chars or len(text) > max_chars:
        return False
    sample = text[:5000]
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t")
    return (printable / max(len(sample), 1)) > 0.92


def clean_cuad_question(raw_question: str, contract_title: str, clause_type: str) -> str:
    """Convert raw CUAD prompt into a clean, contract-scoped question."""
    short_title = contract_title.split("_")[0]
    details_match = re.search(r"Details:\s*(.+)$", raw_question.strip())
    if details_match:
        details = details_match.group(1).strip().rstrip("?.")
        return f"In the {short_title} contract ({clause_type}), {details[0].lower() + details[1:]}?"
    return f"In the {short_title} contract, what are the terms for {clause_type}?"


def extract_distinctive_keywords(
    answers: List[Dict[str, Any]],
    pdf_text_lower: str,
    max_keywords: int = 4,
) -> List[str]:
    """Derive clean, distinctive keywords from CUAD answer spans verified against PDF text."""
    combined = " ".join(a.get("text", "") for a in answers)
    candidates: List[str] = []

    # 1. Extract capitalized multi-word entity names / governing law states / months
    for m in re.finditer(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){1,3})\b", combined):
        phrase = normalize_space(m.group(1))
        words = [w.lower() for w in phrase.split()]
        if not all(w in STOPWORDS for w in words):
            candidates.append(phrase)

    # 2. Extract numbers with units (e.g. "30 days", "5 years", "$100,000", "2020")
    for m in re.finditer(r"\$?\b\d[\d,\.]*(?:\s*(?:days|years|months|percent|%))?\b", combined):
        val = normalize_space(m.group(0))
        if len(val) >= 2 and val not in {"1.", "2.", "3."}:
            candidates.append(val)

    # 3. Extract distinctive single tokens (length >= 5, alphabetic, not stopword)
    tokens = re.findall(r"\b[A-Za-z][A-Za-z\-]{4,}\b", combined)
    for tok in tokens:
        if tok.lower() not in STOPWORDS:
            candidates.append(tok)

    # Deduplicate while preserving order and verifying presence in extracted PDF text
    selected: List[str] = []
    seen_lower = set()
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in seen_lower:
            continue
        if cand_lower in pdf_text_lower:
            selected.append(cand)
            seen_lower.add(cand_lower)
        if len(selected) >= max_keywords:
            break

    return selected


def select_contracts(
    cuad_dir: Path,
    num_contracts: int = 10,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Deterministically select 10 diverse contracts across 10 categories."""
    rng = random.Random(seed)
    cuad_json_path = cuad_dir / "CUAD_v1.json"
    pdf_root = cuad_dir / "full_contract_pdf"

    with cuad_json_path.open("r", encoding="utf-8") as f:
        cuad_data = json.load(f)["data"]

    docs_by_title = {doc["title"]: doc for doc in cuad_data}

    # Index PDFs by category
    pdfs_by_category: Dict[str, List[Path]] = {}
    for pdf_path in sorted(pdf_root.rglob("*.pdf")):
        category = pdf_path.parent.name
        pdfs_by_category.setdefault(category, []).append(pdf_path)

    selected_contracts: List[Dict[str, Any]] = []
    selected_docs: Dict[str, Dict[str, Any]] = {}
    pdf_texts: Dict[str, str] = {}

    for category in PREFERRED_CATEGORIES:
        if len(selected_contracts) >= num_contracts:
            break

        candidates = list(pdfs_by_category.get(category, []))
        rng.shuffle(candidates)

        for pdf_path in candidates:
            title = pdf_path.stem
            doc = docs_by_title.get(title)
            if not doc:
                continue

            qas = doc["paragraphs"][0]["qas"]
            answerable_qas = [q for q in qas if not q["is_impossible"] and q.get("answers")]
            unanswerable_qas = [q for q in qas if q["is_impossible"]]
            if len(answerable_qas) < 8 or len(unanswerable_qas) < 2:
                continue

            text = extract_pdf_text(pdf_path)
            if not is_clean_text(text):
                continue

            rel_path = str(pdf_path.relative_to(cuad_dir))
            selected_contracts.append(
                {
                    "title": title,
                    "filename": pdf_path.name,
                    "category": category,
                    "relative_path": rel_path,
                    "absolute_path": str(pdf_path.resolve()),
                    "char_count": len(text),
                    "answerable_clause_count": len(answerable_qas),
                    "unanswerable_clause_count": len(unanswerable_qas),
                }
            )
            selected_docs[title] = doc
            pdf_texts[title] = text
            break

    if len(selected_contracts) < num_contracts:
        raise RuntimeError(
            f"Only found {len(selected_contracts)} qualifying contracts; expected {num_contracts}."
        )

    return selected_contracts, selected_docs, pdf_texts


def build_cuad_dataset(
    selected_contracts: List[Dict[str, Any]],
    selected_docs: Dict[str, Dict[str, Any]],
    pdf_texts: Dict[str, str],
    num_answerable: int = 42,
    num_unanswerable: int = 8,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build 50-question evaluation dataset (42 answerable, 8 unanswerable)."""
    rng = random.Random(seed)
    priority_rank = {name: idx for idx, name in enumerate(PRIORITY_CLAUSE_TYPES)}

    answerable_pool: Dict[str, List[Dict[str, Any]]] = {}
    unanswerable_pool: Dict[str, List[Dict[str, Any]]] = {}

    for contract in selected_contracts:
        title = contract["title"]
        doc = selected_docs[title]
        pdf_text_lower = normalize_space(pdf_texts[title]).lower()
        qas = doc["paragraphs"][0]["qas"]

        ans_items: List[Tuple[int, Dict[str, Any]]] = []
        unans_items: List[Dict[str, Any]] = []

        for qa in qas:
            clause_type = qa["id"].split("__")[-1]
            if not qa["is_impossible"] and qa.get("answers"):
                keywords = extract_distinctive_keywords(qa["answers"], pdf_text_lower)
                if len(keywords) < 2:
                    continue
                rank = priority_rank.get(clause_type, len(PRIORITY_CLAUSE_TYPES))
                ans_items.append(
                    (
                        rank,
                        {
                            "contract_title": title,
                            "contract_filename": contract["filename"],
                            "category": contract["category"],
                            "clause_type": clause_type,
                            "question": clean_cuad_question(qa["question"], title, clause_type),
                            "expected_keywords": keywords,
                            "expected_chunk_ids": [],
                            "is_impossible": False,
                            "reference_answer": normalize_space(qa["answers"][0]["text"])[:400],
                        },
                    )
                )
            elif qa["is_impossible"]:
                unans_items.append(
                    {
                        "contract_title": title,
                        "contract_filename": contract["filename"],
                        "category": contract["category"],
                        "clause_type": clause_type,
                        "question": clean_cuad_question(qa["question"], title, clause_type),
                        "expected_keywords": [],
                        "expected_chunk_ids": [],
                        "is_impossible": True,
                        "reference_answer": "",
                    }
                )

        # Sort by priority rank, with deterministic tie-breaking
        rng.shuffle(ans_items)
        ans_items.sort(key=lambda pair: pair[0])
        answerable_pool[title] = [item for _, item in ans_items]

        rng.shuffle(unans_items)
        unanswerable_pool[title] = unans_items

    # Round-robin across the 10 contracts to reach 42 answerable questions
    selected_answerable: List[Dict[str, Any]] = []
    idx_per_contract = {c["title"]: 0 for c in selected_contracts}
    while len(selected_answerable) < num_answerable:
        added_in_round = False
        for contract in selected_contracts:
            if len(selected_answerable) >= num_answerable:
                break
            title = contract["title"]
            pos = idx_per_contract[title]
            pool = answerable_pool[title]
            if pos < len(pool):
                selected_answerable.append(pool[pos])
                idx_per_contract[title] = pos + 1
                added_in_round = True
        if not added_in_round:
            break

    # Pick 8 unanswerable questions spread across the contracts
    selected_unanswerable: List[Dict[str, Any]] = []
    contract_order = list(selected_contracts)
    rng.shuffle(contract_order)
    for contract in contract_order:
        if len(selected_unanswerable) >= num_unanswerable:
            break
        pool = unanswerable_pool[contract["title"]]
        if pool:
            selected_unanswerable.append(pool[0])

    if len(selected_answerable) != num_answerable or len(selected_unanswerable) != num_unanswerable:
        raise RuntimeError(
            f"Expected {num_answerable}/{num_unanswerable} questions, "
            f"got {len(selected_answerable)}/{len(selected_unanswerable)}."
        )

    dataset: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected_answerable + selected_unanswerable, start=1):
        entry = {"id": f"cuad_{idx:02d}"}
        entry.update(item)
        dataset.append(entry)

    return dataset


async def ingest_and_index_project(
    selected_contracts: List[Dict[str, Any]],
    project_id: int,
    chunk_size: int = 500,
    overlap_size: int = 50,
) -> Dict[str, int]:
    """Reset project_id, ingest the 10 selected PDFs, and index chunks into PgVector."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    os.chdir(SRC_DIR)

    from controllers import NLPController, ProcessController, ProjectController
    from helpers.config import get_settings
    from models.AssetModel import AssetModel
    from models.ChunkModel import ChunkModel
    from models.ProjectModel import ProjectModel
    from models.db_schemes import Asset, DataChunk
    from models.enums.AssetTypeEnum import AssetTypeEnum
    from sqlalchemy import delete
    from sqlalchemy.sql import text as sql_text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from stores.llm.LLMProviderFactory import LLMProviderFactory
    from stores.llm.templates.template_parser import TemplateParser
    from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

    settings = get_settings()
    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    db_engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    llm_factory = LLMProviderFactory(settings)
    vectordb_factory = VectorDBProviderFactory(config=settings, db_client=db_client)

    embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    vectordb_client = vectordb_factory.create(provider=settings.VECTOR_DB_BACKEND)
    await vectordb_client.connect()

    template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    nlp_controller = NLPController(
        vectordb_client=vectordb_client,
        generation_client=None,
        embedding_client=embedding_client,
        template_parser=template_parser,
    )

    try:
        project_model = await ProjectModel.create_instance(db_client=db_client)
        project = await project_model.get_project_or_create_one(project_id=project_id)

        chunk_model = await ChunkModel.create_instance(db_client=db_client)
        asset_model = await AssetModel.create_instance(db_client=db_client)

        # Clear existing vector collections for project_id (any vector dimension), chunks, assets, and files
        async with db_client() as session:
            async with session.begin():
                res = await session.execute(
                    sql_text("SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix"),
                    {"prefix": f"collection_{project.project_id}_%"},
                )
                for tbl in res.scalars().all():
                    await vectordb_client.delete_collection(collection_name=tbl)
        await chunk_model.delete_chunks_by_project_id(project_id=project.project_id)

        async with db_client() as session:
            async with session.begin():
                await session.execute(
                    delete(Asset).where(Asset.asset_project_id == project.project_id)
                )
                await session.commit()

        project_dir = Path(ProjectController().get_project_path(project_id=project_id))
        if project_dir.exists():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        process_controller = ProcessController(project_id=project_id)

        total_chunks = 0
        total_files = 0

        for contract in selected_contracts:
            src_pdf = Path(contract["absolute_path"])
            dest_pdf = project_dir / contract["filename"]
            shutil.copy2(src_pdf, dest_pdf)

            asset_record = await asset_model.create_asset(
                asset=Asset(
                    asset_project_id=project.project_id,
                    asset_type=AssetTypeEnum.FILE.value,
                    asset_name=contract["filename"],
                    asset_size=dest_pdf.stat().st_size,
                )
            )

            file_content = process_controller.get_file_content(file_id=contract["filename"])
            if not file_content:
                raise RuntimeError(f"Failed to load PDF content: {contract['filename']}")

            file_chunks = process_controller.process_file_content(
                file_content=file_content,
                file_id=contract["filename"],
                chunk_size=chunk_size,
                overlap_size=overlap_size,
            )
            if not file_chunks:
                raise RuntimeError(f"Failed to chunk PDF content: {contract['filename']}")

            chunk_records = [
                DataChunk(
                    chunk_text=chunk.page_content,
                    chunk_metadata={
                        "source_file": contract["filename"],
                        "category": contract["category"],
                    },
                    chunk_order=i + 1,
                    chunk_project_id=project.project_id,
                    chunk_asset_id=asset_record.asset_id,
                )
                for i, chunk in enumerate(file_chunks)
            ]

            total_chunks += await chunk_model.insert_many_chunks(chunks=chunk_records)
            total_files += 1
            print(f"[INGEST] {contract['filename']} -> {len(chunk_records)} chunks")

        # Create vector collection and index all chunks in batches
        collection_name = nlp_controller.create_collection_name(project_id=project.project_id)
        await vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=embedding_client.embedding_size,
            do_reset=True,
        )

        page_no = 1
        indexed_count = 0
        while True:
            page_chunks = await chunk_model.get_project_chunks(
                project_id=project.project_id,
                page_no=page_no,
                page_size=50,
            )
            if not page_chunks:
                break
            page_no += 1
            chunk_ids = [c.chunk_id for c in page_chunks]
            ok = await nlp_controller.index_into_vector_db(
                project=project,
                chunks=page_chunks,
                chunk_ids=chunk_ids,
            )
            if not ok:
                raise RuntimeError("Failed to insert batch into vector DB.")
            indexed_count += len(page_chunks)
            print(f"[INDEX] Indexed {indexed_count}/{total_chunks} chunks into {collection_name}")

        return {
            "files_ingested": total_files,
            "chunks_inserted": total_chunks,
            "chunks_indexed": indexed_count,
        }
    finally:
        await vectordb_client.disconnect()
        await db_engine.dispose()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare deterministic 10-contract CUAD v1 benchmark and ingest into project_id=2."
    )
    parser.add_argument(
        "--cuad-dir",
        default=str(DEFAULT_CUAD_DIR),
        help="Path to CUAD_v1 directory.",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=2,
        help="Target project ID in PostgreSQL/PgVector.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters for ProcessController.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Only generate selected_contracts.json and dataset_cuad.json without DB ingestion.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for preparing and ingesting the CUAD benchmark."""
    args = parse_args()
    cuad_dir = Path(args.cuad_dir).resolve()
    if not cuad_dir.exists():
        print(f"[ERROR] CUAD directory not found: {cuad_dir}", file=sys.stderr)
        return 1

    print(f"[PREP] Selecting 10 diverse contracts from {cuad_dir} (seed={args.seed})...")
    selected_contracts, selected_docs, pdf_texts = select_contracts(
        cuad_dir=cuad_dir,
        num_contracts=10,
        seed=args.seed,
    )

    contracts_out = PROJECT_ROOT / "evals" / "selected_contracts.json"
    contracts_out.parent.mkdir(parents=True, exist_ok=True)
    with contracts_out.open("w", encoding="utf-8") as f:
        json.dump(selected_contracts, f, ensure_ascii=False, indent=2)
    print(f"[PREP] Saved {len(selected_contracts)} selected contracts to {contracts_out}")

    dataset = build_cuad_dataset(
        selected_contracts=selected_contracts,
        selected_docs=selected_docs,
        pdf_texts=pdf_texts,
        num_answerable=42,
        num_unanswerable=8,
        seed=args.seed,
    )
    dataset_out = PROJECT_ROOT / "evals" / "dataset_cuad.json"
    with dataset_out.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"[PREP] Saved 50-question CUAD dataset to {dataset_out}")

    if not args.skip_ingest:
        summary = asyncio.run(
            ingest_and_index_project(
                selected_contracts=selected_contracts,
                project_id=args.project_id,
                chunk_size=args.chunk_size,
            )
        )
        print(f"[DONE] Ingestion summary for project_id={args.project_id}: {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
