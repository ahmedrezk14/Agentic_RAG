# Agentic RAG

An **Agentic RAG** backend built with **FastAPI** and **PostgreSQL** (`pgvector`). It ingests and chunks documents, indexes them with local embeddings and bilingual full-text search, retrieves context via **Hybrid Search (RRF)** + **Cross-Encoder Reranking**, and generates grounded answers with an LLM.

## Core Features

- **Hybrid Search with RRF**: Fuses `pgvector` cosine similarity and PostgreSQL full-text search rankings using Reciprocal Rank Fusion (`k=60`).
- **Bilingual EN/AR Full-Text Search**: Uses a generated `tsvector` column (`to_tsvector('english', ...) || to_tsvector('arabic', ...)`) with a `GIN` index and `ts_rank_cd`.
- **Cross-Encoder Reranking**: Second-stage local reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` (`CrossEncoderReranker`).
- **Strict `chunk_id` Integrity**: Enforces `chunk_id NOT NULL` in vector tables, returns `chunk_id` in all search results, and deduplicates strictly by `chunk_id` during RRF.
- **Local Embeddings & Reranker**: Runs `BAAI/bge-small-en-v1.5` (384-dim) and `CrossEncoderReranker` locally on CPU with no external embedding/reranking API calls.

## Project Structure

```text
.
├── evals/
│   ├── cuad_dataset.json                # 50-example CUAD legal contracts benchmark
│   ├── run_eval.py                      # 4-mode retrieval evaluation script
│   └── results.json                     # Saved benchmark metrics
├── src/
│   ├── main.py                          # FastAPI entrypoint & client initialization
│   ├── .env.example                     # Environment configuration template
│   ├── requirements.txt                 # Python dependencies
│   ├── controllers/                     # Data, Process, Project, NLPController & RetrievalFusion
│   ├── helpers/                         # Pydantic Settings (config.py)
│   ├── models/                          # SQLAlchemy models & Alembic migrations (rag_project/)
│   ├── routes/                          # FastAPI routers (/data/*, /nlp/*) & request schemas
│   └── stores/
│       ├── llm/                         # OpenAI, CoHere & LocalEmbeddingProvider
│       ├── vectordb/                    # PGVectorProvider & QdrantDBProvider
│       └── reranker/                    # CrossEncoderReranker
└── README.md
```

## Quick Start

1. **Configure environment:**
   ```bash
   cp src/.env.example src/.env
   # Set PostgreSQL credentials and LLM API key (OPENAI_API_KEY / OPENAI_API_URL) in src/.env
   ```

2. **Start PostgreSQL (`pgvector`) & run Alembic migrations:**
   ```bash
   cd docker && docker compose up -d pgvector && cd ..
   cd src/models/db_schemes/rag_project
   cp alembic.ini.example alembic.ini   # Set sqlalchemy.url
   alembic upgrade head
   cd ../../../..
   ```

3. **Start the FastAPI server:**
   ```bash
   cd src
   uvicorn main:app --reload --host 0.0.0.0 --port 5000
   ```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/data/upload/{project_id}` | Upload a `.txt` or `.pdf` file |
| `POST` | `/api/v1/data/process/{project_id}` | Extract text and store chunks in PostgreSQL |
| `POST` | `/api/v1/nlp/index/push/{project_id}` | Embed chunks and index into `pgvector` + `GIN` FTS |
| `GET` | `/api/v1/nlp/index/info/{project_id}` | Get vector collection info and record count |
| `POST` | `/api/v1/nlp/index/search/{project_id}` | Search chunks (`dense`, `sparse`, or `hybrid` + reranker) |
| `POST` | `/api/v1/nlp/index/answer/{project_id}` | Retrieve context and generate an LLM answer |

Example request for `POST /api/v1/nlp/index/answer/{project_id}`:

```json
{
  "text": "How does RRF combine dense and sparse rankings?",
  "limit": 5,
  "search_mode": "hybrid",
  "use_reranker": true,
  "rerank_top_k": 20
}
```

## Retrieval Benchmark

Evaluated on **55 queries across 270 contract chunks** from 10 distinct agreements in the **CUAD (Contract Understanding Atticus Dataset)** (`evals/cuad_dataset.json`, `BAAI/bge-base-en-v1.5` 768-dim, `rerank_top_k=25`, `rerank_score_threshold=-1.0`, `w_dense=1.3`, `w_sparse=0.7`, local CPU):

- **`answerable` (29 queries)**: Direct clause retrieval questions over substantive contract terms (payment terms, confidentiality survival, renewal notice windows, audit rights, IP ownership, governing law).
- **`hard_hidden` (15 queries)**: Paraphrased or detail-buried questions (consequential damages exclusions, change-of-control termination, breach notification windows, risk-of-loss transfer, post-termination transition obligations).
- **`unanswerable` (11 queries)**: Out-of-scope queries absent from the corpus, measuring `Rejection Accuracy` and False Positive Rate (`FPR@5`) against score thresholds.

| Mode | Recall@5 | Ans R@5 | Hard R@5 | MRR@5 | Rej Acc | FPR@5 | Mean / p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 0.8523 | 0.7931 | 0.9667 | 0.6852 | 1.0000 | 0.0000 | 120.94 / 120.27 / 170.11 |
| `sparse` | 0.7500 | 0.7241 | 0.8000 | 0.5739 | 1.0000 | 0.0000 | 15.99 / 15.40 / 21.07 |
| `hybrid_rrf` | 0.8750 | 0.8276 | 0.9667 | 0.6902 | 0.0909 | 0.9091 | 154.26 / 148.62 / 237.67 |
| `hybrid_mmr` | 0.8864 | 0.8621 | 0.9333 | 0.7030 | 0.1818 | 0.8182 | 325.39 / 322.61 / 374.87 |
| `hybrid_rerank` | 0.8864 | 0.8621 | 0.9333 | 0.8201 | 1.0000 | 0.0000 | 1332.23 / 1181.93 / 2183.85 |

**Progression Summary**: Native PostgreSQL `sparse` (`websearch_to_tsquery` + `ts_rank_cd`) runs in ~16 ms with `0.7500` `Recall@5` and `1.0000` rejection. `dense` (`BAAI/bge-base-en-v1.5`, 768-dim) reaches `0.8523` `Recall@5` and `0.6852` `MRR@5`. Weighted `hybrid_rrf` (`w_dense=1.3`, `w_sparse=0.7`) improves both `Recall@5` (`0.8750`) and `MRR@5` (`0.6902`) over either single mode alone. `hybrid_mmr` (`lambda_mult=0.78`, reusing stored `pgvector` embeddings) lifts `Recall@5` to `0.8864` and `MRR@5` to `0.7030` at ~325 ms. `hybrid_rerank` (Cross-Encoder over top-25 candidates with `max_length=384` and calibrated `rerank_score_threshold=-1.0`) achieves the highest overall performance (**`0.8864` `Recall@5`**, **`0.8201` `MRR@5`**, and **`1.0000` `Rej Acc`** with `0.0000` `FPR@5`).

Run the benchmark:
```bash
python evals/run_eval.py
```
