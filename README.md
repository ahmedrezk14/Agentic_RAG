# RAG App

A modular Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **PostgreSQL** (`pgvector` + `tsvector` GIN full-text search), local **sentence-transformers** embeddings and cross-encoder reranking, and configurable LLM generation (**OpenAI-compatible / Groq / Cohere**).

## 1. Project Overview

- **Zero-API-Dependency Retrieval**: Uses `BAAI/bge-small-en-v1.5` (384-dim) for local embeddings and `cross-encoder/ms-marco-MiniLM-L-6-v2` for local CPU second-stage reranking.
- **Hybrid Storage in PostgreSQL**: Combines dense vector similarity (`pgvector` HNSW cosine index) with sparse keyword search (`tsvector` generated column + `GIN` index) in per-project collection tables.
- **Two-Stage Retrieval Pipeline**: Fuses dense and sparse candidate pools via **Reciprocal Rank Fusion (RRF, `k=60`)**, then reorders top candidates with a Cross-Encoder reranker.
- **Reproducible Evaluation Harness**: Includes a 10-contract, 50-question CUAD v1 benchmark suite (`evals/`) measuring Hit Rate@5, MRR@5, Keyword Recall@5, latency, and unanswerable query score separation.

---

## 2. Architecture & Retrieval Pipeline

### Pipeline Flow

```text
Document Upload (.txt / .pdf)
        |
        v
Text Extraction & Chunking (ProcessController -> PostgreSQL `chunks`)
        |
        v
Local Embeddings (`LocalEmbeddingProvider`: BAAI/bge-small-en-v1.5, 384-dim)
        |
        v
PostgreSQL Hybrid Collection (`collection_{project_id}_384`)
  |-- `vector(384)`  (pgvector HNSW cosine index)
  +-- `text_tsv`     (generated `tsvector` + GIN index)
        |
        v
Two-Stage Retrieval (`/api/v1/nlp/index/search/{project_id}`)
  |-- Stage 1a: Dense Vector Search  (pgvector `<=>`)
  |-- Stage 1b: Sparse Keyword Search (`ts_rank_cd` on `text_tsv`)
  |-- Fusion:   Reciprocal Rank Fusion (`k=60`) -> top `rerank_top_n` (default 20)
  +-- Stage 2:  `CrossEncoderReranker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) -> top `limit`
        |
        v
Grounded LLM Generation (`/api/v1/nlp/index/answer/{project_id}`)
```

### Directory Structure

```text
.
├── docker/
│   ├── docker-compose.yml               # PostgreSQL (pgvector) & MongoDB services
│   └── .env.example
├── evals/
│   ├── dataset_cuad.json                # 50-question CUAD benchmark (42 ANS, 8 UNANS)
│   ├── selected_contracts.json          # Locked 10-contract CUAD corpus
│   ├── prepare_cuad.py                  # Deterministic CUAD selector, ingestor & indexer
│   ├── run_retrieval.py                 # Retrieval benchmark & comparison runner
│   └── results/                         # Saved JSON evaluation reports
├── src/
│   ├── main.py                          # FastAPI entrypoint & lifespan client setup
│   ├── .env.example                     # Application environment template
│   ├── requirements.txt                 # Python dependencies
│   ├── helpers/
│   │   └── config.py                    # Pydantic Settings
│   ├── controllers/
│   │   ├── DataController.py            # File validation & unique path generation
│   │   ├── ProcessController.py         # PDF/TXT loading & text chunking
│   │   ├── ProjectController.py         # Project filesystem paths
│   │   ├── NLPController.py             # Indexing, dense/sparse/hybrid search & RAG
│   │   └── retrieval_fusion.py          # Reciprocal Rank Fusion (RRF)
│   ├── models/
│   │   ├── ProjectModel.py
│   │   ├── AssetModel.py
│   │   ├── ChunkModel.py
│   │   └── db_schemes/rag_project/      # SQLAlchemy models & Alembic migrations
│   ├── routes/
│   │   ├── base.py                      # Health / app info route
│   │   ├── data.py                      # Upload & chunking routes
│   │   ├── nlp.py                       # Vector index, search & RAG answer routes
│   │   └── schemes/                     # Pydantic request schemas
│   └── stores/
│       ├── llm/                         # OpenAI, CoHere & LocalEmbeddingProvider
│       ├── vectordb/                    # PGVectorProvider (dense + FTS) & QdrantDBProvider
│       └── reranker/                    # CrossEncoderReranker
└── README.md
```

---

## 3. Retrieval Benchmark Results (CUAD Legal Contracts)

Evaluated on `project_id=2` across 10 diverse CUAD v1 contracts (`1,268` chunks in `collection_2_384`, `top_k=5`, `42` answerable + `8` unanswerable queries, running locally on CPU):

| Metric | `dense_bge_small` | `hybrid_bge_rrf` | `hybrid_bge_rerank` |
|---|---:|---:|---:|
| **Hit Rate@5 (Answerable)** | 0.9286 (39/42) | 0.9048 (38/42) | **0.9762 (41/42)** |
| **MRR@5 (Answerable)** | 0.8536 | 0.7766 | **0.8869** |
| **Keyword Recall@5 (Answerable)** | 0.6825 | 0.7183 | **0.7619** |
| **Average Latency (ms)** | 222.72 | **190.19** | 1411.74 *(641.28 ms steady-state)* |
| **Unanswerable Avg Top Score** | 0.7425 | 0.7259 | **-0.8780** |

> **Note on Unanswerable Separation**: The Cross-Encoder outputs raw relevance logits (`-0.8780` average top score on unanswerable queries vs. positive scores on answerable matches), providing a much cleaner threshold signal than cosine similarity or RRF scores alone.

### Reproduce the Benchmark

1. Ingest and index the locked 10 CUAD contracts into `project_id=2`:

```bash
python evals/prepare_cuad.py
```

2. Run the 3-way retrieval benchmarks:

```bash
# 1. Dense baseline
python evals/run_retrieval.py \
  --dataset evals/dataset_cuad.json \
  --project-id 2 \
  --search-mode dense \
  --strategy dense_bge_small \
  --output evals/results/baseline_cuad_dense.json

# 2. Hybrid RRF (Dense + Sparse)
python evals/run_retrieval.py \
  --dataset evals/dataset_cuad.json \
  --project-id 2 \
  --search-mode hybrid \
  --strategy hybrid_bge_rrf \
  --output evals/results/hybrid_cuad_rrf.json

# 3. Hybrid RRF + Cross-Encoder Reranking (with 3-way comparison)
python evals/run_retrieval.py \
  --dataset evals/dataset_cuad.json \
  --project-id 2 \
  --search-mode hybrid \
  --use-reranker \
  --rerank-top-n 20 \
  --strategy hybrid_bge_rerank \
  --output evals/results/hybrid_cuad_rerank.json \
  --compare evals/results/baseline_cuad_dense.json evals/results/hybrid_cuad_rrf.json
```

---

## 4. Setup & Quickstart

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for PostgreSQL with `pgvector`)
- System build libraries (for `psycopg2` if building from source): `libpq-dev`, `gcc`, `python3-dev`

### 1. Install Python Dependencies

```bash
conda create -n mini-rag python=3.10
conda activate mini-rag
pip install -r src/requirements.txt
pip install sentence-transformers
```

### 2. Configure Environment Variables

```bash
cp src/.env.example src/.env
```

Key local retrieval settings in `src/.env` (zero external API keys required for indexing or search):

```env
EMBEDDING_BACKEND="LOCAL"
EMBEDDING_MODEL_ID="BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_SIZE=384

VECTOR_DB_BACKEND="PGVECTOR"
VECTOR_DB_DISTANCE_METHOD="cosine"

RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
```

Set `GENERATION_BACKEND`, `GENERATION_MODEL_ID`, and `OPENAI_API_KEY` (and optionally `OPENAI_API_URL` for Groq or OpenAI-compatible endpoints) if using the `/index/answer/{project_id}` generation endpoint.

### 3. Start PostgreSQL (`pgvector`)

```bash
cd docker
cp .env.example .env
docker compose up -d pgvector
cd ..
```

### 4. Run Alembic Database Migrations

```bash
cd src/models/db_schemes/rag_project
cp alembic.ini.example alembic.ini
# Update sqlalchemy.url in alembic.ini to match your PostgreSQL credentials
alembic upgrade head
cd ../../../..
```

### 5. Start the FastAPI Server

```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

Interactive API documentation is available at `http://localhost:5000/docs`.

---

## 5. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/` | Application name and version |
| `POST` | `/api/v1/data/upload/{project_id}` | Upload a `.txt` or `.pdf` file to a project |
| `POST` | `/api/v1/data/process/{project_id}` | Extract and chunk uploaded files into PostgreSQL |
| `POST` | `/api/v1/nlp/index/push/{project_id}` | Embed chunks and index into `pgvector` + `GIN` FTS |
| `GET` | `/api/v1/nlp/index/info/{project_id}` | Retrieve vector collection metadata and record count |
| `POST` | `/api/v1/nlp/index/search/{project_id}` | Retrieve top chunks (`dense`, `sparse`, or `hybrid` + optional reranker) |
| `POST` | `/api/v1/nlp/index/answer/{project_id}` | Retrieve relevant chunks and generate an LLM answer |

### Example Search Request (`hybrid` + Cross-Encoder Reranking)

```bash
curl -X POST "http://localhost:5000/api/v1/nlp/index/search/2" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Which state law governs the VitalibisInc hosting agreement?",
    "limit": 5,
    "search_mode": "hybrid",
    "use_reranker": true,
    "rerank_top_n": 20
  }'
```

- `search_mode`: `"hybrid"` (default), `"dense"`, or `"sparse"`
- `use_reranker`: `true` (default) or `false`
- `rerank_top_n`: number of first-stage candidates passed to `CrossEncoderReranker` (default `20`)

### Example RAG Answer Request

```bash
curl -X POST "http://localhost:5000/api/v1/nlp/index/answer/2" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Which state law governs the VitalibisInc hosting agreement?",
    "limit": 5
  }'
```
