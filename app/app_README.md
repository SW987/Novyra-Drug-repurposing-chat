# `app/` — Drug Repurposing Chat API

This package is the backend for the Drug Repurposing Chat service. It exposes a FastAPI HTTP API that lets users chat with drug repurposing research papers using Retrieval-Augmented Generation (RAG) backed by Google Gemini and ChromaDB.

---

## Module Overview

| File | Responsibility |
|------|---------------|
| `config.py` | Pydantic settings loaded from environment variables. Single source of truth for all configuration. |
| `schemas.py` | Pydantic request/response models for all API endpoints. |
| `main.py` | FastAPI application, lifespan management, route definitions, drug list caching. |
| `rag.py` | RAG pipeline — query embedding, chunk retrieval, diversity selection, prompt building, answer generation, source citation. |
| `vector_store.py` | ChromaDB client initialisation, custom Gemini embedding function, upsert/query helpers. |
| `ingestion.py` | PDF text extraction, chunking, embedding, and storage. Supports local files, raw bytes, and S3. |
| `ingestion_pipeline.py` | High-level orchestration — validates PDFs, batches ingestion, and optionally fetches papers from PubMed first. |
| `paper_fetcher.py` | PubMed Central integration — searches for open-access articles, downloads PDFs (handles TAR/GZIP), uploads to S3. |
| `drug_resolver.py` | Converts human-readable drug names to canonical underscore IDs stored in ChromaDB. |
| `utils.py` | Shared helpers — PDF text extraction (file and bytes), overlapping text chunking, filename parsing. |

---

## Configuration

All configuration is read from environment variables. Copy `.env.example` at the repository root to `.env` and fill in the required values:

```
GEMINI_API_KEY=<your key>          # Required
GEMINI_EMBEDDING_MODEL=<your Gemini embedding model>   # Required
GEMINI_EMBEDDING_DIMENSION=<vector dim of embedding model>  # Required, must match the model above
GEMINI_CHAT_MODEL=<your Gemini chat model>             # Required
CHROMA_DB_DIR=./data/chroma        # Where ChromaDB persists its index
DOCS_DIR=./data/docs               # Root folder for per-drug PDF subfolders
```

See `.env.example` for the full list of optional variables (S3, cache TTL, etc.).

> **Important:** Never hardcode API keys or model names in source files. Always use the `Settings` object from `config.py`.

---

## Data Flow

```
User message
    │
    ▼
main.py  ──► drug_resolver.py      (resolve drug name → drug_id)
    │
    ▼
rag.py
    ├─► vector_store.py            (embed query with Gemini)
    │       └─► ChromaDB           (retrieve top-K chunks)
    │
    ├─► build_rag_prompt()         (assemble context + history)
    │
    └─► Gemini GenerativeModel     (generate answer)
            │
            ▼
        ChatResponse (answer + PMC source citations)
```

---

## Directory Structure for PDFs

Place PDFs under `DOCS_DIR` in per-drug subfolders. Files must follow the naming convention:

```
<docs_dir>/
├── aspirin/
│   ├── aspirin_repurposing_PMC11242460.pdf
│   └── aspirin_repurposing_PMC9012345.pdf
└── metformin/
    └── metformin_repurposing_PMC8765432.pdf
```

The filename pattern `{drug_name}_repurposing_{pmc_id}.pdf` is parsed by `utils.parse_filename` to extract the drug ID and document ID stored in ChromaDB metadata.

---

## Ingestion

### Ingest PDFs already on disk

```bash
# Via the API endpoint (server must be running)
curl -X POST http://localhost:8000/drug_repurposing_chat/ingest-pdfs

# Or directly via the pipeline script
python -m app.ingestion_pipeline
```

### Download and ingest from PubMed

```python
from app.ingestion_pipeline import PDFIngestionPipeline
from app.config import get_settings

pipeline = PDFIngestionPipeline(get_settings())
result = pipeline.download_and_ingest_drug_papers("aspirin", max_papers=5)
```

### Ingest from S3

Set `S3_BUCKET` (and optionally `S3_PREFIX`, `S3_REGION`) in `.env`, then call the `/ingest-pdfs` endpoint — it will detect the S3 configuration automatically.

---

## API Endpoints

All routes are prefixed with `/drug_repurposing_chat`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check. |
| `POST` | `/chat` | Chat using a canonical drug ID. |
| `POST` | `/chat-by-drug-name` | Chat using a human-readable drug name. |
| `POST` | `/ingest` | Ingest a single document via API (text content). |
| `POST` | `/ingest-pdfs` | Ingest all PDFs from `DOCS_DIR` (or S3). |
| `GET` | `/drugs` | List all available drug IDs in the system. |

Interactive docs are available at `/docs` when the server is running.

---

## Drug Cache

At startup the server loads the drug list from `data/drugs_cache.json` (instant) and then refreshes it from ChromaDB in a background thread. The cache is saved back to disk after each refresh. Set `DRUGS_CACHE_TTL_SECONDS=0` to disable TTL-based expiry.
