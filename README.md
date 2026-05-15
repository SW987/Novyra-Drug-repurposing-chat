# Drug Repurposing Chat — Novyra Integration

An API service that answers natural-language questions about drug repurposing by retrieving evidence from a corpus of PubMed Central research papers, synthesizing across multiple documents, and generating responses using Google Gemini. It is designed to be embedded in the Novyra platform as a FastAPI endpoint.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Corpus Build Pipeline](#3-corpus-build-pipeline)
   - [Phase 2 — Paper Fetching](#phase-2--paper-fetching)
   - [Phase 3 — Ingestion](#phase-3--ingestion)
4. [FastAPI Runtime](#4-fastapi-runtime)
   - [Startup Sequence](#startup-sequence)
   - [Drug Cache](#drug-cache)
   - [API Routes](#api-routes)
5. [RAG Query Pipeline](#5-rag-query-pipeline)
   - [Query Embedding](#query-embedding)
   - [Chunk Retrieval and Diversity](#chunk-retrieval-and-diversity)
   - [Prompt Construction](#prompt-construction)
   - [Answer Generation](#answer-generation)
   - [Citations](#citations)
6. [ChromaDB Schema](#6-chromadb-schema)
7. [Drug Resolution](#7-drug-resolution)
8. [Request and Response Schemas](#8-request-and-response-schemas)
9. [Configuration Reference](#9-configuration-reference)
10. [Running the System](#10-running-the-system)

---

## 1. System Overview

The system runs in two distinct phases: an **offline corpus build** (PubMed Central → PDFs → chunks → Gemini embeddings → ChromaDB) and a **FastAPI runtime** (drug-name resolution → query embedding → ChromaDB retrieval → Gemini answer with PMC citations). The corpus is immutable at query time.

```
   CORPUS BUILD  (offline, one time)            RUNTIME  (FastAPI endpoint)
   ────────────────────────────────             ────────────────────────────
   PubMed Central                               Novyra client
   eSearch + OA API                                   │
        │                                             ▼
        ▼                                      POST /chat-by-drug-name
   paper_fetcher.py                                   │
   download • validate                                ▼
        │                                      drug_resolver.py
        ▼                                      name → canonical drug_id
   [ data/docs/         ]                             │
   [   PDF corpus       ]                             ▼
        │                                      rag.py
        ▼                                      embed query • cosine search
   ingestion.py                                       │
   extract • chunk • embed                            ▼
        │                                      [ ChromaDB             ]
        ▼                                      [ diversity-capped top-k]
   [ data/chroma/       ] ─── persistent ───►         │
   [   vector store     ]      disk                   ▼
                                               Gemini GenerativeModel
                                               synthesize answer
                                                      │
                                                      ▼
                                               ChatResponse
                                               + PMC citations
```

---

## 2. Repository Structure

```
.
├── app/                          Core application package
│   ├── main.py                   FastAPI entry point, lifespan, all routes
│   ├── rag.py                    RAG pipeline: embed → retrieve → generate
│   ├── vector_store.py           ChromaDB client, GeminiEmbeddingFunction, query helpers
│   ├── ingestion.py              PDF → chunks → embeddings → ChromaDB
│   ├── ingestion_pipeline.py     High-level orchestration (Streamlit custom drug feature)
│   ├── paper_fetcher.py          PubMed Central search and PDF download
│   ├── drug_resolver.py          Drug name → canonical drug_id resolution
│   ├── schemas.py                Pydantic request/response models
│   ├── utils.py                  PDF text extraction, chunking, filename parsing
│   └── config.py                 Settings loaded from .env via pydantic-settings
│
├── run_fetch_papers.py           CLI wrapper — drives paper_fetcher.py (Phase 2)
├── run_ingestion.py              CLI wrapper — drives ingestion.py (Phase 3)
│
├── streamlit_showcase/
│   ├── streamlit_demo.py         Streamlit UI (separate deployment mode)
│   └── requirements_streamlit.txt
│
├── data/
│   ├── docs/                     Downloaded PDFs, one subfolder per drug
│   │   └── aspirin repurposing/
│   │       └── aspirin_repurposing_PMC11242460.pdf
│   ├── chroma/                   Persisted ChromaDB vector store
│   └── drugs_cache.json          On-disk drug list cache for fast startup
│
├── Extra_Docs/                   Reference guides (AWS, EC2, deployment)
├── Misc_Files/                   Test scripts and integration demos
├── requirements.txt              FastAPI deployment dependencies
└── .env                          Environment variables (gitignored)
```

`app/ingestion_pipeline.py` and `app/paper_fetcher.py` are corpus-build-only (also used by the Streamlit "custom drug" feature) and are not imported at FastAPI startup. Everything else under `app/` plus `data/chroma/`, `.env`, and `requirements.txt` is needed at runtime.

---

## 3. Corpus Build Pipeline

The corpus must be built before the API can answer any questions. It is a two-phase offline process.

```
   CSV of drug names
         │
         ▼
   ┌─────────────────────────── Phase 2 — Paper Fetching ────────────────────────────┐
   │                                                                                 │
   │   PMC eSearch  ──►  OA API  ──►  download (3 retries)  ──►  is_valid_pdf?       │
   │   '{drug}                        + exponential backoff       >5KB, %PDF-, %%EOF │
   │    repurposing'                                                  │              │
   │                                                                  │ valid        │
   │                                       invalid ◄──────────────────┤              │
   │                                          │                       ▼              │
   │                                          ▼              [ data/docs/            │
   │                                       discard            {drug} repurposing/ ]  │
   │                                                                  │              │
   │                              if S3_BUCKET set ──► [ Optional S3 mirror ]        │
   └─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────── Phase 3 — Ingestion ────────────────────────────┐
   │                                                                                │
   │   parse_filename  ──►  extract_text_from_pdf  ──►  chunk_text                  │
   │   drug_id, doc_id      PyPDF2                      1000 chars, 200 overlap     │
   │   doc_title                                              │                     │
   │                                                          ▼                     │
   │                                                  Gemini embed each chunk       │
   │                                                  (retrieval_document)          │
   │                                                          │                     │
   │                                                          ▼                     │
   │                                                  upsert_chunks                 │
   │                                                  id, text, vector, metadata    │
   │                                                          │                     │
   │                                                          ▼                     │
   │                                                  [ ChromaDB                    │
   │                                                    drug_docs collection ]      │
   └────────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2 — Paper Fetching

**Entry point:** `run_fetch_papers.py` → `app/paper_fetcher.py:PaperFetchPipeline`. Takes a CSV of drug names, queries PMC for open-access repurposing papers, downloads PDFs, and optionally mirrors them to S3.

**APIs used** (search term is always `{drug_name} repurposing`):
```
eSearch:  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
            ?db=pmc&term={drug}+repurposing&retmax={n}&retmode=json
OA API:   https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmc_id}
```
The OA response is parsed for `<link format="pdf">`; if absent, the `tgz` package is downloaded and the PDF extracted from the tarball.

**Validation (`is_valid_pdf`)**: size > 5 KB, header starts with `%PDF-`, last 10 bytes contain `%%EOF`. Invalid files are discarded.

**Output paths:**
```
data/docs/{drug_name} repurposing/{drug_name}_repurposing_PMC{id}.pdf
s3://{S3_BUCKET}/{S3_PREFIX}/{drug_name} repurposing/{filename}   # if S3_BUCKET set
```
`--s3-only` skips local writes. Per-run logs let `run_fetch_papers.py` resume interrupted runs without re-downloading.

---

### Phase 3 — Ingestion

**Entry point:** `run_ingestion.py` → `app/ingestion.py`. Reads PDFs from `data/docs/` (or S3), extracts text, chunks it, embeds each chunk with Gemini, and upserts into ChromaDB.

#### Filename parsing (`utils.parse_filename`)

Filenames follow `{drug_id}_repurposing_{source_id}.pdf`. The parser splits on `_repurposing_`, lowercases `drug_id`, and produces:

```python
DocumentInfo(
    drug_id   = "aspirin",
    doc_id    = "PMC11242460",
    doc_title = "Aspirin Repurposing PMC11242460",
    file_path = "aspirin repurposing/aspirin_repurposing_PMC11242460.pdf"
)
```

#### Text extraction (`utils.extract_text_from_pdf`)

PyPDF2 (`strict=False`) concatenates `page.extract_text()` across all pages; encrypted PDFs are decrypted with an empty password. `extract_text_from_pdf_bytes` does the same from a `BytesIO` buffer for S3 ingestion (no disk write).

#### Chunking (`utils.chunk_text`)

`chunk_size=1000` chars, `overlap=200`. The sliding window snaps to the next sentence boundary (`.`) within 100 chars past the target end, falling back to the nearest space within ±50 chars. Empty chunks are discarded. Typical output: 800–1100 chars per chunk.

#### Embedding (`ingestion.embed_texts`)

```python
genai.embed_content(
    model     = settings.gemini_embedding_model,   # configured via GEMINI_EMBEDDING_MODEL
    content   = chunk_text,
    task_type = "retrieval_document"               # asymmetric with retrieval_query
)
```

One Gemini call per chunk — no batching, so this dominates ingestion time for large corpora.

#### Metadata and upsert

```python
{
    "drug_id":      "aspirin",
    "doc_id":       "PMC11242460",
    "doc_title":    "Aspirin Repurposing PMC11242460",
    "chunk_index":  i,
    "total_chunks": N,
    "file_path":    "/abs/path/...PMC11242460.pdf",
    "drug_folder":  "aspirin repurposing",
    "source_type":  "pdf"          # or "s3" / "pdf_bytes"
}
```

Chunk IDs are `{drug_id}__{doc_id}__chunk_{i}` — deterministic, so re-ingesting the same document collides with existing IDs on `collection.add`. Clear the collection before re-ingesting.

#### Ingestion modes

| Mode | Function | Use case |
|------|----------|----------|
| Local directory | `ingest_pdfs_from_directory` | Standard corpus build from `data/docs/` |
| S3 bucket | `ingest_pdfs_from_s3` | PDF stored in S3, streamed without local disk |
| Single document (API) | `ingest_single_document` | Programmatic ingestion via `POST /ingest` |

`run_ingestion.py` also supports `--watch` mode (polls for new files on an interval), `--max-workers` for parallel processing across drugs, and `--delete-on-success` to remove local PDFs after ingestion.

---

## 4. FastAPI Runtime

### Startup Sequence

**Entry point:** `uvicorn app.main:app`

The startup sequence is managed by the `lifespan` async context manager in `app/main.py`:

1. **Load settings** — `get_settings()` returns a cached `Settings` singleton. All values come from `.env` via `pydantic-settings`. `GEMINI_API_KEY` is the only required field.

2. **Initialise ChromaDB** — `init_vector_store(settings)` creates a `chromadb.PersistentClient` pointed at `CHROMA_DB_DIR`. It calls `get_or_create_collection` with:
   - `metadata={"hnsw:space": "cosine"}` — cosine similarity for all nearest-neighbour searches
   - A `GeminiEmbeddingFunction` instance as the collection's embedding function (used by ChromaDB internally for any direct `collection.add` calls that pass text instead of pre-computed embeddings)

   On first run (empty collection), it calls the embedding function with a test string to verify the dimension matches `GEMINI_EMBEDDING_DIMENSION`. On subsequent starts this check is skipped.

3. **Configure Gemini globally** — `genai.configure(api_key=settings.gemini_api_key)` is called once. All subsequent Gemini calls in `rag.py` and `ingestion.py` inherit this configuration.

4. **Load drug cache** — see next section.

### Drug Cache

Enumerating drug IDs requires scanning all chunk metadata, which is too slow to do on every startup. A two-layer cache makes it near-instant:

- **Disk cache** (`data/drugs_cache.json`): `_load_cache_from_disk` reads `drug_ids`, `drug_aliases`, and a refresh `timestamp`. If present and not expired (`DRUGS_CACHE_TTL_SECONDS`), the API is immediately ready.
- **Background refresh**: a daemon thread runs `_refresh_drugs_cache`, pages through ChromaDB metadata (1000 at a time), rebuilds the alias map, and overwrites the JSON. The main thread is never blocked.

First run (no cache file) does a synchronous refresh before serving requests. `MAX_DRUGS_TO_LOAD > 0` stops the scan early once that many unique drugs are seen. Globals `drug_ids` (`set`) and `drug_aliases` (`dict`) are updated under `_drugs_cache_lock`.

### API Routes

All routes are mounted under the prefix `/drug_repurposing_chat` via an `APIRouter`. FastAPI generates OpenAPI documentation automatically at `/docs`.

---

#### `GET /drug_repurposing_chat/health`

Returns `{"status": "healthy"}`. Used by load balancers and the Streamlit demo's connection indicator.

---

#### `GET /drug_repurposing_chat/drugs`

Returns the list of all drug IDs currently in the vector store.

Query parameter `refresh=true` forces a synchronous cache refresh from ChromaDB before returning.

Response:
```json
{"drugs": ["apomorphine", "aspirin", "insulin"]}
```

---

#### `POST /drug_repurposing_chat/chat`

Primary chat endpoint — the caller supplies a canonical `drug_id` (string or list for multi-alias queries). `doc_id` optionally restricts retrieval to one paper; `conversation_history` enables multi-turn.

```json
{
  "session_id": "session_abc123",
  "drug_id": "aspirin",
  "message": "What evidence supports aspirin for colorectal cancer prevention?",
  "doc_id": null,
  "conversation_history": [
    {"role": "user",      "content": "What is aspirin used for?"},
    {"role": "assistant", "content": "Aspirin is a salicylate..."}
  ]
}
```

---

#### `POST /drug_repurposing_chat/chat-by-drug-name`

**The Novyra-facing endpoint.** Same as `/chat` but accepts a human-readable `drug_name` and resolves it server-side: `resolve_drug_id` → on miss, force a cache refresh and retry once → on second miss, return HTTP 404 `Unknown drug name`.

```json
{
  "session_id": "session_abc123",
  "drug_name": "Aspirin",
  "message": "What evidence supports aspirin for colorectal cancer prevention?",
  "doc_id": null,
  "conversation_history": []
}
```

---

#### `POST /drug_repurposing_chat/ingest`

Ingest a single document supplied as a string. Intended for programmatic corpus updates without running the offline pipeline.

Request body (`IngestRequest`):
```json
{
  "drug_id":    "metformin",
  "doc_id":     "PMC1234567",
  "doc_title":  "Metformin Repurposing PMC1234567",
  "content":    "Full text of the paper..."
}
```

Calls `ingest_single_document`, which chunks and embeds the content and writes to ChromaDB. Returns the number of chunks created.

---

#### `POST /drug_repurposing_chat/ingest-pdfs`

Triggers a full batch ingestion of all PDFs under `DOCS_DIR`. Equivalent to running `run_ingestion.py` but via HTTP. Returns per-file success/failure statistics.

---

## 5. RAG Query Pipeline

`app/rag.py:chat_with_documents` is the single entry point called by both `/chat` and `/chat-by-drug-name`. It executes five steps in sequence.

```
   User message + drug_id + conversation_history
         │
         ▼
   build_enhanced_query                 (fold in last 6 turns)
         │
         ▼
   embed_query                          (Gemini retrieval_query, 768-dim)
         │
         ▼
   collection.query                     (where: drug_id, n_results: top_k * 2)
         │
         ▼
   [ ChromaDB cosine ANN ]
         │
         ▼
   diversity capping                    (max 5 chunks per doc)
         │  top-k = 20 chunks
         ▼
   build_rag_prompt                     (system + history + contexts + question)
         │
         ▼
   Gemini GenerativeModel               (temp=0.1, max_tokens=2000)
         │
         ▼
   strip_context_labels
         │
         ▼
   extract_sources_from_results
         │
         ▼
   append_inline_references             (dedup PMC links)
         │
         ▼
   ChatResponse                         (answer + sources + session_id)
```

### Query Embedding

```python
genai.embed_content(
    model     = settings.gemini_embedding_model,   # configured via GEMINI_EMBEDDING_MODEL
    content   = user_message,
    task_type = "retrieval_query"
)
```

`retrieval_query` is asymmetric with the `retrieval_document` task used at ingest — pairing them improves retrieval accuracy. The returned vector is validated against `GEMINI_EMBEDDING_DIMENSION`.

### Chunk Retrieval and Diversity

```python
collection.query(
    query_embeddings = [query_vector],
    where            = {"drug_id": drug_id},   # or {"$in": [...]} for lists
    n_results        = top_k * 2,              # fetch 2× for diversity capping
    include          = ["documents", "metadatas", "distances"]
)
```

Default `top_k = 20` (so 40 candidates fetched). Distances are cosine (lower = more similar). A per-document counter then caps selection at **5 chunks per `doc_id`** before passing 20 final chunks to the LLM, so a single long paper cannot dominate the context window and answers synthesize across multiple papers.

### Prompt Construction

`build_enhanced_query` prepends the last 6 conversation turns to the current message — used only for retrieval, not in the final prompt. `build_rag_prompt` then assembles:

```
You are a helpful assistant specializing in drug repurposing research.
Use the provided context from scientific papers to answer...

[Last 4 conversation turns — user and assistant alternating]

Context 1:
{chunk_text_1}

Context 2:
{chunk_text_2}
...
Context 20:
{chunk_text_20}

Question: {user_message}

Provide a detailed and comprehensive answer based on the available scientific context:
```

Embedded instructions: synthesize across **all** contexts (not just the top one); do not output `(Context N)` labels — `strip_context_labels` removes any that slip through; for follow-ups, build on previous discussion.

### Answer Generation

```python
model = genai.GenerativeModel(settings.gemini_chat_model)   # configured via GEMINI_CHAT_MODEL
response = model.generate_content(
    prompt,
    generation_config=genai.types.GenerationConfig(
        temperature       = 0.1,    # near-deterministic, grounded in retrieved context
        max_output_tokens = 2000,
    )
)
```

### Citations

`append_inline_references` collects `doc_id`s from retrieved chunks in order, deduplicates them, and appends:

```
This response was generated from looking at the following papers:
[PMC11242460](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11242460/),
[PMC5995787](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5995787/).
```

Each `doc_id` is the PMC accession extracted at ingest. The API response's `sources` field additionally exposes the full chunk-level citation objects (title, distance, preview, file path) — see §8.

---

## 6. ChromaDB Schema

**Client type:** `chromadb.PersistentClient` — data stored on disk at `CHROMA_DB_DIR` (`./data/chroma` by default).

**Collection:** `CHROMA_COLLECTION_NAME` (`drug_docs` by default).

**Similarity metric:** cosine (`hnsw:space=cosine`).

**Embedding dimension:** set via `GEMINI_EMBEDDING_DIMENSION` and determined by the configured Gemini embedding model. This is fixed for the lifetime of the collection — switching to a model with a different output dimension requires re-ingesting into a new collection.

**Per-chunk record:**

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `id` | string | `aspirin__PMC11242460__chunk_0` | Deterministic, used for upsert |
| `document` | string | `"Aspirin has been shown to..."` | Raw chunk text, 800–1100 chars |
| `embedding` | float[768] | `[0.021, -0.003, ...]` | Stored by ChromaDB from `collection.add` |
| `drug_id` | metadata string | `"aspirin"` | Canonical, lower-case, underscored |
| `doc_id` | metadata string | `"PMC11242460"` | PMC accession number |
| `doc_title` | metadata string | `"Aspirin Repurposing PMC11242460"` | Human-readable |
| `chunk_index` | metadata int | `0` | Position within document |
| `total_chunks` | metadata int | `14` | Total chunks for this document |
| `file_path` | metadata string | `/path/to/file.pdf` | Absolute local path or S3 URI |
| `drug_folder` | metadata string | `"aspirin repurposing"` | Parent folder name |
| `source_type` | metadata string | `"pdf"` | `"pdf"`, `"s3"`, or `"pdf_bytes"` |

The `drug_id` metadata field is the primary filter used in all ChromaDB queries. Every chunk for a given drug carries the same `drug_id`, enabling O(1) metadata filtering before the ANN search.

---

## 7. Drug Resolution

`app/drug_resolver.py` converts arbitrary human input ("Aspirin", "ASPIRIN repurposing", "aspirin-repurposing") into the canonical `drug_id` stored in ChromaDB (`"aspirin"`).

**Canonicalisation (`_canonicalize`)**: strips all spaces, hyphens, underscores, the word "repurposing", and non-alphanumeric characters, then lowercases. This means `"Aspirin Repurposing"`, `"aspirin"`, and `"aspirin-repurposing"` all produce the key `"aspirin"`.

**Resolution order (`resolve_drug_id`)**:
1. Exact match: normalise input to lower-case and check against `drug_ids` set.
2. Canonical map lookup: apply `_canonicalize` and look up in `canonical_map`. If exactly one candidate drug_id maps to that key, return it.
3. Ambiguous match: if multiple candidates exist and the normalised input is one of them, return it directly.
4. Fallback: if `allow_fallback=True` (default when `drug_ids` is empty), return the normalised input as-is — ChromaDB will simply return no results if the drug is truly absent, avoiding a hard 404 for unknown drugs when the cache hasn't loaded yet.

The alias map is built from ChromaDB metadata at startup (preferred) or from filesystem scans (fallback). Three alias variants are registered per drug_id: `drug_id`, `drug_id` with underscores replaced by spaces, and `drug_id` with underscores replaced by hyphens.

---

## 8. Request and Response Schemas

All schemas are defined in `app/schemas.py` and enforced by FastAPI's Pydantic validation.

### ChatRequest
```
session_id           str       Caller-supplied session identifier
drug_id              str|list  Canonical drug ID(s) to filter ChromaDB
message              str       User's question
doc_id               str?      Optional: restrict to one paper
conversation_history Message[] Previous turns for multi-turn context
```

### ChatByDrugNameRequest
```
session_id           str       Caller-supplied session identifier
drug_name            str       Human-readable drug name (resolved server-side)
message              str       User's question
doc_id               str?      Optional: restrict to one paper
conversation_history Message[] Previous turns for multi-turn context
```

### Message
```
role     str   "user" or "assistant"
content  str   Message text
```

### ChatResponse
```
answer      str      Full generated answer with inline PMC citations
sources     Source[] Chunks used; includes doc_id, title, distance, preview
session_id  str      Echoed from request
```

### Source
```
doc_id        str    PMC accession number (e.g. "PMC11242460")
doc_title     str    Human-readable title
chunk_id      str    Full chunk ID (e.g. "aspirin__PMC11242460__chunk_3")
distance      float  Cosine distance (lower = more similar)
text_preview  str    First 200 characters of the chunk
file_path     str?   Local path or S3 URI to the source PDF
```

### IngestRequest
```
drug_id    str   Canonical drug identifier
doc_id     str   Document identifier
doc_title  str   Human-readable title
content    str   Full document text
```

### IngestResponse
```
message        str   "Document ingested successfully"
chunks_created int   Number of chunks stored
drug_id        str
doc_id         str
```

---

## 9. Configuration Reference

All values are loaded from `.env` by `app/config.py` using `pydantic-settings`. The only required variable is `GEMINI_API_KEY`.

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — (required) | Google Gemini API key |
| `GEMINI_EMBEDDING_MODEL` | — (required) | Gemini embedding model used for both ingestion and query |
| `GEMINI_EMBEDDING_DIMENSION` | — (required) | Expected vector dimension; must match the embedding model above |
| `GEMINI_CHAT_MODEL` | — (required) | Gemini generative model used to produce answers |
| `CHROMA_DB_DIR` | `./data/chroma` | Directory where ChromaDB persists its index |
| `CHROMA_COLLECTION_NAME` | `drug_docs` | ChromaDB collection name |
| `DOCS_DIR` | `./data/docs` | Root directory of per-drug PDF subfolders |
| `DRUGS_CACHE_TTL_SECONDS` | `300` | Seconds before the disk drug cache is considered stale; `0` = never expires |
| `MAX_DRUGS_TO_LOAD` | `0` | Cap on unique drugs scanned at startup; `0` = unlimited |
| `S3_BUCKET` | unset | S3 bucket for PDF storage; leave unset to disable S3 |
| `S3_PREFIX` | `""` | Key prefix inside the S3 bucket |
| `S3_REGION` | unset | AWS region for the S3 bucket |

Changing the embedding model (or its dimension) after the collection is built causes dimension-mismatch errors at query time — the collection must be deleted and rebuilt.

---

## 10. Running the System

### Prerequisites

```bash
pip install -r requirements.txt
```

Copy `.env` and set `GEMINI_API_KEY` at minimum.

### Build the corpus (one time)

**Phase 2 — fetch papers from PubMed:**
```bash
python run_fetch_papers.py --csv drugs.csv --max-papers 10
```
PDFs are saved to `data/docs/{drug_name} repurposing/`.

**Phase 3 — ingest PDFs into ChromaDB:**
```bash
python run_ingestion.py
```
Reads `DOCS_DIR` from `.env`, processes all drug subfolders, and writes chunks to `data/chroma/`.

Run `python run_ingestion.py --help` and `python run_fetch_papers.py --help` for the full list of options (S3 mode, parallelism, watch mode, etc.).

### Start the FastAPI server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server is ready when the log shows `Startup complete with N drugs`.

Interactive API docs: `http://localhost:8000/docs`

### Example request (Novyra integration)

```bash
curl -X POST http://localhost:8000/drug_repurposing_chat/chat-by-drug-name \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "novyra_session_001",
    "drug_name": "aspirin",
    "message": "What clinical evidence supports aspirin for colorectal cancer prevention?",
    "conversation_history": []
  }'
```

Response:
```json
{
  "answer": "Multiple randomised controlled trials have demonstrated...\n\nThis response was generated from looking at the following papers: [PMC11242460](...), [PMC9876543](...).",
  "sources": [
    {
      "doc_id": "PMC11242460",
      "doc_title": "Aspirin Repurposing PMC11242460",
      "chunk_id": "aspirin__PMC11242460__chunk_4",
      "distance": 0.182,
      "text_preview": "A meta-analysis of 24 randomised trials found that...",
      "file_path": "/data/docs/aspirin repurposing/aspirin_repurposing_PMC11242460.pdf"
    }
  ],
  "session_id": "novyra_session_001"
}
```

### Reset the vector store

```bash
rm -rf data/chroma data/drugs_cache.json
python run_ingestion.py
```

The server must be restarted after a reset.

---

## License

MIT License — see [LICENSE](LICENSE).
