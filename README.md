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

The system has two distinct phases of operation:

**Corpus build** (run once, offline): research papers are fetched from PubMed Central, extracted, chunked, embedded, and stored in a ChromaDB vector database on disk.

**Runtime** (the Novyra-facing API): the FastAPI server loads the pre-built vector store, resolves incoming drug names to canonical IDs, embeds each user query, retrieves relevant chunks across multiple papers, and generates a synthesized answer via Gemini. The corpus does not change at query time.

```
CORPUS BUILD (offline)                     RUNTIME (FastAPI endpoint)
─────────────────────                      ─────────────────────────
PubMed Central                             POST /drug_repurposing_chat/chat-by-drug-name
     │                                              │
     ▼                                              ▼
paper_fetcher.py       →   data/docs/       drug_resolver.py  →  canonical drug_id
     │                                              │
     ▼                                              ▼
ingestion.py           →   data/chroma/     rag.py  →  embed query  →  ChromaDB query
                                                    │
                                                    ▼
                                             Gemini GenerativeModel
                                                    │
                                                    ▼
                                             ChatResponse + PMC citations
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

**Files needed at FastAPI runtime only** (corpus already built):
`app/main.py`, `app/config.py`, `app/vector_store.py`, `app/rag.py`, `app/schemas.py`, `app/utils.py`, `app/ingestion.py`, `app/drug_resolver.py`, `data/chroma/`, `.env`, `requirements.txt`

`app/ingestion_pipeline.py` and `app/paper_fetcher.py` are only used during corpus build and by the Streamlit "custom drug" feature. They are not imported at FastAPI startup.

---

## 3. Corpus Build Pipeline

The corpus must be built before the API can answer any questions. It is a two-phase offline process.

### Phase 2 — Paper Fetching

**Entry point:** `run_fetch_papers.py`  
**Core logic:** `app/paper_fetcher.py` — `PaperFetchPipeline`

Takes a CSV of drug names, queries PubMed Central for open-access repurposing papers, downloads the PDFs, and optionally mirrors them to S3.

#### Step-by-step

**1. PMC Search (`PaperFetchPipeline.search_pmc`)**

Queries the NCBI eSearch API:
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
  ?db=pmc&term={drug_name}+repurposing&retmax={max_papers}&retmode=json
```
Returns a list of PMC IDs (integers). The search term is always `{drug_name} repurposing`.

**2. Open-Access check (`fetch_oa_pdf_url`)**

For each PMC ID, queries the PMC Open Access API to get the FTP package URL:
```
https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmc_id}
```
Parses the XML response for a `<link format="pdf">` or `<link format="tgz">` element. If a direct PDF link is available it is preferred; otherwise the TAR/GZIP archive is downloaded and the PDF extracted from it.

**3. Download and validate**

PDFs are downloaded via `urllib.request` with retry logic (3 attempts, exponential backoff). After download, `is_valid_pdf` checks:
- File size > 5 KB
- First 5 bytes equal `%PDF-`
- Last 10 bytes contain `%%EOF`

Invalid files are discarded. Valid PDFs are saved to:
```
data/docs/{drug_name} repurposing/{drug_name}_repurposing_PMC{id}.pdf
```

**4. Optional S3 upload**

If `S3_BUCKET` is set in `.env`, each validated PDF is uploaded to:
```
s3://{S3_BUCKET}/{S3_PREFIX}/{drug_name} repurposing/{filename}
```
With `--s3-only`, PDFs are not written to local disk.

**5. Resume and logging**

`run_fetch_papers.py` writes per-run logs tracking which drugs have been processed, so interrupted runs can be resumed without re-downloading.

---

### Phase 3 — Ingestion

**Entry point:** `run_ingestion.py`  
**Core logic:** `app/ingestion.py`

Reads every PDF from `data/docs/` (or S3), extracts text, splits into overlapping chunks, embeds each chunk with Gemini, and upserts into ChromaDB.

#### Step 1 — Filename parsing (`app/utils.py:parse_filename`)

Before any content is processed, the filename is parsed to extract identifiers:

```
aspirin_repurposing_PMC11242460.pdf
     │                   │
  drug_id             doc_id
 "aspirin"          "PMC11242460"
```

The filename format is `{drug_id}_repurposing_{source_id}.pdf`. The parser splits on `_repurposing_` and normalises `drug_id` to lower-case with underscores. This produces a `DocumentInfo` namedtuple:

```python
DocumentInfo(
    drug_id   = "aspirin",
    doc_id    = "PMC11242460",
    doc_title = "Aspirin Repurposing PMC11242460",
    file_path = "aspirin repurposing/aspirin_repurposing_PMC11242460.pdf"
)
```

#### Step 2 — Text extraction (`app/utils.py:extract_text_from_pdf`)

Uses **PyPDF2** (`PdfReader`, `strict=False`) to iterate over every page and concatenate `page.extract_text()` output. Encrypted PDFs are decrypted with an empty password. PyPDF2 warnings are suppressed at the logger level and only printed if they occur during extraction. The result is a single string of all page text, stripped of leading/trailing whitespace.

For S3 ingestion, `extract_text_from_pdf_bytes` does the same from an in-memory `BytesIO` buffer, avoiding any disk write.

#### Step 3 — Chunking (`app/utils.py:chunk_text`)

Default parameters: `chunk_size=1000` characters, `overlap=200` characters.

The chunker slides a window across the text. At each step:
1. Tries to extend to the next sentence boundary (`.`) within 100 characters past the target end.
2. If no sentence boundary is found, falls back to the nearest word boundary (space) within ±50 characters.
3. Advances the start pointer by `chunk_size - overlap` to create the overlap.

This produces chunks of approximately 800–1100 characters that preserve sentence integrity at boundaries. Empty chunks are discarded.

#### Step 4 — Embedding (`app/ingestion.py:embed_texts`)

Each chunk text is embedded individually via the Gemini API:

```python
genai.embed_content(
    model  = settings.gemini_embedding_model,   # default: "models/embedding-001"
    content = chunk_text,
    task_type = "retrieval_document"             # optimised for storage-side embedding
)
```

The result is a 768-dimensional float vector. The `retrieval_document` task type instructs Gemini to optimise the embedding for later retrieval — this must be paired with `retrieval_query` on the query side.

There is no batching: each chunk makes one Gemini API call. For large corpora this is the dominant time cost.

#### Step 5 — Metadata construction and upsert

For each chunk `i` in a document, a metadata record is built:

```python
{
    "drug_id":      "aspirin",
    "doc_id":       "PMC11242460",
    "doc_title":    "Aspirin Repurposing PMC11242460",
    "chunk_index":  i,
    "total_chunks": N,
    "file_path":    "/absolute/path/to/aspirin_repurposing_PMC11242460.pdf",
    "drug_folder":  "aspirin repurposing",
    "source_type":  "pdf"          # or "s3" / "pdf_bytes"
}
```

The chunk's unique ID is:
```
aspirin__PMC11242460__chunk_0
aspirin__PMC11242460__chunk_1
...
```

The format is `{drug_id}__{doc_id}__chunk_{i}`. This ID is deterministic: re-ingesting the same document produces the same IDs. ChromaDB's `collection.add` raises on duplicate IDs, so the collection must be cleared before re-ingesting an existing document.

Chunks, metadata, and IDs are passed to `upsert_chunks`, which calls `collection.add`. ChromaDB handles the embedding storage internally alongside the text and metadata.

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

The drug list (which drug IDs exist in ChromaDB) is expensive to compute: it requires scanning all chunk metadata. The system uses a two-layer cache to make startup near-instant:

**Layer 1 — disk cache** (`data/drugs_cache.json`): on startup, `_load_cache_from_disk` reads this JSON file if it exists. The file stores `drug_ids`, `drug_aliases`, a `drugs` list, and the `timestamp` of the last refresh. If the file is present and not expired (TTL controlled by `DRUGS_CACHE_TTL_SECONDS`), the API is immediately ready to serve drug lists and resolve names.

**Layer 2 — background refresh**: after loading from disk, a daemon thread runs `_refresh_drugs_cache` asynchronously. This scans all ChromaDB metadata in pages of 1000, rebuilds the canonical alias map, and overwrites `data/drugs_cache.json`. The main thread is not blocked.

On first run (no cache file), the synchronous refresh runs at startup before any requests are served, then saves to disk. All subsequent startups are fast.

If `MAX_DRUGS_TO_LOAD` is set to a non-zero value, the metadata scan stops early once that many unique drugs have been seen, further reducing startup time for large collections.

Drug IDs and their aliases are stored in two module-level globals: `drug_ids` (a `set`) and `drug_aliases` (a `dict`). These are updated in-place by the background refresh and protected by `_drugs_cache_lock`.

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

The primary chat endpoint. Accepts a `drug_id` (string or list of strings) directly — the caller is responsible for supplying the canonical ID.

Request body (`ChatRequest`):
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

`drug_id` can be a list (e.g. `["aspirin", "aspirin_repurposing"]`) to query across multiple alias IDs simultaneously. `doc_id` optionally restricts retrieval to a single paper. `conversation_history` is passed into the RAG prompt to support multi-turn conversations.

---

#### `POST /drug_repurposing_chat/chat-by-drug-name`

The Novyra-facing endpoint. Accepts a human-readable `drug_name` and performs resolution internally before calling the RAG pipeline. This is the endpoint Novyra should call.

Request body (`ChatByDrugNameRequest`):
```json
{
  "session_id": "session_abc123",
  "drug_name": "Aspirin",
  "message": "What evidence supports aspirin for colorectal cancer prevention?",
  "doc_id": null,
  "conversation_history": []
}
```

Resolution flow:
1. `resolve_drug_id("Aspirin", drug_ids, drug_aliases)` → `"aspirin"`
2. If not found, forces a cache refresh and retries resolution once.
3. If still not found, returns HTTP 404 with `"Unknown drug name"`.

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

### Query Embedding

```python
genai.embed_content(
    model     = settings.gemini_embedding_model,   # "models/embedding-001"
    content   = user_message,
    task_type = "retrieval_query"                  # query-side task type
)
```

The embedding is 768-dimensional. The `retrieval_query` task type produces a query-optimised vector that is asymmetrically paired with the `retrieval_document` vectors stored during ingestion. This asymmetric embedding is a feature of the Gemini embedding model and improves retrieval accuracy compared to embedding both sides identically.

The returned vector is validated against `GEMINI_EMBEDDING_DIMENSION` and raises a `ValueError` on mismatch.

### Chunk Retrieval and Diversity

ChromaDB is queried with:

```python
collection.query(
    query_embeddings = [query_vector],
    where            = {"drug_id": drug_id},        # metadata filter
    n_results        = top_k * 2,                   # fetch 2× for diversity capping
    include          = ["documents", "metadatas", "distances"]
)
```

The default `top_k` is 20, so 40 candidates are fetched. Distances are cosine distances (lower = more similar, since the collection uses `hnsw:space=cosine`).

When `drug_id` is a list, the filter becomes `{"drug_id": {"$in": drug_id}}`, querying across all named aliases simultaneously.

**Diversity capping**: a per-document counter limits how many chunks from the same paper can be selected. The cap is 5 chunks per `doc_id`. This prevents a single long paper from dominating the context window when multiple papers exist for a drug. After diversity filtering, `top_k` (20) chunks are passed to the LLM.

The effect is that answers synthesize evidence across multiple papers rather than paraphrasing one paper repeatedly.

### Prompt Construction

`build_enhanced_query` prepends the last 6 conversation turns to the current message to form a context-aware query string. This enriched string is used only for retrieval — not verbatim in the final prompt.

`build_rag_prompt` assembles the LLM prompt:

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

Key instructions embedded in the prompt:
- Synthesize across **all** provided contexts, not just the most relevant one.
- Do not label answers with "Context N" references (these are stripped post-generation by `strip_context_labels`).
- For follow-up questions, build on previous discussion while adding new detail.

### Answer Generation

```python
model = genai.GenerativeModel(settings.gemini_chat_model)   # "models/gemini-2.0-flash-exp"
response = model.generate_content(
    prompt,
    generation_config=genai.types.GenerationConfig(
        temperature       = 0.1,    # near-deterministic for factual accuracy
        max_output_tokens = 2000,
    )
)
```

Temperature 0.1 keeps responses grounded in the retrieved context rather than generating plausible-sounding but unsupported claims.

After generation, `strip_context_labels` removes any parenthetical `(Context N, M)` strings the model may have produced despite the instruction.

### Citations

`append_inline_references` appends a deduplicated list of PMC citation links at the end of the answer:

```
This response was generated from looking at the following papers:
[PMC11242460](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11242460/),
[PMC5995787](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5995787/).
```

`doc_id` values from all retrieved chunks are collected in retrieval order, deduplicated while preserving order, and formatted as markdown links. Each `doc_id` is the PMC accession number extracted from the original filename during ingestion.

The `sources` field in the API response contains the full `Source` list with `doc_id`, `doc_title`, `chunk_id`, cosine `distance`, a 200-character `text_preview`, and the original `file_path`.

---

## 6. ChromaDB Schema

**Client type:** `chromadb.PersistentClient` — data stored on disk at `CHROMA_DB_DIR` (`./data/chroma` by default).

**Collection:** `CHROMA_COLLECTION_NAME` (`drug_docs` by default).

**Similarity metric:** cosine (`hnsw:space=cosine`).

**Embedding dimension:** 768 (Gemini `models/embedding-001`). This is fixed for the lifetime of the collection and must match `GEMINI_EMBEDDING_DIMENSION`.

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
| `GEMINI_EMBEDDING_MODEL` | `models/embedding-001` | Gemini embedding model used for both ingestion and query |
| `GEMINI_EMBEDDING_DIMENSION` | `768` | Expected vector dimension; must match the model above |
| `GEMINI_CHAT_MODEL` | `models/gemini-2.0-flash-exp` | Gemini generative model used to produce answers |
| `CHROMA_DB_DIR` | `./data/chroma` | Directory where ChromaDB persists its index |
| `CHROMA_COLLECTION_NAME` | `drug_docs` | ChromaDB collection name |
| `DOCS_DIR` | `./data/docs` | Root directory of per-drug PDF subfolders |
| `DRUGS_CACHE_TTL_SECONDS` | `300` | Seconds before the disk drug cache is considered stale; `0` = never expires |
| `MAX_DRUGS_TO_LOAD` | `0` | Cap on unique drugs scanned at startup; `0` = unlimited |
| `S3_BUCKET` | unset | S3 bucket for PDF storage; leave unset to disable S3 |
| `S3_PREFIX` | `""` | Key prefix inside the S3 bucket |
| `S3_REGION` | unset | AWS region for the S3 bucket |

Changing `GEMINI_EMBEDDING_MODEL` or `GEMINI_EMBEDDING_DIMENSION` after a collection has been built will cause dimension mismatch errors. The collection must be deleted and rebuilt from scratch if the embedding model is changed.

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
