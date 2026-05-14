"""
FastAPI application entry point for the Drug Repurposing Chat API.

Startup sequence (managed by the lifespan context manager):
1. Load settings from environment variables.
2. Initialise the ChromaDB vector store.
3. Configure the Gemini client globally.
4. Attempt to load the drug list from the on-disk JSON cache for instant
   startup; fall back to a synchronous ChromaDB scan on first run.
5. Launch a background thread to refresh the cache without blocking requests.

API routes are mounted under the /drug_repurposing_chat prefix via an
APIRouter so the module can be composed with other routers in larger apps.
"""

from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import time
import threading
from datetime import datetime
import json
from pathlib import Path

from .config import Settings, get_settings
from .vector_store import init_vector_store
from .rag import chat_with_documents
from .ingestion import ingest_single_document, ingest_pdfs_from_directory
from .drug_resolver import build_drug_lookup, resolve_drug_id, build_drug_lookup_from_metadatas
from .schemas import (
    ChatRequest, ChatByDrugNameRequest, ChatResponse, IngestRequest, IngestResponse,
    HealthResponse, IngestStatusResponse
)
import google.generativeai as genai # Import genai for global configuration

# Global variables for lifespan management
collection = None
settings = None
drug_ids = set()
drug_aliases = {}
_drugs_cache: dict[str, object] = {
    "drugs": [],
    "timestamp": 0.0,
    "drug_ids": set(),
    "drug_aliases": {},
    "loaded_from_disk": False,
}
_drugs_cache_lock = threading.Lock()
_background_refresh_thread = None

# Cache file path (relative to project root)
CACHE_FILE_PATH = Path("data/drugs_cache.json")


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def _load_cache_from_disk() -> bool:
    """Load drug cache from disk file if it exists. Returns True if loaded successfully."""
    global drug_ids, drug_aliases

    if not CACHE_FILE_PATH.exists():
        _log("No cache file found on disk")
        return False

    try:
        with open(CACHE_FILE_PATH, 'r') as f:
            data = json.load(f)

        drugs_list = data.get("drugs", [])
        timestamp = data.get("timestamp", 0.0)

        # Reconstruct drug_ids and drug_aliases from saved data
        drug_ids_list = data.get("drug_ids", [])
        drug_aliases_dict = data.get("drug_aliases", {})

        # Convert back to proper types
        drug_ids = set(drug_ids_list)
        drug_aliases = {k: set(v) for k, v in drug_aliases_dict.items()}

        _drugs_cache["drugs"] = drugs_list
        _drugs_cache["timestamp"] = timestamp
        _drugs_cache["drug_ids"] = drug_ids
        _drugs_cache["drug_aliases"] = drug_aliases
        _drugs_cache["loaded_from_disk"] = True

        age_seconds = time.time() - timestamp
        _log(f"✅ Loaded {len(drugs_list)} drugs from cache file (age: {age_seconds:.0f}s)")
        return True

    except Exception as e:
        _log(f"⚠️  Failed to load cache from disk: {e}")
        return False


def _save_cache_to_disk() -> None:
    """Save drug cache to disk file."""
    try:
        CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON serialization
        data = {
            "drugs": _drugs_cache.get("drugs", []),
            "timestamp": _drugs_cache.get("timestamp", time.time()),
            "drug_ids": list(_drugs_cache.get("drug_ids", set())),
            "drug_aliases": {
                k: list(v) for k, v in _drugs_cache.get("drug_aliases", {}).items()
            }
        }

        with open(CACHE_FILE_PATH, 'w') as f:
            json.dump(data, f, indent=2)

        _log(f"💾 Saved cache to disk: {len(data['drugs'])} drugs")

    except Exception as e:
        _log(f"⚠️  Failed to save cache to disk: {e}")


def _load_collection_metadatas(collection, max_drugs: int = 0) -> list[dict]:
    """
    Load metadata from collection, optionally stopping after finding max_drugs unique drugs.

    Args:
        collection: ChromaDB collection
        max_drugs: If > 0, stop after finding this many unique drugs (0 = load all)

    Returns:
        List of metadata dictionaries
    """
    try:
        total = collection.count()
    except Exception:
        return []

    if total <= 0:
        return []

    page_size = 1000
    offset = 0
    metadatas = []
    unique_drugs = set()

    _log(f"Loading metadata from {total} chunks (max_drugs={max_drugs or 'unlimited'})...")

    while offset < total:
        batch = collection.get(
            limit=page_size,
            offset=offset,
            include=["metadatas"]
        )
        batch_metas = batch.get("metadatas") or []

        for meta in batch_metas:
            if isinstance(meta, dict):
                metadatas.append(meta)
                # Track unique drugs if limit is set
                if max_drugs > 0:
                    drug_id = (meta.get("drug_id") or "").strip().lower()
                    if drug_id:
                        unique_drugs.add(drug_id)

        offset += page_size

        # Early exit if we've found enough unique drugs
        if max_drugs > 0 and len(unique_drugs) >= max_drugs:
            _log(f"✅ Found {len(unique_drugs)} unique drugs after scanning {len(metadatas)} chunks - stopping early")
            break

        # Progress logging for large collections
        if offset % 10000 == 0:
            _log(f"  Scanned {offset}/{total} chunks... (found {len(unique_drugs)} unique drugs so far)")

    return metadatas


def _refresh_drugs_cache(settings: Settings, save_to_disk: bool = True) -> list[str]:
    """
    Refresh drug cache from Chroma metadata (slow operation).
    Optionally saves to disk after refresh.
    """
    global collection, drug_ids, drug_aliases
    if collection is None:
        _log("Drug cache refresh skipped: collection not initialized")
        return []

    try:
        max_drugs = settings.max_drugs_to_load
        if max_drugs > 0:
            _log(f"🔄 Refreshing drug cache (limited to {max_drugs} drugs for faster startup)")
        else:
            _log("🔄 Refreshing drug cache from Chroma metadata (this may take a while...)")

        start_time = time.time()

        metadatas = _load_collection_metadatas(collection, max_drugs=max_drugs)
        if metadatas:
            drug_ids, drug_aliases = build_drug_lookup_from_metadatas(metadatas)
            elapsed = time.time() - start_time
            _log(f"✅ Loaded {len(drug_ids)} drugs from Chroma metadata in {elapsed:.1f}s")
        else:
            # Fallback to local docs only if no metadata exists yet.
            _log("No Chroma metadata found; falling back to local docs")
            local_ids, local_aliases = build_drug_lookup(settings.docs_dir)
            drug_ids = local_ids
            drug_aliases = local_aliases
            elapsed = time.time() - start_time
            _log(f"✅ Loaded {len(drug_ids)} drugs from local docs in {elapsed:.1f}s")
    except Exception as exc:
        _log(f"❌ Drug cache refresh failed: {exc}")
        # Keep existing cache on failure
        return list(_drugs_cache.get("drugs", []))

    drugs = sorted(drug_ids) if drug_ids else []

    with _drugs_cache_lock:
        _drugs_cache["drugs"] = drugs
        _drugs_cache["timestamp"] = time.time()
        _drugs_cache["drug_ids"] = set(drug_ids)
        _drugs_cache["drug_aliases"] = dict(drug_aliases)

        if save_to_disk:
            _save_cache_to_disk()

    return drugs


def _get_drugs_cached(settings: Settings, refresh: bool = False) -> list[str]:
    """
    Get cached drug list. Uses disk cache if available, otherwise triggers refresh.
    """
    ttl = max(0, settings.drugs_cache_ttl_seconds)
    now = time.time()

    with _drugs_cache_lock:
        cached = _drugs_cache.get("drugs", [])
        last_refresh = _drugs_cache.get("timestamp", 0.0) or 0.0
        loaded_from_disk = _drugs_cache.get("loaded_from_disk", False)

        # If loaded from disk and not expired, return it
        if loaded_from_disk and cached and not refresh and (ttl == 0 or now - last_refresh <= ttl):
            return list(cached)

        if refresh:
            _log("Forced drug cache refresh requested")

        if refresh or not cached or (ttl == 0) or (now - last_refresh > ttl):
            return _refresh_drugs_cache(settings, save_to_disk=True)

        return list(cached)


def _background_refresh_worker(settings: Settings):
    """Background thread worker to refresh drug cache without blocking startup."""
    try:
        _log("🔄 Starting background drug cache refresh...")
        _refresh_drugs_cache(settings, save_to_disk=True)
        _log("✅ Background cache refresh completed")
    except Exception as e:
        _log(f"❌ Background cache refresh failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global collection, settings, drug_ids, drug_aliases, _background_refresh_thread

    # Startup
    _log("🚀 Starting Drug Repurposing Chat API...")
    settings = get_settings()
    collection = init_vector_store(settings)
    genai.configure(api_key=settings.gemini_api_key) # Configure Gemini client globally

    _log(f"📁 Vector store: {settings.chroma_db_dir}")
    _log(f"📚 Collection: {settings.chroma_collection_name}")
    _log(f"📄 PDF source: {settings.docs_dir}")

    # Try to load cache from disk (instant startup!)
    cache_loaded = _load_cache_from_disk()

    if cache_loaded:
        _log(f"✅ Startup complete with {len(drug_ids)} drugs from cache")
        # Start background refresh to update cache
        _background_refresh_thread = threading.Thread(
            target=_background_refresh_worker,
            args=(settings,),
            daemon=True
        )
        _background_refresh_thread.start()
        _log("📡 Background cache refresh started (non-blocking)")
    else:
        # No cache file - do initial refresh (only happens once)
        _log("⚠️  No cache file found - performing initial drug cache refresh...")
        _refresh_drugs_cache(settings, save_to_disk=True)
        _log(f"✅ Startup complete with {len(drug_ids)} drugs")

    yield

    # Shutdown
    _log("Shutting down application")


app = FastAPI(
    title="Drug Repurposing Chat API",
    description="Chat with drug repurposing research papers using RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan
)

router = APIRouter(prefix="/drug_repurposing_chat")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Chat with drug repurposing documents using RAG.

    - Retrieves relevant chunks from PDFs filtered by drug_id (and optionally doc_id)
    - Uses RAG to generate an answer based on the retrieved context
    - Returns answer with source citations from research papers
    """
    global collection

    if collection is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        # Convert conversation history to dict format
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        result = chat_with_documents(
            session_id=request.session_id,
            drug_id=request.drug_id,
            message=request.message,
            collection=collection,
            settings=settings,
            doc_id=request.doc_id,
            conversation_history=conversation_history
        )

        return ChatResponse(**result)

    except Exception as e:
        error_msg = str(e)
        # DEMO GUARANTEE: Handle dimension mismatch gracefully
        if "Embedding dimension" in error_msg and "does not match collection dimensionality" in error_msg:
            raise HTTPException(
                status_code=500,
                detail="System configuration error: Embedding dimensions don't match. Please restart the server."
            )
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {error_msg}")


@router.post("/chat-by-drug-name", response_model=ChatResponse)
async def chat_by_drug_name_endpoint(
    request: ChatByDrugNameRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Chat with drug repurposing documents using a human-readable drug name.
    """
    global collection, drug_ids, drug_aliases

    if collection is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        drug_id = resolve_drug_id(
            request.drug_name,
            drug_ids,
            drug_aliases,
            allow_fallback=not drug_ids
        )
        if not drug_id:
            # Refresh from Chroma metadata in case new drugs were ingested.
            _get_drugs_cached(settings, refresh=True)
            drug_id = resolve_drug_id(
                request.drug_name,
                drug_ids,
                drug_aliases,
                allow_fallback=not drug_ids
            )
        if not drug_id:
            _log(f"Drug name not found: {request.drug_name}")
            raise HTTPException(
                status_code=404,
                detail="Unknown drug name. Use /drugs or /chat with a drug_id."
            )

        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        result = chat_with_documents(
            session_id=request.session_id,
            drug_id=drug_id,
            message=request.message,
            collection=collection,
            settings=settings,
            doc_id=request.doc_id,
            conversation_history=conversation_history
        )

        return ChatResponse(**result)

    except Exception as e:
        error_msg = str(e)
        if "Embedding dimension" in error_msg and "does not match collection dimensionality" in error_msg:
            raise HTTPException(
                status_code=500,
                detail="System configuration error: Embedding dimensions don't match. Please restart the server."
            )
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {error_msg}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Ingest a single document programmatically.

    - Chunks the document content
    - Generates embeddings
    - Stores chunks in vector database
    """
    global collection

    if collection is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        result = ingest_single_document(
            drug_id=request.drug_id,
            doc_id=request.doc_id,
            doc_title=request.doc_title,
            content=request.content,
            settings=settings,
            collection=collection
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return IngestResponse(
            message="Document ingested successfully",
            chunks_created=result["chunks_created"],
            drug_id=result["drug_id"],
            doc_id=result["doc_id"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")


@router.post("/ingest-pdfs", response_model=IngestStatusResponse)
async def ingest_pdfs(
    settings: Settings = Depends(get_settings)
):
    """
    Ingest all PDFs from the drug repurposing directory.

    Processes PDFs from subfolders organized by drug:
    - aspirin repurposing/
    - apomorphine repurposing/
    - insulin repurposing/
    """
    global collection

    if collection is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        result = ingest_pdfs_from_directory(settings.docs_dir, settings, collection)

        return IngestStatusResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")


@router.get("/drugs")
async def list_drugs(
    settings: Settings = Depends(get_settings),
    refresh: bool = False,
):
    """
    List all available drugs in the system.
    """
    global collection
    if collection is None:
        _log("Drug list requested but collection not initialized")
        return {"drugs": []}

    drugs = _get_drugs_cached(settings, refresh=refresh)
    return {"drugs": drugs}


# @app.on_event("startup") is deprecated in FastAPI ≥ 0.93 in favour of the
# lifespan context manager above. Kept here only as a fallback hint for older
# deployments; all real startup logic lives in lifespan().
@app.on_event("startup")
async def startup_event():
    _log("Drug Repurposing Chat API starting up...")
    _log("Make sure GEMINI_API_KEY is set in your .env file")


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        # reload=True # Removed for more stable debugging
    )
