from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import time
import threading

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
}
_drugs_cache_lock = threading.Lock()


def _load_collection_metadatas(collection) -> list[dict]:
    try:
        total = collection.count()
    except Exception:
        return []

    if total <= 0:
        return []

    page_size = 1000
    offset = 0
    metadatas = []

    while offset < total:
        batch = collection.get(
            limit=page_size,
            offset=offset,
            include=["metadatas"]
        )
        batch_metas = batch.get("metadatas") or []
        metadatas.extend([meta for meta in batch_metas if isinstance(meta, dict)])
        offset += page_size

    return metadatas


def _refresh_drugs_cache(settings: Settings) -> list[str]:
    global collection, drug_ids, drug_aliases
    if collection is None:
        return []

    metadatas = _load_collection_metadatas(collection)
    if metadatas:
        drug_ids, drug_aliases = build_drug_lookup_from_metadatas(metadatas)
    else:
        # Fallback to local docs only if no metadata exists yet.
        local_ids, local_aliases = build_drug_lookup(settings.docs_dir)
        drug_ids = local_ids
        drug_aliases = local_aliases

    drugs = sorted(drug_ids) if drug_ids else []

    _drugs_cache["drugs"] = drugs
    _drugs_cache["timestamp"] = time.time()
    _drugs_cache["drug_ids"] = set(drug_ids)
    _drugs_cache["drug_aliases"] = dict(drug_aliases)
    return drugs


def _get_drugs_cached(settings: Settings, refresh: bool = False) -> list[str]:
    ttl = max(0, settings.drugs_cache_ttl_seconds)
    now = time.time()
    with _drugs_cache_lock:
        cached = _drugs_cache.get("drugs", [])
        last_refresh = _drugs_cache.get("timestamp", 0.0) or 0.0

        if refresh or not cached or (ttl == 0) or (now - last_refresh > ttl):
            return _refresh_drugs_cache(settings)

        return list(cached)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global collection, settings, drug_ids, drug_aliases

    # Startup
    settings = get_settings()
    collection = init_vector_store(settings)
    genai.configure(api_key=settings.gemini_api_key) # Configure Gemini client globally
    _refresh_drugs_cache(settings)
    print(f"Initialized vector store at {settings.chroma_db_dir}")
    print(f"Collection: {settings.chroma_collection_name}")
    print(f"PDF source directory: {settings.docs_dir}")
    print(f"Resolved {len(drug_ids)} drug ids for name lookup")

    yield

    # Shutdown
    print("Shutting down application")


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
        return {"drugs": []}

    drugs = _get_drugs_cached(settings, refresh=refresh)
    return {"drugs": drugs}


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("Drug Repurposing Chat API starting up...")
    print("Make sure to set your GEMINI_API_KEY in .env file")


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        # reload=True # Removed for more stable debugging
    )
