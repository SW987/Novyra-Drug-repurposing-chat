from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from .config import Settings, get_settings
from .vector_store import init_vector_store
from .rag import chat_with_documents
from .ingestion import ingest_single_document, ingest_pdfs_from_directory
from .schemas import (
    ChatRequest, ChatResponse, IngestRequest, IngestResponse,
    HealthResponse, IngestStatusResponse
)
import google.generativeai as genai # Import genai for global configuration

# Global variables for lifespan management
collection = None
settings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global collection, settings

    # Startup
    settings = get_settings()
    collection = init_vector_store(settings)
    genai.configure(api_key=settings.gemini_api_key) # Configure Gemini client globally
    print(f"Initialized vector store at {settings.chroma_db_dir}")
    print(f"Collection: {settings.chroma_collection_name}")
    print(f"PDF source directory: {settings.docs_dir}")

    yield

    # Shutdown
    print("Shutting down application")


app = FastAPI(
    title="Drug Repurposing Chat API",
    description="Chat with drug repurposing research papers using RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.post("/chat", response_model=ChatResponse)
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


@app.post("/ingest", response_model=IngestResponse)
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


@app.post("/ingest-pdfs", response_model=IngestStatusResponse)
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


@app.get("/drugs")
async def list_drugs(
    settings: Settings = Depends(get_settings)
):
    """
    List all available drugs in the system.
    """
    from pathlib import Path

    docs_path = Path(settings.docs_dir)
    if not docs_path.exists():
        return {"drugs": []}

    drug_folders = [f.name for f in docs_path.iterdir() if f.is_dir()]
    return {"drugs": drug_folders}


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("Drug Repurposing Chat API starting up...")
    print("Make sure to set your GEMINI_API_KEY in .env file")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        # reload=True # Removed for more stable debugging
    )
