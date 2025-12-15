import os
import google.generativeai as genai
from pathlib import Path
from typing import List, Dict, Any
import time  # Simple timing for chunk/embedding steps
from chromadb import Collection
import chromadb.utils.embedding_functions as embedding_functions # New import for embedding function

from .config import Settings
from .utils import parse_filename, chunk_text, extract_text_from_pdf, DocumentInfo
from .vector_store import upsert_chunks


def embed_texts(texts: List[str], settings: Settings, model: str) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Google Gemini.

    Args:
        texts: List of text strings to embed
        settings: Application settings (for API key and model)
        model: Embedding model name

    Returns:
        List of embedding vectors
    """
    # Ensure genai is configured (can be done globally in main.py lifespan, or here for standalone)
    genai.configure(api_key=settings.gemini_api_key)

    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])
    return embeddings


def process_pdf_file(
    pdf_path: Path,
    drug_folder: str,
    settings: Settings,
    collection: Collection
) -> Dict[str, Any]:
    """
    Process a single PDF file: parse filename, extract text, chunk, embed, and store.

    Args:
        pdf_path: Path to the PDF file
        drug_folder: Name of the drug folder (e.g., "aspirin repurposing")
        settings: Application settings
        collection: ChromaDB collection
        # Removed: client: genai # No longer needed directly, use settings

    Returns:
        Dictionary with processing results
    """
    t0 = time.perf_counter()
    try:
        # Parse filename to get document info
        doc_info = parse_filename(pdf_path.name, drug_folder)

        # Extract text from PDF
        t_extract_start = time.perf_counter()
        content = extract_text_from_pdf(str(pdf_path))
        t_extract = time.perf_counter() - t_extract_start

        if not content.strip():
            return {
                "drug_id": doc_info.drug_id,
                "doc_id": doc_info.doc_id,
                "chunks_created": 0,
                "error": "No text content extracted from PDF",
                "file_path": str(pdf_path)
            }

        # Chunk the content
        t_chunk_start = time.perf_counter()
        chunks = chunk_text(content)
        t_chunk = time.perf_counter() - t_chunk_start

        if not chunks:
            return {
                "drug_id": doc_info.drug_id,
                "doc_id": doc_info.doc_id,
                "chunks_created": 0,
                "error": "No chunks created from PDF content",
                "file_path": str(pdf_path)
            }

        # Generate embeddings for chunks
        t_embed_start = time.perf_counter()
        embeddings = embed_texts(chunks, settings, settings.gemini_embedding_model) # Pass settings instead of client
        t_embed = time.perf_counter() - t_embed_start

        # Prepare metadata for each chunk
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_info.drug_id}__{doc_info.doc_id}__chunk_{i}"
            metadata = {
                "drug_id": doc_info.drug_id,
                "doc_id": doc_info.doc_id,
                "doc_title": doc_info.doc_title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_path": str(pdf_path),
                "drug_folder": drug_folder,
                "source_type": "pdf"
            }
            metadatas.append(metadata)
            ids.append(chunk_id)

        # Store in vector database
        t_upsert_start = time.perf_counter()
        upsert_chunks(collection, chunks, metadatas, ids)
        t_upsert = time.perf_counter() - t_upsert_start

        total_time = time.perf_counter() - t0
        print(
            f"[TIMING] {pdf_path.name} | extract={t_extract:.2f}s chunk={t_chunk:.2f}s "
            f"embed={t_embed:.2f}s upsert={t_upsert:.2f}s total={total_time:.2f}s"
        )

        return {
            "drug_id": doc_info.drug_id,
            "doc_id": doc_info.doc_id,
            "doc_title": doc_info.doc_title,
            "chunks_created": len(chunks),
            "file_path": str(pdf_path),
            "drug_folder": drug_folder,
            "content_length": len(content)
        }

    except Exception as e:
        return {
            "file_path": str(pdf_path),
            "drug_folder": drug_folder,
            "error": str(e)
        }


def ingest_pdfs_from_directory(
    docs_dir: str,
    settings: Settings,
    collection: Collection
) -> Dict[str, Any]:
    """
    Ingest all PDFs from the drug repurposing directory.

    Directory structure expected:
    docs_dir/
    ├── aspirin repurposing/
    │   ├── aspirin_repurposing_PMC11242460.pdf
    │   └── ...
    ├── apomorphine repurposing/
    │   └── ...
    └── insulin repurposing/
        └── ...

    Args:
        docs_dir: Path to directory containing drug folders with PDFs
        settings: Application settings
        collection: ChromaDB collection

    Returns:
        Dictionary with ingestion statistics and results
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    # Get all drug folders (directories containing PDFs)
    drug_folders = [f for f in docs_path.iterdir() if f.is_dir()]

    if not drug_folders:
        return {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "drug_folders": [],
            "results": []
        }

    # Removed: client = get_gemini_client(settings)

    all_results = []
    total_processed = 0
    total_failed = 0
    total_chunks = 0

    for drug_folder in drug_folders:
        print(f"Processing drug folder: {drug_folder.name}")

        # Get all PDF files in this drug folder
        pdf_files = list(drug_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {drug_folder.name}")
            continue

        for pdf_path in pdf_files:
            try:
                result = process_pdf_file(pdf_path, drug_folder.name, settings, collection) # Removed client argument

                if "error" in result:
                    total_failed += 1
                    print(f"❌ Failed: {pdf_path.name} - {result['error']}")
                else:
                    total_processed += 1
                    total_chunks += result["chunks_created"]
                    print(f"✅ Processed: {pdf_path.name} -> {result['chunks_created']} chunks")

                all_results.append(result)

            except Exception as e:
                total_failed += 1
                error_result = {
                    "file_path": str(pdf_path),
                    "drug_folder": drug_folder.name,
                    "error": str(e)
                }
                all_results.append(error_result)
                print(f"❌ Error processing {pdf_path.name}: {e}")

    return {
        "total_files": len(all_results),
        "processed_files": total_processed,
        "failed_files": total_failed,
        "total_chunks": total_chunks,
        "drug_folders": [f.name for f in drug_folders],
        "results": all_results
    }


def ingest_single_document(
    drug_id: str,
    doc_id: str,
    doc_title: str,
    content: str,
    settings: Settings,
    collection: Collection
) -> Dict[str, Any]:
    """
    Ingest a single document programmatically (for API usage).

    Args:
        drug_id: Drug identifier
        doc_id: Document identifier
        doc_title: Human-readable document title
        content: Full document content
        settings: Application settings
        collection: ChromaDB collection

    Returns:
        Processing result
    """
    # Chunk the content
    chunks = chunk_text(content)

    if not chunks:
        return {
            "drug_id": drug_id,
            "doc_id": doc_id,
            "chunks_created": 0,
            "error": "No chunks created from document"
        }

    # Initialize Gemini client and generate embeddings
    embeddings = embed_texts(chunks, settings, settings.gemini_embedding_model) # Pass settings instead of client

    # Prepare metadata and IDs
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{drug_id}__{doc_id}__chunk_{i}"
        metadata = {
            "drug_id": drug_id,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source": "api_upload"
        }
        metadatas.append(metadata)
        ids.append(chunk_id)

    # Store in vector database
    upsert_chunks(collection, chunks, metadatas, ids)

    return {
        "drug_id": drug_id,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "chunks_created": len(chunks)
    }
