import os
import time  # Simple timing for chunk/embedding steps
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

import google.generativeai as genai
from chromadb import Collection
import chromadb.utils.embedding_functions as embedding_functions # New import for embedding function

from .config import Settings
from .utils import (
    DocumentInfo,
    chunk_text,
    extract_text_from_pdf,
    extract_text_from_pdf_bytes,
    parse_filename,
)
from .vector_store import upsert_chunks

try:
    import boto3
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


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


def _ingest_text_content(
    doc_info: DocumentInfo,
    content: str,
    settings: Settings,
    collection: Collection,
    file_path: str,
    drug_folder: str,
    source_type: str,
    source_uri: Optional[str] = None,
) -> Dict[str, Any]:
    if not content.strip():
        return {
            "drug_id": doc_info.drug_id,
            "doc_id": doc_info.doc_id,
            "chunks_created": 0,
            "error": "No text content extracted from PDF",
            "file_path": file_path,
        }

    t_chunk_start = time.perf_counter()
    chunks = chunk_text(content)
    t_chunk = time.perf_counter() - t_chunk_start

    if not chunks:
        return {
            "drug_id": doc_info.drug_id,
            "doc_id": doc_info.doc_id,
            "chunks_created": 0,
            "error": "No chunks created from PDF content",
            "file_path": file_path,
        }

    t_embed_start = time.perf_counter()
    embeddings = embed_texts(chunks, settings, settings.gemini_embedding_model)
    t_embed = time.perf_counter() - t_embed_start

    metadatas = []
    ids = []

    for i, _chunk in enumerate(chunks):
        chunk_id = f"{doc_info.drug_id}__{doc_info.doc_id}__chunk_{i}"
        metadata = {
            "drug_id": doc_info.drug_id,
            "doc_id": doc_info.doc_id,
            "doc_title": doc_info.doc_title,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "file_path": file_path,
            "drug_folder": drug_folder,
            "source_type": source_type,
        }
        if source_uri:
            metadata["source_uri"] = source_uri
        metadatas.append(metadata)
        ids.append(chunk_id)

    t_upsert_start = time.perf_counter()
    upsert_chunks(collection, chunks, metadatas, ids)
    t_upsert = time.perf_counter() - t_upsert_start

    return {
        "drug_id": doc_info.drug_id,
        "doc_id": doc_info.doc_id,
        "doc_title": doc_info.doc_title,
        "chunks_created": len(chunks),
        "file_path": file_path,
        "drug_folder": drug_folder,
        "content_length": len(content),
        "timing": {
            "chunk": t_chunk,
            "embed": t_embed,
            "upsert": t_upsert,
        },
    }


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

    Returns:
        Dictionary with processing results
    """
    t0 = time.perf_counter()
    try:
        doc_info = parse_filename(pdf_path.name, drug_folder)

        t_extract_start = time.perf_counter()
        content = extract_text_from_pdf(str(pdf_path))
        t_extract = time.perf_counter() - t_extract_start

        result = _ingest_text_content(
            doc_info=doc_info,
            content=content,
            settings=settings,
            collection=collection,
            file_path=str(pdf_path),
            drug_folder=drug_folder,
            source_type="pdf",
        )
        if "error" in result:
            return result

        total_time = time.perf_counter() - t0
        timing = result.get("timing", {})
        print(
            f"[TIMING] {pdf_path.name} | extract={t_extract:.2f}s chunk={timing.get('chunk', 0):.2f}s "
            f"embed={timing.get('embed', 0):.2f}s upsert={timing.get('upsert', 0):.2f}s "
            f"total={total_time:.2f}s"
        )
        result.pop("timing", None)
        return result

    except Exception as e:
        return {
            "file_path": str(pdf_path),
            "drug_folder": drug_folder,
            "error": str(e),
        }


def process_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    drug_folder: str,
    settings: Settings,
    collection: Collection,
    source_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a PDF provided as bytes without writing it to disk.
    """
    t0 = time.perf_counter()
    try:
        doc_info = parse_filename(filename, drug_folder)

        t_extract_start = time.perf_counter()
        content = extract_text_from_pdf_bytes(pdf_bytes, source_uri or filename)
        t_extract = time.perf_counter() - t_extract_start

        file_path = source_uri or f"{drug_folder}/{filename}"
        result = _ingest_text_content(
            doc_info=doc_info,
            content=content,
            settings=settings,
            collection=collection,
            file_path=file_path,
            drug_folder=drug_folder,
            source_type="s3" if source_uri else "pdf_bytes",
            source_uri=source_uri,
        )
        if "error" in result:
            return result

        total_time = time.perf_counter() - t0
        timing = result.get("timing", {})
        print(
            f"[TIMING] {filename} | extract={t_extract:.2f}s chunk={timing.get('chunk', 0):.2f}s "
            f"embed={timing.get('embed', 0):.2f}s upsert={timing.get('upsert', 0):.2f}s "
            f"total={total_time:.2f}s"
        )
        result.pop("timing", None)
        return result
    except Exception as e:
        return {
            "file_path": source_uri or filename,
            "drug_folder": drug_folder,
            "error": str(e),
        }


def _normalize_s3_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ""
    return prefix.strip("/")


def _list_s3_pdf_keys(
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> Iterable[str]:
    list_prefix = f"{prefix}/" if prefix else ""
    paginator = s3_client.get_paginator("list_objects_v2")
    kwargs: Dict[str, Any] = {"Bucket": bucket}
    if list_prefix:
        kwargs["Prefix"] = list_prefix

    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if key and key.lower().endswith(".pdf"):
                yield key


def _split_s3_key(prefix: str, key: str) -> tuple[str, str, str]:
    relative_key = key
    if prefix:
        list_prefix = f"{prefix}/"
        if relative_key.startswith(list_prefix):
            relative_key = relative_key[len(list_prefix):]

    path = PurePosixPath(relative_key)
    filename = path.name
    drug_folder = path.parts[0] if len(path.parts) > 1 else "unknown"
    return drug_folder, filename, relative_key


def ingest_pdfs_from_s3(
    bucket: str,
    prefix: str,
    settings: Settings,
    collection: Collection,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest PDFs from S3 without writing them to disk.
    """
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 ingestion. Install boto3 first.")

    normalized_prefix = _normalize_s3_prefix(prefix)
    session = boto3.session.Session(region_name=region) if region else boto3.session.Session()
    s3_client = session.client("s3")

    target_label = f"s3://{bucket}/{normalized_prefix}".rstrip("/")
    _log(f"Starting S3 ingestion from {target_label}")
    _log("Listing PDF objects...")

    all_results = []
    total_processed = 0
    total_failed = 0
    total_chunks = 0
    total_seen = 0

    for key in _list_s3_pdf_keys(s3_client, bucket, normalized_prefix):
        total_seen += 1
        source_uri = f"s3://{bucket}/{key}"
        drug_folder, filename, _relative_key = _split_s3_key(normalized_prefix, key)

        try:
            _log(f"[{total_seen}] Fetching {source_uri}")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content_length = response.get("ContentLength")
            if content_length is not None:
                _log(f"[{total_seen}] Downloaded {content_length} bytes")
            pdf_bytes = response["Body"].read()

            result = process_pdf_bytes(
                pdf_bytes=pdf_bytes,
                filename=filename,
                drug_folder=drug_folder,
                settings=settings,
                collection=collection,
                source_uri=source_uri,
            )

            if "error" in result:
                total_failed += 1
                _log(f"❌ Failed: {source_uri} - {result['error']}")
            else:
                total_processed += 1
                total_chunks += result["chunks_created"]
                _log(f"✅ Processed: {source_uri} -> {result['chunks_created']} chunks")

            all_results.append(result)

        except Exception as e:
            total_failed += 1
            error_result = {
                "file_path": source_uri,
                "drug_folder": drug_folder,
                "error": str(e),
            }
            all_results.append(error_result)
            _log(f"❌ Error processing {source_uri}: {e}")

    _log(
        f"Completed S3 ingestion from {target_label} | "
        f"seen={total_seen} processed={total_processed} failed={total_failed} chunks={total_chunks}"
    )
    return {
        "total_files": len(all_results),
        "processed_files": total_processed,
        "failed_files": total_failed,
        "total_chunks": total_chunks,
        "total_seen": total_seen,
        "s3_bucket": bucket,
        "s3_prefix": normalized_prefix,
        "results": all_results,
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
