"""
Vector store interface built on ChromaDB.

Handles:
- Initialising the persistent ChromaDB client and collection.
- A custom ChromaDB-compatible embedding function backed by Google Gemini.
- Upserting and querying document chunks with metadata filtering.

The Gemini API key and model names are sourced exclusively from the
Settings object — never hardcoded here.
"""

import chromadb
from chromadb import Collection
from typing import Dict, List, Any, Optional
from .config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai


class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """ChromaDB-compatible embedding function that delegates to Google Gemini."""

    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: embedding_functions.Documents) -> embedding_functions.Embeddings:
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings


def init_vector_store(settings: Settings) -> Collection:
    """
    Initialize ChromaDB persistent client and get/create collection.
    Ensures the collection is created with the correct embedding function.
    """
    client = chromadb.PersistentClient(path=settings.chroma_db_dir)

    # Define the custom Gemini embedding function for ChromaDB
    gemini_ef = GeminiEmbeddingFunction(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_embedding_model
    )

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=gemini_ef  # Explicitly set the custom embedding function
    )

    # Only verify dimensions if collection is empty (first-time setup)
    # This avoids expensive API calls on every startup
    count = collection.count()
    if count == 0:
        print("⚠️  Empty collection detected - verifying embedding dimensions...")
        test_embedding = gemini_ef(["test query for dimension verification"])
        expected_dim = settings.gemini_embedding_dimension
        if len(test_embedding[0]) != expected_dim:
            raise ValueError(
                "Embedding dimension mismatch! "
                f"Expected {expected_dim}, got {len(test_embedding[0])}"
            )
        print(f"✅ Verified: Embedding function produces {len(test_embedding[0])}-dimensional vectors")
    else:
        print(f"✅ Collection loaded with {count} chunks (dimension verification skipped)")

    return collection


def upsert_chunks(
    collection: Collection,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str]
) -> None:
    """
    Add or update chunks in the vector store.

    Args:
        collection: ChromaDB collection
        texts: List of text chunks
        metadatas: List of metadata dictionaries for each chunk
        ids: List of unique IDs for each chunk

    Note:
        ChromaDB's `add` raises on duplicate IDs. Re-ingesting the same
        document will produce duplicate-ID errors unless the collection is
        cleared first or IDs are checked beforehand.
    """
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )


class QueryResult:
    """Structured result from vector store query."""

    def __init__(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        distances: List[float],
        ids: List[str]
    ):
        self.documents = documents
        self.metadatas = metadatas
        self.distances = distances
        self.ids = ids


def query_chunks(
    collection: Collection,
    query_embedding: List[float],
    where: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> QueryResult:
    """
    Query the vector store for similar chunks.

    Args:
        collection: ChromaDB collection
        query_embedding: Embedding vector for the query
        where: Optional metadata filter (e.g., {"drug_id": "aspirin"})
        top_k: Number of results to return

    Returns:
        QueryResult with documents, metadatas, distances, and ids
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        where=where,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return QueryResult(
        documents=results["documents"][0] if results["documents"] else [],
        metadatas=results["metadatas"][0] if results["metadatas"] else [],
        distances=results["distances"][0] if results["distances"] else [],
        ids=results["ids"][0] if results["ids"] else []
    )
