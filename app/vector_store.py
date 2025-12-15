import chromadb
from chromadb import Collection
from typing import Dict, List, Any, Optional
from .config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai # Needed for custom embedding function

# Removed: from .ingestion import get_gemini_client # This caused circular import

# Custom Embedding Function for Google Gemini
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
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

    # CRITICAL: Verify and enforce 768-dimensional embeddings for demo guarantee
    # Test embedding to confirm dimensions
    test_embedding = gemini_ef(["test query for dimension verification"])
    if len(test_embedding[0]) != 768:
        raise ValueError(f"Embedding dimension mismatch! Expected 768, got {len(test_embedding[0])}")

    print(f"✅ Verified: Embedding function produces {len(test_embedding[0])}-dimensional vectors")
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

    # Handle ChromaDB result format (may be dict or object)
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    distances = results.get("distances", [])
    ids = results.get("ids", [])

    # Handle nested lists (ChromaDB returns lists of lists for batch queries)
    if documents and isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list):
        documents = documents[0]
    if metadatas and isinstance(metadatas, list) and len(metadatas) > 0 and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if distances and isinstance(distances, list) and len(distances) > 0 and isinstance(distances[0], list):
        distances = distances[0]
    if ids and isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]

    # Ensure we have proper lists
    documents = documents if isinstance(documents, list) else []
    metadatas = metadatas if isinstance(metadatas, list) else []
    distances = distances if isinstance(distances, list) else []
    ids = ids if isinstance(ids, list) else []

    return QueryResult(
        documents=documents,
        metadatas=metadatas,
        distances=distances,
        ids=ids
    )
