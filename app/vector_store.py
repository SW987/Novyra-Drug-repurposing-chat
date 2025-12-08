import chromadb
from chromadb import Collection
from typing import Dict, List, Any, Optional
from .config import Settings


def init_vector_store(settings: Settings) -> Collection:
    """
    Initialize a persistent Chroma client and get/create a collection.

    Args:
        settings: Application settings

    Returns:
        ChromaDB collection for storing document chunks
    """
    # Initialize persistent client
    client = chromadb.PersistentClient(path=settings.chroma_db_dir)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"description": "Drug repurposing PDF documents chunks with embeddings"}
    )

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

    return QueryResult(
        documents=results["documents"][0] if results["documents"] else [],
        metadatas=results["metadatas"][0] if results["metadatas"] else [],
        distances=results["distances"][0] if results["distances"] else [],
        ids=results["ids"][0] if results["ids"] else []
    )
