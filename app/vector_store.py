import chromadb
from chromadb import Collection
from typing import Dict, List, Any, Optional
from pathlib import Path
from .config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai # Needed for custom embedding function
import shutil
import logging

logger = logging.getLogger(__name__)

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


def is_chromadb_corrupted(error: Exception) -> bool:
    """
    Check if an error indicates ChromaDB corruption.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error indicates corruption
    """
    error_str = str(error).lower()
    corruption_indicators = [
        "trailer",
        "not defined",
        "corrupt",
        "database disk image is malformed",
        "file is not a database",
        "sqlite",
        "invalid literal",
        "no such table",
        "could not connect to tenant",
        "operationalerror"
    ]
    return any(indicator in error_str for indicator in corruption_indicators)


def check_database_health(db_path: Path) -> bool:
    """
    Check if ChromaDB database is healthy by looking for required files.
    
    Args:
        db_path: Path to ChromaDB database directory
        
    Returns:
        True if database appears healthy, False if corrupted
    """
    if not db_path.exists():
        return True  # No database yet, that's fine
    
    try:
        # Check if it's a directory
        if not db_path.is_dir():
            return False
        
        # Try to create a test client to see if database is accessible
        test_client = chromadb.PersistentClient(path=str(db_path))
        # Try to list collections (this will fail if database is corrupted)
        test_client.list_collections()
        return True
    except Exception as e:
        error_str = str(e).lower()
        # Check for corruption indicators
        if any(indicator in error_str for indicator in [
            "no such table", "operationalerror", "could not connect",
            "database disk image", "corrupt", "malformed"
        ]):
            return False
        # Other errors might be okay (e.g., no collections yet)
        return True


def reset_chromadb(settings: Settings) -> bool:
    """
    Reset ChromaDB by deleting the database directory and recreating it.
    Use this as a last resort when corruption is detected.
    
    Args:
        settings: Application settings
        
    Returns:
        True if reset was successful
    """
    try:
        db_path = Path(settings.chroma_db_dir)
        if db_path.exists():
            print(f"⚠️ Resetting ChromaDB database at {db_path}")
            # Check health first
            if not check_database_health(db_path):
                print(f"⚠️ Database is corrupted, deleting...")
                shutil.rmtree(db_path)
                print(f"✅ Deleted corrupted database directory")
            else:
                # Database seems healthy, but we're resetting anyway
                shutil.rmtree(db_path)
                print(f"✅ Deleted database directory")
        
        # Recreate directory
        db_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created fresh database directory")
        return True
    except Exception as e:
        print(f"❌ Failed to reset ChromaDB: {e}")
        return False


def init_vector_store(settings: Settings, reset_on_corruption: bool = True) -> Collection:
    """
    Initialize ChromaDB persistent client and get/create collection.
    Ensures the collection is created with the correct embedding function.
    Handles corruption detection and recovery.
    
    Args:
        settings: Application settings
        reset_on_corruption: If True, automatically reset corrupted database
        
    Returns:
        ChromaDB collection
    """
    max_retries = 2
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Ensure database directory exists
            db_path = Path(settings.chroma_db_dir)
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Check database health BEFORE creating client
            if db_path.exists() and not check_database_health(db_path):
                print(f"⚠️ Database corruption detected before initialization")
                if reset_on_corruption and retry_count == 0:
                    print("🔄 Resetting corrupted database...")
                    if reset_chromadb(settings):
                        retry_count += 1
                        continue  # Retry after reset
                    else:
                        raise RuntimeError("Failed to reset corrupted database")
                else:
                    raise RuntimeError("Database is corrupted and reset is disabled")
            
            client = chromadb.PersistentClient(path=settings.chroma_db_dir)

            # Define the custom Gemini embedding function for ChromaDB
            gemini_ef = GeminiEmbeddingFunction(
                api_key=settings.gemini_api_key,
                model_name=settings.gemini_embedding_model
            )

            # Try to get or create collection
            try:
                collection = client.get_or_create_collection(
                    name=settings.chroma_collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=gemini_ef
                )
            except Exception as e:
                if is_chromadb_corrupted(e):
                    print(f"⚠️ ChromaDB corruption detected during collection access: {e}")
                    if reset_on_corruption and retry_count == 0:
                        print("🔄 Attempting to reset corrupted database...")
                        if reset_chromadb(settings):
                            retry_count += 1
                            continue  # Retry after reset
                    raise
                else:
                    raise

            # Test the collection by trying a simple query to ensure it's fully initialized
            try:
                test_result = collection.get(limit=1)
                # If we get here, the collection is accessible
            except Exception as e:
                error_str = str(e).lower()
                # Check for SessionInfo errors specifically (ChromaDB session not ready)
                if "sessioninfo" in error_str or ("session" in error_str and "initialized" in error_str):
                    print(f"⚠️ ChromaDB session not ready: {e}")
                    if retry_count < max_retries - 1:
                        # Recreate client and collection
                        retry_count += 1
                        continue  # Retry with fresh client
                    else:
                        raise RuntimeError(f"ChromaDB session failed to initialize: {e}") from e
                elif is_chromadb_corrupted(e):
                    print(f"⚠️ ChromaDB corruption detected during test query: {e}")
                    if reset_on_corruption and retry_count == 0:
                        print("🔄 Attempting to reset corrupted database...")
                        if reset_chromadb(settings):
                            retry_count += 1
                            continue  # Retry after reset
                    raise
                else:
                    raise

            # CRITICAL: Verify and enforce 768-dimensional embeddings for demo guarantee
            # Test embedding to confirm dimensions
            test_embedding = gemini_ef(["test query for dimension verification"])
            if len(test_embedding[0]) != 768:
                raise ValueError(f"Embedding dimension mismatch! Expected 768, got {len(test_embedding[0])}")

            print(f"[OK] Verified: Embedding function produces {len(test_embedding[0])}-dimensional vectors")
            return collection
            
        except Exception as e:
            if is_chromadb_corrupted(e) and retry_count < max_retries - 1:
                print(f"⚠️ ChromaDB corruption detected: {e}")
                if reset_on_corruption:
                    print("🔄 Attempting to reset corrupted database...")
                    if reset_chromadb(settings):
                        retry_count += 1
                        continue  # Retry after reset
            # If we can't recover, raise the error
            raise
    
    # If we exhausted retries, raise an error
    raise RuntimeError("Failed to initialize ChromaDB after corruption recovery attempts")


def upsert_chunks(
    collection: Collection,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str]
) -> None:
    """
    Add or update chunks in the vector store.
    Handles corruption errors gracefully.

    Args:
        collection: ChromaDB collection
        texts: List of text chunks
        metadatas: List of metadata dictionaries for each chunk
        ids: List of unique IDs for each chunk
        
    Raises:
        RuntimeError: If ChromaDB corruption is detected
        Exception: Other errors from ChromaDB
    """
    try:
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        if is_chromadb_corrupted(e):
            error_msg = f"ChromaDB corruption detected during write: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        else:
            # Re-raise other errors as-is
            raise


def query_chunks(
    collection: Collection,
    query_embedding: List[float],
    where: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> Dict[str, List]:
    """
    Query the vector store for similar chunks.
    Handles corruption errors gracefully.

    Args:
        collection: ChromaDB collection
        query_embedding: Embedding vector for the query
        where: Optional metadata filter (e.g., {"drug_id": "aspirin"})
        top_k: Number of results to return

    Returns:
        Dict with documents, metadatas, distances, and ids (empty lists on corruption)
    """
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            where=where,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        if is_chromadb_corrupted(e):
            error_msg = f"ChromaDB corruption detected during query: {e}"
            logger.error(error_msg)
            # Return empty results instead of crashing to allow app to continue
            # The calling code should handle this gracefully
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }
        else:
            # Re-raise other errors as-is
            raise

    # Extract data from ChromaDB results
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    distances = results.get("distances", [])
    ids = results.get("ids", [])

    # Handle nested lists (ChromaDB returns lists of lists for batch queries)
    if isinstance(documents, list) and documents and isinstance(documents[0], list):
        documents = documents[0]
    if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if isinstance(distances, list) and distances and isinstance(distances[0], list):
        distances = distances[0]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]

    # Return as dict for consistent handling
    return {
        "documents": documents if isinstance(documents, list) else [],
        "metadatas": metadatas if isinstance(metadatas, list) else [],
        "distances": distances if isinstance(distances, list) else [],
        "ids": ids if isinstance(ids, list) else []
    }
