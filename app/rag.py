import google.generativeai as genai
from typing import List, Dict, Any, Optional
from chromadb import Collection

from .config import Settings
from .vector_store import query_chunks, QueryResult
from .schemas import Source


def get_gemini_client(settings: Settings) -> genai:
    """Configure and return Gemini API client."""
    genai.configure(api_key=settings.gemini_api_key)
    return genai


def embed_query(query: str, client: genai, model: str) -> List[float]:
    """
    Generate embedding for a query string using Gemini.

    Args:
        query: Query text
        client: Gemini client
        model: Embedding model name

    Returns:
        Embedding vector
    """
    result = genai.embed_content(
        model=model,
        content=query,
        task_type="retrieval_query"
    )
    return result['embedding']


def build_filter(drug_id: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Build metadata filter for ChromaDB query.

    Args:
        drug_id: Drug identifier (required)
        doc_id: Optional document identifier

    Returns:
        Filter dictionary for ChromaDB
    """
    filter_dict = {"drug_id": drug_id}
    if doc_id:
        filter_dict["doc_id"] = doc_id
    return filter_dict


def retrieve_relevant_chunks(
    query: str,
    drug_id: str,
    collection: Collection,
    settings: Settings,
    doc_id: Optional[str] = None,
    top_k: int = 5
) -> QueryResult:
    """
    Retrieve relevant document chunks for a query.

    Args:
        query: User query
        drug_id: Drug identifier to filter by
        collection: ChromaDB collection
        settings: Application settings
        doc_id: Optional document ID to further filter
        top_k: Number of chunks to retrieve

    Returns:
        QueryResult with relevant chunks
    """
    # Get Gemini client and embed query
    client = get_gemini_client(settings)
    query_embedding = embed_query(query, client, settings.gemini_embedding_model)

    # Build filter
    where_filter = build_filter(drug_id, doc_id)

    # Query vector store
    results = query_chunks(collection, query_embedding, where_filter, top_k)

    return results


def build_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Build a RAG prompt with retrieved context.

    Args:
        query: User query
        context_chunks: List of relevant text chunks

    Returns:
        Formatted prompt for LLM
    """
    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

    prompt = f"""You are a helpful assistant specializing in drug repurposing research.
Use the provided context from scientific papers to answer the user's question accurately and comprehensively.

If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information is available.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def generate_answer(
    query: str,
    context_chunks: List[str],
    client: genai,
    model: str,
    temperature: float = 0.1
) -> str:
    """
    Generate an answer using Gemini with retrieved context.

    Args:
        query: User query
        context_chunks: Relevant context chunks
        client: Gemini client
        model: Chat model name
        temperature: Sampling temperature

    Returns:
        Generated answer
    """
    prompt = build_rag_prompt(query, context_chunks)

    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1000,
        )
    )

    return response.text.strip()


def extract_sources_from_results(results: QueryResult) -> List[Source]:
    """
    Extract source information from query results.

    Args:
        results: QueryResult from vector store

    Returns:
        List of Source objects
    """
    sources = []

    for i, (doc, metadata, distance, chunk_id) in enumerate(zip(
        results.documents, results.metadatas, results.distances, results.ids
    )):
        # Create text preview (first 200 characters)
        text_preview = doc[:200] + "..." if len(doc) > 200 else doc

        source = Source(
            doc_id=metadata["doc_id"],
            doc_title=metadata["doc_title"],
            chunk_id=chunk_id,
            distance=distance,
            text_preview=text_preview
        )
        sources.append(source)

    return sources


def chat_with_documents(
    session_id: str,
    drug_id: str,
    message: str,
    collection: Collection,
    settings: Settings,
    doc_id: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Main RAG pipeline: retrieve relevant chunks and generate answer.

    Args:
        session_id: Session identifier
        drug_id: Drug identifier
        message: User query
        collection: ChromaDB collection
        settings: Application settings
        doc_id: Optional document ID filter
        top_k: Number of chunks to retrieve

    Returns:
        Dictionary with answer and sources
    """
    # Retrieve relevant chunks
    results = retrieve_relevant_chunks(message, drug_id, collection, settings, doc_id, top_k)

    if not results.documents:
        return {
            "answer": f"I couldn't find any relevant information about '{message}' in the documents for drug {drug_id}." +
                     (f" (filtered to document {doc_id})" if doc_id else ""),
            "sources": [],
            "session_id": session_id
        }

    # Generate answer using LLM
    client = get_gemini_client(settings)
    context_chunks = results.documents
    answer = generate_answer(message, context_chunks, client, settings.gemini_chat_model)

    # Extract sources
    sources = extract_sources_from_results(results)

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }
