import google.generativeai as genai
from typing import List, Dict, Any, Optional
from chromadb import Collection

from .config import Settings
from .vector_store import query_chunks, QueryResult
from .schemas import Source


def get_gemini_client(settings: Settings) -> genai:
    """Return Gemini API client."""
    # genai.configure(api_key=settings.gemini_api_key) # Configured globally in main.py lifespan
    return genai


def embed_query(query: str, client: genai, model: str) -> List[float]:
    """
    Generate embedding for a query string using Gemini.

    Args:
        query: Query text
        client: Gemini client
        model: Embedding model name

    Returns:
        Embedding vector (guaranteed 768 dimensions)
    """
    result = genai.embed_content(
        model=model,
        content=query,
        task_type="retrieval_query"
    )
    embedding = result['embedding']
    # DEMO GUARANTEE: Verify dimensions
    if len(embedding) != 768:
        raise ValueError(f"Query embedding dimension mismatch! Expected 768, got {len(embedding)}")
    return embedding


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


def build_enhanced_query(current_message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Build an enhanced query that includes conversation context.

    Args:
        current_message: Current user message
        conversation_history: List of previous {role: content} messages

    Returns:
        Enhanced query string with context
    """
    if not conversation_history:
        return current_message

    # Build context from recent conversation
    context_parts = []
    for msg in conversation_history[-6:]:  # Last 6 messages for context
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            context_parts.append(f"User: {content}")
        elif role == "assistant":
            context_parts.append(f"Assistant: {content}")

    conversation_context = "\n".join(context_parts)

    enhanced_query = f"""Previous conversation:
{conversation_context}

Current question: {current_message}

Please provide a comprehensive answer that builds on our previous discussion and synthesizes information from all available research documents."""

    return enhanced_query

def build_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Build a RAG prompt with retrieved context.

    Args:
        query: User query (may include conversation context)
        context_chunks: List of relevant text chunks

    Returns:
        Formatted prompt for LLM
    """
    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

    prompt = f"""You are a helpful assistant specializing in drug repurposing research.
Use the provided context from scientific papers to answer the user's question accurately and comprehensively.

IMPORTANT: Synthesize information across ALL provided contexts to give a complete, unified answer. Do not just repeat information from individual contexts - combine and summarize the key findings from all relevant documents.

If you have conversation context, build upon previous answers while incorporating new information.

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
    top_k: int = 15,  # Increased for better coverage
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Main RAG pipeline: retrieve relevant chunks and generate answer with conversation context.

    Args:
        session_id: Session identifier
        drug_id: Drug identifier
        message: User query
        collection: ChromaDB collection
        settings: Application settings
        doc_id: Optional document ID filter
        top_k: Number of chunks to retrieve
        conversation_history: Previous messages for context

    Returns:
        Dictionary with answer and sources
    """
    # Retrieve relevant chunks (increased top_k for better coverage)
    results = retrieve_relevant_chunks(message, drug_id, collection, settings, doc_id, top_k)

    if not results.documents:
        return {
            "answer": f"I couldn't find any relevant information about '{message}' in the documents for drug {drug_id}." +
                     (f" (filtered to document {doc_id})" if doc_id else ""),
            "sources": [],
            "session_id": session_id
        }

    # Generate answer using LLM with conversation context
    client = get_gemini_client(settings)
    context_chunks = results.documents

    # Build enhanced prompt with conversation history
    enhanced_query = build_enhanced_query(message, conversation_history)
    answer = generate_answer(enhanced_query, context_chunks, client, settings.gemini_chat_model)

    # Extract sources
    sources = extract_sources_from_results(results)

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }
