import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
from chromadb import Collection

from .config import Settings
from .vector_store import query_chunks, QueryResult
from .schemas import Source, Message


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

    # Query vector store with higher top_k for diversity
    initial_results = query_chunks(collection, query_embedding, where_filter, top_k * 2)

    # Implement diversity selection - avoid selecting too many chunks from the same document
    selected_chunks = []
    doc_counts = {}

    for i, (doc_id, distance, document, metadata) in enumerate(zip(
        initial_results.ids,
        initial_results.distances,
        initial_results.documents,
        initial_results.metadatas
    )):
        current_doc = metadata.get('doc_id', 'unknown')
        if doc_counts.get(current_doc, 0) < 5:  # Limit to 5 chunks per document
            selected_chunks.append((doc_id, distance, document, metadata))
            doc_counts[current_doc] = doc_counts.get(current_doc, 0) + 1

        if len(selected_chunks) >= top_k:
            break

    # Reconstruct results with diverse chunks
    diverse_results = {
        'ids': [chunk[0] for chunk in selected_chunks],
        'distances': [chunk[1] for chunk in selected_chunks],
        'documents': [chunk[2] for chunk in selected_chunks],
        'metadatas': [chunk[3] for chunk in selected_chunks]
    }

    return diverse_results


def build_enhanced_query(current_message: str, conversation_history: Optional[List[Union[Message, Dict[str, str]]]] = None) -> str:
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
        # Handle both Message objects and dict formats
        if hasattr(msg, 'role'):
            role = msg.role
            content = msg.content
        else:
            role = msg.get("role", "user")
            content = msg.get("content", "")

        if role == "user":
            context_parts.append(f"User: {content}")
        elif role == "assistant":
            context_parts.append(f"Assistant: {content}")

    conversation_context = "\n".join(context_parts)

    # Create a more specific context-aware query
    if "second" in current_message.lower() or "expand" in current_message.lower():
        # Try to infer what "second" refers to from the assistant's last response
        last_assistant_msg = None
        for msg in reversed(conversation_history):
            if (hasattr(msg, 'role') and msg.role == 'assistant') or msg.get('role') == 'assistant':
                content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                last_assistant_msg = content
                break

        if last_assistant_msg and any(keyword in last_assistant_msg.lower() for keyword in ['cardiovascular', 'anti-aging', 'cancer', 'covid']):
            # Make the query more specific based on context
            current_message = f"Based on our previous discussion about metformin repurposing opportunities, {current_message}"

    enhanced_query = f"""Previous conversation context:
{conversation_context}

Current user question: {current_message}

IMPORTANT: This appears to be a follow-up question. Provide a detailed, comprehensive answer that builds on our previous discussion. If the user is asking about a specific item from a list, clearly identify what they're referring to and provide detailed information about that topic from the research documents."""

    return enhanced_query

def build_rag_prompt(query: str, context_chunks: List[str], conversation_history: Optional[List[Union[Message, Dict[str, str]]]] = None) -> str:
    """
    Build a RAG prompt with retrieved context.

    Args:
        query: User query (may include conversation context)
        context_chunks: List of relevant text chunks

    Returns:
        Formatted prompt for LLM
    """
    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

    # Add conversation history to the prompt
    history_str = ""
    if conversation_history:
        history_parts = []
        for msg in conversation_history[-4:]:  # Last 4 messages for context
            # Handle both Message objects and dict formats
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")

            if role == "user":
                history_parts.append(f"User: {content}")
            elif role == "assistant":
                history_parts.append(f"Assistant: {content}")

        if history_parts:
            history_str = "\n".join(history_parts) + "\n\n"

    prompt = f"""You are a helpful assistant specializing in drug repurposing research.
Use the provided context from scientific papers to answer the user's question accurately and comprehensively.

IMPORTANT INSTRUCTIONS:
1. Synthesize information across ALL provided contexts to give a complete, unified answer
2. Do not just repeat information - combine and summarize key findings from all documents
3. If the context seems limited, provide what information is available and note any gaps
4. For follow-up questions, build upon previous context while adding new details
5. Be comprehensive but concise - provide detailed explanations within reasonable length

{history_str}Context:
{context}

Question: {query}

Provide a detailed and comprehensive answer based on the available scientific context:"""

    return prompt


def generate_answer(
    query: str,
    context_chunks: List[str],
    client: genai,
    model: str,
    conversation_history: Optional[List[Union[Message, Dict[str, str]]]] = None,
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
    prompt = build_rag_prompt(query, context_chunks, conversation_history)

    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=2000,  # Increased for more detailed responses
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
    top_k: int = 20,  # Increased for better coverage and diversity
    conversation_history: Optional[List[Union[Message, Dict[str, str]]]] = None
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
    answer = generate_answer(enhanced_query, context_chunks, client, settings.gemini_chat_model, conversation_history)

    # Extract sources
    sources = extract_sources_from_results(results)

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }
