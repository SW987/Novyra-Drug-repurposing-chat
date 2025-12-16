import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
from chromadb import Collection

from .config import Settings
from .vector_store import query_chunks
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
) -> Dict[str, List]:
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
        Dict with relevant chunks
    """
    # Get Gemini client and embed query
    client = get_gemini_client(settings)
    query_embedding = embed_query(query, client, settings.gemini_embedding_model)

    # Build filter
    where_filter = build_filter(drug_id, doc_id)

    # Query vector store with higher top_k for diversity
    initial_results = query_chunks(collection, query_embedding, where_filter, top_k * 2)

    # If no results with drug filter, try broader search (remove drug filter)
    if not initial_results.get("documents") or len(initial_results.get("documents", [])) == 0:
        print(f"DEBUG: No results with drug filter, trying broader search")
        broader_results = query_chunks(collection, query_embedding, None, top_k * 2)  # No filter
        if broader_results.get("documents") and len(broader_results.get("documents", [])) > 0:
            print(f"DEBUG: Found {len(broader_results.get('documents', []))} documents with broader search")
            initial_results = broader_results

    # Extract data from dict (query_chunks now returns consistent dict format)
    ids_list = initial_results.get("ids", [])
    distances_list = initial_results.get("distances", [])
    documents_list = initial_results.get("documents", [])
    metadatas_list = initial_results.get("metadatas", [])

    # Handle nested lists (ChromaDB returns lists of lists for batch queries)
    if ids_list and isinstance(ids_list, list) and len(ids_list) > 0 and isinstance(ids_list[0], list):
        ids_list = ids_list[0]
    if distances_list and isinstance(distances_list, list) and len(distances_list) > 0 and isinstance(distances_list[0], list):
        distances_list = distances_list[0]
    if documents_list and isinstance(documents_list, list) and len(documents_list) > 0 and isinstance(documents_list[0], list):
        documents_list = documents_list[0]
    if metadatas_list and isinstance(metadatas_list, list) and len(metadatas_list) > 0 and isinstance(metadatas_list[0], list):
        metadatas_list = metadatas_list[0]

    # Implement diversity selection - avoid selecting too many chunks from the same document
    selected_chunks = []
    doc_counts = {}

    for i, (doc_id, distance, document, metadata) in enumerate(zip(
        ids_list,
        distances_list,
        documents_list,
        metadatas_list
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
    for msg in conversation_history[-8:]:  # Last 8 messages for broader context
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

    # Create a more specific context-aware query for follow-up questions
    if any(keyword in current_message.lower() for keyword in ["second", "third", "first", "expand", "more on", "tell me more"]):
        # Try to infer what the user is referring to from the assistant's last response
        last_assistant_msg = None
        for msg in reversed(conversation_history):
            if (hasattr(msg, 'role') and msg.role == 'assistant') or msg.get('role') == 'assistant':
                content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                last_assistant_msg = content
                break

        if last_assistant_msg:
            # Extract numbered items or topics from the last response
            lines = last_assistant_msg.split('\n')
            numbered_items = []
            for line in lines:
                if any(line.strip().startswith(f"{i}.") or line.strip().startswith(f"{i})") for i in range(1, 10)):
                    numbered_items.append(line.strip())
            
            # If user asks about "second one" and we found numbered items, enhance the query
            if "second" in current_message.lower() and len(numbered_items) >= 2:
                current_message = f"Provide detailed, comprehensive information about: {numbered_items[1]}. {current_message}"
            elif "first" in current_message.lower() and len(numbered_items) >= 1:
                current_message = f"Provide detailed, comprehensive information about: {numbered_items[0]}. {current_message}"
            elif "third" in current_message.lower() and len(numbered_items) >= 3:
                current_message = f"Provide detailed, comprehensive information about: {numbered_items[2]}. {current_message}"

    enhanced_query = f"""Previous conversation context:
{conversation_context}

Current user question: {current_message}

IMPORTANT: This appears to be a follow-up question. Provide a detailed, comprehensive, and extensive answer that builds on our previous discussion. If the user is asking about a specific item from a list, clearly identify what they're referring to and provide extensive, detailed information about that topic from the research documents. Do not provide condensed or brief answers - be thorough and comprehensive."""

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
        for msg in conversation_history[-8:]:  # Last 8 messages for broader context
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
4. For follow-up questions (like "more on the second one"), carefully review the previous conversation to understand what the user is referring to, then provide detailed, expanded information about that specific topic
5. ALWAYS provide detailed, comprehensive answers - do not be overly brief or condensed
6. Include specific details, mechanisms, findings, and evidence from the research papers
7. If the user asks about a numbered item from a previous list, clearly identify which item they mean and provide extensive detail about it

{history_str}Context:
{context}

Question: {query}

Provide a detailed, comprehensive, and informative answer based on the available scientific context. Include specific details and evidence from the research papers:"""

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
            max_output_tokens=4000,  # Increased for comprehensive, detailed responses
        )
    )

    return response.text.strip()


def extract_sources_from_results(results: Dict[str, List]) -> List[Source]:
    """
    Extract source information from query results.

    Args:
        results: Dict from vector store

    Returns:
        List of Source objects
    """
    sources = []

    for i, (doc, metadata, distance, chunk_id) in enumerate(zip(
        results["documents"], results["metadatas"], results["distances"], results["ids"]
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
    # Check for conversational/introductory messages
    message_lower = message.lower().strip()

    # Handle common conversational messages
    conversational_responses = {
        "hello": "Hello! I'm a drug repurposing research assistant. I can help you explore scientific research about drug repurposing opportunities. What drug would you like to learn about?",
        "hi": "Hi there! I'm here to help you discover drug repurposing research from scientific papers. Which drug interests you?",
        "hey": "Hey! Ready to explore drug repurposing research? Ask me about any drug and I'll search through scientific literature to find repurposing opportunities.",
        "help": "I'm a drug repurposing research assistant powered by AI and scientific literature. I can:\n\n• Answer questions about any drug's repurposing potential\n• Search through research papers automatically\n• Provide evidence-based answers with citations\n• Analyze up to 10 scientific papers per drug\n\nTry asking: 'What are the repurposing opportunities for aspirin?' or enter any drug name!",
        "what can you do": "I can help you explore drug repurposing research! I search through scientific papers to find:\n\n• New medical uses for existing drugs\n• Research evidence and clinical studies\n• Cancer treatments, neurological applications, and more\n\nJust ask about any drug, and I'll analyze the latest research for you.",
        "how does this work": "I work by:\n\n1. **Searching** PubMed for research papers about your chosen drug\n2. **Downloading** open-access scientific papers (up to 10 per drug)\n3. **Analyzing** the content using AI to find repurposing opportunities\n4. **Answering** your questions with evidence from the research\n\nThe system learns about new drugs dynamically - no pre-loaded database needed!"
    }

    # Check for exact matches or close matches
    for key, response in conversational_responses.items():
        if key in message_lower or message_lower in [key, f"{key}?", f"{key}!", f"{key}."]:
            return {
                "answer": response,
                "sources": [],
                "session_id": session_id
            }

    # Also check for very short messages that are likely conversational
    if len(message_lower.split()) <= 3 and message_lower not in ["cancer", "research", "papers", "drugs", "drug"]:
        return {
            "answer": "I'm a drug repurposing research assistant. I help explore scientific research about using existing drugs for new medical purposes. Try asking me about a specific drug like 'aspirin' or 'metformin', or ask 'help' to learn more about what I can do!",
            "sources": [],
            "session_id": session_id
        }

    # Retrieve relevant chunks (increased top_k for better coverage)
    results = retrieve_relevant_chunks(message, drug_id, collection, settings, doc_id, top_k)

    # Debug: Log what we found
    print(f"DEBUG: Query '{message}' for drug '{drug_id}' found {len(results.get('documents', []))} documents")

    if not results["documents"]:
        # Try to provide more helpful guidance
        suggestions = [
            f"Try asking about '{drug_id}' and 'cancer', 'clinical trials', or 'neurological applications'",
            f"Ask about specific medical conditions or therapeutic areas",
            f"The research papers may not contain information about this exact topic",
            f"Try a different drug or more specific question"
        ]

        return {
            "answer": f"I searched through the available research papers but couldn't find specific information about '{message}' in relation to {drug_id}." +
                     (f" (filtered to document {doc_id})" if doc_id else "") +
                     "\n\n💡 **Suggestions:**\n" + "\n".join(f"• {s}" for s in suggestions[:2]),
            "sources": [],
            "session_id": session_id
        }

    # Generate answer using LLM with conversation context
    client = get_gemini_client(settings)
    context_chunks = results["documents"]

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
