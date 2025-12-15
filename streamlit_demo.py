#!/usr/bin/env python3
"""
Self-Contained Streamlit Demo for Drug Repurposing Chat System
Combines frontend and backend in one deployable application
"""

import streamlit as st
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import our backend modules
try:
    from app.config import Settings, get_settings
    from app.vector_store import init_vector_store
    from app.rag import chat_with_documents
    from app.schemas import Message
    settings = get_settings()
    collection = init_vector_store(settings)
except ImportError as e:
    st.error(f"❌ Failed to import backend modules: {e}")
    st.error("Make sure all backend files are in the app/ directory")
    st.stop()

# Available drugs (pre-loaded)
AVAILABLE_DRUGS = {
    "aspirin": "Aspirin - Cancer Prevention & Cardiovascular",
    "apomorphine": "Apomorphine - Parkinson's & Addiction Treatment",
    "insulin": "Insulin - Metabolic & Research Applications"
}

# Function to persist processed drugs across sessions (limited on free tier)
@st.cache_data
def load_persistent_drugs():
    """Load previously processed drugs (limited persistence on free tier)"""
    return set()

@st.cache_data
def save_persistent_drugs(drugs_set):
    """Save processed drugs (limited persistence on free tier)"""
    return drugs_set.copy()

def init_session_state():
    """Initialize session state for chat history"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_drug" not in st.session_state:
        st.session_state.current_drug = "aspirin"
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"streamlit_{int(time.time())}"
    if "processed_drugs" not in st.session_state:
        # Load any persistently saved drugs and merge with session
        persistent_drugs = load_persistent_drugs()
        # Also discover drugs that exist in the vector store
        existing_drugs = discover_existing_drugs()
        st.session_state.processed_drugs = persistent_drugs.union(existing_drugs)  # Track which custom drugs have been processed

        # Add existing drugs to AVAILABLE_DRUGS so they appear as pre-loaded options
        global AVAILABLE_DRUGS
        for drug in existing_drugs:
            if drug not in AVAILABLE_DRUGS:
                AVAILABLE_DRUGS[drug] = f"{drug.title()} - Custom Analysis"

@st.cache_resource
def get_collection():
    """Get or create the vector collection (cached to avoid reinitialization)"""
    try:
        return collection
    except NameError:
        # Fallback if global collection not available
        settings = get_settings()
        return init_vector_store(settings)

@st.cache_data
def discover_existing_drugs():
    """Discover drugs that already have data in the vector store"""
    try:
        collection = get_collection()
        # Get all metadata to find unique drug_ids
        results = collection.get(include=["metadatas"], limit=10000)  # Large limit to get all
        existing_drugs = set()

        if results.get('metadatas'):
            for metadata in results['metadatas']:
                if metadata and 'drug_id' in metadata:
                    existing_drugs.add(metadata['drug_id'])

        return existing_drugs
    except Exception:
        return set()

def process_custom_drug(drug_name):
    """Process a custom drug: download papers, extract text, and make available for chat"""
    try:
        # Check if drug is already processed in current session
        if drug_name in st.session_state.processed_drugs:
            st.success(f"✅ '{drug_name.title()}' already processed and ready for chat!")
            return True

        # Check if drug data already exists in vector store
        collection = get_collection()
        try:
            # Query for existing drug data
            existing_docs = collection.get(where={"drug_id": drug_name}, limit=1)
            if existing_docs['documents']:
                st.success(f"✅ '{drug_name.title()}' data found in database - ready for chat!")
                st.session_state.processed_drugs.add(drug_name)
                # Add to available drugs for future sessions
                AVAILABLE_DRUGS[drug_name] = f"{drug_name.title()} - Custom Analysis"
                return True
        except Exception:
            # Vector store might be empty or corrupted, continue with download
            pass

        # Import the integrated download system
        from app.ingestion_pipeline import PDFIngestionPipeline
        from app.config import get_settings

        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Initialize RAG system
        status_text.text("🔧 Initializing RAG system...")
        progress_bar.progress(10)

        settings = get_settings()
        pipeline = PDFIngestionPipeline(settings)

        progress_bar.progress(20)

        # Step 2: Search and download papers
        status_text.text(f"🔍 Searching PubMed for '{drug_name}' repurposing papers...")
        progress_bar.progress(30)

        # Run the integrated workflow
        result = pipeline.download_and_ingest_drug_papers(drug_name, max_papers=3)

        progress_bar.progress(80)

        # Step 3: Process results
        status_text.text("🧠 Processing downloaded papers...")
        progress_bar.progress(90)

        if result["downloaded"] > 0:
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")

            # Show results
            st.success(f"🎉 Successfully processed {result['downloaded']} papers for '{drug_name}'!")
            st.info(f"📊 Found: {result['papers_found']} papers, Downloaded: {result['downloaded']}, Ingested: {result['ingested']}")

            # Mark as processed and save persistently
            st.session_state.processed_drugs.add(drug_name)
            save_persistent_drugs(st.session_state.processed_drugs)

            # Add to AVAILABLE_DRUGS so it appears as pre-loaded option for future sessions
            AVAILABLE_DRUGS[drug_name] = f"{drug_name.title()} - Custom Analysis"

            # Show that chat is now available
            st.info("💬 You can now chat about this drug using the research papers!")

            return True
        else:
            progress_bar.progress(100)
            status_text.text("❌ No papers downloaded for this drug.")
            st.warning(f"No research papers could be downloaded for '{drug_name}' repurposing. Try a different drug name or check the spelling.")
            return False

    except Exception as e:
        st.error(f"❌ Error processing drug: {str(e)}")
        return False
    finally:
        # Clean up progress indicators
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass

def make_chat_request(drug_id: str, message: str, session_id: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Make a chat request using the integrated RAG system"""
    try:
        # Convert conversation history to Message objects
        message_history = []
        if conversation_history:
            for msg in conversation_history:
                message_history.append(Message(
                    role=msg["role"],
                    content=msg["content"]
                ))

        # Get the vector collection
        vector_collection = get_collection()

        # Call the RAG system directly
        result = chat_with_documents(
            session_id=session_id,
            drug_id=drug_id,
            message=message,
            collection=vector_collection,
            settings=settings,
            conversation_history=message_history if message_history else None
        )

        return result

    except Exception as e:
        return {"error": f"RAG system error: {str(e)}"}

def display_source(source, index: int) -> None:
    """Display a source with proper formatting"""
    # Handle both Source objects and dictionaries for compatibility
    if hasattr(source, 'doc_id'):  # Source object
        doc_id = source.doc_id
        doc_title = source.doc_title
        distance = source.distance
        text_preview = source.text_preview
    else:  # Dictionary (legacy support)
        doc_id = source['doc_id']
        doc_title = source['doc_title']
        distance = source['distance']
        text_preview = source['text_preview']

    # Generate truly unique key using timestamp and random component
    import time
    import random
    unique_key = f"preview_{doc_id}_{index}_{int(time.time()*1000)}_{random.randint(1000,9999)}"

    with st.expander(f"📄 {doc_title} (Distance: {distance:.3f})", expanded=False):
        st.write(f"**Document ID:** {doc_id}")
        st.write(f"**Text Preview:**")
        st.text_area("", text_preview, height=100, disabled=True, key=unique_key)

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper formatting"""
    if is_user:
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["answer"])

            if "sources" in message and message["sources"]:
                st.write("**📚 Sources:**")
                for i, source in enumerate(message["sources"]):
                    display_source(source, i)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Drug Repurposing Chat",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("💊 Drug Repurposing Research Chat")
    st.markdown("*Powered by RAG (Retrieval-Augmented Generation) - Ask about ANY drug!*")
    st.markdown("*Supports pre-loaded drugs (aspirin, apomorphine, insulin) + custom drug analysis*")

    # Initialize session state
    init_session_state()

    # Sidebar for drug selection and info
    with st.sidebar:
        st.header("🔬 Drug Selection")

        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["Pre-loaded Drugs", "Custom Drug (Download & Process)"],
            key="mode_selector",
            help="Choose between drugs with existing data or enter any drug for analysis"
        )

        st.markdown("---")

        if mode == "Pre-loaded Drugs":
            # Drug selector for pre-loaded drugs
            selected_drug_display = st.selectbox(
                "Choose a drug to discuss:",
                options=list(AVAILABLE_DRUGS.keys()),
                format_func=lambda x: AVAILABLE_DRUGS[x],
                index=list(AVAILABLE_DRUGS.keys()).index(st.session_state.current_drug),
                key="drug_selector"
            )

            # Update current drug if changed
            if selected_drug_display != st.session_state.current_drug:
                st.session_state.current_drug = selected_drug_display
                st.session_state.messages = []  # Clear chat when switching drugs
                st.session_state.session_id = f"streamlit_{int(time.time())}"

        else:  # Custom Drug Mode
            st.subheader("🔍 Custom Drug Analysis")

            custom_drug = st.text_input(
                "Enter any drug name:",
                placeholder="e.g., metformin, ibuprofen, hydroxychloroquine",
                key="custom_drug_input"
            )

            # Debug: Show current input
            if custom_drug:
                st.write(f"💡 You entered: **{custom_drug}**")

            # Show analysis button
            analyze_clicked = st.button("🚀 Analyze Drug", key="analyze_button", type="primary")

            if analyze_clicked:
                if custom_drug and custom_drug.strip():
                    custom_drug_clean = custom_drug.strip().lower()

                    # Update session state immediately
                    st.session_state.current_drug = custom_drug_clean
                    st.session_state.messages = []
                    st.session_state.session_id = f"streamlit_custom_{int(time.time())}"

                    # Show immediate feedback
                    st.success(f"✅ Drug '{custom_drug_clean.title()}' selected for analysis!")
                    st.balloons()  # Celebration effect

                    # Trigger automatic processing
                    process_custom_drug(custom_drug_clean)

                else:
                    st.error("❌ Please enter a drug name first")

            # Always show current drug status if it's a custom drug
            if st.session_state.current_drug and st.session_state.current_drug not in AVAILABLE_DRUGS:
                drug_title = st.session_state.current_drug.title()
                if st.session_state.current_drug in st.session_state.processed_drugs:
                    st.success(f"📊 **Active Drug Analysis**: {drug_title} ✅ (Data Ready)")
                    st.info("💬 You can now chat about this drug using the research papers below!")
                else:
                    st.info(f"📊 **Active Drug Analysis**: {drug_title}")
                    st.warning("⚠️ Click '🚀 Analyze Drug' to automatically download papers and enable chat.")

            if st.session_state.current_drug and st.session_state.current_drug not in AVAILABLE_DRUGS:
                st.info(f"📊 Currently analyzing: **{st.session_state.current_drug.title()}**")
                st.warning("⚠️ This drug hasn't been processed yet. In full implementation, this would trigger automatic PDF download and RAG ingestion.")

        st.markdown("---")

        # Current drug info
        if st.session_state.current_drug in AVAILABLE_DRUGS:
            drug_name = AVAILABLE_DRUGS[st.session_state.current_drug].split(' - ')[0]
            drug_focus = AVAILABLE_DRUGS[st.session_state.current_drug].split(' - ')[1]
        else:
            # Custom drug
            drug_name = st.session_state.current_drug.title()
            drug_focus = "Custom Drug Analysis (Processing Required)"

        st.subheader(f"Current Drug: {drug_name}")
        st.write(f"**Focus:** {drug_focus}")

        # System status
        st.markdown("---")
        st.subheader("🔍 System Status")

        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ Chat service is online")
            else:
                st.error("❌ Chat service is offline")
        except:
            st.error("❌ Cannot connect to chat service")
            st.info("💡 **Required**: Run `python -m app.main` in a separate terminal")
            st.info("🔗 This starts the FastAPI backend for chat functionality")

        # Available drugs list
        st.markdown("---")
        st.subheader("📚 Available Options")

        with st.expander("✅ Pre-loaded Drugs (Ready to chat)", expanded=True):
            for drug_id, description in AVAILABLE_DRUGS.items():
                st.write(f"• **{description.split(' - ')[0]}**: {description.split(' - ')[1]}")

        with st.expander("🔧 Custom Drugs (Dynamic Download)", expanded=False):
            st.write("• **Any drug name** you enter")
            st.write("• Automatic PubMed search & PDF download")
            st.write("• Real-time text extraction & RAG processing")
            st.info("💡 Enter any drug in the custom mode above to automatically download and process research papers!")

    # Main chat interface
    if st.session_state.current_drug in AVAILABLE_DRUGS:
        drug_display = AVAILABLE_DRUGS[st.session_state.current_drug].split(' - ')[0]
    else:
        drug_display = st.session_state.current_drug.title()

    st.header(f"💬 Chat about {drug_display} Repurposing")

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message, message.get("is_user", False))

    # Get current drug display name and data availability
    if st.session_state.current_drug in AVAILABLE_DRUGS:
        drug_display = AVAILABLE_DRUGS[st.session_state.current_drug].split(' - ')[0]
        has_data = True
    elif st.session_state.current_drug in st.session_state.processed_drugs:
        drug_display = st.session_state.current_drug.title()
        has_data = True  # Custom drug has been processed
    else:
        drug_display = st.session_state.current_drug.title()
        has_data = False  # Custom drugs don't have data yet

    # Chat input
    if has_data:
        placeholder_text = f"Ask about {drug_display} repurposing..."
    else:
        placeholder_text = f"Process '{drug_display}' first using the sidebar, then you can chat!"
    if prompt := st.chat_input(placeholder_text):
        # Add user message to history
        user_message = {"content": prompt, "is_user": True}
        st.session_state.messages.append(user_message)
        display_chat_message(user_message, True)

        if has_data:
            # Build conversation history for API (exclude current message)
            conversation_history = []
            for msg in st.session_state.messages[:-1]:  # Exclude the current user message
                if msg.get("is_user"):
                    conversation_history.append({"role": "user", "content": msg["content"]})
                elif "answer" in msg:
                    conversation_history.append({"role": "assistant", "content": msg["answer"]})

            # Make API request with conversation history
            with st.spinner("🔍 Searching research documents..."):
                response = make_chat_request(
                    st.session_state.current_drug,
                    prompt,
                    st.session_state.session_id,
                    conversation_history
                )

            # Handle response
            if "error" in response:
                error_message = {
                    "answer": f"❌ Error: {response['error']}",
                    "sources": []
                }
                st.session_state.messages.append(error_message)
                display_chat_message(error_message, False)
            else:
                # Add assistant response to history
                st.session_state.messages.append(response)
                display_chat_message(response, False)
        else:
            # Custom drug without data - show informative placeholder response
            placeholder_response = {
                "answer": f"I understand you're asking about **{drug_display}** repurposing. In the full implementation, I would automatically:\n\n"
                        f"1. 🔍 **Search PubMed** for '{st.session_state.current_drug} repurposing' papers\n"
                        f"2. 📥 **Download relevant PDFs** from research databases\n"
                        f"3. 📖 **Extract and chunk** the scientific content\n"
                        f"4. 🧠 **Generate embeddings** for semantic search\n"
                        f"5. 💬 **Provide detailed answers** based on the research literature\n\n"
                        f"This system is designed to handle **any drug name** you enter - from common medications "
                        f"like ibuprofen and metformin to experimental compounds. The AI would analyze "
                        f"thousands of research papers to give you evidence-based insights about drug repurposing opportunities!",
                "sources": []
            }
            st.session_state.messages.append(placeholder_response)
            display_chat_message(placeholder_response, False)

    # Footer
    st.markdown("---")
    st.markdown("*Built with FastAPI, ChromaDB, Google Gemini, and Streamlit*")
    st.markdown("*RAG system retrieves and synthesizes information from scientific research papers*")

if __name__ == "__main__":
    main()
