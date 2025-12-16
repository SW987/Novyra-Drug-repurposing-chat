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
    from app.vector_store import init_vector_store, is_chromadb_corrupted, reset_chromadb
    from app.rag import chat_with_documents
    from app.schemas import Message
    settings = get_settings()
    # IMPORTANT: do NOT initialize Chroma at import-time.
    # If init fails, Streamlit never renders and cannot show recovery UI.
    collection = None
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

# Global variable to store processed drugs across sessions (limited on free tier)
_persistent_drugs_cache = set()
@st.cache_data
def load_persistent_drugs():
    """Load previously processed drugs (limited persistence on free tier)"""
    global _persistent_drugs_cache
    return _persistent_drugs_cache.copy()

@st.cache_data
def save_persistent_drugs(drugs_set):
    """Save processed drugs (limited persistence on free tier)"""
    global _persistent_drugs_cache
    _persistent_drugs_cache = drugs_set.copy()
    # Force cache invalidation by returning a new value
    return _persistent_drugs_cache.copy()

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
    global collection
    settings = get_settings()
    
    try:
        # Check if global collection exists and is valid
        if 'collection' in globals() and collection is not None:
            # Test if collection is accessible
            try:
                collection.get(limit=1)  # Quick test query
                return collection
            except Exception as test_error:
                # Collection exists but may be corrupted
                if is_chromadb_corrupted(test_error):
                    print("⚠️ Collection corrupted, resetting...")
                    if reset_chromadb(settings):
                        # Reinitialize after reset
                        collection = init_vector_store(settings, reset_on_corruption=False)
                        return collection
                # Reinitialize for other errors
                pass
    except NameError:
        pass
    
    # Fallback: reinitialize collection
    try:
        collection = init_vector_store(settings)
        return collection
    except Exception as e:
        if is_chromadb_corrupted(e):
            print("⚠️ Database corrupted during get_collection, resetting...")
            if reset_chromadb(settings):
                collection = init_vector_store(settings, reset_on_corruption=False)
                return collection
        raise

@st.cache_data
def discover_existing_drugs():
    """Discover drugs that already have data in the vector store"""
    try:
        collection = get_collection()
        # Get all metadata to find unique drug_ids
        results = collection.get(include=["metadatas"], limit=10000)  # Large limit to get all
        existing_drugs = set()

        metadatas = results.get('metadatas', [])
        if metadatas:
            # Handle nested lists (ChromaDB can return lists of lists)
            if isinstance(metadatas, list) and len(metadatas) > 0:
                # Check if first element is a list (nested structure)
                if isinstance(metadatas[0], list):
                    # Flatten nested structure
                    for metadata_list in metadatas:
                        for metadata in metadata_list:
                            if metadata and isinstance(metadata, dict) and 'drug_id' in metadata:
                                existing_drugs.add(str(metadata['drug_id']).strip().lower())
                else:
                    # Flat structure
                    for metadata in metadatas:
                        if metadata and isinstance(metadata, dict) and 'drug_id' in metadata:
                            existing_drugs.add(str(metadata['drug_id']).strip().lower())

        return existing_drugs
    except Exception as e:
        print(f"Error discovering existing drugs: {e}")
        return set()

def process_custom_drug(drug_name):
    """Process a custom drug: download papers, extract text, and make available for chat"""
    # Initialize progress indicators early
    progress_bar = None
    status_text = None
    
    try:
        # Normalize drug name (internal IDs are always lowercase)
        drug_name = (drug_name or "").strip().lower()
        if not drug_name:
            st.error("❌ Please enter a drug name.")
            return False

        # Check if drug is already processed in current session
        if drug_name in st.session_state.processed_drugs:
            st.success(f"✅ '{drug_name.title()}' already processed and ready for chat!")
            return True

        # Check if drug data already exists in vector store
        try:
            collection = get_collection()
            # Query for existing drug data
            existing_docs = collection.get(where={"drug_id": drug_name}, limit=1)
            if existing_docs.get('documents') and len(existing_docs.get('documents', [])) > 0:
                st.success(f"✅ '{drug_name.title()}' data found in database - ready for chat!")
                st.session_state.processed_drugs.add(drug_name)
                save_persistent_drugs(st.session_state.processed_drugs)  # Save persistently
                # Add to available drugs for future sessions
                AVAILABLE_DRUGS[drug_name] = f"{drug_name.title()} - Custom Analysis"
                return True
        except Exception as check_error:
            # Vector store might be empty or corrupted, continue with download
            print(f"Note: Could not check existing data: {check_error}")
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
        pipeline = PDFIngestionPipeline(settings, collection=get_collection())

        progress_bar.progress(20)

        # Step 2: Search and download papers
        status_text.text(f"🔍 Searching PubMed for '{drug_name}' repurposing papers (searching 50 results, downloading 2)...")
        progress_bar.progress(30)

        # Run the integrated workflow
        try:
            result = pipeline.download_and_ingest_drug_papers(drug_name, max_papers=2)
        except Exception as pipeline_error:
            progress_bar.progress(100)
            status_text.text("❌ Pipeline error")
            error_msg = str(pipeline_error)
            print(f"ERROR in pipeline: {error_msg}")
            st.error(f"❌ **Error during download/processing:** {error_msg[:300]}")
            if "corrupt" in error_msg.lower() or "database" in error_msg.lower():
                st.warning("💡 Database issue detected. The app will attempt to recover automatically.")
            return False

        progress_bar.progress(80)

        # Step 3: Process results
        status_text.text("🧠 Processing downloaded papers...")
        progress_bar.progress(90)

        if result and result.get("downloaded", 0) > 0:
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")

            # Show results
            st.success(f"🎉 Successfully processed {result['downloaded']} papers for '{drug_name}'!")
            st.info(f"📊 Found: {result.get('papers_found', 0)} papers, Downloaded: {result['downloaded']}, Ingested: {result.get('ingested', 0)}")

            # Mark as processed and save persistently
            st.session_state.processed_drugs.add(drug_name)
            save_persistent_drugs(st.session_state.processed_drugs)

            # Add to AVAILABLE_DRUGS so it appears as pre-loaded option
            AVAILABLE_DRUGS[drug_name] = f"{drug_name.title()} - Custom Analysis"
            
            # Also add to processed_drugs in session state to ensure it's tracked
            st.session_state.processed_drugs.add(drug_name)
            
            # Rediscover all drugs to ensure dropdown is up to date
            try:
                existing_drugs = discover_existing_drugs()
                for drug in existing_drugs:
                    if drug not in AVAILABLE_DRUGS:
                        AVAILABLE_DRUGS[drug] = f"{drug.title()} - Custom Analysis"
            except Exception as e:
                print(f"Note: Could not rediscover drugs: {e}")

            # Show that chat is now available
            st.success("💬 You can now chat about this drug using the research papers!")
            st.success(f"✅ **'{drug_name.title()}' is now available in the 'Pre-loaded Drugs' dropdown!** Switch modes to see it.")

            return True
        else:
            progress_bar.progress(100)
            status_text.text("❌ No papers downloaded for this drug.")
            papers_found = result.get('papers_found', 0) if result else 0
            if papers_found > 0:
                st.warning(f"⚠️ Found {papers_found} papers but couldn't download open-access PDFs for '{drug_name}'. Try a different drug name.")
            else:
                st.warning(f"⚠️ No research papers found for '{drug_name}' repurposing. Try a different drug name or check the spelling.")
            return False

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in process_custom_drug: {error_details}")
        try:
            progress_bar.progress(100)
            status_text.text("❌ Error occurred")
        except:
            pass
        error_msg = str(e)
        st.error(f"❌ **Error processing drug:** {error_msg[:300]}")
        if len(error_msg) > 300:
            st.info("💡 Check the console/logs for full error details.")
        return False
    finally:
        # Clean up progress indicators
        try:
            if progress_bar is not None:
                progress_bar.empty()
            if status_text is not None:
                status_text.empty()
        except:
            pass

def make_chat_request(drug_id: str, message: str, session_id: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Make a chat request using the integrated RAG system"""
    try:
        # Get the vector collection (with retry for SessionInfo errors)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                vector_collection = get_collection()
                # Test if collection is accessible
                vector_collection.get(limit=1)
                break  # Success, exit retry loop
            except Exception as coll_error:
                error_str = str(coll_error).lower()
                if "sessioninfo" in error_str or ("session" in error_str and "initialized" in error_str):
                    print(f"⚠️ ChromaDB session error on attempt {attempt + 1}, retrying...")
                    if attempt < max_retries - 1:
                        # Clear cache and retry
                        get_collection.clear()
                        continue
                    else:
                        return {"error": f"Database session error: {str(coll_error)}. Please refresh the page."}
                else:
                    raise  # Re-raise if not a session error

        # Call the RAG system directly
        # conversation_history is already in dict format, which chat_with_documents accepts
        result = chat_with_documents(
            session_id=session_id,
            drug_id=drug_id,
            message=message,
            collection=vector_collection,
            settings=settings,
            conversation_history=conversation_history
        )

        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in make_chat_request: {error_details}")
        error_str = str(e).lower()
        if "sessioninfo" in error_str or ("session" in error_str and "initialized" in error_str):
            return {"error": "Database session error. Please refresh the page and try again."}
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
    
    # Discover existing drugs from vector store and add to AVAILABLE_DRUGS
    # This ensures custom drugs appear in the dropdown
    try:
        existing_drugs = discover_existing_drugs()
        for drug in existing_drugs:
            if drug not in AVAILABLE_DRUGS:
                AVAILABLE_DRUGS[drug] = f"{drug.title()} - Custom Analysis"
        # Update processed_drugs with discovered drugs
        st.session_state.processed_drugs.update(existing_drugs)
    except Exception as e:
        print(f"Note: Could not discover existing drugs: {e}")

    # Auto-initialize pre-loaded drugs if database is empty (check persistent vector store)
    # Use session state to prevent running on every Streamlit rerun
    if "db_init_checked" not in st.session_state:
        st.session_state.db_init_checked = False
    if "db_init_in_progress" not in st.session_state:
        st.session_state.db_init_in_progress = False
        
    if not st.session_state.db_init_checked:
        # Create container for initialization messages
        init_container = st.container()
        
        try:
            # Global lock to prevent multiple sessions triggering initialization concurrently
            # (common on Railway: multiple connections + Streamlit reruns)
            init_lock_path = Path("/tmp") / "nuvyra_preload_init.lock"
            # If the container restarted mid-init, the lock can be left behind.
            # Treat old locks as stale and auto-clear them.
            if init_lock_path.exists():
                stale_seconds = 2 * 60  # 2 minutes (fast recovery for deployments)
                is_stale = False
                try:
                    ts = float(init_lock_path.read_text().strip())
                    if time.time() - ts > stale_seconds:
                        is_stale = True
                except Exception:
                    # If we can't parse, assume stale
                    is_stale = True

                if is_stale:
                    try:
                        init_lock_path.unlink()
                        print("ℹ️ Cleared stale init lock file")
                    except Exception:
                        pass
                else:
                    with init_container:
                        st.info("🔄 Database initialization is already running in another session. Please wait a moment and refresh.")
                        # Allow manual recovery if init crashed or got wedged
                        if st.button("🔓 Force unlock & retry initialization", key="force_unlock_init"):
                            try:
                                init_lock_path.unlink()
                            except Exception:
                                pass
                            # Reset flags so init can run again
                            st.session_state.db_init_checked = False
                            st.session_state.db_init_in_progress = False
                            try:
                                get_collection.clear()
                            except Exception:
                                pass
                            st.rerun()
                    st.session_state.db_init_in_progress = True
                    return
            try:
                init_lock_path.write_text(str(time.time()))
            except Exception:
                # If we can't write the lock, proceed anyway (best-effort)
                pass

            # Mark in session immediately so reruns don't start another init
            st.session_state.db_init_in_progress = True
            st.session_state.db_init_checked = True

            # Get collection (local variable, not modifying global)
            collection = get_collection()
            print("🔍 Checking for existing drug data in persistent vector store...")
            
            # Check if ANY drug data exists (not just aspirin)
            try:
                # First, try to get any documents
                all_results = collection.get(limit=1)
                ids = all_results.get('ids', [])
                print(f"🔍 Database check: Got {len(ids) if ids else 0} IDs from collection.get(limit=1)")
                
                # Check if any IDs exist (more reliable than checking documents)
                if ids and isinstance(ids, list) and len(ids) > 0:
                    # Handle nested lists from ChromaDB
                    if isinstance(ids[0], list):
                        # Nested structure - count all inner IDs
                        total_docs = sum(len(id_list) for id_list in ids if isinstance(id_list, list))
                        print(f"📊 Nested structure detected: {total_docs} documents")
                    else:
                        # Flat structure
                        total_docs = len(ids)
                        print(f"📊 Flat structure: {total_docs} documents")
                else:
                    total_docs = 0
                    print("📊 No IDs found in initial check")
                
                # If no results from get(), try querying for aspirin specifically as a fallback
                if total_docs == 0:
                    try:
                        print("🔍 Trying fallback: querying for aspirin specifically...")
                        aspirin_results = collection.get(where={"drug_id": "aspirin"}, limit=1)
                        aspirin_ids = aspirin_results.get('ids', [])
                        if aspirin_ids and len(aspirin_ids) > 0:
                            total_docs = 1  # At least one document exists
                            print("📊 Found data by querying for aspirin")
                    except Exception as fallback_error:
                        print(f"⚠️ Fallback query failed: {fallback_error}")
                        pass  # If query fails, assume no data
                
                print(f"📊 Final check: Total documents in database: {total_docs}")
            except Exception as db_error:
                # Handle ChromaDB corruption or errors
                from app.vector_store import is_chromadb_corrupted, reset_chromadb, init_vector_store
                error_str = str(db_error)
                
                if is_chromadb_corrupted(db_error):
                    print(f"⚠️ ChromaDB database corruption detected: {error_str}")
                    with init_container:
                        st.error("❌ **Database Corruption Detected**")
                        st.warning("⚠️ The ChromaDB database appears to be corrupted. Attempting to reset...")
                    
                    # Try to reset the database
                    try:
                        if reset_chromadb(settings):
                            with init_container:
                                st.success("✅ Database reset successful. Re-initializing collection...")
                            # Clear the cached collection to force re-initialization
                            get_collection.clear()  # Clear Streamlit cache
                            # Re-initialize the collection (local variable)
                            collection = init_vector_store(settings, reset_on_corruption=False)
                            # Re-check after reset
                            try:
                                all_results = collection.get(limit=1)
                                total_docs = 0  # Will trigger re-initialization
                            except Exception:
                                total_docs = 0
                        else:
                            with init_container:
                                st.error("❌ Failed to reset database. Please contact support.")
                            total_docs = 0
                            st.session_state.db_init_checked = True  # Prevent infinite loop
                    except Exception as reset_error:
                        print(f"❌ Failed to reset database: {reset_error}")
                        with init_container:
                            st.error(f"❌ Cannot recover from database corruption: {str(reset_error)[:200]}")
                        total_docs = 0
                        st.session_state.db_init_checked = True  # Prevent infinite loop
                else:
                    print(f"❌ Error checking database: {error_str}")
                    total_docs = 0

            if total_docs == 0:
                # Database is completely empty - initialize pre-loaded drugs
                print("🚀 Database is empty - initializing pre-loaded drugs...")
                # Use existing init_container, don't create a new one
                with init_container:
                    st.info("🔄 Setting up drug database (this may take a few minutes)...")
                    st.info("💡 This only happens once. Data will persist for future sessions.")

                try:
                    from app.ingestion_pipeline import PDFIngestionPipeline
                    pipeline = PDFIngestionPipeline(settings, collection=get_collection())
                    preloaded_drugs = ["aspirin", "apomorphine", "insulin"]

                    success_count = 0
                    failed_drugs = []
                    
                    for drug in preloaded_drugs:
                        try:
                            # Check if drug already has data in database
                            try:
                                drug_check = collection.get(where={"drug_id": drug}, limit=1)
                                if drug_check.get('documents') and len(drug_check.get('documents', [])) > 0:
                                    print(f"✅ {drug}: Already has data in database, skipping...")
                                    with init_container:
                                        st.success(f"✅ **{drug.title()}**: Already loaded (skipping)")
                                    st.session_state.processed_drugs.add(drug)
                                    success_count += 1
                                    continue  # Skip this drug, it's already processed
                            except Exception:
                                # If check fails, continue with download
                                pass
                            
                            print(f"📥 Processing {drug}...")
                            with init_container:
                                st.info(f"📥 Loading research for {drug}...")
                            result = pipeline.download_and_ingest_drug_papers(drug, max_papers=2)
                            downloaded = result.get('downloaded', 0)
                            ingested = result.get('ingested', 0)
                            papers_found = result.get('papers_found', 0)
                            print(f"📊 {drug}: Found {papers_found} papers, downloaded {downloaded}, ingested {ingested} chunks")
                            
                            if downloaded > 0 and ingested > 0:
                                with init_container:
                                    st.success(f"✅ **{drug.title()}**: {downloaded} papers downloaded, {ingested} chunks saved")
                                st.session_state.processed_drugs.add(drug)
                                success_count += 1
                                # Data is automatically persisted in vector store
                            elif downloaded > 0 and ingested == 0:
                                # Papers downloaded but failed to save (likely ChromaDB issue)
                                # Get detailed error from results
                                error_details = []
                                for res in result.get('results', []):
                                    if res.get('downloaded') and not res.get('ingested'):
                                        ingest_result = res.get('ingest_result', {})
                                        error_msg = ingest_result.get('error', 'Unknown ingestion error')
                                        error_details.append(error_msg)
                                
                                error_summary = error_details[0] if error_details else 'Unknown error'
                                print(f"⚠️ {drug}: Papers downloaded but failed to save to database. Error: {error_summary}")
                                with init_container:
                                    st.warning(f"⚠️ **{drug.title()}**: Downloaded {downloaded} papers but failed to save.")
                                    st.error(f"❌ **Error**: {error_summary[:200]}")
                                    if "corrupt" in error_summary.lower() or "database" in error_summary.lower():
                                        st.info("💡 Solution: The ChromaDB database may need to be reset. Contact support or check logs.")
                                failed_drugs.append(drug)
                            elif papers_found > 0 and downloaded == 0:
                                # Papers found but couldn't download (no OA PDFs)
                                with init_container:
                                    st.warning(f"⚠️ **{drug.title()}**: Found {papers_found} papers but no open-access PDFs available for download")
                                failed_drugs.append(drug)
                            else:
                                # No papers found at all
                                with init_container:
                                    st.warning(f"⚠️ **{drug.title()}**: No papers found in PubMed Central")
                                failed_drugs.append(drug)
                        except Exception as e:
                            error_msg = str(e)
                            print(f"❌ {drug} failed: {error_msg}")
                            failed_drugs.append(drug)
                            # Check if it's a ChromaDB corruption error
                            if "trailer" in error_msg.lower() or "not defined" in error_msg.lower() or "corrupt" in error_msg.lower():
                                with init_container:
                                    st.error(f"❌ **{drug.title()}**: Database corruption detected. ChromaDB may need to be reset.")
                            else:
                                with init_container:
                                    st.warning(f"⚠️ **{drug.title()}**: {error_msg[:100]}...")
                    
                    # Summary
                    with init_container:
                        st.markdown("---")
                        if success_count > 0:
                            st.success(f"🎉 **Initialization Summary**: {success_count} of {len(preloaded_drugs)} drugs loaded successfully")
                            st.info(f"📊 **Process**: For each drug, searched 50 papers and downloaded 2 open-access PDFs")
                        if failed_drugs:
                            st.warning(f"⚠️ **Failed to load**: {', '.join([d.title() for d in failed_drugs])}. You can still use custom drug search.")
                        st.markdown("---")

                    with init_container:
                        st.success("🎉 Pre-loaded drugs initialized! Data will persist for future sessions.")
                    st.session_state.db_init_in_progress = False
                    try:
                        if init_lock_path.exists():
                            init_lock_path.unlink()
                    except Exception:
                        pass
                except Exception as e:
                    error_msg = str(e)[:100]
                    print(f"❌ Drug initialization failed: {error_msg}")
                    with init_container:
                        st.warning(f"⚠️ Auto-initialization failed: {error_msg}")
                        st.info("💡 You can still use custom drug search below to load specific drugs manually.")
                    st.session_state.db_init_in_progress = False
                    try:
                        if init_lock_path.exists():
                            init_lock_path.unlink()
                    except Exception:
                        pass
            else:
                print(f"✅ Database already has {total_docs} documents - skipping auto-initialization")
                # Discover and add any existing drugs to processed_drugs
                existing_drugs = discover_existing_drugs()
                st.session_state.processed_drugs.update(existing_drugs)
                # Add existing drugs to AVAILABLE_DRUGS
                for drug in existing_drugs:
                    if drug not in AVAILABLE_DRUGS:
                        AVAILABLE_DRUGS[drug] = f"{drug.title()} - Custom Analysis"
                st.session_state.db_init_in_progress = False
                try:
                    if init_lock_path.exists():
                        init_lock_path.unlink()
                except Exception:
                    pass
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"❌ Initialization check error: {error_msg}")
            st.session_state.db_init_in_progress = False
            try:
                if 'init_lock_path' in locals() and init_lock_path.exists():
                    init_lock_path.unlink()
            except Exception:
                pass
            # Don't show error to user - just log it, app can still work

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
            # Rebuild AVAILABLE_DRUGS from discovered drugs to ensure it's up to date
            try:
                discovered_drugs = discover_existing_drugs()
                for drug in discovered_drugs:
                    if drug not in AVAILABLE_DRUGS:
                        AVAILABLE_DRUGS[drug] = f"{drug.title()} - Custom Analysis"
            except Exception as e:
                print(f"Note: Could not discover drugs for dropdown: {e}")

            # Also include any drugs processed in this session (even if discover fails)
            try:
                for drug in st.session_state.get("processed_drugs", set()):
                    d = str(drug).strip().lower()
                    if d and d not in AVAILABLE_DRUGS:
                        AVAILABLE_DRUGS[d] = f"{d.title()} - Custom Analysis"
            except Exception:
                pass
            
            # Drug selector for pre-loaded drugs
            # Safely determine the index for the current drug
            available_drugs_list = list(AVAILABLE_DRUGS.keys())
            current_index = 0  # Default to first drug
            current_drug_is_in_list = st.session_state.current_drug in available_drugs_list
            if current_drug_is_in_list:
                current_index = available_drugs_list.index(st.session_state.current_drug)

            selected_drug_display = st.selectbox(
                "Choose a drug to discuss:",
                options=available_drugs_list,
                format_func=lambda x: AVAILABLE_DRUGS[x],
                index=current_index,
                key="drug_selector"
            )

            # IMPORTANT:
            # If the current drug is a custom drug not present in the dropdown yet,
            # Streamlit will default-select the first option (e.g., aspirin). Do NOT treat
            # that as a user-initiated drug switch, or chats will appear to "hallucinate"
            # about the previous drug.
            if current_drug_is_in_list and selected_drug_display != st.session_state.current_drug:
                st.session_state.current_drug = selected_drug_display
                st.session_state.messages = []  # Clear chat when switching drugs
                st.session_state.session_id = f"streamlit_{int(time.time())}"
            elif not current_drug_is_in_list:
                st.info(
                    f"💡 You're currently chatting about custom drug **{st.session_state.current_drug.title()}**. "
                    f"Switch to **Custom Drug** mode to change or re-run analysis."
                )

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

        st.subheader("Available Options")


    # Chat Interface - Always visible in main content
    st.markdown("---")
    st.subheader("💬 Chat about Drug Repurposing")
    
    # Show database status (only check once, cache result - non-blocking)
    if "db_status_checked" not in st.session_state:
        st.session_state.db_status_checked = False
        st.session_state.db_has_data = None
    
    # Only check once, don't block on errors
    if not st.session_state.db_status_checked:
        try:
            # Quick non-blocking check
            status_collection = get_collection()
            if status_collection:
                status_check = status_collection.get(limit=1)
                has_data = status_check.get('documents') and len(status_check.get('documents', [])) > 0
                st.session_state.db_has_data = has_data
            else:
                st.session_state.db_has_data = None
            st.session_state.db_status_checked = True
        except Exception as e:
            # Don't block on errors, just mark as checked
            print(f"Status check error (non-blocking): {e}")
            st.session_state.db_has_data = None
            st.session_state.db_status_checked = True
    
    # Show status only if needed (quick display, no blocking)
    if st.session_state.db_has_data is False:
        if not st.session_state.db_init_checked:
            st.info("🔄 **Database is initializing...** Please wait for pre-loaded drugs to download (2-3 minutes).")
        else:
            st.warning("⚠️ **Database is empty.** Pre-loaded drugs may have failed to initialize. Try using custom drug search.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about drug repurposing research..."):
        if not st.session_state.current_drug:
            st.error("❌ Please select a drug first!")
            st.stop()

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing research papers..."):
                try:
                    result = make_chat_request(
                        drug_id=str(st.session_state.current_drug).strip().lower(),
                        message=prompt,
                        session_id=st.session_state.session_id,
                        conversation_history=st.session_state.messages
                    )

                    # Handle both success and error responses
                    if not result:
                        error_msg = "❌ **No response from server.** Please try again or refresh the page."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    elif "error" in result:
                        error_msg = f"❌ {result['error']}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    elif "answer" in result:
                        response_text = result["answer"]
                        st.markdown(response_text)

                        # Add assistant message to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                    else:
                        error_msg = f"❌ Unexpected response format: {list(result.keys()) if result else 'None'}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except KeyError as e:
                    error_msg = f"❌ Error: Missing key in response - {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"ERROR in chat: {error_details}")
                    error_msg = f"❌ Error: {str(e)[:300]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()



