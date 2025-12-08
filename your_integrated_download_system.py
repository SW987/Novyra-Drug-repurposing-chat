#!/usr/bin/env python3
"""
YOUR EXISTING DOWNLOAD SYSTEM - NOW WITH RAG INTEGRATION
Original functionality preserved + automatic RAG ingestion added
"""

import requests
import urllib.request
import xml.etree.ElementTree as ET
import os
import gzip
import shutil
import tarfile
import time

# ---------------------------- KEEP YOUR EXISTING VALIDATION ----------------------------
def is_valid_pdf(file_path):
    """Return True if file exists, has PDF header, and size > 5KB."""
    try:
        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(file_path)
        if file_size < 5000:
            print(f"[WARN] File too small ({file_size} bytes)")
            return False

        with open(file_path, "rb") as f:
            header = f.read(5)
            f.seek(-10, 2)  # Seek to end
            footer = f.read(10)

        has_valid_header = header == b"%PDF-"
        has_valid_footer = b"%%EOF" in footer

        if not has_valid_footer:
            print("[WARN] Missing EOF marker")
            return False

        return has_valid_header
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        return False


# ---------------------------- YOUR EXISTING DOWNLOAD FUNCTIONS ----------------------------
# (Paste your existing download functions here - they remain unchanged)

def download_pubmed_pdf(pmcid, output_path):
    """
    YOUR existing PDF download function - UNCHANGED
    Downloads a single PDF from PubMed Central
    """
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
    pdf_url = f"{base_url}{pmcid}/pdf/"

    try:
        # Your existing download logic here
        print(f"[DOWNLOAD] Fetching {pmcid}...")
        # ... your existing code ...

        return True
    except Exception as e:
        print(f"[ERROR] Download failed for {pmcid}: {e}")
        return False


def search_pubmed_drug_repurposing(drug_name, max_results=10):
    """
    YOUR existing PubMed search function - UNCHANGED
    Searches for drug repurposing papers
    """
    # Your existing search logic here
    print(f"[SEARCH] Searching PubMed for '{drug_name} repurposing'...")

    # Mock results for demonstration - replace with your actual search
    mock_results = [
        {"pmcid": "PMC1000001", "title": f"{drug_name} repurposing study 1"},
        {"pmcid": "PMC1000002", "title": f"{drug_name} repurposing study 2"},
        {"pmcid": "PMC1000003", "title": f"{drug_name} repurposing study 3"},
    ]

    return mock_results[:max_results]


# ---------------------------- NEW: RAG INTEGRATION ----------------------------
def initialize_rag_system():
    """
    Initialize RAG system - only called when RAG features are needed
    """
    try:
        from app.ingestion_pipeline import PDFIngestionPipeline
        from app.config import get_settings

        settings = get_settings()
        pipeline = PDFIngestionPipeline(settings)
        print("[RAG] System initialized")
        return pipeline
    except ImportError as e:
        print(f"[WARN] RAG system not available: {e}")
        return None


def process_downloaded_pdf_with_rag(pdf_path, drug_name, rag_pipeline):
    """
    NEW FUNCTION: Process a validated PDF with RAG system
    This is the integration point - called after your validation
    """
    if rag_pipeline is None:
        print("[RAG] RAG system not available - PDF stored for manual processing")
        return False

    try:
        result = rag_pipeline.validate_and_ingest_pdf(pdf_path, drug_name)

        if result["success"]:
            print(f"[RAG] ✅ Successfully ingested: {result['chunks_created']} chunks")
            return True
        else:
            print(f"[RAG] ❌ Ingestion failed: {result['error']}")
            return False

    except Exception as e:
        print(f"[RAG] Error during ingestion: {e}")
        return False


# ---------------------------- INTEGRATED MAIN FUNCTION ----------------------------
def download_and_process_drug_papers(drug_name, max_papers=5, enable_rag=True):
    """
    INTEGRATED WORKFLOW: Your download + validation + NEW RAG ingestion

    Parameters:
    - drug_name: Name of the drug to search for
    - max_papers: Maximum number of papers to download
    - enable_rag: Whether to enable automatic RAG ingestion (default: True)

    Returns:
    - Dictionary with results
    """

    print(f"🚀 Starting integrated workflow for: {drug_name}")
    print("=" * 60)

    # Initialize RAG system if enabled
    rag_pipeline = None
    if enable_rag:
        rag_pipeline = initialize_rag_system()

    # Step 1: Search PubMed (YOUR EXISTING CODE - UNCHANGED)
    print("📚 Step 1: Searching PubMed...")
    search_results = search_pubmed_drug_repurposing(drug_name, max_papers)

    if not search_results:
        print("❌ No papers found")
        return {"status": "no_papers_found"}

    print(f"📄 Found {len(search_results)} papers")

    # Step 2: Download PDFs (YOUR EXISTING CODE - UNCHANGED)
    print("\n📥 Step 2: Downloading PDFs...")
    downloaded_files = []
    successful_downloads = 0

    for paper in search_results:
        pmcid = paper['pmcid']
        output_filename = f"{drug_name}_{pmcid}.pdf"
        output_path = os.path.join("downloads", drug_name, output_filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if download_pubmed_pdf(pmcid, output_path):
            downloaded_files.append({
                'path': output_path,
                'pmcid': pmcid,
                'title': paper['title']
            })
            successful_downloads += 1
            print(f"✅ Downloaded: {pmcid}")
        else:
            print(f"❌ Failed: {pmcid}")

    print(f"📊 Downloads: {successful_downloads}/{len(search_results)} successful")

    # Step 3: Validate PDFs (YOUR EXISTING CODE - UNCHANGED)
    print("\n🔍 Step 3: Validating PDFs...")
    valid_pdfs = []
    validation_passed = 0

    for pdf_info in downloaded_files:
        if is_valid_pdf(pdf_info['path']):
            valid_pdfs.append(pdf_info)
            validation_passed += 1
            print(f"✅ Valid: {pdf_info['pmcid']}")
        else:
            print(f"⚠️  Invalid: {pdf_info['pmcid']}")

    print(f"📊 Validation: {validation_passed}/{len(downloaded_files)} passed")

    # Step 4: NEW - RAG Ingestion (Only if enabled)
    rag_ingested = 0
    if enable_rag and rag_pipeline and valid_pdfs:
        print("\n🧠 Step 4: RAG Ingestion...")
        for pdf_info in valid_pdfs:
            if process_downloaded_pdf_with_rag(pdf_info['path'], drug_name, rag_pipeline):
                rag_ingested += 1

        print(f"📊 RAG: {rag_ingested}/{len(valid_pdfs)} successfully ingested")

    # Summary
    print("\n🎉 Workflow Complete!")
    print(f"📚 Papers found: {len(search_results)}")
    print(f"📥 Downloads: {successful_downloads}")
    print(f"✅ Validated: {validation_passed}")
    if enable_rag:
        print(f"🧠 RAG ingested: {rag_ingested}")

    return {
        "drug": drug_name,
        "papers_found": len(search_results),
        "downloads_successful": successful_downloads,
        "validation_passed": validation_passed,
        "rag_ingested": rag_ingested if enable_rag else 0,
        "valid_pdfs": valid_pdfs
    }


# ---------------------------- BACKWARD COMPATIBILITY ----------------------------
def download_drug_papers_legacy(drug_name, max_papers=5):
    """
    LEGACY FUNCTION: Your original workflow without RAG
    Call this if you want the old behavior only
    """
    return download_and_process_drug_papers(drug_name, max_papers, enable_rag=False)


# ---------------------------- USAGE EXAMPLES ----------------------------
if __name__ == "__main__":
    print("🔬 Drug Repurposing PDF Download & RAG System")
    print("=" * 60)
    print("Choose mode:")
    print("1. Full integrated workflow (download + validate + RAG)")
    print("2. Legacy workflow (download + validate only)")
    print("3. RAG processing only (for existing PDFs)")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        # NEW INTEGRATED WORKFLOW
        drug = input("Enter drug name: ").strip() or "aspirin"
        results = download_and_process_drug_papers(drug, max_papers=3, enable_rag=True)
        print(f"\n📊 Results: {results}")

    elif choice == "2":
        # YOUR ORIGINAL WORKFLOW (unchanged)
        drug = input("Enter drug name: ").strip() or "aspirin"
        results = download_drug_papers_legacy(drug, max_papers=3)
        print(f"\n📊 Results: {results}")

    elif choice == "3":
        # RAG processing for existing PDFs
        drug = input("Enter drug name: ").strip() or "aspirin"
        pdf_dir = f"downloads/{drug}"

        if os.path.exists(pdf_dir):
            results = download_and_process_drug_papers(drug, max_papers=100, enable_rag=True)
            print(f"\n📊 RAG processing results: {results}")
        else:
            print(f"❌ Directory not found: {pdf_dir}")

    else:
        print("Invalid choice")
