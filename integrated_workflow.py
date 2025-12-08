#!/usr/bin/env python3
"""
INTEGRATED WORKFLOW: Your Download Code + RAG System
This shows how both systems work together
"""

import requests
import urllib.request
import xml.etree.ElementTree as ET
import os
import gzip
import shutil
import tarfile
import time

# Your existing validation function
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


# YOUR EXISTING DOWNLOAD SYSTEM (template)
def download_pubmed_pdfs(drug_name, max_papers=5):
    """
    YOUR existing download function - this is a template
    showing how it would integrate with RAG
    """
    print(f"🔍 Searching PubMed for '{drug_name} repurposing'...")

    # This is where YOUR existing PubMed API code would go
    # For demo, we'll simulate finding papers

    downloaded_pdfs = []

    # Simulate download results (replace with your actual code)
    for i in range(max_papers):
        paper_info = {
            'pmcid': f'PMC{1000000 + i}',
            'title': f'{drug_name.title()} repurposing study {i+1}',
            'file_path': f'/tmp/{drug_name}_paper_{i+1}.pdf',  # Your actual path
            'drug_name': drug_name
        }
        downloaded_pdfs.append(paper_info)

    return downloaded_pdfs


# INTEGRATED WORKFLOW
def integrated_download_and_ingest(drug_name):
    """
    COMPLETE workflow: Download → Validate → RAG Ingest
    """
    print(f"🚀 Starting integrated workflow for: {drug_name}")
    print("=" * 60)

    # STEP 1: DOWNLOAD (Your existing system)
    print("📥 STEP 1: Downloading PDFs...")
    downloaded_pdfs = download_pubmed_pdfs(drug_name, max_papers=3)

    if not downloaded_pdfs:
        print("❌ No papers found")
        return

    print(f"📄 Found {len(downloaded_pdfs)} papers")

    # STEP 2: VALIDATE + INGEST (Integrated)
    print("\n🔍 STEP 2: Validating and ingesting PDFs...")

    # Import RAG system (only when needed)
    from app.ingestion_pipeline import PDFIngestionPipeline
    from app.config import get_settings

    settings = get_settings()
    pipeline = PDFIngestionPipeline(settings)

    successful_ingests = 0
    total_chunks = 0

    for paper in downloaded_pdfs:
        pdf_path = paper['file_path']

        # Your existing validation
        if not is_valid_pdf(pdf_path):
            print(f"⚠️  Skipping invalid PDF: {paper['title']}")
            continue

        print(f"✅ Valid PDF: {paper['title']}")

        # NEW: RAG ingestion
        result = pipeline.validate_and_ingest_pdf(pdf_path, drug_name)

        if result["success"]:
            successful_ingests += 1
            total_chunks += result["chunks_created"]
            print(f"🧠 RAG: Created {result['chunks_created']} searchable chunks")
        else:
            print(f"❌ RAG failed: {result['error']}")

    print("\n🎉 Workflow Complete!")
    print(f"📊 Papers downloaded: {len(downloaded_pdfs)}")
    print(f"✅ Successfully ingested: {successful_ingests}")
    print(f"📝 Total chunks created: {total_chunks}")

    return {
        "downloaded": len(downloaded_pdfs),
        "ingested": successful_ingests,
        "chunks": total_chunks
    }


def demonstrate_current_separate_systems():
    """
    Show how the systems currently work separately
    """
    print("📋 CURRENT SYSTEM STATUS:")
    print("=" * 50)

    print("🔄 SYSTEM 1: Your Download + Validation")
    print("   ✅ Downloads PDFs from PubMed/PMC")
    print("   ✅ Validates PDF integrity")
    print("   ❌ PDFs are just stored on disk")

    print("\n🔄 SYSTEM 2: RAG Chat System")
    print("   ✅ Processes PDFs into searchable chunks")
    print("   ✅ Creates embeddings for semantic search")
    print("   ✅ Provides AI chat about drug repurposing")
    print("   ❌ Requires manual PDF ingestion")

    print("\n🔗 INTEGRATION NEEDED:")
    print("   🔄 Connect download output → RAG input")
    print("   📝 Add RAG ingestion after validation")


def show_integration_code():
    """
    Show the exact code to add to integrate the systems
    """
    print("\n💻 CODE TO ADD FOR INTEGRATION:")
    print("=" * 50)

    integration_code = '''
# Add these imports to your existing download script
from app.ingestion_pipeline import PDFIngestionPipeline
from app.config import get_settings

# Initialize RAG pipeline (add at top of your script)
settings = get_settings()
rag_pipeline = PDFIngestionPipeline(settings)

# Add this after your PDF validation
if is_valid_pdf(pdf_path):
    # Your existing validation passed
    
    # NEW: Add RAG ingestion
    result = rag_pipeline.validate_and_ingest_pdf(pdf_path, drug_name)
    
    if result["success"]:
        print(f"🧠 RAG: {result['chunks_created']} chunks ready for chat")
    else:
        print(f"❌ RAG Error: {result['error']}")
'''

    print(integration_code)


if __name__ == "__main__":
    print("🔍 Understanding Current System State")
    print("=" * 50)

    demonstrate_current_separate_systems()
    show_integration_code()

    print("\n🧪 Test integrated workflow? (y/n): ", end="")
    test = input().strip().lower()

    if test == 'y':
        drug = input("Enter drug name (aspirin/metformin/etc): ").strip()
        if drug:
            integrated_download_and_ingest(drug)
        else:
            print("Using demo drug: aspirin")
            integrated_download_and_ingest("aspirin")
