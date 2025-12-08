#!/usr/bin/env python3
"""
Integration script showing how to use your existing PDF validation
and download code with the new RAG system.
"""

import requests
import urllib.request
import xml.etree.ElementTree as ET
import os
import gzip
import shutil
import tarfile
import time

# Import your existing validation function
from app.ingestion_pipeline import is_valid_pdf, PDFIngestionPipeline
from app.config import get_settings


# ---------------------------- YOUR EXISTING VALIDATION CODE ----------------------------
# (Copy your validation function here - already imported above)


def integrate_with_existing_pipeline():
    """
    Example showing how to integrate your existing PDF processing
    with the new RAG system.
    """

    # Initialize the RAG ingestion pipeline
    settings = get_settings()
    pipeline = PDFIngestionPipeline(settings)

    print("🚀 Starting integrated PDF processing pipeline")
    print("=" * 60)

    # Example: Process your existing directories
    drug_directories = {
        "aspirin": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\aspirin repurposing",
        "apomorphine": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\apomorphine repurposing",
        "insulin": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\insulin repurposing"
    }

    total_processed = 0
    total_valid = 0
    total_chunks = 0

    for drug_name, directory in drug_directories.items():
        print(f"\n🏥 Processing {drug_name} PDFs from: {directory}")

        if not os.path.exists(directory):
            print(f"❌ Directory not found: {directory}")
            continue

        # Get all PDFs in directory
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        print(f"📁 Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)

            # Use YOUR existing validation function
            if not is_valid_pdf(pdf_path):
                print(f"⚠️  Skipping invalid PDF: {pdf_file}")
                continue

            # Use our RAG pipeline to process the valid PDF
            result = pipeline.validate_and_ingest_pdf(pdf_path, drug_name)

            if result["success"]:
                total_valid += 1
                total_chunks += result["chunks_created"]
                print(f"✅ {pdf_file} -> {result['chunks_created']} chunks")
            else:
                print(f"❌ {pdf_file} -> {result['error']}")

            total_processed += 1

    print("\n🎉 Integration Complete!")
    print(f"📊 PDFs processed: {total_processed}")
    print(f"✅ Successfully ingested: {total_valid}")
    print(f"📝 Total chunks created: {total_chunks}")

    return {
        "total_processed": total_processed,
        "total_valid": total_valid,
        "total_chunks": total_chunks
    }


def example_download_and_ingest_workflow():
    """
    Example showing how your download workflow can integrate with RAG ingestion.

    This is a template based on typical PubMed/PMC download patterns.
    """

    # Initialize RAG pipeline
    settings = get_settings()
    pipeline = PDFIngestionPipeline(settings)

    print("🔬 Example: Download → Validate → Ingest Workflow")
    print("=" * 60)

    # Example drugs to search for
    drugs_to_search = ["aspirin", "metformin", "apomorphine", "insulin"]

    for drug in drugs_to_search:
        print(f"\n🏥 Processing drug: {drug}")

        # STEP 1: Your existing download code would go here
        # Example pseudocode:
        """
        search_results = search_pubmed(f"{drug} repurposing", max_results=5)

        for paper in search_results:
            # Download PDF using your existing code
            pdf_path = download_pdf_from_pmc(paper['pmcid'])

            # STEP 2: Validate using your existing function
            if not is_valid_pdf(pdf_path):
                print(f"⚠️  Invalid PDF: {paper['title']}")
                continue

            # STEP 3: Ingest into RAG system
            result = pipeline.validate_and_ingest_pdf(pdf_path, drug)

            if result["success"]:
                print(f"✅ Ingested: {paper['title']}")
            else:
                print(f"❌ Failed: {paper['title']} - {result['error']}")
        """

        print(f"📝 Template ready for {drug} - integrate your download code above")

    return {"status": "template_ready"}


if __name__ == "__main__":
    print("Choose integration method:")
    print("1. Process existing PDF directories")
    print("2. View download workflow template")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        results = integrate_with_existing_pipeline()
        print(f"\n📊 Results: {results}")
    elif choice == "2":
        example_download_and_ingest_workflow()
    else:
        print("Invalid choice")
