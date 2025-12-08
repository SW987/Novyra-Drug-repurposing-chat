#!/usr/bin/env python3
"""
Automated PDF Ingestion Pipeline for Drug Repurposing Research
Integrates with existing PDF download/validation systems
"""

import os
import requests
import urllib.request
import xml.etree.ElementTree as ET
import gzip
import shutil
import tarfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai

from .config import Settings, get_settings
from .vector_store import init_vector_store
from .ingestion import process_pdf_file
from .utils import parse_filename


# ---------------------------- VALIDATE PDF ----------------------------
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


class PDFIngestionPipeline:
    """Automated pipeline for ingesting drug repurposing PDFs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection = init_vector_store(settings)
        genai.configure(api_key=settings.gemini_api_key)

    def validate_and_ingest_pdf(self, pdf_path: str, drug_name: str) -> Dict[str, Any]:
        """
        Validate a PDF and ingest it into the vector database.

        Args:
            pdf_path: Path to the PDF file
            drug_name: Drug name (e.g., 'aspirin', 'metformin')

        Returns:
            Processing results
        """
        print(f"🔄 Processing: {pdf_path}")

        # Validate PDF
        if not is_valid_pdf(pdf_path):
            return {
                "success": False,
                "error": "Invalid PDF",
                "file_path": pdf_path
            }

        # Create drug folder structure
        drug_folder = f"{drug_name} repurposing"

        try:
            # Process the PDF using our existing pipeline
            result = process_pdf_file(
                pdf_path=Path(pdf_path),
                drug_folder=drug_folder,
                settings=self.settings,
                collection=self.collection,
                client=genai  # Pass configured Gemini client
            )

            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "file_path": pdf_path
                }

            return {
                "success": True,
                "drug_id": result["drug_id"],
                "doc_id": result["doc_id"],
                "chunks_created": result["chunks_created"],
                "file_path": pdf_path
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": pdf_path
            }

    def ingest_directory_batch(self, directory_path: str, drug_name: str) -> Dict[str, Any]:
        """
        Ingest all PDFs from a directory for a specific drug.

        Args:
            directory_path: Path to directory containing PDFs
            drug_name: Drug name for categorization

        Returns:
            Batch processing results
        """
        directory = Path(directory_path)
        if not directory.exists():
            return {"error": f"Directory {directory_path} does not exist"}

        pdf_files = list(directory.glob("*.pdf"))
        print(f"📁 Found {len(pdf_files)} PDFs in {directory_path}")

        results = []
        successful = 0
        failed = 0

        for pdf_path in pdf_files:
            result = self.validate_and_ingest_pdf(str(pdf_path), drug_name)
            results.append(result)

            if result["success"]:
                successful += 1
                print(f"✅ {pdf_path.name} -> {result['chunks_created']} chunks")
            else:
                failed += 1
                print(f"❌ {pdf_path.name} -> {result['error']}")

        return {
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }

    def ingest_from_pubmed_search(self, drug_name: str, max_papers: int = 10) -> Dict[str, Any]:
        """
        Example integration with PubMed search and download.
        This is a template - customize based on your existing download system.
        """
        print(f"🔍 Searching PubMed for '{drug_name} repurposing'...")

        # This would integrate with your existing PubMed API code
        # For now, return a template structure

        return {
            "drug": drug_name,
            "papers_found": max_papers,
            "status": "integration_template",
            "note": "Customize this method to integrate with your PubMed download system"
        }


def run_ingestion_pipeline(drug_directories: Dict[str, str]) -> Dict[str, Any]:
    """
    Run the complete ingestion pipeline for multiple drugs.

    Args:
        drug_directories: Dict mapping drug names to directory paths
                          e.g., {"aspirin": "/path/to/aspirin/pdfs"}

    Returns:
        Complete pipeline results
    """
    settings = get_settings()
    pipeline = PDFIngestionPipeline(settings)

    overall_results = {
        "timestamp": time.time(),
        "drugs_processed": [],
        "total_files": 0,
        "total_successful": 0,
        "total_failed": 0,
        "drug_results": []
    }

    for drug_name, directory_path in drug_directories.items():
        print(f"\n🏥 Processing drug: {drug_name}")
        print("=" * 50)

        drug_result = pipeline.ingest_directory_batch(directory_path, drug_name)
        drug_result["drug_name"] = drug_name
        drug_result["directory"] = directory_path

        overall_results["drugs_processed"].append(drug_name)
        overall_results["total_files"] += drug_result["total_files"]
        overall_results["total_successful"] += drug_result["successful"]
        overall_results["total_failed"] += drug_result["failed"]
        overall_results["drug_results"].append(drug_result)

    print("\n🎉 Pipeline Complete!")
    print(f"📊 Total PDFs processed: {overall_results['total_files']}")
    print(f"✅ Successful: {overall_results['total_successful']}")
    print(f"❌ Failed: {overall_results['total_failed']}")

    return overall_results


# Example usage functions
def example_usage_your_directory():
    """Example using your existing directory structure."""
    drug_directories = {
        "aspirin": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\aspirin repurposing",
        "apomorphine": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\apomorphine repurposing",
        "insulin": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\insulin repurposing"
    }

    results = run_ingestion_pipeline(drug_directories)
    return results


def example_usage_integrated_download():
    """
    Example of how to integrate with your download system.
    Modify this to work with your existing PDF download pipeline.
    """
    # This would be integrated into your existing download workflow

    # Pseudocode for integration:
    """
    # 1. Your existing download system finds and downloads PDFs
    downloaded_pdfs = your_download_function(drug_name, search_terms)

    # 2. For each downloaded PDF, validate and ingest
    pipeline = PDFIngestionPipeline(get_settings())

    for pdf_info in downloaded_pdfs:
        result = pipeline.validate_and_ingest_pdf(
            pdf_info['file_path'],
            pdf_info['drug_name']
        )

        if result['success']:
            print(f"✅ Ingested {pdf_info['title']}")
        else:
            print(f"❌ Failed to ingest {pdf_info['title']}: {result['error']}")
    """

    return {"status": "integration_example"}


if __name__ == "__main__":
    # Run with your existing directory
    results = example_usage_your_directory()
    print("\n📋 Final Results:")
    print(results)
