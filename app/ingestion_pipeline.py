#!/usr/bin/env python3
"""
High-level orchestration pipeline for ingesting drug repurposing PDFs.

Wraps the lower-level ingestion functions from ingestion.py and the paper
download utilities from paper_fetcher.py into two main entry points:

- PDFIngestionPipeline: class-based interface for validating and ingesting
  individual files or entire directories, with optional PubMed paper fetching.
- run_ingestion_pipeline: convenience function for batch-processing multiple
  drug directories in one call.

Directory structure expected under settings.docs_dir:
    <docs_dir>/
    ├── aspirin/
    │   └── aspirin_repurposing_PMC11242460.pdf
    └── metformin/
        └── metformin_repurposing_PMC9876543.pdf
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
# Removed direct import of google.generativeai as genai here

from .config import Settings, get_settings
from .vector_store import init_vector_store
from .ingestion import process_pdf_file # Removed get_gemini_client import
from .paper_fetcher import PaperFetchPipeline, is_valid_pdf


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}")


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[no items]"
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


class PDFIngestionPipeline:
    """Automated pipeline for ingesting drug repurposing PDFs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection = init_vector_store(settings)
        # Removed: self.gemini_client = get_gemini_client(settings) # No longer needed here

    def validate_and_ingest_pdf(self, pdf_path: str, drug_name: str) -> Dict[str, Any]:
        """
        Validate a PDF and ingest it into the vector database.

        Args:
            pdf_path: Path to the PDF file
            drug_name: Drug name (e.g., 'aspirin', 'metformin')

        Returns:
            Processing results
        """
        _log(f"Processing PDF: {pdf_path}")

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
                collection=self.collection
                # Removed: client=self.gemini_client  # No longer passed directly
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
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "error": f"Directory {directory_path} does not exist",
            }

        pdf_files = list(directory.glob("*.pdf"))
        _log(f"Found {len(pdf_files)} PDFs in {directory_path}")

        results = []
        successful = 0
        failed = 0

        total_files = len(pdf_files)
        for index, pdf_path in enumerate(pdf_files, start=1):
            result = self.validate_and_ingest_pdf(str(pdf_path), drug_name)
            results.append(result)

            progress = _render_progress(index, total_files)
            if result["success"]:
                successful += 1
                _log(f"{progress} Ingested {pdf_path.name} -> {result['chunks_created']} chunks")
            else:
                failed += 1
                _log(f"{progress} Failed {pdf_path.name} -> {result['error']}")

        return {
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }

    def download_and_ingest_drug_papers(
        self, drug_name: str, max_papers: int = 3, max_search_results: int = 50
    ) -> Dict[str, Any]:
        """
        Download drug papers and ingest them into the vector database.

        Args:
            drug_name: Drug name to search for
            max_papers: Maximum number of papers to successfully download
            max_search_results: Maximum number of search results to look through

        Returns:
            Complete processing results
        """
        fetcher = PaperFetchPipeline(
            self.settings,
            s3_bucket=self.settings.s3_bucket,
            s3_prefix=self.settings.s3_prefix,
            s3_region=self.settings.s3_region,
        )
        fetch_result = fetcher.fetch_drug_papers(
            drug_name, max_papers=max_papers, max_search_results=max_search_results
        )

        downloaded_files = fetch_result.get("downloaded_files", [])
        ingested_count = 0
        ingest_results = []

        total_files = len(downloaded_files)
        if total_files:
            _log(f"Starting ingestion for {total_files} downloaded PDFs")

        for index, file_path in enumerate(downloaded_files, start=1):
            ingest_result = self.validate_and_ingest_pdf(file_path, drug_name)
            ingest_results.append(
                {
                    "file_path": file_path,
                    "ingested": ingest_result.get("success", False),
                    "ingest_result": ingest_result,
                }
            )
            if ingest_result.get("success"):
                ingested_count += 1
                progress = _render_progress(index, total_files)
                _log(f"{progress} Ingested {Path(file_path).name}")
            elif total_files:
                progress = _render_progress(index, total_files)
                _log(f"{progress} Failed {Path(file_path).name} -> {ingest_result.get('error')}")

        return {
            **fetch_result,
            "ingested": ingested_count,
            "ingest_results": ingest_results,
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
        _log(f"Processing drug: {drug_name}")
        _log("=" * 50)

        drug_result = pipeline.ingest_directory_batch(directory_path, drug_name)
        drug_result["drug_name"] = drug_name
        drug_result["directory"] = directory_path

        overall_results["drugs_processed"].append(drug_name)
        overall_results["total_files"] += drug_result["total_files"]
        overall_results["total_successful"] += drug_result["successful"]
        overall_results["total_failed"] += drug_result["failed"]
        overall_results["drug_results"].append(drug_result)

    _log("Pipeline complete")
    _log(f"Total PDFs processed: {overall_results['total_files']}")
    _log(f"Successful: {overall_results['total_successful']}")
    _log(f"Failed: {overall_results['total_failed']}")

    return overall_results


def example_usage_from_settings():
    """
    Example: ingest all PDFs that are already placed under settings.docs_dir.

    Set DOCS_DIR in your .env file (or environment) to the root folder that
    contains per-drug subdirectories, then run:

        python -m app.ingestion_pipeline
    """
    settings = get_settings()
    docs_dir = settings.docs_dir

    # Build the drug→directory map from whatever subfolders already exist
    from pathlib import Path as _Path
    drug_directories = {
        folder.name: str(folder)
        for folder in _Path(docs_dir).iterdir()
        if folder.is_dir()
    }

    if not drug_directories:
        _log(f"No drug subdirectories found under {docs_dir}")
        return {}

    results = run_ingestion_pipeline(drug_directories)
    return results


if __name__ == "__main__":
    results = example_usage_from_settings()
    _log("Final results:")
    print(results)
