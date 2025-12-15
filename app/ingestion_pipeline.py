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
# Removed direct import of google.generativeai as genai here

from .config import Settings, get_settings
from .vector_store import init_vector_store
from .ingestion import process_pdf_file # Removed get_gemini_client import
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


# ----------------------------
# PMC SEARCH
# ----------------------------
def search_pmc_articles(query, max_results=50):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance"
    }

    response = requests.get(url, params=params, timeout=15)
    data = response.json()

    pmc_ids = data.get("esearchresult", {}).get("idlist", [])
    links = [f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{id}/" for id in pmc_ids]
    return pmc_ids, links


# ----------------------------
# GET PDF LINK FROM PMC OA API
# ----------------------------
def get_pdf_link_from_pmcid(pmcid):
    api_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmcid}"

    try:
        r = requests.get(api_url, timeout=10)
        root = ET.fromstring(r.text)

        for link in root.findall(".//link"):
            if link.attrib.get("format") == "pdf":
                return link.attrib["href"]
    except Exception as e:
        print(f"[ERROR] OA fetch failed for PMC{pmcid}: {e}")

    return None


# ----------------------------
# DOWNLOAD FILE STREAM (HTTP or FTP)
# ----------------------------
def download_stream(url, destination, timeout=20):
    """Reliable binary download for HTTP and FTP."""
    if url.startswith("ftp://"):
        with urllib.request.urlopen(url, timeout=timeout) as response, open(destination, "wb") as out:
            shutil.copyfileobj(response, out)
    else:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


# ----------------------------
# PARSE TAR.GZ ARCHIVES
# ----------------------------
def extract_pdf_from_tar_gz(tar_path, output_path):
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".pdf"):
                    tar.extract(member, path=".")
                    os.rename(member.name, output_path)
                    return True
    except Exception as e:
        print("[ERROR] TAR extraction failed:", e)
    return False


# ----------------------------
# HANDLE GZIP DECOMPRESSION SAFELY
# ----------------------------
def safe_gunzip(data):
    """Safely decompress gzip data with fallback."""
    try:
        return gzip.decompress(data)
    except Exception as e:
        print(f"[WARN] GZIP decompress failed: {e}")
        return None


# ----------------------------
# DOWNLOAD PDF (gzip-aware, tar-aware, fallback-aware)
# ----------------------------
def download_pdf(pdf_url, save_path, retries=3):
    for attempt in range(1, retries + 1):
        temp_file = save_path + ".tmp"

        try:
            print(f"[INFO] Attempt {attempt}: {pdf_url}")

            download_stream(pdf_url, temp_file)
            time.sleep(0.5)  # Avoid rate limiting

            with open(temp_file, "rb") as f:
                raw = f.read()

            if len(raw) < 100:
                print("[ERROR] Downloaded file too small")
                continue

            # ---- Detect .tar.gz archive ----
            if pdf_url.endswith(".tar.gz") or raw[:2] == b"\x1f\x8b":
                if pdf_url.endswith(".tar.gz"):
                    print("[INFO] Detected TAR.GZ archive. Extracting...")
                    if extract_pdf_from_tar_gz(temp_file, save_path):
                        os.remove(temp_file)
                        if is_valid_pdf(save_path):
                            print("[SUCCESS] Extracted valid PDF")
                            return True
                    print("[SKIP] No valid PDF inside TAR.GZ")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    return False

            # ---- Detect gzip-wrapped PDFs ----
            if raw[:2] == b"\x1f\x8b":
                print("[INFO] Detected GZIPPED content → decompressing")
                decompressed = safe_gunzip(raw)
                if decompressed:
                    raw = decompressed
                else:
                    print("[SKIP] GZIP decompression failed")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    return False

            # ---- Save actual PDF ----
            with open(save_path, "wb") as f:
                f.write(raw)

            if os.path.exists(temp_file):
                os.remove(temp_file)

            if is_valid_pdf(save_path):
                print(f"[SUCCESS] Valid PDF saved: {save_path}")
                return True
            else:
                print("[SKIP] Invalid PDF content")
                os.remove(save_path)
                return False

        except requests.exceptions.Timeout:
            print("[ERROR] Download timeout")
        except requests.exceptions.ConnectionError:
            print("[ERROR] Connection error")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")

        if os.path.exists(temp_file):
            os.remove(temp_file)

        if attempt < retries:
            print("[INFO] Retrying in 2 seconds...\n")
            time.sleep(2)

    print("[SKIP] Invalid OA PDF link")
    return False


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

    def download_and_ingest_drug_papers(self, drug_name: str, max_papers: int = 3, max_search_results: int = 50) -> Dict[str, Any]:
        """
        Search PubMed for drug repurposing papers, download PDFs, and ingest them.
 
        Args:
            drug_name: Drug name to search for
            max_papers: Maximum number of papers to successfully download
            max_search_results: Maximum number of search results to look through
 
        Returns:
            Complete processing results
        """
        print(f"🔍 Searching PubMed for '{drug_name} repurposing'...")
 
        # Create output directory
        full_query = f"{drug_name} repurposing"
        output_folder = Path(self.settings.docs_dir) / full_query
        output_folder.mkdir(exist_ok=True)
 
        print(f"📁 Saving PDFs to: {output_folder}")
 
        # Search PMC for articles
        pmc_ids, article_links = search_pmc_articles(full_query, max_results=max_search_results)
        print(f"📋 Found {len(pmc_ids)} PMC articles for '{drug_name}'")
 
        if not pmc_ids:
            return {
                "success": False,
                "drug": drug_name,
                "papers_found": 0,
                "downloaded": 0,
                "ingested": 0,
                "error": "No papers found in PubMed Central"
            }
 
        downloaded_count = 0
        ingested_count = 0
        results = []
 
        # Download and process papers until we reach max_papers successful downloads
        for pmcid in pmc_ids:
            # Stop if we've downloaded enough papers
            if downloaded_count >= max_papers:
                print(f"\n✅ Reached target of {max_papers} papers. Stopping search.")
                break
 
            print(f"\n📥 Processing PMC{pmcid} (Downloaded: {downloaded_count}/{max_papers})")
 
            # Get PDF link
            pdf_url = get_pdf_link_from_pmcid(pmcid)
            if not pdf_url:
                print(f"[SKIP] No OA PDF available for PMC{pmcid}")
                results.append({"pmcid": pmcid, "status": "no_pdf_available"})
                continue
 
            # Download PDF
            save_path = output_folder / f"{drug_name}_repurposing_PMC{pmcid}.pdf"
            if download_pdf(pdf_url, str(save_path)):
                downloaded_count += 1
                print(f"[SUCCESS] Downloaded: {save_path}")
 
                # Ingest the downloaded PDF
                ingest_result = self.validate_and_ingest_pdf(str(save_path), drug_name)
                results.append({
                    "pmcid": pmcid,
                    "downloaded": True,
                    "ingested": ingest_result["success"],
                    "ingest_result": ingest_result
                })
 
                if ingest_result["success"]:
                    ingested_count += 1
                    print(f"[SUCCESS] Ingested into vector DB: {ingest_result['chunks_created']} chunks")
                else:
                    print(f"[ERROR] Failed to ingest: {ingest_result.get('error', 'Unknown error')}")
            else:
                print(f"[FAILED] Could not download PDF for PMC{pmcid}")
                results.append({"pmcid": pmcid, "status": "download_failed"})
 
        return {
            "success": downloaded_count > 0,
            "drug": drug_name,
            "papers_found": len(pmc_ids),
            "links_searched": len(results),
            "downloaded": downloaded_count,
            "ingested": ingested_count,
            "output_folder": str(output_folder),
            "results": results
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
