import gzip
import os
import re
import shutil
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import Settings

try:
    import boto3
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None


def sanitize_drug_name(drug_name: str) -> str:
    """Normalize a drug name for folder/filename use."""
    clean = re.sub(r"[^a-z0-9]+", "_", drug_name.lower()).strip("_")
    return clean or "unknown_drug"


def is_valid_pdf(file_path: str) -> bool:
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


def _get_with_retries(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Optional[requests.Response]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                time.sleep(backoff_seconds * attempt)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            last_error = e
            time.sleep(backoff_seconds * attempt)

    if last_error:
        print(f"[ERROR] Request failed after retries: {last_error}")
    return None


def search_pmc_articles(
    query: str,
    max_results: int = 50,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Tuple[List[str], List[str]]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
    }

    response = _get_with_retries(
        url, params=params, timeout=15, max_retries=max_retries, backoff_seconds=backoff_seconds
    )
    if not response:
        return [], []

    data = response.json()
    pmc_ids = data.get("esearchresult", {}).get("idlist", [])
    links = [f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{id}/" for id in pmc_ids]
    return pmc_ids, links


def get_pdf_link_from_pmcid(
    pmcid: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Optional[str]:
    api_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmcid}"

    response = _get_with_retries(
        api_url, timeout=10, max_retries=max_retries, backoff_seconds=backoff_seconds
    )
    if not response:
        return None

    try:
        root = ET.fromstring(response.text)
    except Exception as e:
        print(f"[ERROR] OA parse failed for PMC{pmcid}: {e}")
        return None

    for link in root.findall(".//link"):
        if link.attrib.get("format") == "pdf":
            return link.attrib["href"]

    return None


def download_stream(url: str, destination: str, timeout: int = 25) -> None:
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


def extract_pdf_from_tar_gz(tar_path: str, output_path: str) -> bool:
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".pdf"):
                    tar.extract(member, path=".")
                    os.rename(member.name, output_path)
                    return True
    except Exception as e:
        print(f"[ERROR] TAR extraction failed: {e}")
    return False


def safe_gunzip(data: bytes) -> Optional[bytes]:
    """Safely decompress gzip data with fallback."""
    try:
        return gzip.decompress(data)
    except Exception as e:
        print(f"[WARN] GZIP decompress failed: {e}")
        return None


def download_pdf(
    pdf_url: str,
    save_path: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> bool:
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

            if raw[:2] == b"\x1f\x8b":
                print("[INFO] Detected gzipped content, decompressing")
                decompressed = safe_gunzip(raw)
                if decompressed:
                    raw = decompressed
                else:
                    print("[SKIP] GZIP decompression failed")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    return False

            with open(save_path, "wb") as f:
                f.write(raw)

            if os.path.exists(temp_file):
                os.remove(temp_file)

            if is_valid_pdf(save_path):
                print(f"[SUCCESS] Valid PDF saved: {save_path}")
                return True

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
            wait_seconds = retry_delay * attempt
            print(f"[INFO] Retrying in {wait_seconds:.1f} seconds...")
            time.sleep(wait_seconds)

    print("[SKIP] Invalid OA PDF link")
    return False


def _normalize_s3_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ""
    return prefix.strip("/")


def _build_s3_key(prefix: str, relative_path: Path) -> str:
    if prefix:
        return f"{prefix}/{relative_path.as_posix()}"
    return relative_path.as_posix()


class PaperFetchPipeline:
    """Download drug repurposing PDFs without ingestion."""

    def __init__(
        self,
        settings: Settings,
        request_delay: float = 1.0,
        error_delay: float = 5.0,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        s3_region: Optional[str] = None,
    ):
        self.settings = settings
        self.request_delay = request_delay
        self.error_delay = error_delay
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.s3_bucket = s3_bucket
        self.s3_prefix = _normalize_s3_prefix(s3_prefix)
        self.s3_client = None

        if self.s3_bucket:
            if boto3 is None:
                raise RuntimeError("boto3 is required for S3 uploads. Install boto3 or omit --s3-bucket.")
            session = boto3.session.Session(region_name=s3_region) if s3_region else boto3.session.Session()
            self.s3_client = session.client("s3")

    def _upload_to_s3(self, local_path: Path) -> Optional[str]:
        if not self.s3_client or not self.s3_bucket:
            return None

        docs_root = Path(self.settings.docs_dir).resolve()
        try:
            relative_path = local_path.resolve().relative_to(docs_root)
        except ValueError:
            relative_path = Path(local_path.name)

        key = _build_s3_key(self.s3_prefix, relative_path)
        try:
            self.s3_client.upload_file(str(local_path), self.s3_bucket, key)
        except Exception as exc:
            print(f"[WARN] S3 upload failed for {local_path}: {exc}")
            return None

        return f"s3://{self.s3_bucket}/{key}"

    def fetch_drug_papers(
        self,
        drug_name: str,
        max_papers: int = 3,
        max_search_results: int = 50,
        upload_to_s3: bool = True,
        upload_after: Optional[int] = None,
    ) -> Dict[str, Any]:
        print(f"[INFO] Searching PubMed for '{drug_name} repurposing'...")

        drug_slug = sanitize_drug_name(drug_name)
        output_folder = Path(self.settings.docs_dir) / drug_slug

        full_query = f"{drug_name} repurposing"
        pmc_ids, _ = search_pmc_articles(
            full_query,
            max_results=max_search_results,
            max_retries=self.max_retries,
            backoff_seconds=self.backoff_seconds,
        )
        print(f"[INFO] Found {len(pmc_ids)} PMC articles for '{drug_name}'")

        if not pmc_ids:
            return {
                "success": False,
                "drug": drug_name,
                "papers_found": 0,
                "downloaded": 0,
                "output_folder": str(output_folder),
                "downloaded_files": [],
                "error": "No papers found in PubMed Central",
            }

        output_folder.mkdir(parents=True, exist_ok=True)

        downloaded_count = 0
        results = []
        downloaded_files = []
        downloaded_s3 = []
        uploaded_paths = set()
        result_by_path: Dict[str, Dict[str, Any]] = {}
        candidates: List[Tuple[str, str]] = []
        started_downloads = False
        next_candidate_index = 0

        for pmcid in pmc_ids:
            if downloaded_count >= max_papers:
                print(f"[INFO] Reached target of {max_papers} papers. Stopping.")
                break

            print(f"[INFO] Processing PMC{pmcid} (Downloaded: {downloaded_count}/{max_papers})")

            pdf_url = get_pdf_link_from_pmcid(
                pmcid, max_retries=self.max_retries, backoff_seconds=self.backoff_seconds
            )
            if not pdf_url:
                print(f"[SKIP] No OA PDF available for PMC{pmcid}")
                results.append({"pmcid": pmcid, "status": "no_pdf_available"})
                time.sleep(self.error_delay)
                continue

            candidates.append((pmcid, pdf_url))
            if not started_downloads and len(candidates) >= max_papers:
                started_downloads = True
                print(f"[INFO] Found {max_papers} OA PDFs for '{drug_name}'. Starting downloads.")

            if started_downloads:
                while next_candidate_index < len(candidates) and downloaded_count < max_papers:
                    candidate_pmcid, candidate_url = candidates[next_candidate_index]
                    next_candidate_index += 1

                    save_path = output_folder / f"{drug_slug}_repurposing_PMC{candidate_pmcid}.pdf"
                    if download_pdf(candidate_url, str(save_path), retries=self.max_retries, retry_delay=self.backoff_seconds):
                        downloaded_count += 1
                        downloaded_files.append(str(save_path))
                        result_entry = {
                            "pmcid": candidate_pmcid,
                            "downloaded": True,
                            "file_path": str(save_path),
                        }
                        result_by_path[str(save_path)] = result_entry
                        results.append(result_entry)
                        print(f"[SUCCESS] Downloaded: {save_path}")

                        if upload_to_s3 and self.s3_bucket:
                            if upload_after is None:
                                print(f"[INFO] Uploading to S3: {save_path.name}")
                                s3_uri = self._upload_to_s3(save_path)
                                if s3_uri:
                                    downloaded_s3.append(s3_uri)
                                    result_entry["s3_uri"] = s3_uri
                                    uploaded_paths.add(str(save_path))
                                    print(f"[SUCCESS] Uploaded: {s3_uri}")
                            elif downloaded_count >= upload_after:
                                pending = [path for path in downloaded_files if path not in uploaded_paths]
                                if pending:
                                    print(
                                        f"[INFO] Uploading {len(pending)} papers to S3 for '{drug_name}'"
                                    )
                                for pending_path in pending:
                                    pending_file = Path(pending_path)
                                    print(f"[INFO] Uploading to S3: {pending_file.name}")
                                    s3_uri = self._upload_to_s3(pending_file)
                                    if s3_uri:
                                        downloaded_s3.append(s3_uri)
                                        entry = result_by_path.get(pending_path)
                                        if entry is not None:
                                            entry["s3_uri"] = s3_uri
                                        uploaded_paths.add(pending_path)
                                        print(f"[SUCCESS] Uploaded: {s3_uri}")
                                    else:
                                        print(f"[WARN] S3 upload failed for {pending_file}")
                    else:
                        results.append({"pmcid": candidate_pmcid, "status": "download_failed"})
                        time.sleep(self.error_delay)

            time.sleep(self.request_delay)

        if downloaded_count == 0 and output_folder.exists():
            try:
                if not any(output_folder.iterdir()):
                    output_folder.rmdir()
            except OSError:
                pass

        return {
            "success": downloaded_count > 0,
            "drug": drug_name,
            "papers_found": len(pmc_ids),
            "links_searched": len(results),
            "downloaded": downloaded_count,
            "output_folder": str(output_folder),
            "downloaded_files": downloaded_files,
            "downloaded_s3": downloaded_s3,
            "results": results,
        }

    def fetch_papers_for_drugs(
        self,
        drugs: List[str],
        max_papers_per_drug: int = 3,
        max_search_results: int = 100,
        upload_to_s3: bool = True,
        upload_after: Optional[int] = None,
    ) -> Dict[str, Any]:
        overall = {
            "timestamp": time.time(),
            "drugs_processed": [],
            "total_downloaded": 0,
            "drug_results": [],
        }

        for drug_name in drugs:
            result = self.fetch_drug_papers(
                drug_name,
                max_papers=max_papers_per_drug,
                max_search_results=max_search_results,
                upload_to_s3=upload_to_s3,
                upload_after=upload_after,
            )
            overall["drugs_processed"].append(drug_name)
            overall["total_downloaded"] += result.get("downloaded", 0)
            overall["drug_results"].append(result)

            time.sleep(self.request_delay)

        return overall
