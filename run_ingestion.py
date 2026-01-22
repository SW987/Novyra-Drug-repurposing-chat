import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.absolute()))

from dotenv import load_dotenv

from app.config import get_settings
from app.ingestion_pipeline import PDFIngestionPipeline

DEFAULT_STORAGE_PATH = r"C:\Users\daud.haider\Desktop\DRUG_REPURPOSING_CHAT_LATEST_WORKING\data\testdata"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str, job: Optional[str] = None) -> None:
    prefix = f"[{_timestamp()}]"
    if job:
        prefix = f"{prefix} [{job}]"
    print(f"{prefix} {message}")


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[no items]"
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def _parse_storage_paths(raw_paths: List[str]) -> List[str]:
    storage_paths: List[str] = []
    for entry in raw_paths:
        if not entry:
            continue
        for part in entry.split(","):
            cleaned = part.strip()
            if cleaned:
                storage_paths.append(cleaned)
    return storage_paths


def _collect_drug_dirs(docs_dir: Path) -> Dict[str, str]:
    drug_dirs: Dict[str, str] = {}
    if docs_dir.exists():
        for subdir in docs_dir.iterdir():
            if not subdir.is_dir():
                continue
            drug_dirs[subdir.name] = str(subdir)
    return drug_dirs


def _run_ingestion_batch(
    pipeline: PDFIngestionPipeline,
    drug_dirs: Dict[str, str],
    job_label: str,
) -> Dict[str, Any]:
    overall_results: Dict[str, Any] = {
        "timestamp": time.time(),
        "drugs_processed": [],
        "total_files": 0,
        "total_successful": 0,
        "total_failed": 0,
        "drug_results": [],
    }

    total_drugs = len(drug_dirs)
    for index, (drug_name, directory_path) in enumerate(drug_dirs.items(), start=1):
        progress = _render_progress(index, total_drugs)
        _log(f"{progress} Processing drug: {drug_name}", job_label)
        _log("=" * 50, job_label)

        drug_result = pipeline.ingest_directory_batch(directory_path, drug_name)
        drug_result["drug_name"] = drug_name
        drug_result["directory"] = directory_path

        overall_results["drugs_processed"].append(drug_name)
        overall_results["total_files"] += drug_result["total_files"]
        overall_results["total_successful"] += drug_result["successful"]
        overall_results["total_failed"] += drug_result["failed"]
        overall_results["drug_results"].append(drug_result)

    _log("Ingestion complete", job_label)
    _log(f"Total PDFs processed: {overall_results['total_files']}", job_label)
    _log(f"Successful: {overall_results['total_successful']}", job_label)
    _log(f"Failed: {overall_results['total_failed']}", job_label)
    return overall_results


def _run_ingestion_job(storage_path: str) -> Dict[str, Any]:
    job_label = Path(storage_path).name or storage_path
    _log(f"Starting ingestion job for {storage_path}", job_label)
    load_dotenv()
    _log("Environment variables loaded", job_label)

    settings = get_settings()
    if storage_path:
        settings.docs_dir = str(Path(storage_path))

    _log(
        f"Settings loaded: Chroma DB Dir = {settings.chroma_db_dir}, Docs Dir = {settings.docs_dir}",
        job_label,
    )

    docs_dir = Path(settings.docs_dir)
    drug_dirs = _collect_drug_dirs(docs_dir)

    if not drug_dirs:
        message = f"No drug folders found under {docs_dir}"
        _log(message, job_label)
        _log("Update run_ingestion.py to point at your PDF folders", job_label)
        return {
            "timestamp": time.time(),
            "storage_path": str(docs_dir),
            "drugs_processed": [],
            "total_files": 0,
            "total_successful": 0,
            "total_failed": 0,
            "drug_results": [],
            "error": message,
        }

    pipeline = PDFIngestionPipeline(settings)
    results = _run_ingestion_batch(pipeline, drug_dirs, job_label)
    results["storage_path"] = str(docs_dir)
    return results


def _error_result(storage_path: str, message: str) -> Dict[str, Any]:
    return {
        "timestamp": time.time(),
        "storage_path": storage_path,
        "drugs_processed": [],
        "total_files": 0,
        "total_successful": 0,
        "total_failed": 0,
        "drug_results": [],
        "error": message,
    }


def _combine_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {
        "timestamp": time.time(),
        "total_jobs": len(results),
        "drugs_processed": [],
        "total_files": 0,
        "total_successful": 0,
        "total_failed": 0,
        "jobs": results,
    }

    for result in results:
        combined["drugs_processed"].extend(result.get("drugs_processed", []))
        combined["total_files"] += result.get("total_files", 0)
        combined["total_successful"] += result.get("total_successful", 0)
        combined["total_failed"] += result.get("total_failed", 0)

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDF ingestion for downloaded drug papers.")
    parser.add_argument(
        "--storage-path",
        action="append",
        default=[],
        help="Folder containing drug subfolders (repeat or comma-separate for multiple jobs)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel ingestion jobs (1 = sequential)",
    )
    args = parser.parse_args()

    storage_paths = _parse_storage_paths(args.storage_path)
    if not storage_paths:
        storage_paths = [DEFAULT_STORAGE_PATH]

    max_workers = max(1, args.max_workers)
    _log("Starting dedicated ingestion script")

    results_by_path: Dict[str, Dict[str, Any]] = {}
    if len(storage_paths) > 1 and max_workers > 1:
        worker_count = min(max_workers, len(storage_paths))
        _log(f"Running {len(storage_paths)} ingestion jobs with {worker_count} workers")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_run_ingestion_job, path): path for path in storage_paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    results_by_path[path] = future.result()
                except Exception as exc:
                    message = f"Job failed for {path}: {exc}"
                    _log(message)
                    results_by_path[path] = _error_result(path, message)
    else:
        for storage_path in storage_paths:
            results_by_path[storage_path] = _run_ingestion_job(storage_path)

    ordered_results = [results_by_path[path] for path in storage_paths]
    if len(ordered_results) == 1:
        print(ordered_results[0])
    else:
        combined = _combine_results(ordered_results)
        print(combined)


if __name__ == "__main__":
    main()
