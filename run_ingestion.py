import argparse
import csv
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
from app.ingestion import ingest_pdfs_from_s3
from app.ingestion_pipeline import PDFIngestionPipeline
from app.vector_store import init_vector_store

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


def _append_ingestion_log(csv_path: Path, row: Dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = ["drug", "total_files", "successful", "failed", "processing_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _load_processed_drugs(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        processed = set()
        for row in reader:
            drug = (row.get("drug") or "").strip()
            if drug:
                processed.add(drug)
    return processed


def _prompt_resume_choice() -> str:
    prompt = (
        "Ingestion log found. Choose an option:\n"
        "  1) Resume (skip already processed drugs)\n"
        "  2) Restart (process all drugs)\n"
        "Enter 1 or 2: "
    )
    while True:
        choice = input(prompt).strip()
        if choice in {"1", "2"}:
            return choice
        print("Please enter 1 or 2.")


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
    log_path: Optional[Path] = None,
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

        drug_start = time.time()
        drug_result = pipeline.ingest_directory_batch(directory_path, drug_name)
        drug_duration = time.time() - drug_start
        drug_result["drug_name"] = drug_name
        drug_result["directory"] = directory_path

        overall_results["drugs_processed"].append(drug_name)
        overall_results["total_files"] += drug_result["total_files"]
        overall_results["total_successful"] += drug_result["successful"]
        overall_results["total_failed"] += drug_result["failed"]
        overall_results["drug_results"].append(drug_result)

        if log_path:
            _append_ingestion_log(
                log_path,
                {
                    "drug": drug_name,
                    "total_files": str(drug_result["total_files"]),
                    "successful": str(drug_result["successful"]),
                    "failed": str(drug_result["failed"]),
                    "processing_seconds": f"{drug_duration:.1f}",
                },
            )

    _log("Ingestion complete", job_label)
    _log(f"Total PDFs processed: {overall_results['total_files']}", job_label)
    _log(f"Successful: {overall_results['total_successful']}", job_label)
    _log(f"Failed: {overall_results['total_failed']}", job_label)
    return overall_results


def _run_ingestion_job(storage_path: str, resume_mode: str = "ask") -> Dict[str, Any]:
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
    log_path = docs_dir / "ingestion_log.csv"

    if log_path.exists():
        if resume_mode == "ask":
            choice = _prompt_resume_choice()
        else:
            choice = "1" if resume_mode == "resume" else "2"

        if choice == "1":
            processed = _load_processed_drugs(log_path)
            if processed:
                original_count = len(drug_dirs)
                drug_dirs = {k: v for k, v in drug_dirs.items() if k not in processed}
                skipped = original_count - len(drug_dirs)
                _log(f"Resuming: skipping {skipped} already-processed drugs", job_label)
            else:
                _log("Ingestion log was empty; continuing with all drugs", job_label)
        else:
            _log("Restarting: clearing ingestion log for a fresh run", job_label)
            log_path.unlink()

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
    results = _run_ingestion_batch(pipeline, drug_dirs, job_label, log_path=log_path)
    results["storage_path"] = str(docs_dir)
    return results


def _run_s3_ingestion_job(bucket: str, prefix: str, region: Optional[str]) -> Dict[str, Any]:
    job_label = f"s3://{bucket}/{prefix}".rstrip("/")
    _log(f"Starting S3 ingestion job for {job_label}", job_label)
    load_dotenv()
    _log("Environment variables loaded", job_label)

    settings = get_settings()
    _log(
        f"Settings loaded: Chroma DB Dir = {settings.chroma_db_dir}, Docs Dir = {settings.docs_dir}",
        job_label,
    )

    collection = init_vector_store(settings)
    results = ingest_pdfs_from_s3(bucket, prefix, settings, collection, region=region)
    results["storage_path"] = job_label
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
    parser.add_argument(
        "--resume-mode",
        choices=["ask", "resume", "restart"],
        default="ask",
        help="How to handle existing ingestion logs",
    )
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket containing PDFs")
    parser.add_argument("--s3-prefix", default="", help="S3 prefix to scan for PDFs")
    parser.add_argument("--s3-region", default=None, help="AWS region for S3 (optional)")
    args = parser.parse_args()

    if args.s3_bucket:
        if args.storage_path:
            _log("S3 ingestion selected; ignoring --storage-path values")
        result = _run_s3_ingestion_job(args.s3_bucket, args.s3_prefix, args.s3_region)
        print(result)
        return

    storage_paths = _parse_storage_paths(args.storage_path)
    if not storage_paths:
        storage_paths = [DEFAULT_STORAGE_PATH]

    resume_mode = args.resume_mode
    if resume_mode == "ask":
        any_logs = any((Path(path) / "ingestion_log.csv").exists() for path in storage_paths)
        if any_logs:
            choice = _prompt_resume_choice()
            resume_mode = "resume" if choice == "1" else "restart"
        else:
            resume_mode = "resume"

    max_workers = max(1, args.max_workers)
    _log("Starting dedicated ingestion script")

    results_by_path: Dict[str, Dict[str, Any]] = {}
    if len(storage_paths) > 1 and max_workers > 1:
        worker_count = min(max_workers, len(storage_paths))
        _log(f"Running {len(storage_paths)} ingestion jobs with {worker_count} workers")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_ingestion_job, path, resume_mode): path
                for path in storage_paths
            }
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
            results_by_path[storage_path] = _run_ingestion_job(storage_path, resume_mode)

    ordered_results = [results_by_path[path] for path in storage_paths]
    if len(ordered_results) == 1:
        print(ordered_results[0])
    else:
        combined = _combine_results(ordered_results)
        print(combined)


if __name__ == "__main__":
    main()
