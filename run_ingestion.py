import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.absolute()))

from dotenv import load_dotenv

from app.config import get_settings
from app.ingestion_pipeline import PDFIngestionPipeline


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDF ingestion for downloaded drug papers.")
    parser.add_argument(
        "--storage-path",
        default=r"C:\Users\daud.haider\Desktop\DRUG_REPURPOSING_CHAT_LATEST_WORKING\data\testdata",
        help="Folder containing drug subfolders",
    )
    args = parser.parse_args()

    _log("Starting dedicated ingestion script")

    load_dotenv()
    _log("Environment variables loaded")

    settings = get_settings()
    if args.storage_path:
        settings.docs_dir = str(Path(args.storage_path))

    _log(f"Settings loaded: Chroma DB Dir = {settings.chroma_db_dir}, Docs Dir = {settings.docs_dir}")

    docs_dir = Path(settings.docs_dir)
    drug_dirs = {}

    if docs_dir.exists():
        for subdir in docs_dir.iterdir():
            if not subdir.is_dir():
                continue
            drug_dirs[subdir.name] = str(subdir)

    if not drug_dirs:
        _log(f"No drug folders found under {docs_dir}")
        _log("Update run_ingestion.py to point at your PDF folders")
        return

    pipeline = PDFIngestionPipeline(settings)

    overall_results = {
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
        _log(f"{progress} Processing drug: {drug_name}")
        _log("=" * 50)

        drug_result = pipeline.ingest_directory_batch(directory_path, drug_name)
        drug_result["drug_name"] = drug_name
        drug_result["directory"] = directory_path

        overall_results["drugs_processed"].append(drug_name)
        overall_results["total_files"] += drug_result["total_files"]
        overall_results["total_successful"] += drug_result["successful"]
        overall_results["total_failed"] += drug_result["failed"]
        overall_results["drug_results"].append(drug_result)

    _log("Ingestion complete")
    _log(f"Total PDFs processed: {overall_results['total_files']}")
    _log(f"Successful: {overall_results['total_successful']}")
    _log(f"Failed: {overall_results['total_failed']}")

    print(overall_results)


if __name__ == "__main__":
    main()
