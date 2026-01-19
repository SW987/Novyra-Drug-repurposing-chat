import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.absolute()))

from dotenv import load_dotenv

from app.config import get_settings
from app.paper_fetcher import PaperFetchPipeline


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


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
    except csv.Error:
        return ","


def _normalize_fieldnames(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        return []
    cleaned = []
    for name in fieldnames:
        cleaned.append(name.replace("\ufeff", "").strip())
    return cleaned


def _load_drugs_from_csv(csv_path: Path) -> list[str]:
    drugs = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        delimiter = _detect_delimiter(sample)
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = _normalize_fieldnames(reader.fieldnames)
        reader.fieldnames = fieldnames
        if "drug" not in fieldnames:
            raise ValueError("CSV must contain a 'drug' column")
        for row in reader:
            drug = (row.get("drug") or "").strip()
            if drug:
                drugs.append(drug)
    return drugs


def _append_no_papers_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = ["drug", "processing_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _append_retrieval_log(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = ["drug", "downloaded", "papers_downloaded", "processing_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch drug repurposing papers from a CSV list.")
    parser.add_argument(
        "--csv-path",
        default=r"C:\\Users\\daud.haider\\Desktop\\DRUG_REPURPOSING_CHAT_LATEST_WORKING\\data\\drug_Data\\drugs.csv",
        help="Path to CSV with a 'drug' column",
    )
    parser.add_argument(
        "--storage-path",
        default=r"C:\Users\daud.haider\Desktop\DRUG_REPURPOSING_CHAT_LATEST_WORKING\data\testdata",
        help="Folder to store downloaded PDFs",
    )
    parser.add_argument("--max-papers-per-drug", type=int, default=3)
    parser.add_argument("--max-search-results", type=int, default=100)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--error-delay", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-retry-delay", type=float, default=10.0)
    args = parser.parse_args()

    _log("Starting paper fetch job")

    load_dotenv()
    settings = get_settings()
    settings.docs_dir = str(Path(args.storage_path))

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        _log(f"CSV not found: {csv_path}")
        sys.exit(1)

    storage_path = Path(args.storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    drugs = _load_drugs_from_csv(csv_path)
    if not drugs:
        _log("No drugs found in CSV")
        sys.exit(1)

    _log(f"Loaded {len(drugs)} drugs from {csv_path}")
    _log(f"Storing PDFs under {storage_path}")

    fetcher = PaperFetchPipeline(
        settings,
        request_delay=args.request_delay,
        error_delay=args.error_delay,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
    )

    job_start = time.time()
    overall = {
        "timestamp": time.time(),
        "drugs_processed": [],
        "total_downloaded": 0,
        "drug_results": [],
    }

    no_papers_rows = []
    retrieval_log_path = storage_path / "drug_retrieval_log.csv"
    total_drugs = len(drugs)
    for index, drug_name in enumerate(drugs, start=1):
        drug_start = time.time()
        progress = _render_progress(index, total_drugs)
        _log(f"{progress} Fetching papers for {drug_name}")

        result = None
        for attempt in range(1, args.api_retries + 1):
            try:
                result = fetcher.fetch_drug_papers(
                    drug_name,
                    max_papers=args.max_papers_per_drug,
                    max_search_results=args.max_search_results,
                )
                if result.get("success"):
                    break
                _log(f"Attempt {attempt} failed for {drug_name}")
            except Exception as exc:
                _log(f"Attempt {attempt} errored for {drug_name}: {exc}")

            if attempt < args.api_retries:
                wait_seconds = args.api_retry_delay * attempt
                _log(f"Waiting {wait_seconds:.1f}s before retrying {drug_name}")
                time.sleep(wait_seconds)

        if result is None:
            result = {
                "success": False,
                "drug": drug_name,
                "downloaded": 0,
                "error": "fetch_failed",
            }

        drug_duration = time.time() - drug_start
        job_elapsed = time.time() - job_start
        _log(
            f"Completed {drug_name} in {drug_duration:.1f}s "
            f"(elapsed {job_elapsed:.1f}s since start)"
        )

        if result.get("papers_found", 0) == 0:
            no_papers_rows.append(
                {"drug": drug_name, "processing_seconds": f"{drug_duration:.1f}"}
            )

        _append_retrieval_log(
            retrieval_log_path,
            {
                "drug": drug_name,
                "downloaded": "yes" if result.get("downloaded", 0) > 0 else "no",
                "papers_downloaded": str(result.get("downloaded", 0)),
                "processing_seconds": f"{drug_duration:.1f}",
            },
        )

        overall["drugs_processed"].append(drug_name)
        overall["total_downloaded"] += result.get("downloaded", 0)
        overall["drug_results"].append(result)

        time.sleep(args.request_delay)

    if no_papers_rows:
        no_papers_path = storage_path / "no_papers_found.csv"
        _append_no_papers_csv(no_papers_path, no_papers_rows)
        _log(f"Wrote {len(no_papers_rows)} entries to {no_papers_path}")

    _log("Paper fetch complete")
    print(overall)


if __name__ == "__main__":
    main()
