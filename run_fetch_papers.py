"""
PubMed paper fetcher — Phase 2 of the corpus build pipeline.

Reads a CSV of drug names and drives ``app.paper_fetcher.PaperFetchPipeline``
to:

1. Search PubMed for each drug's repurposing papers.
2. Download open-access PDFs from PubMed Central.
3. Optionally mirror PDFs to S3 (or store only in S3 with ``--s3-only``).
4. Write per-run logs next to the storage path:
   - ``drug_retrieval_log_<csv>.csv`` — outcome per drug (downloaded count,
     duration).
   - ``no_papers_found_<csv>.csv`` — drugs whose search returned zero hits.

Supports resume/restart by replaying the retrieval log so an interrupted
run can pick up where it stopped. Drugs that yield fewer than three papers
are discarded to keep the corpus dense.

The folder layout produced here is the input expected by
``run_ingestion.py`` (Phase 3), which embeds the PDFs into ChromaDB.
"""

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
    """Best-effort delimiter sniff so the CSV works with comma/tab/semicolon exports."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
    except csv.Error:
        return ","


def _normalize_fieldnames(fieldnames: list[str] | None) -> list[str]:
    """Strip the UTF-8 BOM and surrounding whitespace from CSV headers."""
    if not fieldnames:
        return []
    cleaned = []
    for name in fieldnames:
        cleaned.append(name.replace("\ufeff", "").strip())
    return cleaned


def _load_drugs_from_csv(csv_path: Path) -> list[str]:
    """Return drug names from the ``drug`` column of ``csv_path``."""
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
    """Append drugs that returned zero PubMed hits to the no-papers log."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = ["drug", "processing_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _append_retrieval_log(csv_path: Path, row: dict[str, str]) -> None:
    """Record the outcome of one drug to the retrieval log (drives resume logic)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = ["drug", "downloaded", "papers_downloaded", "processing_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _load_processed_drugs(csv_path: Path) -> set[str]:
    """Replay the retrieval log to find which drugs were already attempted."""
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
    """Interactive resume/restart prompt when an existing log is detected."""
    prompt = (
        "Retrieval log found. Choose an option:\n"
        "  1) Resume (skip already processed drugs)\n"
        "  2) Restart (process all drugs)\n"
        "Enter 1 or 2: "
    )
    while True:
        choice = input(prompt).strip()
        if choice in {"1", "2"}:
            return choice
        print("Please enter 1 or 2.")


def main() -> None:
    """Parse CLI args, load the drug CSV, and run the fetch loop end-to-end.

    Per drug: search PubMed, download up to ``--max-papers-per-drug`` PDFs,
    discard the batch if fewer than three papers were retrieved, and append
    the result to the retrieval log. Retries are applied per drug with a
    linear backoff (``--api-retries`` × ``--api-retry-delay``).
    """
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
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Folder to store retrieval logs (defaults to storage-path)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier to avoid log conflicts",
    )
    parser.add_argument(
        "--s3-only",
        action="store_true",
        help="Store PDFs only in S3 (no local files). Requires S3 settings.",
    )
    parser.add_argument("--max-papers-per-drug", type=int, default=3)
    parser.add_argument("--max-search-results", type=int, default=100)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--error-delay", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-retry-delay", type=float, default=10.0)
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket to mirror downloaded PDFs")
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="S3 prefix to store PDFs (e.g. llm-docs/testdata)",
    )
    parser.add_argument("--s3-region", default=None, help="AWS region for S3 (optional)")
    parser.add_argument(
        "--resume-mode",
        choices=["ask", "resume", "restart"],
        default="ask",
        help="How to handle existing retrieval logs",
    )
    args = parser.parse_args()

    _log("Starting paper fetch job")

    min_papers_for_upload = 3
    load_dotenv()
    settings = get_settings()
    settings.docs_dir = str(Path(args.storage_path))

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        _log(f"CSV not found: {csv_path}")
        sys.exit(1)

    storage_path = Path(args.storage_path)
    if not args.s3_only:
        storage_path.mkdir(parents=True, exist_ok=True)

    drugs = _load_drugs_from_csv(csv_path)
    if not drugs:
        _log("No drugs found in CSV")
        sys.exit(1)

    if not args.s3_bucket and settings.s3_bucket:
        args.s3_bucket = settings.s3_bucket
        if args.s3_prefix == "" and settings.s3_prefix:
            args.s3_prefix = settings.s3_prefix
        if not args.s3_region and settings.s3_region:
            args.s3_region = settings.s3_region
    if args.s3_only and not args.s3_bucket:
        _log("S3-only mode requires an S3 bucket (set --s3-bucket or S3_BUCKET).")
        sys.exit(1)

    _log(f"Loaded {len(drugs)} drugs from {csv_path}")
    if args.s3_only:
        _log("Storing PDFs only in S3 (no local files)")
    else:
        _log(f"Storing PDFs under {storage_path}")
    _log(f"Uploading to S3 only when a drug has >= {min_papers_for_upload} papers")
    if args.s3_bucket:
        prefix = args.s3_prefix.strip("/")
        target = f"s3://{args.s3_bucket}/{prefix}" if prefix else f"s3://{args.s3_bucket}"
        _log(f"Mirroring PDFs to {target}")

    fetcher = PaperFetchPipeline(
        settings,
        request_delay=args.request_delay,
        error_delay=args.error_delay,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_region=args.s3_region,
        store_local=not args.s3_only,
    )

    job_start = time.time()
    overall = {
        "timestamp": time.time(),
        "drugs_processed": [],
        "total_downloaded": 0,
        "drug_results": [],
    }

    no_papers_rows = []
    log_dir = Path(args.log_dir) if args.log_dir else storage_path
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_stem = csv_path.stem.replace(" ", "_")
    run_suffix = f"_{args.run_id}" if args.run_id else ""
    retrieval_log_path = log_dir / f"drug_retrieval_log_{csv_stem}{run_suffix}.csv"
    if retrieval_log_path.exists():
        if args.resume_mode == "ask":
            choice = _prompt_resume_choice()
        else:
            choice = "1" if args.resume_mode == "resume" else "2"

        if choice == "1":
            processed = _load_processed_drugs(retrieval_log_path)
            if processed:
                original_count = len(drugs)
                drugs = [drug for drug in drugs if drug not in processed]
                skipped = original_count - len(drugs)
                _log(f"Resuming: skipping {skipped} already-processed drugs")
            else:
                _log("Retrieval log was empty; continuing with all drugs")
        else:
            _log("Restarting: clearing retrieval log for a fresh run")
            retrieval_log_path.unlink()

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
                    upload_to_s3=bool(args.s3_bucket),
                    upload_after=min_papers_for_upload,
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

        downloaded_files = [Path(p) for p in result.get("downloaded_files", [])]
        downloaded_count = result.get("downloaded", 0)

        if downloaded_count < min_papers_for_upload and downloaded_files:
            _log(
                f"Discarding {downloaded_count} papers for {drug_name} "
                f"(needs >= {min_papers_for_upload})"
            )
            for file_path in downloaded_files:
                try:
                    file_path.unlink()
                except OSError as exc:
                    _log(f"Failed to delete {file_path}: {exc}")

            output_folder = Path(result.get("output_folder", ""))
            if output_folder.exists():
                try:
                    if not any(output_folder.iterdir()):
                        output_folder.rmdir()
                except OSError:
                    pass

            result["downloaded"] = 0
            result["downloaded_files"] = []
            result["downloaded_s3"] = []
            result["discarded"] = downloaded_count
            result["success"] = False

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
        no_papers_path = log_dir / f"no_papers_found_{csv_stem}{run_suffix}.csv"
        _append_no_papers_csv(no_papers_path, no_papers_rows)
        _log(f"Wrote {len(no_papers_rows)} entries to {no_papers_path}")

    _log("Paper fetch complete")
    print(overall)


if __name__ == "__main__":
    main()
