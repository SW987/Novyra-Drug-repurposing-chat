# EC2 Parallel S3-Only Fetch Guide

Goal: run 4 simultaneous `run_fetch_papers.py` jobs (one per CSV), **store PDFs only in S3**, and keep **separate logs** so there are no conflicts.

## 1) Prereqs

- `.env` has your S3 settings:
  - `S3_BUCKET=your-bucket`
  - `S3_PREFIX=llm-docs/testdata`
  - `S3_REGION=us-east-1`
- AWS credentials are available (recommended: EC2 IAM role).
- Virtual env activated.

## 2) Create a log folder

```bash
mkdir -p logs
```

## 3) Start 4 parallel jobs (S3-only)

Replace the CSV paths with your 4 files. Each run gets a unique `--run-id` and log file.

```bash
nohup python run_fetch_papers.py \
  --csv-path "data/drug_Data/Drug Repurposing Papers (Drug Names Sets)/Drug Repurposing (Set 1).csv" \
  --s3-only \
  --log-dir logs \
  --run-id set1 \
  > logs/fetch_set1.out 2>&1 &

nohup python run_fetch_papers.py \
  --csv-path "data/drug_Data/Drug Repurposing Papers (Drug Names Sets)/Drug Repurposing (Set 2).csv" \
  --s3-only \
  --log-dir logs \
  --run-id set2 \
  > logs/fetch_set2.out 2>&1 &

nohup python run_fetch_papers.py \
  --csv-path "data/drug_Data/Drug Repurposing Papers (Drug Names Sets)/Drug Repurposing (Set 3).csv" \
  --s3-only \
  --log-dir logs \
  --run-id set3 \
  > logs/fetch_set3.out 2>&1 &

nohup python run_fetch_papers.py \
  --csv-path "data/drug_Data/Drug Repurposing Papers (Drug Names Sets)/Drug Repurposing (Set 4).csv" \
  --s3-only \
  --log-dir logs \
  --run-id set4 \
  > logs/fetch_set4.out 2>&1 &
```

Notes:
- `--s3-only` means no PDFs are written to disk. Downloads happen in memory and are uploaded to S3 once 3 PDFs are found.
- If you don’t pass `--s3-bucket`, the script uses `S3_BUCKET` from `.env`.

## 4) Check running processes

```bash
ps aux | grep run_fetch_papers.py
```

## 5) Watch logs (print statements)

```bash
tail -f logs/fetch_set1.out
```

If you started the job while inside `logs/`, the output file will be created in
the current directory (e.g. `./fetch_set1.out`). In that case:

```bash
tail -f fetch_set1.out
```

If you see no output, it may be buffered or waiting for the resume prompt. Use
this unbuffered, non-interactive command:

```bash
nohup env PYTHONUNBUFFERED=1 python -u run_fetch_papers.py \
  --csv-path "data/drug_Data/Drug Repurposing Papers (Drug Names Sets)/Drug Repurposing (Set 1).csv" \
  --s3-only \
  --log-dir logs \
  --run-id set1 \
  --resume-mode resume \
  > logs/fetch_set1.out 2>&1 &
```

Then watch it:

```bash
tail -f logs/fetch_set1.out
```

Check the process:

```bash
ps aux | grep run_fetch_papers.py
```

Check the output file exists:

```bash
ls -lh logs/fetch_set1.out
tail -n 50 logs/fetch_set1.out
```

Each run also writes a **retrieval log** file (for resume tracking):

```
logs/drug_retrieval_log_Drug_Repurposing_(Set_1)_set1.csv
logs/drug_retrieval_log_Drug_Repurposing_(Set_2)_set2.csv
...
```

## 6) Resume a specific CSV later

Use the same `--run-id` so it reads the correct log file:

```bash
python run_fetch_papers.py \
  --csv-path "data/drug_Data/Drug Repurposing Papers (Drug Names Sets)/Drug Repurposing (Set 1).csv" \
  --s3-only \
  --log-dir logs \
  --run-id set1 \
  --resume-mode resume
```

## 7) Why logs don’t conflict

Each run writes a unique log file based on:
- CSV name (stem)
- `--run-id`

That means 4 parallel processes won’t collide, and resume works per CSV.
