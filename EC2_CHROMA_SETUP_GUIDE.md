# EC2 ChromaDB + App Setup Guide

This guide sets up ChromaDB persistence, ingestion, and the Streamlit + FastAPI apps
on an AWS EC2 instance.

## 1) Launch EC2

- AMI: Ubuntu 22.04 LTS
- Instance size: t3.large or larger
- Disk: 50+ GB (more if you store many PDFs locally)
- Security group:
  - TCP 22 (SSH) from your IP
  - TCP 8501 (Streamlit) from your IP
  - TCP 8000 (FastAPI) from your IP

## 2) Connect

```bash
ssh -i /path/to/key.pem ubuntu@EC2_PUBLIC_IP
```

## 3) Install system deps

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
```

## 4) Get the code

```bash
git clone <YOUR_REPO_URL>
cd DRUG_REPURPOSING_CHAT_LATEST_WORKING
```

## 5) Create venv + install deps

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_streamlit.txt
```

## 6) Configure environment (.env)

Create `.env` in the repo root:

```bash
cat > .env << 'EOF'
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_EMBEDDING_MODEL=models/embedding-001
GEMINI_CHAT_MODEL=models/gemini-2.0-flash-exp
CHROMA_DB_DIR=./data/chroma
CHROMA_COLLECTION_NAME=drug_docs
DOCS_DIR=./data/docs
S3_BUCKET=your-bucket
S3_PREFIX=llm-docs/testdata
S3_REGION=us-east-1
EOF
```

Notes:
- `CHROMA_DB_DIR` is the persistent store on disk.
- `DOCS_DIR` is used for local PDFs (if you keep them).
- `S3_*` enable S3 mirroring and S3 ingestion.

## 7) Create local data folders (if needed)

```bash
mkdir -p data/chroma data/docs data/testdata
```

If you are running in **S3-only** mode (no local PDFs), you only need:

```bash
mkdir -p data/chroma
```

## 8) Fetch papers (local + S3 mirror)

Example using a CSV with spaces in the name:

```bash
python run_fetch_papers.py \
  --csv-path "data/drug_Data/Drug Repurposing Papers (Drug Names Sets)/Drug Repurposing (Set SAMPLE).csv" \
  --storage-path "data/testdata"
```

This:
- searches PubMed Central
- downloads OA PDFs into `data/testdata/<drug_slug>/`
- uploads to S3 when a drug reaches 3 papers
- discards drugs with fewer than 3 papers

## 9) Fetch papers (S3-only, no local PDFs)

```bash
python run_fetch_papers.py \
  --csv-path "data/drug_Data/Drug Repurposing Papers (Drug Names Sets)/Drug Repurposing (Set SAMPLE).csv" \
  --s3-only \
  --log-dir logs \
  --run-id sample
```

This downloads PDFs into memory and uploads directly to:
`s3://<bucket>/<prefix>/<drug_slug>/<file>.pdf`

## 10) Ingest PDFs into ChromaDB (local)

```bash
python run_ingestion.py --storage-path data/testdata
```

Process multiple drugs concurrently (example: 5 workers):

```bash
python run_ingestion.py --storage-path data/testdata --drug-workers 5
```

Watch mode (poll for new drugs, delete PDFs after ingest):

```bash
python run_ingestion.py --storage-path data/testdata --drug-workers 5 --watch --poll-interval 60 --delete-on-success
```

Resume / restart ingestion using the log:

```bash
python run_ingestion.py --resume-mode resume
python run_ingestion.py --resume-mode restart
```

## 11) Ingest PDFs directly from S3 (no local download)

```bash
python run_ingestion.py --s3-bucket your-bucket --s3-prefix llm-docs/testdata --s3-region us-east-1
```

## 12) Start FastAPI

```bash
python -m app.main
```

FastAPI will listen on port 8000.

## 13) Start Streamlit (optional, local testing only)

Streamlit is **not required** on EC2 for the FastAPI service. Use it only for
local testing or if you explicitly want the UI on the instance.

```bash
python -m streamlit run streamlit_demo.py
```

Streamlit will listen on port 8501.

## 14) (Optional) Keep services running

Basic background run:

```bash
nohup python -m app.main > fastapi.log 2>&1 &

# Optional (UI only)
nohup python -m streamlit run streamlit_demo.py > streamlit.log 2>&1 &
```

## 15) Verify

- FastAPI health:
  ```bash
  curl http://localhost:8000/health
  ```
- Streamlit UI:
  - `http://EC2_PUBLIC_IP:8501`

If you didn't open the ports, use SSH tunnels:

```bash
ssh -i /path/to/key.pem -L 8000:localhost:8000 -L 8501:localhost:8501 ubuntu@EC2_PUBLIC_IP
```

Then open:
- `http://localhost:8000/health`
- `http://localhost:8501`

## 16) Notes on storage strategy

- Local: PDFs live under `DOCS_DIR`.
- S3: PDFs are stored as `s3://<bucket>/<prefix>/<drug_slug>/<file>.pdf`.
- ChromaDB: Persistent vectors are stored in `CHROMA_DB_DIR`.

If you want to rebuild the vector store:

```bash
rm -rf data/chroma
```

Then re-run ingestion.

## 17) Parallel fetches (4 CSVs)

See `EC2_PARALLEL_S3_FETCH_GUIDE.md` for a ready-to-run set of `nohup` commands
that launch 4 S3-only fetch processes with separate logs.
