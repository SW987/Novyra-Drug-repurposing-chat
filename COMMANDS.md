# Commands

This file lists the main commands to install, fetch papers, ingest PDFs, and run the app.

## Environment setup (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements_streamlit.txt
```

If you are running backend-only scripts:

```powershell
pip install -r requirements.txt
```

## Configure environment variables

Create `.env` in the project root:

```bash
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_EMBEDDING_MODEL=models/embedding-001
GEMINI_CHAT_MODEL=models/gemini-2.0-flash-exp
CHROMA_DB_DIR=./data/chroma
CHROMA_COLLECTION_NAME=drug_docs
DOCS_DIR=./data/docs
S3_BUCKET=your-bucket
S3_PREFIX=llm-docs/testdata
S3_REGION=us-east-1
```

## Run the Streamlit app

```powershell
streamlit run streamlit_demo.py
```

If the Streamlit launcher path is broken, use:

```powershell
python -m streamlit run streamlit_demo.py
```

## Run the FastAPI API

```powershell
python -m app.main
```

Or with uvicorn directly:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Fetch papers (downloads PDFs only)

Default (prompts to resume/restart if a retrieval log exists):

```powershell
python run_fetch_papers.py
```

Resume (skip drugs already in the retrieval log):

```powershell
python run_fetch_papers.py --resume-mode resume
```

Restart (reprocess everything and clear the log):

```powershell
python run_fetch_papers.py --resume-mode restart
```

Optional log controls (avoid conflicts when running multiple CSVs):

```powershell
python run_fetch_papers.py --csv-path .\data\drug_Data\drugs.csv --log-dir .\logs --run-id set1
```

Custom CSV and output folder:

```powershell
python run_fetch_papers.py --csv-path .\data\drug_Data\drugs.csv --storage-path .\data\testdata
```

Sample CSV with spaces/parentheses:

```powershell
python run_fetch_papers.py --csv-path "data\drug_Data\Drug Repurposing Papers (Drug Names Sets)\Drug Repurposing (Set SAMPLE).csv" --storage-path "data\testdata"
```

Mirror PDFs to S3 while downloading:

```powershell
python run_fetch_papers.py --csv-path .\data\drug_Data\drugs.csv --storage-path .\data\testdata --s3-bucket novyra --s3-prefix llm-docs/testdata
```

S3-only (no local PDFs) with separate logs:

```powershell
python run_fetch_papers.py --csv-path .\data\drug_Data\drugs.csv --s3-only --log-dir .\logs --run-id set1
```

## Ingest downloaded PDFs into the vector DB

```powershell
python run_ingestion.py
```

Custom storage path:

```powershell
python run_ingestion.py --storage-path .\data\testdata
```

Process drugs concurrently (5 workers):

```powershell
python run_ingestion.py --storage-path .\data\testdata --drug-workers 5
```

Watch mode (poll for new drugs, delete PDFs after ingest):

```powershell
python run_ingestion.py --storage-path .\data\testdata --drug-workers 5 --watch --poll-interval 60 --delete-on-success
```

Multiple storage paths (run jobs in parallel):

```powershell
python run_ingestion.py --storage-path .\data\run1 --storage-path .\data\run2 --max-workers 2
```

Resume (skip already ingested drugs in `ingestion_log.csv`):

```powershell
python run_ingestion.py --resume-mode resume
```

Restart (reprocess everything and clear the log):

```powershell
python run_ingestion.py --resume-mode restart
```

## Ingest PDFs directly from S3 (no local download)

```powershell
python run_ingestion.py --s3-bucket your-bucket --s3-prefix llm-docs/testdata
```

Optional region:
```powershell
python run_ingestion.py --s3-bucket your-bucket --s3-prefix llm-docs/testdata --s3-region us-east-1
```

## Integrated workflow demo (optional)

```powershell
python integrated_workflow.py
```

## Run test scripts (optional)

```powershell
python simple_test.py
python test_integration.py
python test_dynamic_drug.py
python test_pubmed_integration.py
```
