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
```

## Run the Streamlit app

```powershell
streamlit run streamlit_demo.py
```

## Fetch papers (downloads PDFs only)

Default (prompts to resume/restart if a retrieval log exists):

```powershell
python run_fetch_papers.py
```

Resume (skip drugs already in `drug_retrieval_log.csv`):

```powershell
python run_fetch_papers.py --resume-mode resume
```

Restart (reprocess everything and clear the log):

```powershell
python run_fetch_papers.py --resume-mode restart
```

Custom CSV and output folder:

```powershell
python run_fetch_papers.py --csv-path .\data\drug_Data\drugs.csv --storage-path .\data\testdata
```

## Ingest downloaded PDFs into the vector DB

```powershell
python run_ingestion.py
```

Custom storage path:

```powershell
python run_ingestion.py --storage-path .\data\testdata
```

Multiple storage paths (run jobs in parallel):

```powershell
python run_ingestion.py --storage-path .\data\run1 --storage-path .\data\run2 --max-workers 2
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
