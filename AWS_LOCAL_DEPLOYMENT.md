# AWS local deployment guide (EC2 + local paper storage)

This guide runs the app on an AWS EC2 instance and stores PDFs on the instance disk
(EBS). No S3 is required or used.

## 1) Launch an EC2 instance

- AMI: Ubuntu 22.04 LTS (or similar)
- Instance size: t3.large or larger (8 GB+ recommended)
- Storage (EBS): 50+ GB if you plan to store many PDFs
- Security group inbound:
  - TCP 22 from your IP (SSH)
  - TCP 8501 from your IP (Streamlit), or keep closed and use SSH tunneling

## 2) Connect to the instance

```bash
ssh -i /path/to/key.pem ubuntu@EC2_PUBLIC_IP
```

## 3) Install system dependencies

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
```

## 4) Get the code onto the instance

Option A: clone from your git repo
```bash
git clone <YOUR_REPO_URL>
cd DRUG_REPURPOSING_CHAT_LATEST_WORKING
```

Option B: copy from local machine
```bash
scp -i /path/to/key.pem -r /local/path/DRUG_REPURPOSING_CHAT_LATEST_WORKING \
  ubuntu@EC2_PUBLIC_IP:/home/ubuntu/
cd /home/ubuntu/DRUG_REPURPOSING_CHAT_LATEST_WORKING
```

## 5) Create the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_streamlit.txt
```

## 6) Configure local storage (no S3)

The app stores PDFs under `DOCS_DIR`. Keep it on the EC2 disk.

Create directories:
```bash
mkdir -p data/docs data/chroma
```

Update `.env`:
```bash
cat > .env << 'EOF'
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_EMBEDDING_MODEL=models/embedding-001
GEMINI_EMBEDDING_DIMENSION=768
GEMINI_CHAT_MODEL=models/gemini-2.0-flash-exp
CHROMA_DB_DIR=./data/chroma
CHROMA_COLLECTION_NAME=drug_docs
DOCS_DIR=./data/docs
EOF
```

Important: Do not set any S3 options and do not pass `--s3-bucket` in scripts.
If you use a different embedding model, update `GEMINI_EMBEDDING_DIMENSION`
accordingly and rebuild Chroma.
If you want S3-only fetching or parallel S3 runs, use `EC2_PARALLEL_S3_FETCH_GUIDE.md`.

## 7) (Optional) Preload or download PDFs locally

If you already have PDFs, copy them into:
```
data/docs/<drug_name>/your_paper.pdf
```

To download papers locally from PubMed (no S3):
```bash
python run_fetch_papers.py \
  --csv-path ./data/drug_Data/drugs.csv \
  --storage-path ./data/docs
```

To ingest the downloaded PDFs into Chroma:
```bash
python run_ingestion.py --storage-path ./data/docs
```

## 8) Start the Streamlit app

```bash
streamlit run streamlit_demo.py --server.address 0.0.0.0 --server.port 8501
```

Access it in your browser:
- `http://EC2_PUBLIC_IP:8501`

If you did not open port 8501, use SSH tunneling:
```bash
ssh -i /path/to/key.pem -L 8501:localhost:8501 ubuntu@EC2_PUBLIC_IP
```
Then open `http://localhost:8501` on your machine.

## 9) Keep the app running (optional)

Simple background run:
```bash
nohup streamlit run streamlit_demo.py --server.address 0.0.0.0 --server.port 8501 \
  > streamlit.log 2>&1 &
```

## 10) Notes about local storage

- PDFs are stored under `DOCS_DIR` on the EC2 disk (EBS).
- Back up your data by snapshotting the EBS volume if needed.
- S3 upload is optional and disabled unless you pass `--s3-bucket`.

## 11) Troubleshooting

- If downloads fail, check outbound internet access from EC2.
- If the app loads but chat errors, verify `GEMINI_API_KEY` in `.env`.
- If the port is unreachable, confirm security group rules.
