# 🧬 Drug Repurposing Chat API

AI-powered chat system for drug repurposing research. Ask questions about scientific papers and get evidence-based answers with source citations.

## ⚡ Quick Start (5 minutes)

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Get API Key
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create API key and copy it

### 3. Configure
```bash
copy .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here
```

### 4. Process PDFs
```bash
python -c "
from app.ingestion_pipeline import run_ingestion_pipeline
drug_dirs = {
    'aspirin': r'C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\aspirin repurposing',
    'apomorphine': r'C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\apomorphine repurposing',
    'insulin': r'C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\insulin repurposing'
}
run_ingestion_pipeline(drug_dirs)
"
```

### 5. Start Chat
```bash
python app/main.py
```

### 6. Ask Questions
```bash
# Aspirin cancer benefits
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_1", "drug_id": "aspirin", "message": "What are aspirin benefits for cancer?"}'

# Apomorphine applications
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_2", "drug_id": "apomorphine", "message": "How is apomorphine repurposed?"}'
```

## 💬 How to Use

### Start the Server
```bash
python app/main.py
# Server runs at http://localhost:8000
```

### Chat API
```bash
# Basic chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_1",
    "drug_id": "aspirin",
    "message": "What are aspirin benefits for cancer prevention?"
  }'

# Chat about specific drug
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_2",
    "drug_id": "apomorphine",
    "message": "How is apomorphine being repurposed?"
  }'
```

### Response Format
```json
{
  "answer": "Aspirin has shown promise in cancer prevention through COX-2 inhibition and anti-inflammatory effects...",
  "sources": [
    {
      "doc_id": "PMC11242460",
      "doc_title": "Aspirin Repurposing PMC11242460",
      "text_preview": "Low-dose aspirin for the prevention of...",
      "distance": 0.123
    }
  ],
  "session_id": "session_1"
}
```

### Ingest More PDFs
```bash
# Ingest PDFs from directories
python -c "
from app.ingestion_pipeline import run_ingestion_pipeline
drug_dirs = {'new_drug': '/path/to/new_drug/pdfs'}
run_ingestion_pipeline(drug_dirs)
"
```

## ⚙️ Configuration

Create `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here
# Optional settings below (defaults shown)
GEMINI_EMBEDDING_MODEL=models/embedding-001
GEMINI_CHAT_MODEL=models/gemini-2.0-flash-exp
CHROMA_DB_DIR=./data/chroma
```

## 📁 PDF Organization

Put PDFs in folders like this:
```
downloads/
├── aspirin repurposing/
│   ├── aspirin_repurposing_PMC11242460.pdf
│   └── aspirin_repurposing_PMC11866938.pdf
├── apomorphine repurposing/
│   └── apomorphine_repurposing_PMC5995787.pdf
└── insulin repurposing/
    └── insulin_repurposing_PMC11919260.pdf
```

## 🔗 Integration

### With Existing Download Systems

Add this to your existing download script:
```python
from app.ingestion_pipeline import PDFIngestionPipeline
from app.config import get_settings

# After downloading + validating PDF
settings = get_settings()
pipeline = PDFIngestionPipeline(settings)
result = pipeline.validate_and_ingest_pdf(pdf_path, drug_name)
```

### Automated Processing
```python
from app.ingestion_pipeline import run_ingestion_pipeline

drug_dirs = {
    "aspirin": "/path/to/aspirin/pdfs",
    "metformin": "/path/to/metformin/pdfs"
}
results = run_ingestion_pipeline(drug_dirs)
```

## 🧪 Testing

```bash
# Test basic functionality
python simple_test.py

# Test integration
python test_integration.py
```

## 🐛 Troubleshooting

**API Key Issues:**
- Verify key in `.env` file
- Check Google AI Studio account

**PDF Processing:**
- PDFs must not be password-protected
- Ensure PDFs contain selectable text

**Module Errors:**
```bash
pip install -r requirements.txt
```

**Vector DB Issues:**
```bash
# Clear and restart
rmdir /s data\chroma
python -c "from app.ingestion_pipeline import run_ingestion_pipeline; run_ingestion_pipeline(your_dirs)"
```

## 📋 API Reference

- `POST /chat` - Ask questions about drugs
- `POST /ingest-pdfs` - Process PDF directories
- `GET /health` - Check system status
- `GET /docs` - Interactive API documentation

## 📄 License

MIT License - see LICENSE file.

---

**Ready to chat with your drug repurposing research! 🧬💬**
