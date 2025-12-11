# 🧬 Drug Repurposing Chat System

**AI-powered research assistant** that answers questions about ANY drug using scientific literature via RAG (Retrieval-Augmented Generation). Supports pre-loaded drugs + dynamic research paper downloads for custom drugs.

## ⚡ Quick Start (3 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Get Gemini API Key
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create API key and copy it

### 3. Configure Environment
```bash
# Create .env file
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_EMBEDDING_MODEL=models/embedding-001
GEMINI_CHAT_MODEL=models/gemini-2.0-flash-exp
CHROMA_DB_DIR=./data/chroma
CHROMA_COLLECTION_NAME=drug_docs
DOCS_DIR=./data/docs
```

### 4. Launch the App
```bash
streamlit run streamlit_demo.py
```
**That's it!** Opens at `http://localhost:8501` 🎉

## ✨ Key Features

- **🎯 Dual Drug Support**:
  - **Pre-loaded Drugs**: Aspirin, apomorphine, insulin (instant chat)
  - **Custom Drugs**: Enter ANY drug name → auto-download research papers → enable chat

- **🔄 Smart Research Pipeline**:
  - PubMed Central search for repurposing studies
  - Open access PDF download and validation
  - Automatic text extraction and chunking
  - Vector embeddings for semantic search

- **💬 Intelligent Chat**:
  - RAG-powered answers from scientific literature
  - Conversation history and context awareness
  - Source citations with research paper previews
  - Multi-document synthesis for comprehensive answers

- **🚀 Unified Deployment**: Single Streamlit app (no separate backend needed)

## 💬 How to Use

### Launch the Complete System
```bash
streamlit run streamlit_demo.py
```
**Opens at:** `http://localhost:8501`

### Interface Options

#### Option A: Pre-loaded Drugs (Instant Access)
- Select from dropdown: **Aspirin**, **Apomorphine**, or **Insulin**
- Start chatting immediately with existing research data
- Perfect for demos and presentations

#### Option B: Custom Drug Research (Dynamic)
- Enter ANY drug name (e.g., metformin, hydroxychloroquine, statins)
- System automatically:
  - Searches PubMed for repurposing research
  - Downloads open access PDFs
  - Processes and indexes the content
  - Enables intelligent chat about that drug

### Demo Features
- **💊 Drug Selection**: Dropdown + custom input
- **💬 Chat Interface**: Message history, context awareness
- **📚 Source Citations**: Expandable paper previews with relevance scores
- **⚡ Real-time Processing**: Progress bars for PDF downloads
- **🔍 Research Integration**: Direct PubMed connectivity

### Programmatic Usage
```python
# Use the integrated system programmatically
from app.ingestion_pipeline import PDFIngestionPipeline
from app.config import get_settings

# Initialize
settings = get_settings()
pipeline = PDFIngestionPipeline(settings)

# Download and process papers for any drug
result = pipeline.download_and_ingest_drug_papers("metformin", max_papers=5)
print(f"Downloaded: {result['downloaded']}, Ingested: {result['ingested']}")

# Now chat about the drug using the integrated RAG system
from streamlit_demo import make_chat_request

response = make_chat_request(
    drug_id="metformin",
    message="What are metformin repurposing applications?",
    session_id="demo_session"
)
print(response["answer"])
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:
```bash
# Required
GEMINI_API_KEY=your_actual_api_key_here

# Optional (defaults shown)
GEMINI_EMBEDDING_MODEL=models/embedding-001
GEMINI_CHAT_MODEL=models/gemini-2.0-flash-exp
CHROMA_DB_DIR=./data/chroma
CHROMA_COLLECTION_NAME=drug_docs
DOCS_DIR=./data/docs
```

### Pre-loaded Research Data
The system comes with research papers for:
- **Aspirin**: Cancer prevention & cardiovascular studies
- **Apomorphine**: Parkinson's & addiction repurposing
- **Insulin**: Metabolic research applications

### Dynamic Research Downloads
For custom drugs, the system automatically:
- Searches PubMed Central for repurposing studies
- Downloads open access PDFs
- Validates and processes the content
- Makes it available for intelligent chat

**Note:** Not all drugs have open access research papers. Try: aspirin, metformin, hydroxychloroquine, statins.

## 🧪 Testing & Validation

```bash
# Test the complete unified system
python test_unified_system.py

# Test basic backend functionality
python test_self_contained.py

# Test specific components
python simple_test.py
```

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → Set main file: `streamlit_demo.py`
4. Add secrets: `GEMINI_API_KEY`
5. Deploy! 🎉

### Local Development
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Configure .env file with your Gemini API key

# Launch app
streamlit run streamlit_demo.py
```

## 🐛 Troubleshooting

### API Key Issues
```bash
# Verify key in .env file
echo $GEMINI_API_KEY

# Check Google AI Studio account
# Visit: https://makersuite.google.com/app/apikey
```

### PDF Download Issues
- **"No OA PDF available"**: Normal for many drugs - try aspirin, metformin, hydroxychloroquine
- **Network timeouts**: Check internet connection, try different drug
- **Validation failures**: PDFs must contain selectable text, not images

### Module Installation
```bash
# For deployment issues
pip install -r requirements_streamlit.txt

# Clear cache if needed
pip cache purge
```

### Vector Database Issues
```bash
# Reset database
rm -rf data/chroma

# Restart the app - it will rebuild automatically
streamlit run streamlit_demo.py
```

### Common Issues
- **"Module not found"**: Run from project root directory
- **"API key invalid"**: Verify key in Google AI Studio
- **"No papers found"**: Try different drug names or check spelling
- **"Streamlit not working"**: Ensure port 8501 is available

## 📊 System Architecture

```
🖥️  Streamlit App (Single Deployment)
├── 🎨 Frontend UI
│   ├── Drug Selection (pre-loaded + custom)
│   ├── Chat Interface & History
│   └── Progress Indicators
├── 🔬 Research Pipeline
│   ├── PubMed Search & Download
│   ├── PDF Validation & Processing
│   └── Vector Database Integration
└── 🤖 RAG System
    ├── Semantic Retrieval
    ├── Context Synthesis
    └── Gemini LLM Integration
```

## 📄 License

MIT License - see LICENSE file.

---

## 🎯 System Capabilities & Limitations

### ✅ What It Does Well
- **Pre-loaded Drugs**: Instant access to aspirin, apomorphine, insulin research
- **Custom Drugs**: Automatic research paper discovery for any drug
- **Intelligent Answers**: Evidence-based responses with source citations
- **Research Integration**: Direct PubMed connectivity for latest studies
- **User-Friendly**: No technical expertise required

### ⚠️ Current Limitations
- **Open Access Only**: Only free research papers (not paywalled content)
- **PDF Format**: Requires open access PDFs, not all research is available
- **English Only**: Currently optimized for English research papers
- **Research Quality**: Answers based on available literature quality

### 🔮 Future Enhancements
- Additional research databases (bioRxiv, ResearchGate)
- Multi-language support
- Advanced filtering (date ranges, study types)
- Collaborative features

---

**🧬 Ready to explore drug repurposing research with AI-powered insights!**

**Launch:** `streamlit run streamlit_demo.py` 🚀

**Demo Drugs:** aspirin, metformin, hydroxychloroquine, statins
