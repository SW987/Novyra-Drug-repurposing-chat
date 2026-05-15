# 🚀 Unified Deployment Demo Guide

## One-Click Deploy to Streamlit Cloud

### ✨ What's New
**Complete self-contained deployment** - Frontend, backend, AND dynamic PDF downloading all in one app!

### Step 1: Prepare Files
- `streamlit_demo.py` - Main app (includes full RAG backend + PubMed integration)
- `requirements_streamlit.txt` - All dependencies
- `app/` directory - Complete backend modules
- `.env` - Environment variables (API keys)

### Step 2: Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file: `streamlit_demo.py`
4. Add secrets:
   ```
   GEMINI_API_KEY = "your_actual_api_key_here"
   GEMINI_EMBEDDING_MODEL = "<your Gemini embedding model>"
   GEMINI_EMBEDDING_DIMENSION = "<vector dim of embedding model>"
   GEMINI_CHAT_MODEL = "<your Gemini chat model>"
   CHROMA_DB_DIR = "./data/chroma"
   CHROMA_COLLECTION_NAME = "drug_docs"
   DOCS_DIR = "./data/docs"
   ```
5. Click Deploy!

### Demo Features
- ✅ **Single deployment** - Everything in one app
- ✅ **Pre-loaded drugs** (aspirin, apomorphine, insulin)
- ✅ **Custom drug analysis** - Enter ANY drug name!
- ✅ **Automatic PubMed search** & PDF download
- ✅ **Real-time RAG processing** of downloaded papers
- ✅ **Intelligent chat** with research-backed answers
- ✅ **Source citations** with expandable previews
- ✅ **Conversation history** and context
- ✅ **Professional UI** for presentations

### Requirements
- Gemini API key configured
- Internet access for PubMed searches
- All backend modules in `app/` directory

### 🚀 Demo URL
**After deployment:** `https://your-app.streamlit.app`

### 🔧 Complete Architecture
```
Streamlit App (Single Deployment)
├── Frontend UI
│   ├── Drug Selection (pre-loaded + custom)
│   ├── Chat Interface
│   └── Progress Indicators
├── PubMed Integration
│   ├── PMC Article Search
│   ├── PDF Link Extraction
│   └── Smart PDF Download
├── RAG Backend
│   ├── PDF Validation & Text Extraction
│   ├── Document Chunking
│   ├── Vector Store (ChromaDB)
│   ├── Semantic Retrieval
│   └── Gemini LLM Integration
└── Research Data (dynamic + pre-loaded)
```

### 🎯 User Experience
1. **Choose pre-loaded drug** → Instant chat
2. **Enter custom drug** → Auto-download research → Chat enabled
3. **Ask questions** → Get evidence-based answers from papers
4. **Explore sources** → See original research citations
