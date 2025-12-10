# 🚀 Streamlit Demo Deployment Guide

## Quick Deploy to Streamlit Cloud

### Step 1: Prepare Files
- `streamlit_demo.py` - Main app
- `requirements_streamlit.txt` - Dependencies
- `.env` - Environment variables (API keys)

### Step 2: Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file: `streamlit_demo.py`
4. Add secrets:
   ```
   GEMINI_API_KEY = "your_api_key_here"
   API_BASE_URL = "https://your-fastapi-backend.onrender.com"
   ```
5. Click Deploy!

### Step 3: Backend Setup
Deploy FastAPI backend separately:
```bash
# Use Render, Railway, or Heroku
pip install -r requirements.txt
python -m app.main
```

### Demo Features
- ✅ Drug selection (aspirin, apomorphine, insulin)
- ✅ Real-time chat with research papers
- ✅ Source citations with expandable previews
- ✅ Conversation history
- ✅ Professional UI for presentations

### Requirements
- FastAPI backend must be running
- Gemini API key configured
- Vector database pre-populated with drug data

**Demo URL will be:** `https://your-app.streamlit.app`
