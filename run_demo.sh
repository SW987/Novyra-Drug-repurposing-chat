#!/bin/bash
# Drug Repurposing Chat Demo Launcher

echo "🚀 Starting Drug Repurposing Chat Demo..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv_fresh" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_fresh/Scripts/activate

# Check if server is running
echo "🔍 Checking server status..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Server is already running"
else
    echo "❌ Server not running. Please start with: python -m app.main"
    echo "Then run this script again."
    exit 1
fi

# Launch Streamlit demo
echo "🌟 Launching Streamlit demo..."
echo "📱 Demo will open at: http://localhost:8501"
echo ""
echo "💡 Features:"
echo "   • Drug selection (aspirin, apomorphine, insulin)"
echo "   • Chat interface with source citations"
echo "   • RAG-powered answers from research papers"
echo ""

streamlit run streamlit_demo.py


