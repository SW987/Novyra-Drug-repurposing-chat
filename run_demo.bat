@echo off
REM Drug Repurposing Chat Demo Launcher (Windows)

echo 🚀 Starting Drug Repurposing Chat Demo...
echo.

REM Check if virtual environment exists
if not exist "venv_fresh" (
    echo ❌ Virtual environment not found. Please run setup first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv_fresh\Scripts\activate

REM Check if server is running
echo 🔍 Checking server status...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Server is already running
) else (
    echo ❌ Server not running. Please start with: python -m app.main
    echo Then run this script again.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Launch Streamlit demo
echo 🌟 Launching Streamlit demo...
echo 📱 Demo will open at: http://localhost:8501
echo.
echo 💡 Features:
echo    • Drug selection (aspirin, apomorphine, insulin)
echo    • Chat interface with source citations
echo    • RAG-powered answers from research papers
echo.

streamlit run streamlit_demo.py

pause


