#!/usr/bin/env python3
"""
Setup script for Drug Repurposing Research Assistant
Automates the initial setup process
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.11+")
        return False


def create_virtual_environment():
    """Create virtual environment."""
    if os.path.exists("venv"):
        print("⚠️  Virtual environment already exists")
        return True

    return run_command("python -m venv venv", "Creating virtual environment")


def activate_and_install():
    """Activate virtual environment and install dependencies."""
    # Note: Activation in scripts is tricky, so we'll install directly
    pip_path = "venv\\Scripts\\pip.exe" if os.name == 'nt' else "venv/bin/pip"

    if not os.path.exists(pip_path):
        print(f"❌ Pip not found at {pip_path}")
        return False

    commands = [
        (f"{pip_path} install --upgrade pip", "Upgrading pip"),
        (f"{pip_path} install -r requirements.txt", "Installing dependencies")
    ]

    for command, description in commands:
        if not run_command(command, description):
            return False

    return True


def setup_environment_file():
    """Set up environment configuration."""
    if os.path.exists(".env"):
        print("⚠️  .env file already exists - skipping creation")
        return True

    if not os.path.exists(".env.example"):
        print("❌ .env.example not found")
        return False

    try:
        shutil.copy(".env.example", ".env")
        print("✅ Created .env file from template")
        print("⚠️  IMPORTANT: Edit .env and add your GEMINI_API_KEY")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    python_exe = "venv\\Scripts\\python.exe" if os.name == 'nt' else "venv/bin/python"

    if not os.path.exists(python_exe):
        print(f"❌ Python executable not found at {python_exe}")
        return False

    # Test import
    test_commands = [
        (f'{python_exe} -c "from app.config import get_settings; print(\'Config import: OK\')"', "Testing configuration import"),
        (f'{python_exe} -c "from app.utils import is_valid_pdf; print(\'Utils import: OK\')"', "Testing utilities import")
    ]

    for command, description in test_commands:
        if not run_command(command, description):
            return False

    return True


def show_next_steps():
    """Show user what to do next."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Edit .env file and add your Gemini API key:")
    print("   GEMINI_API_KEY=your_actual_api_key_here")
    print("\n2. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("\n3. Test the system:")
    print("   python simple_test.py")
    print("\n4. Ingest your PDF data:")
    print("   python -c \"from app.ingestion_pipeline import run_ingestion_pipeline; run_ingestion_pipeline({'aspirin': 'path/to/aspirin/pdfs'})\"")
    print("\n5. Start the API server:")
    print("   python app/main.py")
    print("\n6. Test chat functionality:")
    print("   curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"session_id\": \"test\", \"drug_id\": \"aspirin\", \"message\": \"What are aspirin benefits?\"}'")


def main():
    """Main setup function."""
    print("🚀 Drug Repurposing Research Assistant - Setup")
    print("=" * 60)

    steps = [
        ("Checking Python version", check_python_version),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", activate_and_install),
        ("Setting up environment", setup_environment_file),
        ("Testing basic functionality", test_basic_functionality)
    ]

    completed_steps = 0
    total_steps = len(steps)

    for step_name, step_function in steps:
        print(f"\n[{completed_steps + 1}/{total_steps}] {step_name}")
        if step_function():
            completed_steps += 1
        else:
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please check the error messages above and try again.")
            return False

    show_next_steps()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
