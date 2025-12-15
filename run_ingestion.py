import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.absolute()))

from dotenv import load_dotenv
from app.config import get_settings
from app.ingestion_pipeline import run_ingestion_pipeline

def main():
    print("🚀 Starting dedicated ingestion script...")

    # Explicitly load environment variables
    load_dotenv()
    print("✅ Environment variables loaded.")

    # Get settings
    settings = get_settings()
    print(f"⚙️ Settings loaded: Chroma DB Dir = {settings.chroma_db_dir}, Docs Dir = {settings.docs_dir}")

    drug_dirs = {
        'aspirin': r'C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\aspirin repurposing',
        'apomorphine': r'C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\apomorphine repurposing',
        'insulin': r'C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\insulin repurposing'
    }

    print("Starting PDF ingestion pipeline...")
    results = run_ingestion_pipeline(drug_dirs)
    print("Ingestion Complete!")
    print("\n📋 Ingestion Results:")
    print(results)

if __name__ == "__main__":
    main()


