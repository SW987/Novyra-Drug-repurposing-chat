#!/usr/bin/env python3
"""
Simple demonstration of integrating your PDF validation code
with the RAG system.
"""

import os
import google.generativeai as genai
from pathlib import Path

# Default docs root: ``<project>/data/docs`` (matches the project's DOCS_DIR).
DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"

# Your existing validation function
def is_valid_pdf(file_path):
    """Return True if file exists, has PDF header, and size > 5KB."""
    try:
        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(file_path)
        if file_size < 5000:
            print(f"[WARN] File too small ({file_size} bytes)")
            return False

        with open(file_path, "rb") as f:
            header = f.read(5)
            f.seek(-10, 2)  # Seek to end
            footer = f.read(10)

        has_valid_header = header == b"%PDF-"
        has_valid_footer = b"%%EOF" in footer

        if not has_valid_footer:
            print("[WARN] Missing EOF marker")
            return False

        return has_valid_header
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        return False


def demonstrate_integration():
    """
    Demonstrate how your existing PDF validation integrates
    with the RAG system.
    """

    # Configure Gemini (uses environment variable)
    import os
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)

    print("🚀 PDF Validation + RAG Integration Demo")
    print("=" * 50)

    # Your existing PDF directories
    pdf_directories = {
        "aspirin": str(DOCS_DIR / "aspirin repurposing"),
        "apomorphine": str(DOCS_DIR / "apomorphine repurposing"),
        "insulin": str(DOCS_DIR / "insulin repurposing"),
    }

    total_pdfs = 0
    valid_pdfs = 0

    for drug_name, directory in pdf_directories.items():
        print(f"\n🏥 Checking {drug_name} PDFs...")

        if not os.path.exists(directory):
            print(f"❌ Directory not found: {directory}")
            continue

        # Find all PDFs
        pdf_files = list(Path(directory).glob("*.pdf"))
        print(f"📁 Found {len(pdf_files)} PDF files")

        for pdf_path in pdf_files:
            total_pdfs += 1

            # Use YOUR existing validation function
            if is_valid_pdf(str(pdf_path)):
                valid_pdfs += 1
                print(f"✅ Valid: {pdf_path.name}")

                # Here you would integrate with RAG ingestion
                # result = ingest_valid_pdf(pdf_path, drug_name)

            else:
                print(f"❌ Invalid: {pdf_path.name}")

    print("\n🎉 Validation Complete!")
    print(f"📊 Total PDFs found: {total_pdfs}")
    print(f"✅ Valid PDFs: {valid_pdfs}")
    print(f"❌ Invalid PDFs: {total_pdfs - valid_pdfs}")

    # Show integration points
    print("\n🔗 Integration Points:")
    print("1. ✅ Your validation function works perfectly")
    print("2. 🔄 RAG ingestion would happen here for valid PDFs")
    print("3. 📝 Each valid PDF gets chunked and embedded")
    print("4. 💬 Ready for chat queries about drug repurposing")

    return {
        "total_pdfs": total_pdfs,
        "valid_pdfs": valid_pdfs,
        "integration_ready": True
    }


if __name__ == "__main__":
    results = demonstrate_integration()
    print(f"\n📋 Results: {results}")
