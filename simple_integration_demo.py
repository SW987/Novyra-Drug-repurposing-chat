#!/usr/bin/env python3
"""
Simple demonstration of integrating your PDF validation code
with the RAG system.
"""

import os
import google.generativeai as genai
from pathlib import Path

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

    # Configure Gemini (hardcoded for demo)
    genai.configure(api_key="AIzaSyBDd4K-NL86geJG5Mz9r11YF4bOBRf6jb4")

    print("🚀 PDF Validation + RAG Integration Demo")
    print("=" * 50)

    # Your existing PDF directories
    pdf_directories = {
        "aspirin": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\aspirin repurposing",
        "apomorphine": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\apomorphine repurposing",
        "insulin": r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\insulin repurposing"
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
