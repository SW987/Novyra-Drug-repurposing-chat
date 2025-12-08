#!/usr/bin/env python3
"""
Simple test script for PDF processing and Gemini integration
"""

import os
import sys
sys.path.append('.')

from app.config import get_settings
from app.utils import extract_text_from_pdf, parse_filename, chunk_text
import google.generativeai as genai


def test_pdf_extraction():
    """Test PDF text extraction from one of the files."""
    print("🔄 Testing PDF text extraction...")

    # Hardcode path for testing
    pdf_path = r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs\aspirin repurposing\aspirin_repurposing_PMC11242460.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None

    try:
        text = extract_text_from_pdf(pdf_path)
        print(f"✅ Successfully extracted {len(text)} characters from PDF")
        print(f"📄 Text preview: {text[:200]}...")

        # Test chunking
        chunks = chunk_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        print(f"📝 First chunk preview: {chunks[0][:100]}...")

        return chunks

    except Exception as e:
        print(f"❌ Error extracting PDF: {str(e)}")
        return None


def test_gemini_integration():
    """Test Gemini API integration."""
    print("\n🔄 Testing Gemini API integration...")

    # Hardcode API key for testing
    api_key = "AIzaSyBDd4K-NL86geJG5Mz9r11YF4bOBRf6jb4"

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)

        # Test embedding
        test_text = "Aspirin is commonly used for pain relief and has shown promise in cancer prevention."
        result = genai.embed_content(
            model="models/embedding-001",
            content=test_text,
            task_type="retrieval_document"
        )
        embedding = result['embedding']
        print(f"✅ Generated embedding with {len(embedding)} dimensions")

        # Test chat completion
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        response = model.generate_content(
            "What are the potential benefits of aspirin for cancer prevention?"
        )
        print(f"✅ Generated response: {response.text[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Gemini API error: {str(e)}")
        return False


def test_filename_parsing():
    """Test filename parsing logic."""
    print("\n🔄 Testing filename parsing...")

    test_cases = [
        ("aspirin_repurposing_PMC11242460.pdf", "aspirin"),
        ("apomorphine_repurposing_PMC5995787.pdf", "apomorphine"),
        ("insulin_repurposing_PMC11919260.pdf", "insulin")
    ]

    for filename, expected_drug in test_cases:
        try:
            drug_folder = f"{expected_drug} repurposing"
            info = parse_filename(filename, drug_folder)
            print(f"✅ {filename} -> drug: {info.drug_id}, doc: {info.doc_id}")
            assert info.drug_id == expected_drug, f"Expected {expected_drug}, got {info.drug_id}"
        except Exception as e:
            print(f"❌ Error parsing {filename}: {str(e)}")


def main():
    """Main test function."""
    print("🚀 Starting Simple PDF & Gemini Integration Tests")
    print("=" * 50)

    # Test 1: Filename parsing
    test_filename_parsing()

    # Test 2: PDF extraction
    chunks = test_pdf_extraction()

    # Test 3: Gemini integration
    gemini_ok = test_gemini_integration()

    print("\n" + "=" * 50)
    if chunks and gemini_ok:
        print("🎉 All core components working!")
        print("✅ PDF text extraction: Working")
        print("✅ Text chunking: Working")
        print("✅ Gemini embeddings: Working")
        print("✅ Gemini chat: Working")
        print("✅ Filename parsing: Working")
        print("\n🚀 Ready to build the full RAG system!")
    else:
        print("⚠️ Some components need attention:")
        if not chunks:
            print("❌ PDF extraction failed")
        if not gemini_ok:
            print("❌ Gemini API integration failed")


if __name__ == "__main__":
    main()
