#!/usr/bin/env python3
"""
Test script for the Drug Repurposing Chat System
Tests PDF ingestion and chat functionality with Gemini API
"""

import sys
import os
sys.path.append('.')

from app.config import get_settings
from app.vector_store import init_vector_store
from app.ingestion import ingest_pdfs_from_directory
from app.rag import chat_with_documents


def test_pdf_ingestion():
    """Test PDF ingestion from the user's directory."""
    print("🔄 Testing PDF ingestion...")

    settings = get_settings()
    collection = init_vector_store(settings)

    # Ingest PDFs from the user's directory
    result = ingest_pdfs_from_directory(settings.docs_dir, settings, collection)

    print("✅ Ingestion completed!"    print(f"📊 Total files found: {result['total_files']}")
    print(f"📊 Successfully processed: {result['processed_files']}")
    print(f"📊 Failed: {result['failed_files']}")
    print(f"📊 Total chunks created: {result['total_chunks']}")

    for drug_folder in result['drug_folders']:
        print(f"📁 Found drug folder: {drug_folder}")

    return result


def test_chat_functionality():
    """Test chat functionality with sample queries."""
    print("\n🔄 Testing chat functionality...")

    settings = get_settings()
    collection = init_vector_store(settings)

    test_cases = [
        {
            "drug_id": "aspirin",
            "message": "What are the benefits of aspirin for cancer prevention?",
            "description": "Aspirin cancer prevention query"
        },
        {
            "drug_id": "apomorphine",
            "message": "How is apomorphine being repurposed?",
            "description": "Apomorphine repurposing query"
        },
        {
            "drug_id": "insulin",
            "message": "What are the clinical applications of insulin repurposing?",
            "description": "Insulin repurposing query"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")

        try:
            result = chat_with_documents(
                session_id=f"test_session_{i}",
                drug_id=test_case["drug_id"],
                message=test_case["message"],
                collection=collection,
                settings=settings
            )

            print("✅ Chat response generated"            print(f"📝 Answer preview: {result['answer'][:100]}...")
            print(f"📚 Sources found: {len(result['sources'])}")

            if result['sources']:
                print(f"📄 Top source: {result['sources'][0].doc_title}")

        except Exception as e:
            print(f"❌ Error in test {i}: {str(e)}")


def main():
    """Main test function."""
    print("🚀 Starting Drug Repurposing Chat System Tests")
    print("=" * 50)

    try:
        # Test 1: PDF Ingestion
        ingestion_result = test_pdf_ingestion()

        if ingestion_result['processed_files'] == 0:
            print("⚠️  No PDFs were processed. Please check your PDF directory.")
            return

        # Test 2: Chat Functionality
        test_chat_functionality()

        print("\n" + "=" * 50)
        print("🎉 All tests completed!")

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
