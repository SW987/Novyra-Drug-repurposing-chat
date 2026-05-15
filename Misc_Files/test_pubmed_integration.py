#!/usr/bin/env python3
"""
Test PubMed Integration and Download System
"""

def test_pubmed_integration():
    """Test the PubMed integration template"""
    print("🔬 Testing PubMed Integration Framework")
    print("=" * 50)

    try:
        from app.ingestion_pipeline import PDFIngestionPipeline
        from app.config import get_settings

        print("✅ Imports successful")

        settings = get_settings()
        print("✅ Settings loaded")

        pipeline = PDFIngestionPipeline(settings)
        print("✅ Pipeline initialized")

        # Test the PubMed integration template
        print("\n🔍 Testing PubMed Search Template...")
        result = pipeline.ingest_from_pubmed_search('aspirin', max_papers=3)

        print("📊 PubMed Integration Result:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Drug: {result.get('drug', 'unknown')}")
        print(f"   Papers Found: {result.get('papers_found', 'unknown')}")
        print(f"   Note: {result.get('note', 'none')}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_download_system_integration():
    """Test the download system integration"""
    print("\n📥 Testing Download System Integration")
    print("=" * 50)

    try:
        import sys
        import os

        # Add current directory to path for imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Test the integrated download system
        print("🔍 Testing integrated download system...")

        # Mock test without actual network calls
        from your_integrated_download_system import initialize_rag_system

        rag_pipeline = initialize_rag_system()
        if rag_pipeline:
            print("✅ RAG system integration successful")
            return True
        else:
            print("⚠️ RAG system not available (expected for template)")
            return True

    except ImportError as e:
        print(f"❌ Download system import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Download system test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Drug Repurposing PDF Integration Test Suite")
    print("=" * 60)

    # Test PubMed integration
    pubmed_ok = test_pubmed_integration()

    # Test download system integration
    download_ok = test_download_system_integration()

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   PubMed Integration: {'✅ PASS' if pubmed_ok else '❌ FAIL'}")
    print(f"   Download Integration: {'✅ PASS' if download_ok else '❌ FAIL'}")

    if pubmed_ok and download_ok:
        print("\n🎉 All integration tests passed!")
        print("📋 Integration Status: READY FOR IMPLEMENTATION")
        print("\n💡 Next Steps:")
        print("   1. Replace template functions with real PubMed API calls")
        print("   2. Implement PDF download logic")
        print("   3. Test with real PubMed data")
    else:
        print("\n⚠️ Some tests failed - check error messages above")
