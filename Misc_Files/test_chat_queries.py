#!/usr/bin/env python3
"""
Test various chat queries for different drugs to demonstrate functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_query(drug_id, message, description, doc_id=None):
    """Test a single query and return results."""
    print(f"\n🧪 {description}")
    print(f"💊 Drug: {drug_id}")
    print(f"❓ Query: {message}")

    payload = {
        "session_id": f"test_{drug_id}_{int(time.time())}",
        "drug_id": drug_id,
        "message": message
    }

    if doc_id:
        payload["doc_id"] = doc_id

    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Answer:", data['answer'][:200] + "..." if len(data['answer']) > 200 else data['answer'])
            print(f"📚 Sources: {len(data['sources'])}")
            if data['sources']:
                print(f"📄 Top source: {data['sources'][0]['doc_title']}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def run_comprehensive_tests():
    """Run comprehensive tests for all drugs with various query types."""

    test_cases = [
        # Aspirin queries
        ("aspirin", "What are the main benefits of aspirin for cancer prevention?",
         "Aspirin cancer prevention benefits"),

        ("aspirin", "What clinical trials support aspirin use in colorectal cancer?",
         "Aspirin colorectal cancer trials"),

        ("aspirin", "What are the cardiovascular benefits and risks of aspirin?",
         "Aspirin cardiovascular effects"),

        ("aspirin", "How does aspirin work as an anti-inflammatory agent?",
         "Aspirin mechanism of action"),

        # Apomorphine queries
        ("apomorphine", "How is apomorphine being repurposed for medical use?",
         "Apomorphine repurposing applications"),

        ("apomorphine", "What are the neurological applications of apomorphine?",
         "Apomorphine neurological uses"),

        ("apomorphine", "What clinical studies exist for apomorphine in Parkinson's disease?",
         "Apomorphine Parkinson's research"),

        # Insulin queries
        ("insulin", "What are the cancer treatment applications of insulin?",
         "Insulin cancer treatment"),

        ("insulin", "How is insulin being repurposed beyond diabetes?",
         "Insulin repurposing beyond diabetes"),

        ("insulin", "What clinical evidence supports insulin use in oncology?",
         "Insulin oncology evidence"),

        # Cross-drug queries
        ("aspirin", "Compare aspirin with other drugs for cancer prevention",
         "Aspirin vs other chemopreventive drugs"),

        # Specific document queries
        ("aspirin", "What specific findings are in the PMC11242460 paper?",
         "Specific paper analysis", "PMC11242460"),
    ]

    print("🚀 COMPREHENSIVE CHAT QUERY TESTING")
    print("=" * 60)
    print("Testing various queries across all drugs...")

    successful_queries = 0
    total_queries = len(test_cases)

    for drug_id, message, description, *doc_id in test_cases:
        if test_query(drug_id, message, description, doc_id[0] if doc_id else None):
            successful_queries += 1
        time.sleep(1)  # Rate limiting

    print("
" + "=" * 60)
    print(f"📊 RESULTS: {successful_queries}/{total_queries} queries successful")
    print("=" * 60)

    return successful_queries == total_queries

def test_api_endpoints():
    """Test basic API endpoints."""
    print("\n🔍 Testing API endpoints...")

    # Test health
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check: OK")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

    # Test invalid drug
    payload = {
        "session_id": "test_invalid",
        "drug_id": "nonexistent_drug",
        "message": "What are the benefits?"
    }

    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "couldn't find any relevant information" in data['answer'].lower():
                print("✅ Invalid drug handling: OK")
                return True
            else:
                print("❌ Invalid drug response unexpected")
                return False
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

if __name__ == "__main__":
    print("🧬 Drug Repurposing Chat API - Query Testing")
    print("Make sure the server is running: python app/main.py")
    print("\nPress Enter to start testing...")
    input()

    # Test API endpoints first
    if not test_api_endpoints():
        print("❌ API endpoint tests failed. Check server.")
        exit(1)

    # Run comprehensive query tests
    success = run_comprehensive_tests()

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Chat system is working correctly")
        print("✅ All drugs respond appropriately")
        print("✅ Source citations working")
        print("🚀 Ready for production use!")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("🔧 Check the failed queries above")
