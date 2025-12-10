#!/usr/bin/env python3
"""
Live chat testing to demonstrate actual query responses
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_query(drug_id, message, description, doc_id=None):
    """Test a query and show the response."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"💊 Drug: {drug_id}")
    print(f"❓ Query: {message}")
    print(f"{'='*60}")

    payload = {
        "session_id": f"live_test_{drug_id}_{int(time.time())}",
        "drug_id": drug_id,
        "message": message
    }

    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print("✅ RESPONSE:")
            print(data['answer'])
            print("\n📚 SOURCES:")
            for i, source in enumerate(data['sources'], 1):
                print(f"  {i}. {source['doc_title']} (distance: {source['distance']:.3f})")
                print(f"     Preview: {source['text_preview'][:100]}...")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is server running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_live_tests():
    """Run comprehensive live tests."""

    test_cases = [
        # Aspirin - unified from all documents
        ("aspirin", "What are the main benefits of aspirin for cancer prevention?",
         "ASPIRIN: Cancer prevention benefits (unified from all docs)"),

        ("aspirin", "What are the cardiovascular benefits and risks of aspirin?",
         "ASPIRIN: Cardio benefits & risks (unified from all docs)"),

        ("aspirin", "What clinical trials support aspirin use in cancer?",
         "ASPIRIN: Clinical trials evidence (unified from all docs)"),

        # Apomorphine - unified from all documents
        ("apomorphine", "How is apomorphine being repurposed for medical use?",
         "APOMORPHINE: Repurposing applications (unified from all docs)"),

        ("apomorphine", "What are the neurological applications of apomorphine?",
         "APOMORPHINE: Neurological uses (unified from all docs)"),

        # Insulin - unified from all documents
        ("insulin", "What are the cancer treatment applications of insulin?",
         "INSULIN: Cancer treatment applications (unified from all docs)"),

        ("insulin", "How is insulin being repurposed beyond diabetes?",
         "INSULIN: Beyond diabetes repurposing (unified from all docs)"),

        # Specific document queries
        ("aspirin", "What specific findings are in the PMC11242460 paper?",
         "ASPIRIN: Specific paper analysis", "PMC11242460"),

        # Cross-drug comparison
        ("aspirin", "Compare aspirin with other drugs for cancer prevention",
         "ASPIRIN: Comparative analysis (unified from all docs)"),
    ]

    print("🚀 LIVE CHAT SYSTEM TESTING")
    print("Testing actual responses from ingested documents...")
    print("Make sure server is running: python app/main.py")

    successful = 0
    total = len(test_cases)

    for drug_id, message, description, *doc_id in test_cases:
        if test_query(drug_id, message, description, doc_id[0] if doc_id else None):
            successful += 1
        time.sleep(2)  # Rate limiting

    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS: {successful}/{total} queries successful")
    print(f"{'='*60}")

    if successful == total:
        print("🎉 ALL TESTS PASSED - System working perfectly!")
    else:
        print(f"⚠️ {total - successful} queries failed - check server/logs")

if __name__ == "__main__":
    run_live_tests()
