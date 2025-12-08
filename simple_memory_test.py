#!/usr/bin/env python3
"""
Simple in-memory test without ChromaDB to verify query responses
"""

import google.generativeai as genai
from app.utils import extract_text_from_pdf
from pathlib import Path
import json

# Configure Gemini
genai.configure(api_key="AIzaSyBDd4K-NL86geJG5Mz9r11YF4bOBRf6jb4")

def load_pdf_content(drug_name):
    """Load PDF content for a drug."""
    base_path = Path(r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs")

    drug_dir = base_path / f"{drug_name} repurposing"
    if not drug_dir.exists():
        return []

    pdf_files = list(drug_dir.glob("*.pdf"))
    content = []

    for pdf_path in pdf_files:
        try:
            text = extract_text_from_pdf(str(pdf_path))
            content.append({
                'filename': pdf_path.name,
                'content': text[:5000],  # First 5000 chars for testing
                'path': str(pdf_path)
            })
            print(f"✅ Loaded: {pdf_path.name} ({len(text)} chars)")
        except Exception as e:
            print(f"❌ Failed to load {pdf_path.name}: {e}")

    return content

def simple_rag_query(drug_name, query, pdf_contents):
    """Simple RAG query using Gemini directly."""

    if not pdf_contents:
        return "No documents found for this drug."

    # Combine all content
    combined_content = "\n\n".join([doc['content'] for doc in pdf_contents])

    # Truncate if too long (Gemini has limits)
    if len(combined_content) > 15000:
        combined_content = combined_content[:15000] + "..."

    prompt = f"""
You are a helpful assistant specializing in drug repurposing research.

Use the following research paper content to answer the user's question about {drug_name}:

CONTENT:
{combined_content}

QUESTION: {query}

Answer based on the provided content. If the content doesn't contain enough information, say so.
"""

    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        return f"Error generating response: {e}"

def test_drug_queries():
    """Test queries for each drug."""

    drugs_to_test = ['aspirin', 'apomorphine', 'insulin']

    test_queries = {
        'aspirin': [
            "What are the main benefits of aspirin for cancer prevention?",
            "What clinical trials support aspirin use in colorectal cancer?",
            "What are the cardiovascular benefits and risks of aspirin?"
        ],
        'apomorphine': [
            "How is apomorphine being repurposed for medical use?",
            "What are the neurological applications of apomorphine?",
            "What clinical studies exist for apomorphine in Parkinson's disease?"
        ],
        'insulin': [
            "What are the cancer treatment applications of insulin?",
            "How is insulin being repurposed beyond diabetes?",
            "What clinical evidence supports insulin use in oncology?"
        ]
    }

    print("🧪 TESTING DRUG QUERY RESPONSES")
    print("=" * 60)

    for drug in drugs_to_test:
        print(f"\n🏥 TESTING {drug.upper()}")
        print("-" * 40)

        # Load PDF content
        pdf_contents = load_pdf_content(drug)

        if not pdf_contents:
            print(f"❌ No PDFs found for {drug}")
            continue

        print(f"📄 Loaded {len(pdf_contents)} documents")

        # Test queries
        for i, query in enumerate(test_queries[drug], 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 30)

            response = simple_rag_query(drug, query, pdf_contents)

            print("RESPONSE:")
            print(response)
            print()

            # Don't overwhelm the API
            import time
            time.sleep(2)

if __name__ == "__main__":
    test_drug_queries()
