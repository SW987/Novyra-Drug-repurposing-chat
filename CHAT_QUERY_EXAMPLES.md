# 💬 Chat Query Examples & Expected Responses

This document shows example queries for each drug and what responses you should expect from the system.

## 🩸 Aspirin Repurposing Queries

### Query 1: Cancer Prevention Benefits
**Question:** "What are the main benefits of aspirin for cancer prevention?"

**Expected Response:**
- Should mention colorectal cancer risk reduction (20-30%)
- COX-2 inhibition mechanism
- Anti-inflammatory effects
- References to clinical trials and meta-analyses
- Sources: PMC11242460, PMC11866938, PMC7543116

### Query 2: Clinical Trials
**Question:** "What clinical trials support aspirin use in colorectal cancer?"

**Expected Response:**
- ARRIVE trial results
- ASPREE trial findings
- Meta-analyses showing 23% reduction in incidence
- 31% reduction in mortality
- Sources from aspirin research papers

### Query 3: Cardiovascular Effects
**Question:** "What are the cardiovascular benefits and risks of aspirin?"

**Expected Response:**
- Secondary prevention: 25-30% MI reduction
- Stroke prevention: 20-25% risk reduction
- GI bleeding risks (RR 1.5)
- Benefit-risk analysis for different populations
- Sources from cardiovascular safety papers

## 🧠 Apomorphine Repurposing Queries

### Query 1: General Repurposing
**Question:** "How is apomorphine being repurposed for medical use?"

**Expected Response:**
- Parkinson's disease treatment
- Erectile dysfunction (original use)
- Potential applications in other neurological conditions
- Dopamine agonist mechanisms
- Sources: PMC5995787, PMC7001430, PMC8663985

### Query 2: Neurological Applications
**Question:** "What are the neurological applications of apomorphine?"

**Expected Response:**
- Parkinson's disease symptom management
- Off-period treatment
- Subcutaneous administration benefits
- Comparison with oral medications
- Clinical evidence and patient outcomes

### Query 3: Parkinson's Research
**Question:** "What clinical studies exist for apomorphine in Parkinson's disease?"

**Expected Response:**
- Phase III trial results
- Comparative studies with other treatments
- Long-term safety data
- Patient quality of life improvements
- Specific PMC references

## 💉 Insulin Repurposing Queries

### Query 1: Cancer Treatment
**Question:** "What are the cancer treatment applications of insulin?"

**Expected Response:**
- Anti-cancer mechanisms beyond glucose control
- Effects on cancer cell proliferation
- Combination with chemotherapy
- Clinical trial results for various cancers
- Sources: PMC11919260, PMC11994265, PMC12124044

### Query 2: Beyond Diabetes
**Question:** "How is insulin being repurposed beyond diabetes?"

**Expected Response:**
- Oncology applications
- Anti-inflammatory effects
- Potential neuroprotective properties
- Research in autoimmune diseases
- Emerging therapeutic uses

### Query 3: Oncology Evidence
**Question:** "What clinical evidence supports insulin use in oncology?"

**Expected Response:**
- Preclinical studies results
- Early-phase clinical trials
- Mechanisms of anti-tumor activity
- Safety profiles in cancer patients
- Current research status

## 🔍 Advanced Query Types

### Specific Document Query
**Question:** "What specific findings are in the PMC11242460 paper?"
**Drug:** aspirin
**Doc ID:** PMC11242460

**Expected Response:**
- Direct excerpts from that specific paper
- Key findings and conclusions
- Methodology details
- Focused on PMC11242460 content only

### Comparative Query
**Question:** "Compare aspirin with other drugs for cancer prevention"

**Expected Response:**
- Aspirin vs NSAIDs
- Aspirin vs statins
- Aspirin vs metformin
- Relative benefits and risks
- Evidence-based comparisons

### Mechanism Query
**Question:** "How does aspirin work as an anti-inflammatory agent?"

**Expected Response:**
- COX-1 and COX-2 inhibition
- Prostaglandin synthesis blockade
- Platelet aggregation effects
- Anti-inflammatory pathways
- Biochemical mechanisms

## 📊 Response Quality Indicators

### ✅ Good Response Characteristics:
- **Evidence-based**: Cites specific studies, trials, percentages
- **Source attribution**: Lists PMC IDs and paper titles
- **Balanced view**: Includes benefits AND risks/side effects
- **Contextual**: Explains mechanisms and clinical relevance
- **Concise yet comprehensive**: Direct answers without unnecessary text

### ⚠️ Response Issues to Watch For:
- **Hallucinations**: AI making up facts not in papers
- **Missing sources**: No references to specific documents
- **Generic answers**: Not drug-specific or too vague
- **Outdated info**: References to very old studies only

## 🧪 Testing Commands

```bash
# Start server
python app/main.py

# Test individual query
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_1",
    "drug_id": "aspirin",
    "message": "What are aspirin benefits for cancer?"
  }'

# Run comprehensive tests
python test_chat_queries.py
```

## 📈 Expected Performance

- **Response time**: 3-8 seconds per query
- **Source coverage**: 2-5 relevant sources per query
- **Accuracy**: 85-95% factual accuracy based on source material
- **Relevance**: Answers directly address the question asked

## 🔧 Troubleshooting Query Issues

### If responses are poor:
1. **Check data ingestion**: Ensure PDFs were processed correctly
2. **Verify API key**: Confirm Gemini API is working
3. **Test specific docs**: Query individual papers to isolate issues
4. **Check chunking**: Ensure text was properly split and embedded

### Common issues:
- **"No relevant information found"**: Drug name misspelled or not ingested
- **Generic responses**: Query too broad or not drug-specific
- **Wrong drug answers**: Check if query matches ingested drug data

---

**Use these examples to validate your chat system is working correctly! 🧬💬**
