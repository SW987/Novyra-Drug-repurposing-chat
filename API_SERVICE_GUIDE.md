# FastAPI Service Access + Payload Guide

Small reference for backend integration and testing.

## Base URL

- Local (SSH tunnel): `http://localhost:8123/drug_discovery`
- Public (if SG allows): `http://<EC2_PUBLIC_IP>:8123/drug_discovery`

## Common headers

```
Content-Type: application/json
```

## Endpoints

### 1) Health check

**GET** `/health`

Response:
```json
{"status":"healthy","version":"1.0.0"}
```

### 2) List drugs

**GET** `/drugs`

Response:
```json
{"drugs":["aspirin","apomorphine","insulin"]}
```

### 3) Chat by drug name (recommended for external clients)

**POST** `/chat-by-drug-name`

Request:
```json
{
  "session_id": "client-123",
  "drug_name": "aspirin",
  "message": "What is the name of drug being discussed here?",
  "doc_id": null,
  "conversation_history": [
    {"role":"user","content":"Hi"},
    {"role":"assistant","content":"Hello!"}
  ]
}
```

Notes:
- `doc_id` is optional.
- `conversation_history` is optional (array of `{role, content}`).

Response:
```json
{
  "answer": "Aspirin is the drug being discussed...",
  "sources": [
    {
      "doc_id": "aspirin_repurposing_PMC12345",
      "doc_title": "Aspirin Repurposing PMC12345",
      "chunk_id": "aspirin_repurposing_PMC12345::chunk_3",
      "distance": 0.21,
      "text_preview": "Aspirin has been studied for...",
      "file_path": null
    }
  ],
  "session_id": "client-123"
}
```

### 4) Chat by drug_id (internal use when you already know IDs)

**POST** `/chat`

Request:
```json
{
  "session_id": "client-123",
  "drug_id": "aspirin",
  "message": "Summarize the main findings.",
  "doc_id": null,
  "conversation_history": []
}
```

Notes:
- `drug_id` can be a string **or** list of strings.

### 5) Ingest a single document (programmatic)

**POST** `/ingest`

Request:
```json
{
  "drug_id": "aspirin",
  "doc_id": "aspirin_repurposing_PMC12345",
  "doc_title": "Aspirin Repurposing PMC12345",
  "content": "Full document text here..."
}
```

Response:
```json
{"message":"Document ingested successfully","chunks_created":42,"drug_id":"aspirin","doc_id":"aspirin_repurposing_PMC12345"}
```

### 6) Ingest all PDFs from local DOCS_DIR

**POST** `/ingest-pdfs`

Response:
```json
{
  "total_files": 10,
  "processed_files": 8,
  "failed_files": 2,
  "total_chunks": 320,
  "drug_folders": ["aspirin","apomorphine"],
  "results": []
}
```

## Example curl

```bash
curl -s http://localhost:8123/drug_discovery/chat-by-drug-name \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-1","drug_name":"aspirin","message":"What is the name of drug being discussed here?"}'
```

## Access notes (EC2)

- If you use SSH tunneling:
  ```bash
  ssh -i /path/to/key.pem -L 8123:localhost:8123 ubuntu@<EC2_PUBLIC_IP>
  ```
- If you want public access, add an inbound rule for TCP 8123 in the EC2 security group.
