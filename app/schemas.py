"""
Pydantic request/response schemas for the Drug Repurposing Chat API.

All API input and output shapes are defined here so that FastAPI can
auto-generate OpenAPI documentation and enforce runtime validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


class Message(BaseModel):
    """Individual message in conversation history."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    drug_id: Union[str, List[str]] = Field(..., description="Drug identifier(s) to filter documents")
    message: str = Field(..., description="User's chat message")
    doc_id: Optional[str] = Field(None, description="Optional specific document ID to query")
    conversation_history: Optional[List[Message]] = Field(default_factory=list, description="Previous conversation messages for context")


class ChatByDrugNameRequest(BaseModel):
    """Request model for chat-by-drug-name endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    drug_name: str = Field(..., description="Drug name (human readable)")
    message: str = Field(..., description="User's chat message")
    doc_id: Optional[str] = Field(None, description="Optional specific document ID to query")
    conversation_history: Optional[List[Message]] = Field(default_factory=list, description="Previous conversation messages for context")


class Source(BaseModel):
    """Source information for a retrieved chunk."""
    doc_id: str = Field(..., description="Document identifier")
    doc_title: str = Field(..., description="Human-readable document title")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    distance: float = Field(..., description="Similarity distance score")
    text_preview: str = Field(..., description="Preview of the chunk text (first 200 chars)")
    file_path: Optional[str] = Field(None, description="Path to the original PDF if available")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="AI-generated answer")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")
    session_id: str = Field(..., description="Echo of the session identifier")


class IngestRequest(BaseModel):
    """Request model for document ingestion endpoint."""
    drug_id: str = Field(..., description="Drug identifier")
    doc_id: str = Field(..., description="Document identifier")
    doc_title: str = Field(..., description="Human-readable document title")
    content: str = Field(..., description="Full document content")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    message: str = Field(..., description="Success message")
    chunks_created: int = Field(..., description="Number of chunks created")
    drug_id: str
    doc_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field("1.0.0", description="API version")


class IngestStatusResponse(BaseModel):
    """Response for ingestion status."""
    total_files: int = Field(..., description="Total PDF files found")
    processed_files: int = Field(..., description="Successfully processed files")
    failed_files: int = Field(..., description="Failed to process files")
    total_chunks: int = Field(..., description="Total chunks created")
    drug_folders: List[str] = Field(..., description="Drug folders found")
    results: List[Dict[str, Any]] = Field(..., description="Detailed results per file")
