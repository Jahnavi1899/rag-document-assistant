from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentSummary(BaseModel):
    short_summary: str
    detailed_summary: str
    key_topics: List[str]
    summary_method: str
    chunks_processed: Optional[int] = None

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    processing_time: Optional[float] = None

class QueryRequest(BaseModel):
    document_id: str
    question: str
    max_results: int = Field(default=5, ge=1, le=20)
    include_summary: bool = Field(default=False, description="Include document summary in response")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    response_time: float
    llm_model_used: str
    document_summary: Optional[DocumentSummary] = None

class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    status: str  # processing, summarizing, ready, error
    chunk_count: Optional[int] = None
    created_at: datetime
    error_message: Optional[str] = None
    summary: Optional[DocumentSummary] = None

class DocumentDetails(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: Optional[int] = None
    created_at: datetime
    summary: Optional[DocumentSummary] = None