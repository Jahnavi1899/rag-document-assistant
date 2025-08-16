from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import time
import uuid
import tempfile
import os
from datetime import datetime
from typing import Dict

# Import our services
from models.schemas import *
from services.document_processor import DocumentProcessor
from services.embeddings import EmbeddingService
from services.llm_service import LLMServiceFactory
from storage.vector_store import VectorStore
from services.summarization_service import SummarizationService
from config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Assistant",
    description="A simple RAG-based document assistant for AI and backend learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()
llm_service = LLMServiceFactory.create_llm_service()
summarization_service = SummarizationService()

# In-memory storage for document status (in production, use a database)
document_registry: Dict[str, DocumentStatus] = {}

def process_document_async(document_id: str, file_path: str, filename: str):
    """Background task to process uploaded document with summarization."""
    try:
        logger.info(f"Starting processing for document {document_id}")
        
        # Update status to processing
        document_registry[document_id].status = "processing"
        
        # Extract text
        text = document_processor.extract_text(file_path, filename)
        logger.info(f"Extracted {len(text)} characters from {filename}")
        
        # Create metadata
        metadata = {
            "document_id": document_id,
            "filename": filename,
            "upload_time": datetime.now().isoformat()
        }
        
        # Update status to summarizing
        document_registry[document_id].status = "summarizing"
        
        # Generate document summary
        logger.info(f"Generating summary for {filename}")
        summary_data = summarization_service.generate_document_summary(text, filename)
        summary = DocumentSummary(**summary_data)
        
        # Chunk text for RAG
        logger.info(f"Chunking document {filename}")
        chunks = document_processor.chunk_text(text, metadata)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)
        
        # Store in vector database
        logger.info(f"Storing embeddings in vector database")
        vector_store.create_collection(document_id)
        vector_store.add_documents(document_id, chunks, embeddings)
        
        # Update status to ready with summary
        document_registry[document_id].status = "ready"
        document_registry[document_id].chunk_count = len(chunks)
        document_registry[document_id].summary = summary
        
        logger.info(f"Successfully processed document {document_id} with {len(chunks)} chunks and summary")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        document_registry[document_id].status = "error"
        document_registry[document_id].error_message = str(e)
    
    finally:
        # Clean up temp file
        try:
            os.unlink(file_path)
        except:
            pass

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a document."""
    start_time = time.time()
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Check file extension
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    # Register document
    document_registry[document_id] = DocumentStatus(
        document_id=document_id,
        filename=file.filename,
        status="processing",
        created_at=datetime.now()
    )
    
    # Start background processing
    background_tasks.add_task(process_document_async, document_id, temp_file_path, file.filename)
    
    processing_time = time.time() - start_time
    
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="processing",
        message="Document uploaded successfully. Processing started.",
        processing_time=processing_time
    )

@app.get("/status/{document_id}", response_model=DocumentStatus)
async def get_document_status(document_id: str):
    """Get the processing status of a document."""
    if document_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document_registry[document_id]

@app.post("/query", response_model=QueryResponse)
async def query_document(query: QueryRequest):
    """Query a document using RAG."""
    start_time = time.time()
    
    # Check if document exists and is ready
    if query.document_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_status = document_registry[query.document_id]
    if doc_status.status != "ready":
        raise HTTPException(
            status_code=400, 
            detail=f"Document not ready. Current status: {doc_status.status}"
        )
    
    try:
        # Generate query embedding
        query_embedding = embedding_service.embed_text(query.question)
        
        # Search for similar chunks
        results = vector_store.similarity_search(
            query.document_id, 
            query_embedding, 
            k=query.max_results
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant content found")
        
        # Prepare context from search results
        context_parts = []
        sources = []
        
        for text, metadata, distance in results:
            context_parts.append(text)
            sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "chunk_index": metadata.get("chunk_index", 0),
                "similarity_score": 1 - distance  # Convert distance to similarity
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate response using LLM
        answer = llm_service.generate_response(query.question, context)
        
        response_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            response_time=response_time,
            model_used=settings.DEFAULT_MODEL
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "RAG Document Assistant API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    return {
        "documents": list(document_registry.values())
    }

@app.get("/documents/{document_id}", response_model=DocumentDetails)
async def get_document_details(document_id: str):
    """Get detailed information about a document including summary."""
    if document_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_status = document_registry[document_id]
    
    return DocumentDetails(
        document_id=doc_status.document_id,
        filename=doc_status.filename,
        status=doc_status.status,
        chunk_count=doc_status.chunk_count,
        created_at=doc_status.created_at,
        summary=doc_status.summary
    )

@app.get("/documents/{document_id}/summary", response_model=DocumentSummary)
async def get_document_summary(document_id: str):
    """Get just the summary of a document."""
    if document_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_status = document_registry[document_id]
    
    if doc_status.status != "ready" or not doc_status.summary:
        raise HTTPException(
            status_code=400, 
            detail="Document summary not available"
        )
    
    return doc_status.summary

@app.post("/query", response_model=QueryResponse)
async def query_document(query: QueryRequest):
    """Query a document using RAG, optionally including summary."""
    start_time = time.time()
    
    # Check if document exists and is ready
    if query.document_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_status = document_registry[query.document_id]
    if doc_status.status != "ready":
        raise HTTPException(
            status_code=400, 
            detail=f"Document not ready. Current status: {doc_status.status}"
        )
    
    try:
        # Generate query embedding
        query_embedding = embedding_service.embed_text(query.question)
        
        # Search for similar chunks
        results = vector_store.similarity_search(
            query.document_id, 
            query_embedding, 
            k=query.max_results
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant content found")
        
        # Prepare context from search results
        context_parts = []
        sources = []
        
        for text, metadata, distance in results:
            context_parts.append(text)
            sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "chunk_index": metadata.get("chunk_index", 0),
                "similarity_score": 1 - distance
            })
        
        context = "\n\n".join(context_parts)
        
        # Add document summary to context if requested
        if query.include_summary and doc_status.summary:
            context = f"Document Summary: {doc_status.summary.short_summary}\n\nRelevant Content:\n{context}"
        
        # Generate response using LLM
        answer = llm_service.generate_response(query.question, context)
        
        response_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            response_time=response_time,
            model_used=settings.DEFAULT_MODEL,
            document_summary=doc_status.summary if query.include_summary else None
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )