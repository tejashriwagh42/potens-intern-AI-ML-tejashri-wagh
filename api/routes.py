import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from pydantic import BaseModel

from ingestion.loader import extract_text_from_file
from ingestion.chunker import create_chunks

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic schemas for request validation
class AskRequest(BaseModel):
    question: str

class AskResponseCitation(BaseModel):
    source_file: str
    page: int
    chunk_id: str
    snippet: str

class AskResponse(BaseModel):
    answer: str
    confidence_score: float
    warning: Optional[str] = None
    citations: List[AskResponseCitation]

class ContradictRequest(BaseModel):
    doc1: str
    doc2: str
    topic: str

class ContradictResponse(BaseModel):
    conflict: Any  # True, False, or "unknown"
    reasoning: str

class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    file_type: str
    chunks_count: int

class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    llm_provider: str


@router.post("/upload", response_model=Dict[str, Any])
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Ingests a document: stores it on disk, extracts pages, chunks text,
    generates embeddings, and stores them in ChromaDB.
    """
    store = request.app.state.store
    docs_dir = request.app.state.docs_dir
    
    # Ensure documents directory exists
    os.makedirs(docs_dir, exist_ok=True)
    
    # Save the file locally
    temp_path = os.path.join(docs_dir, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    try:
        # Check if already ingested in ChromaDB to avoid duplication
        if store.check_document_exists(file.filename):
            # Clean up saved file if we decide not to duplicate
            # To be safe, we keep the file on disk but skip indexing
            doc_info = [d for d in store.list_documents() if d["file_name"] == file.filename]
            doc_id = doc_info[0]["document_id"] if doc_info else "existing_doc"
            
            return {
                "status": "success",
                "document_id": doc_id,
                "message": f"File '{file.filename}' already ingested. Skipping indexing to avoid duplicates."
            }
            
        # Extract pages
        pages = extract_text_from_file(temp_path)
        if not pages:
            raise ValueError("No text could be extracted from this document.")
            
        # Chunk pages
        chunks = create_chunks(pages)
        if not chunks:
            raise ValueError("No text chunks could be created.")
            
        # Index in ChromaDB
        doc_id = store.add_documents(chunks)
        
        return {
            "status": "success",
            "document_id": doc_id,
            "chunks_count": len(chunks),
            "message": f"File '{file.filename}' ingested successfully."
        }
    except Exception as e:
        # Clean up local file on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Error during ingestion of {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(e)}")


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: Request, body: AskRequest):
    """
    Accepts a user question, performs search/reranking, and generates an LLM answer with citations.
    Supports English, Hindi, Marathi, Spanish, French queries.
    """
    pipeline = request.app.state.pipeline
    try:
        result = pipeline.ask(body.question)
        return result
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query model: {str(e)}")


@router.post("/contradict", response_model=ContradictResponse)
async def detect_contradiction(request: Request, body: ContradictRequest):
    """
    Compares two documents on a specific topic and returns whether they contradict.
    """
    detector = request.app.state.detector
    try:
        result = detector.detect_contradiction(
            doc1_id=body.doc1,
            doc2_id=body.doc2,
            topic=body.topic
        )
        return result
    except Exception as e:
        logger.error(f"Error in /contradict endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Contradiction detection failed: {str(e)}")


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(request: Request):
    """
    Lists all documents currently indexed in ChromaDB.
    """
    store = request.app.state.store
    try:
        docs = store.list_documents()
        return docs
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def check_health(request: Request):
    """
    Simple health check verifying DB connection and LLM configurations.
    """
    store = request.app.state.store
    pipeline = request.app.state.pipeline
    
    try:
        # Check DB by querying collections list
        collections = store.client.list_collections()
        db_connected = True
    except Exception:
        db_connected = False
        
    return {
        "status": "healthy",
        "db_connected": db_connected,
        "llm_provider": pipeline.llm_provider
    }
