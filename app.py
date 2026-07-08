import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

# Load env variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_project")

# Import custom components
from vectorstore.chroma_store import ChromaStore
from retrieval.reranker import Reranker
from retrieval.retriever import RetrievalPipeline
from contradiction.detector import ContradictionDetector
from api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events to manage component initialization and cleanups.
    """
    logger.info("Initializing RAG Application components...")
    
    # Retrieve configuration variables
    db_path = os.environ.get("CHROMA_DB_PATH", "chroma_db")
    docs_dir = os.environ.get("DOCUMENTS_DIR", "documents")
    reranking_enabled = os.environ.get("RERANKING_ENABLED", "true").lower() == "true"
    
    # Ensure standard directories exist
    os.makedirs(docs_dir, exist_ok=True)
    
    # Initialize Persistent Chroma Store
    store = ChromaStore(db_path=db_path)
    
    # Initialize Reranker if enabled
    reranker = None
    if reranking_enabled:
        try:
            reranker = Reranker()
        except Exception as e:
            logger.error(f"Failed to initialize Cross-Encoder Reranker: {e}. Reranking will be disabled.")
            
    # Initialize Retrieval pipeline & contradiction detector
    pipeline = RetrievalPipeline(chroma_store=store, reranker=reranker)
    detector = ContradictionDetector(chroma_store=store)
    
    # Save objects to app state for access in routes
    app.state.store = store
    app.state.reranker = reranker
    app.state.pipeline = pipeline
    app.state.detector = detector
    app.state.docs_dir = docs_dir
    
    logger.info("RAG Application components initialized successfully.")
    yield
    logger.info("Shutting down RAG Application...")

# Initialize FastAPI App
app = FastAPI(
    title="Multilingual Document Q&A with Citations and Contradiction Detection",
    description="FastAPI Backend for high-precision vector retrieval, reranking, and citation generation.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multilingual Document Q&A API. Visit /docs for documentation."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
