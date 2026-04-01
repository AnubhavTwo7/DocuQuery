import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production-Level RAG API",
    description="A Retrieval-Augmented Generation API with FastAPI, FAISS, BM25, and Gemini.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes import document, query, evaluate
from app.services.retrieval import get_hybrid_retriever
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app.include_router(document.router, prefix="/api", tags=["Documents"])
app.include_router(query.router, prefix="/api", tags=["Query and Generation"])
app.include_router(evaluate.router, prefix="/api", tags=["Evaluation"])

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "UI not found, please create static/index.html"}

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing RAG System...")
    # This will trigger the _load_indices() inside the HybridRetriever singleton
    get_hybrid_retriever()
    logger.info("RAG System initialization complete.")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
