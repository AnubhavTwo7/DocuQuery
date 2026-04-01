from fastapi import APIRouter, File, UploadFile, HTTPException
import shutil
import os
from typing import Dict, Any

from app.services.ingestion import ingest_pdf
from app.services.retrieval import get_hybrid_retriever
from app.core.config import settings

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    file_path = os.path.join(settings.DATA_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Ingest PDF and Chunk
        ingested_doc = ingest_pdf(file_path, file.filename)
        
        # 2. Add to Hybrid Retriever
        retriever = get_hybrid_retriever()
        retriever.add_chunks(ingested_doc.chunks)
        
        return {
            "message": "Document ingested successfully",
            "document_id": ingested_doc.document_id,
            "filename": ingested_doc.filename,
            "chunks_created": len(ingested_doc.chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def list_documents():
    try:
        retriever = get_hybrid_retriever()
        return {"documents": retriever.get_all_documents()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    try:
        retriever = get_hybrid_retriever()
        success = retriever.delete_document(filename)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": f"Successfully deleted {filename}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
