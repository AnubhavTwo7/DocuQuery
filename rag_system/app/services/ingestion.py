import os
import uuid
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.domain_models import DocumentChunk, DocumentMetadata, IngestedDocument
from app.core.config import settings

def ingest_pdf(file_path: str, original_filename: str) -> IngestedDocument:
    """
    Ingests a PDF document, extracts text, and applies smart chunking.
    Returns an IngestedDocument containing the chunks with metadata.
    """
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. Smart Chunking (Recursive Character Splitting)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    doc_id = str(uuid.uuid4())
    chunks: List[DocumentChunk] = []

    for page_num, page in enumerate(pages, start=1):
        # We process page by page to keep page numbers in metadata accurate
        page_chunks = text_splitter.split_text(page.page_content)
        
        for i, chunk_text in enumerate(page_chunks):
            chunk_metadata = DocumentMetadata(
                source=original_filename,
                page=page_num
            )
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}-p{page_num}-c{i}",
                text=chunk_text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)

    return IngestedDocument(
        document_id=doc_id,
        filename=original_filename,
        chunks=chunks
    )
