from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DocumentMetadata(BaseModel):
    source: str
    page: int
    additional_info: Dict[str, Any] = Field(default_factory=dict)

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: DocumentMetadata

class IngestedDocument(BaseModel):
    document_id: str
    filename: str
    chunks: List[DocumentChunk]
