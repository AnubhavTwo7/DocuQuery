from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class SourceAttribution(BaseModel):
    source: str
    page: int
    text_snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceAttribution]

class EvaluationRequest(BaseModel):
    query: str
    context: str
    answer: str

class EvaluationResponse(BaseModel):
    evaluation_score: float  # 0.0 to 1.0
    reasoning: str
