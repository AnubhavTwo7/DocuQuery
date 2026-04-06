from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json

from app.models.api_models import QueryRequest, SourceAttribution
from app.services.retrieval import get_hybrid_retriever
from app.services.generation import generate_rag_response, rewrite_query
from app.core.config import settings

router = APIRouter()

@router.post("/query")
async def query_system(request: QueryRequest):
    retriever = get_hybrid_retriever()
    
    # 1. Rewrite Query (Optional/Automated)
    # This expands a short user query into a better search phrase
    rewritten_query = await rewrite_query(request.query)
    print(f"DEBUG: Original Query: {request.query}")
    print(f"DEBUG: Rewritten Query: {rewritten_query}")

    # 2. Retrieve Chunks
    top_k = request.top_k or settings.FINAL_TOP_K
    # Use the REWRITTEN query for search, but keep original for response generation
    chunks = retriever.search(rewritten_query, top_k=top_k)
    
    # 3. Prepare Source Attribution
    sources = [
        SourceAttribution(
            source=chunk.metadata.source,
            page=chunk.metadata.page,
            text_snippet=chunk.text
        ).model_dump() for chunk in chunks
    ]
    
    # 4. Stream Response
    async def generate_stream():
        # First send the sources
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        
        # Then stream the generated text
        async for chunk_text in generate_rag_response(request.query, chunks):
            yield f"data: {json.dumps({'type': 'content', 'data': chunk_text})}\n\n"
            
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

