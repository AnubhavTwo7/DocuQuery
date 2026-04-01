from typing import AsyncGenerator, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.core.config import settings
from app.models.domain_models import DocumentChunk

# Initialize LLM
llm = ChatOpenAI(
    model="qwen/qwen3.6-plus-preview:free",
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
    streaming=True
)

rewrite_prompt = ChatPromptTemplate.from_template(
    "You are an AI assistant tasked with reformulating a user's query for a semantic search engine. "
    "Given the user's query, rewrite it to be a more complete and descriptive query that will yield better search results. "
    "Do not answer the query, just rewrite it into a single, well-formed question or search phrase. "
    "Do not output anything other than the rewritten query.\n\n"
    "User Query: {query}\n"
    "Rewritten Query:"
)

rag_prompt = ChatPromptTemplate.from_template(
    "You are a helpful and accurate assistant. Answer the user's question based strictly on the provided context. "
    "If the answer is not contained in the context, say 'I cannot answer this based on the provided documents.' "
    "Do not make up any information. "
    "IMPORTANT: For all mathematical formulas, you MUST use \\( and \\) for inline math, and \\[ and \\] for display math. Never use $ or $$ directly.\n\n"
    "Context:\n{context}\n\n"
    "User Question: {query}\n"
    "Answer:"
)

async def rewrite_query(original_query: str) -> str:
    """Uses LLM to rewrite and expand the query for better retrieval."""
    chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = await chain.ainvoke({"query": original_query})
    return rewritten.strip()

async def generate_rag_response(query: str, chunks: List[DocumentChunk]) -> AsyncGenerator[str, None]:
    """Generates a streaming response using the retrieved chunks as context."""
    
    # Format context
    context_text = ""
    for chunk in chunks:
        context_text += f"---\nSource: {chunk.metadata.source} (Page {chunk.metadata.page})\nContext: {chunk.text}\n"

    chain = rag_prompt | llm | StrOutputParser()
    
    # Use stream to return an async generator
    async for chunk_str in chain.astream({"context": context_text, "query": query}):
        yield chunk_str
