from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from app.core.config import settings
from app.models.api_models import EvaluationResponse

eval_llm = ChatOpenAI(
    model="qwen/qwen3.6-plus-preview:free",
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0
)

class EvalResult(BaseModel):
    score: float = Field(description="A score between 0.0 and 1.0 indicating answer relevance and correctness based solely on the context.")
    reasoning: str = Field(description="The reasoning behind the score.")

parser = PydanticOutputParser(pydantic_object=EvalResult)

eval_prompt = ChatPromptTemplate.from_template(
    "You are an expert evaluator. Given a user query, a context document, and an answer, evaluate how good the answer is based strictly on the context. "
    "Provide a score between 0.0 and 1.0, and a brief reasoning. Follow the output instructions exactly.\n\n"
    "Query: {query}\n"
    "Context: {context}\n"
    "Answer: {answer}\n\n"
    "{format_instructions}"
)

async def evaluate_answer(query: str, context: str, answer: str) -> EvaluationResponse:
    """Evaluates the quality of a generated answer using an LLM-as-a-judge approach."""
    chain = eval_prompt | eval_llm | parser
    
    result: EvalResult = await chain.ainvoke({
        "query": query,
        "context": context,
        "answer": answer,
        "format_instructions": parser.get_format_instructions()
    })
    
    return EvaluationResponse(
        evaluation_score=result.score,
        reasoning=result.reasoning
    )
