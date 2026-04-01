from fastapi import APIRouter
from app.models.api_models import EvaluationRequest, EvaluationResponse
from app.services.evaluation import evaluate_answer

router = APIRouter()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    response = await evaluate_answer(
        query=request.query,
        context=request.context,
        answer=request.answer
    )
    return response
