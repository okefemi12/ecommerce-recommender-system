from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.model import ModelService

router = APIRouter()

class UserState(BaseModel):
    history: list[int]

@router.post("/recommend")
def get_recommendation(data: UserState):
    """Get a product recommendation"""
    if not ModelService.interpreter:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        recommended_id, q_values = ModelService.predict(data.history)
        return {
            "recommended_id": recommended_id,
            "all_q_values": q_values
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))