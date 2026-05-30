from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., example="user_123")
    top_n: int = Field(default=5, ge=1, le=20)
    metadata: Optional[Dict[str, Any]] = None

class ConversationalRequest(BaseModel):
    user_id: str = Field(..., example="user_123")
    query: str = Field(..., example="I'm looking for a mystery novel with a twist ending.")
    top_n: int = Field(default=5, ge=1, le=20)
    chat_history: Optional[List[Dict[str, Any]]] = []

class RecommendationItem(BaseModel):
    item_id: str
    score: float
    title: Optional[str] = None
    description: Optional[str] = None
    price: Optional[str] = "N/A"
    category: Optional[str] = "N/A"
    material: Optional[str] = "N/A"
    color: Optional[str] = "N/A"
    style: Optional[str] = "N/A"

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]
    model_used: str

class ConversationalResponse(BaseModel):
    user_id: str
    response: str
    recommendations: Optional[List[RecommendationItem]] = None
