import math
from fastapi import APIRouter, HTTPException
from schemas import RecommendationRequest, RecommendationResponse, RecommendationItem
from services import get_hybrid_recommender

router = APIRouter()

def sigmoid(x: float) -> float:
    """Normalizes raw scores into a 0-1 range for UI display."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0

@router.post("/hybrid", response_model=RecommendationResponse)
async def get_hybrid_recommendations(request: RecommendationRequest):
    """
    Endpoint for Hybrid recommendations.
    Combines Collaborative Filtering and Embedding-Based results with LTR ranking.
    Scores are normalized for UI display using a sigmoid function.
    """
    try:
        recommender = get_hybrid_recommender()
        
        # Determine user_id to use. In this dataset, user_ids are integers.
        # Fallback to a valid integer ID if user_123 is provided (demo purpose)
        internal_user_id = 1
        try:
            internal_user_id = int(request.user_id)
        except ValueError:
            # If user_id is not an integer (like 'user_123'), use a default valid one
            internal_user_id = 1
            
        # Get recommendations from the hybrid model
        results = recommender.hybrid_recommend(
            user_id=internal_user_id,
            top_k=request.top_n
        )
        
        # Map to RecommendationResponse with sigmoid-normalized scores
        recommendations = [
            RecommendationItem(
                item_id=str(res["product_id"]),
                score=round(sigmoid(float(res.get("ltr_score", res.get("hybrid_signal_score", 0.0)))), 4),
                title=str(res.get("title", f"Product {res['product_id']}")),
                description=str(res.get("description", "No description available.")),
                price=str(res.get("price", "N/A")),
                category=str(res.get("category", "N/A")),
                material=str(res.get("material", "N/A")),
                color=str(res.get("color", "N/A")),
                style=str(res.get("style", "N/A"))
            ) for res in results
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used="Hybrid Recommendation (LTR-Ranked)"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hybrid recommendation error: {str(e)}")
