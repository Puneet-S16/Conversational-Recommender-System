from fastapi import APIRouter, HTTPException
from schemas import RecommendationRequest, RecommendationResponse, RecommendationItem
from services import get_embedding_recommender

router = APIRouter()

@router.post("/embedding", response_model=RecommendationResponse)
async def get_embedding_recommendations(request: RecommendationRequest):
    """
    Endpoint for Embedding-Based recommendations.
    Uses existing vector search and embedding logic.
    """
    try:
        recommender = get_embedding_recommender()
        
        # Mock logic to get purchased products for a user if metadata is empty
        # In a real app, this would come from a database.
        purchased_ids = []
        if request.metadata and "purchased_product_ids" in request.metadata:
            purchased_ids = request.metadata["purchased_product_ids"]
        else:
            # Fallback for demo: if user_id is 'user_123', use some default product IDs
            if request.user_id == "user_123":
                purchased_ids = ["1"] # Valid example ID from debug
            else:
                # If no purchased IDs, we might want to recommend based on a generic query
                # or return empty. Here we'll try to find similar to a generic 'furniture' query
                # if no history is provided.
                pass

        if not purchased_ids:
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[],
                model_used="Embedding-Based Recommendation (No history provided)"
            )

        # Get recommendations using the integrated logic
        results = recommender["recommend_for_user"](
            purchased_ids, 
            top_k=request.top_n
        )
        
        # Map to RecommendationResponse
        recommendations = [
            RecommendationItem(
                item_id=str(res.get("product_id")),
                score=float(res.get("score", 0.0)),
                title=str(res.get("title", res.get("name", "Unknown Product"))),
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
            model_used="Embedding-Based Recommendation"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding recommendation error: {str(e)}")
