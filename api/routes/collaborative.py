from fastapi import APIRouter, HTTPException
from schemas import RecommendationRequest, RecommendationResponse, RecommendationItem
from services import get_collaborative_model
from api.vector_store import get_products

router = APIRouter()

@router.post("/collaborative", response_model=RecommendationResponse)
async def get_collaborative_recommendations(request: RecommendationRequest):
    """
    Endpoint for Collaborative Filtering recommendations.
    Uses the integrated CollaborativeFiltering implementation.
    Enriched with catalog metadata.
    """
    try:
        model = get_collaborative_model()
        
        # Get recommendations from the model
        item_ids = model.recommend_user_based(request.user_id, top_n=request.top_n)
        
        if not item_ids:
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[],
                model_used="Collaborative Filtering"
            )

        # Load metadata catalog to enrich IDs
        catalog = {str(p["product_id"]): p for p in get_products()}
        
        recommendations = []
        for item_id in item_ids:
            item_id_str = str(item_id)
            meta = catalog.get(item_id_str, {})
            
            recommendations.append(
                RecommendationItem(
                    item_id=item_id_str,
                    score=1.0,  # Placeholder
                    title=str(meta.get("title", meta.get("name", f"Product {item_id_str}"))),
                    description=str(meta.get("description", "No description available.")),
                    price=str(meta.get("price", "N/A")),
                    category=str(meta.get("category", "N/A")),
                    material=str(meta.get("material", "N/A")),
                    color=str(meta.get("color", "N/A")),
                    style=str(meta.get("style", "N/A"))
                )
            )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used="Collaborative Filtering"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collaborative recommendation error: {str(e)}")
