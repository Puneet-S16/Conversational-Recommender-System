from fastapi import APIRouter, HTTPException
from schemas import ConversationalRequest, ConversationalResponse, RecommendationItem
from services import get_rag_chain

router = APIRouter()

@router.post("/conversational", response_model=ConversationalResponse)
async def get_conversational_recommendations(request: ConversationalRequest):
    """
    Endpoint for Conversational RAG recommendations.
    Uses LLM to generate a response and FAISS to retrieve relevant items.
    """
    # Debug logging
    print(f"DEBUG: Incoming RAG request: {request.dict()}")
    
    try:
        rag_services = get_rag_chain()
        generate_recommendation = rag_services["generate_recommendation"]
        retriever = rag_services["retriever"]
        
        # 1. Generate the LLM response
        history_summary = ""
        if request.chat_history:
            history_summary = " ".join([m.get("content", "") for m in request.chat_history[-2:]])
            
        ai_response = generate_recommendation(request.query, history_summary)
        
        # 2. Get the items for the UI table using top_n from request
        retrieved_products = retriever.get_relevant_products(request.query)
        # Apply top_n slice if retriever doesn't handle it directly or to be sure
        retrieved_products = retrieved_products[:request.top_n]
        
        # Map to RecommendationItem
        recommendations = [
            RecommendationItem(
                item_id=str(res.get("product_id")),
                score=float(res.get("score", res.get("mean_score", 1.0))),
                title=str(res.get("title", "Product")),
                description=str(res.get("description", "No description available.")),
                price=str(res.get("price", "N/A")),
                category=str(res.get("category", "N/A")),
                material=str(res.get("material", "N/A")),
                color=str(res.get("color", "N/A")),
                style=str(res.get("style", "N/A"))
            ) for res in retrieved_products
        ]
        
        return ConversationalResponse(
            user_id=request.user_id,
            response=ai_response,
            recommendations=recommendations
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Conversational RAG error: {str(e)}")
