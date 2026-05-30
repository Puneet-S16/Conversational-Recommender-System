import pandas as pd
from collaborative_filter import CollaborativeFiltering
import os
from pathlib import Path

# Initialize models as singletons
_collab_model = None
_embedding_recommender = None
_hybrid_recommender = None
_rag_chain = None

def get_collaborative_model():
    global _collab_model
    if _collab_model is None:
        _collab_model = CollaborativeFiltering()
        # Mock data for initialization if no data source is provided
        data = {
            'user_id': ['user_123', 'user_123', 'user_456', 'user_456', 'user_789', 'user_123', 'user_456'],
            'product_id': ['prod_1', 'prod_2', 'prod_1', 'prod_3', 'prod_2', 'prod_3', 'prod_2'],
            'score': [5, 4, 5, 2, 4, 3, 5]
        }
        df = pd.DataFrame(data)
        _collab_model.fit(df)
    return _collab_model

def get_embedding_recommender():
    """
    Returns functions from the integrated embedding-based recommender.
    Ensures that the index and embeddings are built if they don't exist.
    """
    global _embedding_recommender
    if _embedding_recommender is None:
        from content_recommender import recommend_for_user, recommend_from_query
        from vector_store import ensure_index_artifacts
        
        # Ensure artifacts exist on first call
        ensure_index_artifacts()
        
        _embedding_recommender = {
            "recommend_for_user": recommend_for_user,
            "recommend_from_query": recommend_from_query
        }
    return _embedding_recommender

def get_hybrid_recommender():
    """
    Returns the integrated HybridRecommender instance.
    """
    global _hybrid_recommender
    if _hybrid_recommender is None:
        from hybrid_recommender import HybridRecommender
        _hybrid_recommender = HybridRecommender(load_ranker=True)
    return _hybrid_recommender

def get_rag_chain():
    """
    Returns the integrated RAG chain functions.
    """
    global _rag_chain
    if _rag_chain is None:
        from rag.rag_chain import generate_recommendation
        from rag.retriever import FAISSRetriever
        
        _rag_chain = {
            "generate_recommendation": generate_recommendation,
            "retriever": FAISSRetriever(top_k=3)
        }
    return _rag_chain
