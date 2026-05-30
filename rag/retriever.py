import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

try:
    from api.content_recommender import recommend_from_query
except ImportError:
    # Fallback for different execution contexts
    import os
    sys.path.append(os.getcwd())
    from api.content_recommender import recommend_from_query

class FAISSRetriever:
    """
    A bridge class that reuses the existing FAISS-based retrieval pipeline.
    """
    def __init__(self, top_k=3):
        self.top_k = top_k

    def get_relevant_products(self, query):
        """
        Retrieves products using the existing content_recommender logic.
        """
        try:
            return recommend_from_query(query, top_k=self.top_k)
        except Exception as e:
            print(f"Error in FAISSRetriever: {e}")
            return []

if __name__ == "__main__":
    # Test the retriever bridge
    retriever = FAISSRetriever()
    results = retriever.get_relevant_products("modern sofa")
    print(f"Retrieved {len(results)} products.")
    for r in results:
        print(f"- {r['title']}")
