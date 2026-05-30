# 🛠 GenAI Recommender System: API Technical Reference

This document provides a detailed technical specification for the GenAI Recommender System API. The API is built using **FastAPI** and is designed to be modular, scalable, and easy to integrate with various frontend applications.

---

## 🚀 Getting Started

### Base URL
`http://localhost:8000`

### Interactive Documentation (Swagger UI)
Once the server is running, you can access the interactive Swagger documentation at:
*   **Swagger UI**: `http://localhost:8000/docs`
*   **ReDoc**: `http://localhost:8000/redoc`

Use the Swagger UI to test endpoints directly from your browser without writing any code.

---

## 🔒 Authentication (Placeholder)

Currently, the API does not require authentication for internal testing. In a production environment, use the `Authorization` header with a Bearer token:

```http
Authorization: Bearer <your_access_token>
```

---

## 📡 API Endpoints

### 1. Collaborative Filtering
**POST** `/recommend/collaborative`

Returns items based on user-item interaction patterns.

**Request Schema:**
```json
{
  "user_id": "1",
  "top_n": 5
}
```

**Response Schema:**
```json
{
  "user_id": "1",
  "recommendations": [
    {
      "item_id": "173",
      "score": 1.0,
      "title": "Product 173",
      "description": null
    }
  ],
  "model_used": "Collaborative Filtering"
}
```

---

### 2. Embedding-Based Recommendation
**POST** `/recommend/embedding`

Uses vector similarity (FAISS) to find products semantically similar to the user's purchase history.

**Request Schema:**
```json
{
  "user_id": "user_123",
  "top_n": 3,
  "metadata": {
    "purchased_product_ids": ["1", "2"]
  }
}
```

**Note**: If no history is provided in metadata, the system uses a demo fallback for `user_123`.

---

### 3. Hybrid Recommendation (LTR)
**POST** `/recommend/hybrid`

The most advanced engine. It pools candidates from multiple sources and ranks them using an **XGBoost Learning-to-Rank (LTR)** model.

**Explanation**:
This endpoint calculates features like `cf_score`, `embedding_similarity`, `price_sensitivity`, and `historical_ctr` for 50+ candidates before picking the top $N$ using the LTR ranker.

---

### 4. Conversational RAG Flow
**POST** `/recommend/conversational`

Interactive AI assistant flow.

**The Workflow**:
1.  **Retrieval**: The system uses FAISS to find the top 3 items matching the user's natural language `query`.
2.  **Augmentation**: Item metadata (titles, prices, materials) is injected into a specific **Prompt Template** (e.g., "Luxury" or "Budget").
3.  **Generation**: The LLM (OpenAI/Ollama) generates a professional recommendation response.

**Request Schema:**
```json
{
  "user_id": "user_123",
  "query": "I need a modern wooden sofa for my office.",
  "chat_history": [
    {"role": "user", "content": "Hi, I'm looking for furniture."},
    {"role": "assistant", "content": "Sure! What specifically are you looking for?"}
  ]
}
```

---

## ⚠️ Error Handling

The API uses standard HTTP status codes:

| Code | Meaning | Description |
| :--- | :--- | :--- |
| `200` | OK | Success. |
| `400` | Bad Request | Validation error (check your JSON schema). |
| `404` | Not Found | Resource not found. |
| `500` | Internal Error | Server-side logic failure (check logs). |

**Error Response Format:**
```json
{
  "detail": "Error message describing the failure."
}
```

---

## 🛠 Running Instructions

1.  **Start API**: `uvicorn api.main:app --reload`
2.  **Verify**: Access `http://localhost:8000/health`
3.  **Test**: Use `curl`, Postman, or the built-in Swagger UI.
