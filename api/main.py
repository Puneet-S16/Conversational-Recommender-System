from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from routes import collaborative, embedding, hybrid, conversational

app = FastAPI(
    title="GenAI Recommender System API",
    description="API for various recommendation algorithms including Collaborative, Embedding-based, Hybrid, and Conversational RAG.",
    version="1.0.0"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"DEBUG: Validation Error Details: {exc.errors()}")
    print(f"DEBUG: Request Body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(await request.body())},
    )

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(collaborative.router, prefix="/recommend", tags=["Collaborative"])
app.include_router(embedding.router, prefix="/recommend", tags=["Embedding"])
app.include_router(hybrid.router, prefix="/recommend", tags=["Hybrid"])
app.include_router(conversational.router, prefix="/recommend", tags=["Conversational"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
