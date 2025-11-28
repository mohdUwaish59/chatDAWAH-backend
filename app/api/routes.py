"""API routes"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    StatsResponse,
    ContextItem
)
from app.services import chatbot_service
from app.core.config import settings

router = APIRouter()


@router.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def home():
    """Serve the main HTML page"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Template not found</h1><p>Please ensure templates/index.html exists</p>",
            status_code=404
        )


@router.post("/query", response_model=QueryResponse, tags=["Chatbot"])
async def query(request: QueryRequest):
    """
    Query the chatbot with a question
    
    - **question**: The user's question
    - **top_k**: Number of similar documents to retrieve (default: 5)
    """
    print("\n" + "="*60)
    print("🌐 DEBUG: Received POST /query request")
    print(f"   Question: {request.question}")
    print(f"   Top K: {request.top_k}")
    print("="*60)
    
    if not chatbot_service.is_ready:
        print("❌ DEBUG: Chatbot service not ready!")
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        print("🔄 DEBUG: Calling chatbot service...")
        result = await chatbot_service.query(
            request.question,
            top_k=request.top_k
        )
        
        print(f"✅ DEBUG: Query successful, building response...")
        response = QueryResponse(
            answer=result['answer'],
            context=[ContextItem(**item) for item in result.get('context', [])],
            question=result['question']
        )
        print(f"📤 DEBUG: Sending response ({len(response.answer)} chars)")
        return response
        
    except Exception as e:
        print(f"❌ DEBUG: Error in query endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint"""
    print("🏥 DEBUG: Health check requested")
    is_ready = chatbot_service.is_ready
    status = "healthy" if is_ready else "initializing"
    print(f"   Status: {status}")
    print(f"   Ready: {is_ready}")
    
    return HealthResponse(
        status=status,
        chatbot_ready=is_ready,
        version=settings.APP_VERSION
    )


@router.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats():
    """Get chatbot statistics"""
    print("📊 DEBUG: Stats endpoint requested")
    
    if not chatbot_service.is_ready:
        print("❌ DEBUG: Chatbot not ready for stats")
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        print("🔄 DEBUG: Getting stats from service...")
        stats_data = chatbot_service.get_stats()
        print(f"✅ DEBUG: Stats retrieved successfully")
        return StatsResponse(**stats_data)
    except Exception as e:
        print(f"❌ DEBUG: Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", tags=["System"])
async def get_config():
    """
    Get current configuration settings
    This allows frontend to use backend configuration dynamically
    """
    return {
        "top_k": settings.TOP_K,
        "max_tokens": settings.MAX_TOKENS,
        "temperature": settings.TEMPERATURE,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "llm_provider": settings.LLM_PROVIDER,
        "model": settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.HUGGINGFACE_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "collection_name": settings.COLLECTION_NAME
    }
