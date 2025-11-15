"""Main FastAPI application entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api import router
from app.services import chatbot_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    # Startup
    print("\n" + "="*60)
    print(f"🚀 DEBUG: Starting {settings.APP_NAME}")
    print(f"   Version: {settings.APP_VERSION}")
    print(f"   Debug mode: {settings.DEBUG}")
    print(f"   LLM Provider: {settings.LLM_PROVIDER}")
    print("="*60)
    
    try:
        print("🔧 DEBUG: Initializing chatbot service...")
        await chatbot_service.initialize()
        print("✅ DEBUG: Chatbot service initialized successfully")
        print("\n" + "="*60)
        print(f"🌐 DEBUG: Server ready at: http://{settings.HOST}:{settings.PORT}")
        print(f"📚 DEBUG: API docs at: http://{settings.HOST}:{settings.PORT}/docs")
        print(f"🔍 DEBUG: Health check: http://{settings.HOST}:{settings.PORT}/health")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ DEBUG: Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Shutdown
    print("\n" + "="*60)
    print("👋 DEBUG: Shutting down...")
    print("="*60 + "\n")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Retrieval-Augmented Generation Chatbot with ChromaDB and OpenAI",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
# Allow all origins for now (can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
