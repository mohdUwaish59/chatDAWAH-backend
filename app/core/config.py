"""Application configuration"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    APP_NAME: str = "RAG Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "7860"))  # HF Spaces uses 7860
    
    # LLM Provider Settings
    LLM_PROVIDER: str = "openai"  # Options: "openai", "huggingface"
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    # Hugging Face Settings
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # Popular GPT-style models on Hugging Face:
    # - "openai-community/gpt2" (Small, fast)
    # - "openai-community/gpt2-large" (Better quality)
    # - "EleutherAI/gpt-neo-2.7B" (GPT-3 style, 2.7B params)
    # - "EleutherAI/gpt-j-6B" (GPT-3 style, 6B params)
    # - "EleutherAI/gpt-neox-20b" (GPT-3 style, 20B params)
    # Other excellent models:
    # - "mistralai/Mistral-7B-Instruct-v0.2" (Recommended)
    # - "meta-llama/Llama-2-7b-chat-hf" (Meta's Llama)
    # - "tiiuae/falcon-7b-instruct" (Falcon)
    # - "google/flan-t5-xxl" (Google T5)
    # - "bigscience/bloom-7b1" (Multilingual)
    
    # Common LLM Settings
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    
    # Retrieval Settings
    TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.3
    
    # Qdrant Settings
    QDRANT_URL: Optional[str] = None  # e.g., "https://xyz.cloud.qdrant.io"
    QDRANT_API_KEY: Optional[str] = None
    COLLECTION_NAME: str = "instructions"
    
    # Embedding Model (FastEmbed)
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # Data Settings
    DATA_PATH: str = "data/data.json"
    
    # CORS Settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://chat-dawah-frontend.vercel.app",
        "https://*.vercel.app",  # Allow all Vercel preview deployments
        "https://mohdwaish59-chatdawah.hf.space",  # HF Space backend
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
