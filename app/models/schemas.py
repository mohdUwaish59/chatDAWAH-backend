"""Pydantic models for request/response validation"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from app.core.config import settings


class ContextItem(BaseModel):
    """Context item from retrieval"""
    instruction: str
    output: str
    similarity: float


class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., min_length=1, description="User's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve")
    
    @field_validator('top_k', mode='before')
    @classmethod
    def set_default_top_k(cls, v):
        """Use settings.TOP_K if not provided"""
        return v if v is not None else settings.TOP_K


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    context: List[ContextItem] = []
    question: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    chatbot_ready: bool
    version: str


class StatsResponse(BaseModel):
    """Statistics response"""
    total_documents: int
    model: str
    embedding_model: str
    max_tokens: int
    default_top_k: int
