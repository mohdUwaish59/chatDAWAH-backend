"""Pydantic models for request/response validation"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ContextItem(BaseModel):
    """Context item from retrieval"""
    instruction: str
    output: str
    similarity: float


class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., min_length=1, description="User's question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents to retrieve")


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
