"""Services module"""
# Use Qdrant Cloud with FastEmbed
from .chatbot_qdrant import chatbot_service

__all__ = ["chatbot_service"]
