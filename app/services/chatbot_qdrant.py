"""Chatbot service with RAG implementation using Qdrant Cloud"""
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed.embedding import DefaultEmbedding

from app.core.config import settings
from app.services.llm_provider import get_llm_provider, LLMProvider


class ChatbotService:
    """RAG Chatbot Service with Qdrant Cloud"""
    
    def __init__(self):
        self.data: List[Dict[str, str]] = []
        self.llm_provider: LLMProvider = None
        self.qdrant_client: QdrantClient = None
        self.embedding_model: DefaultEmbedding = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the chatbot service"""
        if self._initialized:
            return
        
        print("Loading data...")
        with open(settings.DATA_PATH, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Initializing LLM provider: {settings.LLM_PROVIDER}...")
        self.llm_provider = get_llm_provider(
            provider_name=settings.LLM_PROVIDER,
            openai_key=settings.OPENAI_API_KEY,
            openai_model=settings.OPENAI_MODEL,
            huggingface_key=settings.HUGGINGFACE_API_KEY,
            huggingface_model=settings.HUGGINGFACE_MODEL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE
        )
        
        if not self.llm_provider.is_available():
            raise ValueError(f"LLM provider '{settings.LLM_PROVIDER}' is not properly configured. Check your API keys.")
        
        print("Initializing Qdrant client...")
        if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        
        print(f"Initializing embedding model: {settings.EMBEDDING_MODEL}...")
        self.embedding_model = DefaultEmbedding(model_name=settings.EMBEDDING_MODEL)
        
        # Get vector dimension from the model
        sample_embedding = list(self.embedding_model.embed(["test"]))[0]
        vector_size = len(sample_embedding)
        print(f"Vector dimension: {vector_size}")
        
        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(c.name == settings.COLLECTION_NAME for c in collections)
        
        if collection_exists:
            collection_info = self.qdrant_client.get_collection(settings.COLLECTION_NAME)
            print(f"Loaded existing collection with {collection_info.points_count} items")
        else:
            print("Creating new collection...")
            self.qdrant_client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            await self._populate_collection()
        
        self._initialized = True
        print("Chatbot service initialized successfully!")
    
    async def _populate_collection(self):
        """Add all instruction-output pairs to Qdrant"""
        print("Generating embeddings and uploading to Qdrant...")
        
        instructions = [item['instruction'] for item in self.data]
        
        # Generate embeddings in batches
        embeddings = list(self.embedding_model.embed(instructions))
        
        # Create points for Qdrant
        points = []
        for idx, (item, embedding) in enumerate(zip(self.data, embeddings)):
            # Build payload with all available data
            payload = {
                "instruction": item['instruction'],
                "output": item['output']
            }
            
            # Add optional metadata if available
            if 'channel_username' in item and item['channel_username']:
                payload['channel_username'] = item['channel_username']
            if 'video_id' in item and item['video_id']:
                payload['video_id'] = item['video_id']
            if 'input' in item and item['input']:
                payload['input'] = item['input']
            
            # Add source identifier
            payload['source'] = item.get('source', 'data.json')
            
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=batch
            )
            print(f"Uploaded {min(i + batch_size, len(points))}/{len(points)} items")
        
        print("Collection populated successfully!")
    
    def retrieve_context(self, user_question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve top K most similar documents"""
        if not self._initialized:
            raise RuntimeError("Chatbot service not initialized")
        
        top_k = top_k or settings.TOP_K
        
        # Generate embedding for the query
        query_embedding = list(self.embedding_model.embed([user_question]))[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        context_items = []
        for result in search_results:
            if result.score >= settings.SIMILARITY_THRESHOLD:
                item = {
                    'instruction': result.payload['instruction'],
                    'output': result.payload['output'],
                    'similarity': result.score
                }
                
                # Add optional metadata if available
                if 'channel_username' in result.payload:
                    item['channel_username'] = result.payload['channel_username']
                if 'video_id' in result.payload:
                    item['video_id'] = result.payload['video_id']
                if 'source' in result.payload:
                    item['source'] = result.payload['source']
                
                context_items.append(item)
        
        return context_items
    
    async def generate_response(self, user_question: str, context_items: List[Dict[str, Any]]) -> str:
        """Generate response using LLM with retrieved context"""
        if not self._initialized:
            raise RuntimeError("Chatbot service not initialized")
        
        # Build context string
        context_text = "\n\n".join([
            f"Q: {item['instruction']}\nA: {item['output']}"
            for item in context_items
        ])
        
        # Create prompt
        prompt = f"""You are a knowledgeable assistant. Answer the user's question based on the following relevant information from the knowledge base.

Relevant Information:
{context_text}

User Question: {user_question}

Instructions:
- Answer based on the provided information and context only
- Provide comprehensive, detailed responses when the question requires it
- If the information doesn't fully answer the question, say so
- Synthesize information from multiple sources when relevant
- Maintain the tone and style of the knowledge base
- Use examples and explanations where helpful
- Never give any reference from quran

Answer:"""
        
        # System prompt
        system_prompt = "You are a helpful assistant that answers questions based on provided context."
        
        # Call LLM provider
        response = await self.llm_provider.generate(prompt, system_prompt)
        
        return response
    
    async def query(self, user_question: str, top_k: int = None) -> Dict[str, Any]:
        """Main query method: retrieve context and generate response"""
        if not self._initialized:
            raise RuntimeError("Chatbot service not initialized")
        
        # Retrieve relevant documents
        context_items = self.retrieve_context(user_question, top_k)
        
        if not context_items:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'context': [],
                'question': user_question
            }
        
        # Generate response using LLM
        answer = await self.generate_response(user_question, context_items)
        
        return {
            'answer': answer,
            'context': context_items,
            'question': user_question
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        if not self._initialized:
            raise RuntimeError("Chatbot service not initialized")
        
        # Get model name based on provider
        if settings.LLM_PROVIDER == "openai":
            model_name = settings.OPENAI_MODEL
        elif settings.LLM_PROVIDER == "huggingface":
            model_name = settings.HUGGINGFACE_MODEL
        else:
            model_name = "unknown"
        
        collection_info = self.qdrant_client.get_collection(settings.COLLECTION_NAME)
        
        return {
            "total_documents": collection_info.points_count,
            "llm_provider": settings.LLM_PROVIDER,
            "model": model_name,
            "embedding_model": settings.EMBEDDING_MODEL,
            "max_tokens": settings.MAX_TOKENS,
            "default_top_k": settings.TOP_K,
            "vector_db": "Qdrant Cloud"
        }
    
    @property
    def is_ready(self) -> bool:
        """Check if chatbot is ready"""
        return self._initialized


# Global chatbot instance
chatbot_service = ChatbotService()
