"""LLM Provider abstraction for multiple AI backends"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from openai import AsyncOpenAI
import httpx


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using OpenAI API"""
        if not self.is_available():
            raise ValueError("OpenAI provider not configured. Set OPENAI_API_KEY.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content


class HuggingFaceProvider(LLMProvider):
    """Hugging Face provider using OpenAI-compatible API"""
    
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        
        # Initialize OpenAI client with HuggingFace router
        if api_key:
            self.client = AsyncOpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=api_key
            )
            print(f"✅ Hugging Face OpenAI-compatible client initialized with model: {model}")
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using Hugging Face OpenAI-compatible API"""
        if not self.is_available():
            raise ValueError("Hugging Face provider not configured. Set HUGGINGFACE_API_KEY")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Hugging Face API error: {str(e)}")


def get_llm_provider(
    provider_name: str,
    openai_key: str = None,
    openai_model: str = "gpt-3.5-turbo",
    huggingface_key: str = None,
    huggingface_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> LLMProvider:
    """Factory function to get the appropriate LLM provider"""
    
    provider_name = provider_name.lower()
    
    if provider_name == "openai":
        return OpenAIProvider(
            api_key=openai_key,
            model=openai_model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    elif provider_name == "huggingface":
        return HuggingFaceProvider(
            api_key=huggingface_key,
            model=huggingface_model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}. Use 'openai' or 'huggingface'.")
