from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import json

import requests
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from config import settings

logger = logging.getLogger(__name__)

class BaseLLMService(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: str) -> str:
        pass

class OllamaLLMService(BaseLLMService):
    def __init__(self, model: str = settings.DEFAULT_MODEL):
        self.model = model
        self.base_url = settings.OLLAMA_BASE_URL
    
    def generate_response(self, prompt: str, context: str) -> str:
        """Generate response using Ollama."""
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
        Use only the information from the context to answer questions. If the context doesn't contain 
        enough information to answer the question, say so clearly."""
        
        full_prompt = f"""Context: {context}

        Question: {prompt}

        Answer:"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise

class OpenAILLMService(BaseLLMService):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
    
    def generate_response(self, prompt: str, context: str) -> str:
        """Generate response using OpenAI API."""
        # Implementation for OpenAI would go here
        # This is a placeholder for the structure
        pass

class LLMServiceFactory:
    @staticmethod
    def create_llm_service(provider: str = settings.DEFAULT_LLM_PROVIDER) -> BaseLLMService:
        if provider == "ollama":
            return OllamaLLMService()
        elif provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not provided")
            return OpenAILLMService(settings.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")