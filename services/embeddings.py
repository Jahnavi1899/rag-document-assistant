from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import logging

from config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]