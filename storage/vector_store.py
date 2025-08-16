import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api import ClientAPI
import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging
import os

from langchain.schema import Document as LangchainDocument
from config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # Connect to external ChromaDB server
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMADB_HOST,
                port=settings.CHROMADB_PORT
            )
            
            # Test connection
            self.client.heartbeat()
            logger.info(f"Connected to ChromaDB at {settings.CHROMADB_HTTP_URL}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            logger.info("Falling back to in-memory ChromaDB")
            # Fallback to in-memory client for development
            self.client = chromadb.EphemeralClient()
    
    def create_collection(self, document_id: str) -> None:
        """Create a new collection for a document."""
        try:
            collection_name = f"doc_{document_id.replace('-', '_')}"
            
            # Delete collection if it exists (for re-uploads)
            try:
                self.client.delete_collection(name=collection_name)
            except:
                pass
            
            self.client.create_collection(name=collection_name)
            logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def add_documents(self, document_id: str, documents: List[LangchainDocument], 
                     embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store."""
        collection_name = f"doc_{document_id.replace('-', '_')}"
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            ids = [f"{document_id}_{i}" for i in range(len(documents))]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection {collection_name}")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, document_id: str, query_embedding: List[float], k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for similar documents."""
        collection_name = f"doc_{document_id.replace('-', '_')}"
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    text = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'][0] else 0.0
                    
                    formatted_results.append((text, metadata, distance))
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def delete_collection(self, document_id: str) -> None:
        """Delete a document collection."""
        collection_name = f"doc_{document_id.replace('-', '_')}"
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
