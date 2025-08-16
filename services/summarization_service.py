from typing import List, Dict, Optional
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument

from services.llm_service import LLMServiceFactory
from config import settings

logger = logging.getLogger(__name__)

class SummarizationService:
    """Service for generating document summaries using LLMs."""
    
    def __init__(self):
        self.llm_service = LLMServiceFactory.create_llm_service()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Larger chunks for summarization
            chunk_overlap=200,
            length_function=len,
        )
    
    def generate_document_summary(self, text: str, filename: str) -> Dict[str, str]:
        """
        Generate a comprehensive summary of the entire document.
        Returns both short and detailed summaries.
        """
        try:
            # If document is short, summarize directly
            if len(text) <= 4000:
                return self._summarize_single_chunk(text, filename)
            
            # For longer documents, use hierarchical summarization
            return self._hierarchical_summarization(text, filename)
            
        except Exception as e:
            logger.error(f"Error generating summary for {filename}: {str(e)}")
            return {
                "short_summary": "Summary generation failed",
                "detailed_summary": f"Error: {str(e)}",
                "key_topics": [],
                "summary_method": "failed"
            }
    
    def _summarize_single_chunk(self, text: str, filename: str) -> Dict[str, str]:
        """Summarize a single chunk of text."""
        
        # Short summary prompt
        short_prompt = f"""
        Please provide a concise summary of this document in 2-3 sentences:
        
        Document: {filename}
        Content: {text[:3000]}...
        
        Summary:"""
        
        # Detailed summary prompt
        detailed_prompt = f"""
        Please provide a comprehensive summary of this document including:
        1. Main purpose and topic
        2. Key points and findings
        3. Important details
        4. Conclusions or recommendations
        
        Document: {filename}
        Content: {text}
        
        Detailed Summary:"""
        
        # Key topics prompt
        topics_prompt = f"""
        Extract the main topics and themes from this document. List them as comma-separated keywords:
        
        Document: {filename}
        Content: {text[:2000]}...
        
        Key topics:"""
        
        try:
            short_summary = self.llm_service.generate_response(short_prompt, "")
            detailed_summary = self.llm_service.generate_response(detailed_prompt, "")
            key_topics_str = self.llm_service.generate_response(topics_prompt, "")
            
            # Parse key topics
            key_topics = [topic.strip() for topic in key_topics_str.split(',') if topic.strip()]
            
            return {
                "short_summary": short_summary.strip(),
                "detailed_summary": detailed_summary.strip(),
                "key_topics": key_topics[:10],  # Limit to 10 topics
                "summary_method": "single_chunk"
            }
            
        except Exception as e:
            logger.error(f"Error in single chunk summarization: {str(e)}")
            raise
    
    def _hierarchical_summarization(self, text: str, filename: str) -> Dict[str, str]:
        """
        Use hierarchical summarization for long documents:
        1. Split into chunks
        2. Summarize each chunk
        3. Combine chunk summaries into final summary
        """
        
        # Split text into manageable chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Processing {len(chunks)} chunks for hierarchical summarization")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks[:10]):  # Limit to 10 chunks to avoid token limits
            chunk_prompt = f"""
            Summarize this section of the document in 2-3 sentences, focusing on the main points:
            
            Section {i+1} of {filename}:
            {chunk}
            
            Section Summary:"""
            
            try:
                chunk_summary = self.llm_service.generate_response(chunk_prompt, "")
                chunk_summaries.append(chunk_summary.strip())
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {i+1}: {str(e)}")
                chunk_summaries.append(f"Section {i+1}: Summary unavailable")
        
        # Combine chunk summaries into final summary
        combined_text = "\n\n".join(chunk_summaries)
        
        # Generate final summaries
        short_prompt = f"""
        Based on these section summaries from document "{filename}", provide a concise 2-3 sentence overview:
        
        Section Summaries:
        {combined_text}
        
        Overall Summary:"""
        
        detailed_prompt = f"""
        Based on these section summaries from document "{filename}", provide a comprehensive summary including:
        1. Document purpose and main topic
        2. Key findings and important points
        3. Major themes and conclusions
        
        Section Summaries:
        {combined_text}
        
        Comprehensive Summary:"""
        
        topics_prompt = f"""
        Extract the main topics from this document based on the section summaries. List as comma-separated keywords:
        
        {combined_text}
        
        Key topics:"""
        
        try:
            short_summary = self.llm_service.generate_response(short_prompt, "")
            detailed_summary = self.llm_service.generate_response(detailed_prompt, "")
            key_topics_str = self.llm_service.generate_response(topics_prompt, "")
            
            key_topics = [topic.strip() for topic in key_topics_str.split(',') if topic.strip()]
            
            return {
                "short_summary": short_summary.strip(),
                "detailed_summary": detailed_summary.strip(),
                "key_topics": key_topics[:10],
                "summary_method": "hierarchical",
                "chunks_processed": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical summarization: {str(e)}")
            raise