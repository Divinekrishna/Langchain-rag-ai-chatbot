from typing import List, Optional, Dict
from langchain_core.documents import Document
from .llm_handler import LLMHandler
from .document_handler import DocumentHandler


class RAGSystem:
    """Retrieval-Augmented Generation system."""
    
    def __init__(self, llm_handler: Optional[LLMHandler] = None):
        self.llm_handler = llm_handler or LLMHandler()
        self.document_handler = DocumentHandler()
        self.documents = []
        self.processed_chunks = []
    
    def ingest_documents(self, file_paths: List[str]) -> Dict:
        """Ingest and process documents."""
        result = self.document_handler.process_documents(file_paths)
        self.documents.append(result)
        
        # Create chunks for retrieval
        chunks = self.document_handler.chunk_text(result['content'])
        self.processed_chunks.extend(chunks)
        
        return {
            'status': 'success',
            'documents_processed': result['total_documents'],
            'chunks_created': len(chunks),
            'content_size': len(result['content'])
        }
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant chunks based on query (simple similarity)."""
        if not self.processed_chunks:
            return []
        
        # Simple retrieval - can be enhanced with embeddings
        relevant_chunks = []
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk in self.processed_chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            scored_chunks.append((overlap, chunk))
        
        # Sort by overlap score and return top-k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        relevant_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
        
        return relevant_chunks
    
    def answer_question(self, query: str) -> str:
        """Answer a question using RAG approach."""
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            return "No relevant documents found to answer the question."
        
        # Combine relevant chunks as context
        context = "\n\n".join(relevant_chunks)
        
        # Use LLM to answer based on context
        answer = self.llm_handler.answer_question(context, query)
        
        return answer
    
    def summarize_documents(self, max_tokens: int = 500) -> str:
        """Summarize all ingested documents."""
        if not self.documents:
            return "No documents to summarize."
        
        # Combine all document content
        all_content = "\n\n".join([doc['content'] for doc in self.documents])
        
        # Limit content size for summarization
        if len(all_content) > 5000:
            all_content = all_content[:5000] + "..."
        
        summary = self.llm_handler.summarize(all_content, max_tokens)
        
        return summary
    
    def clear_documents(self):
        """Clear all stored documents and chunks."""
        self.documents = []
        self.processed_chunks = []
