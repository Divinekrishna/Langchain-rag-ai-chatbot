import os
from typing import List, Optional
from pathlib import Path
import PyPDF2


class DocumentHandler:
    """Handle document ingestion and processing."""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.docx'}
        self.documents = []
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported."""
        return Path(filename).suffix.lower() in self.supported_formats
    
    def load_pdf(self, file_path: str) -> str:
        """Load and extract text from PDF."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return ""
    
    def load_text(self, file_path: str) -> str:
        """Load text from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading text file: {e}")
            return ""
    
    def load_document(self, file_path: str) -> Optional[str]:
        """Load document based on file type."""
        file_path = str(file_path)
        
        if not self.is_supported_format(file_path):
            print(f"Unsupported format: {file_path}")
            return None
        
        suffix = Path(file_path).suffix.lower()
        
        if suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.txt':
            return self.load_text(file_path)
        
        return None
    
    def process_documents(self, file_paths: List[str]) -> dict:
        """Process multiple documents and return combined content."""
        combined_text = ""
        metadata = []
        
        for file_path in file_paths:
            content = self.load_document(file_path)
            if content:
                combined_text += f"\n\n--- Document: {Path(file_path).name} ---\n\n{content}"
                metadata.append({
                    'filename': Path(file_path).name,
                    'size': len(content),
                    'format': Path(file_path).suffix
                })
        
        return {
            'content': combined_text,
            'metadata': metadata,
            'total_documents': len(metadata)
        }
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks for processing."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
