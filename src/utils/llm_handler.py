import os
import time
from typing import Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


class LLMHandler:
    """Handle interactions with Google Gemini LLM using LangChain."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize LangChain ChatGoogleGenerativeAI
        self.model = model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self.api_key,
            temperature=0.7,
            max_output_tokens=500,
            timeout=60
        )
        self.last_request_time = 0
        self.min_request_interval = 1
    
    def _rate_limit(self):
        """Rate limiting to avoid quota exhaustion."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def chat(self, messages: List[dict], model: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Send chat message to Gemini and get response using LangChain."""
        try:
            self._rate_limit()
            
            # Use provided model or default
            if model:
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=self.api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    google_api_key=self.api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                role = msg.get('role', 'user').lower()
                content = msg.get('content', '')
                
                if role == 'system':
                    langchain_messages.append(SystemMessage(content=content))
                elif role == 'user':
                    langchain_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    langchain_messages.append(HumanMessage(content=content))
            
            # If no messages, use the last message as fallback
            if not langchain_messages:
                prompt = messages[-1]['content'] if messages else "Hello"
                langchain_messages = [HumanMessage(content=prompt)]
            
            # Invoke the model
            response = llm.invoke(langchain_messages)
            
            if response.content:
                return response.content
            else:
                return "I couldn't generate a response. Please try again."
        
        except Exception as e:
            error_msg = str(e)
            print(f"Error in chat request: {error_msg}")
            
            if "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                return "⚠️ API quota exceeded. Please wait a moment and try again."
            elif "429" in error_msg:
                return "⚠️ Too many requests. Please wait a moment and try again."
            else:
                return f"Error: {error_msg}"
    
    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text based on a prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature)
    
    def answer_question(self, context: str, question: str) -> str:
        """Answer a question based on provided context using LangChain."""
        template = """Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        try:
            self._rate_limit()
            result = chain.invoke({"context": context, "question": question})
            return result.content
        except Exception as e:
            error_msg = str(e)
            print(f"Error in answer_question: {error_msg}")
            return f"Error: {error_msg}"
    
    def summarize(self, text: str, max_tokens: int = 300) -> str:
        """Summarize provided text using LangChain."""
        prompt = f"Please summarize the following text in a concise manner:\n\n{text}"
        
        llm_with_tokens = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=0.7,
            max_output_tokens=max_tokens
        )
        
        try:
            self._rate_limit()
            result = llm_with_tokens.invoke([HumanMessage(content=prompt)])
            return result.content
        except Exception as e:
            error_msg = str(e)
            print(f"Error in summarize: {error_msg}")
            return f"Error: {error_msg}"
