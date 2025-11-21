"""
LLM Interface Module
Handles interaction with Groq API for answer generation (FREE & FAST)
"""

import os
import time
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMInterface:
    """
    Interface for Groq LLM (Free & Fast)
    """
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize LLM interface with Groq
        
        Args:
            model_name: Name of the Groq model to use
            Available models:
            - llama-3.1-8b-instant (fastest, recommended)
            - llama-3.3-70b-versatile (more powerful)
            - mixtral-8x7b-32768 (good for long context)
        """
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Please create a .env file with your Groq API key.\n"
                "Get a free key at: https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=api_key)
        print(f"🤖 LLM Interface initialized with model: {model_name}")
        print("   Using Groq API (FREE & FAST) 🚀")
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        metadatas: List[Dict]
    ) -> Dict:
        """
        Generate answer using retrieved context
        
        Args:
            query: User's question
            context_chunks: List of relevant text chunks
            metadatas: List of metadata for each chunk
            
        Returns:
            Dictionary with answer, generation time, and context used
        """
        start_time = time.time()
        
        # Handle case with no context
        if not context_chunks:
            return {
                "answer": "I don't have enough information to answer that question. The retrieved context doesn't contain relevant information.",
                "generation_time": 0.0,
                "context_used": [],
                "model": self.model_name
            }
        
        # Build context with metadata
        context = self._build_context(context_chunks, metadatas)
        
        # Create messages for chat completion
        messages = [
            {
                "role": "system",
                "content": """You are a helpful technical support assistant. Your job is to answer questions about a technical manual based ONLY on the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the context provided
2. If the context doesn't contain the answer, say "I don't have enough information in the manual to answer that question."
3. DO NOT make up information or use knowledge outside the provided context
4. Be concise and specific
5. Reference the chapter or section when relevant
6. If multiple sources are provided, synthesize the information clearly"""
            },
            {
                "role": "user",
                "content": f"""CONTEXT FROM MANUAL:
{context}

USER QUESTION: {query}

Please answer based only on the context above."""
            }
        ]
        
        # Generate response using Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            answer = f"Error generating answer: {str(e)}"
        
        generation_time = time.time() - start_time
        
        return {
            "answer": answer,
            "generation_time": generation_time,
            "context_used": context_chunks,
            "model": self.model_name
        }
    
    def _build_context(
        self,
        chunks: List[str],
        metadatas: List[Dict]
    ) -> str:
        """
        Format context with metadata for the prompt
        
        Args:
            chunks: List of text chunks
            metadatas: List of metadata dictionaries
            
        Returns:
            Formatted context string
        """
        formatted_contexts = []
        
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
            chapter = meta.get('chapter', 'Unknown')
            page = meta.get('page', 'Unknown')
            
            formatted_contexts.append(
                f"[Source {i} - Chapter: {chapter}, Page: {page}]\n{chunk}"
            )
        
        return "\n\n---\n\n".join(formatted_contexts)
    
    def test_connection(self) -> bool:
        """
        Test if Groq API is accessible
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test to verify API key works
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print(f"✅ Connected to Groq API successfully")
            print(f"   Model: {self.model_name}")
            return True
                
        except Exception as e:
            print(f"❌ Cannot connect to Groq API: {e}")
            print("\nPlease check:")
            print("  1. Your API key is correct in .env file")
            print("  2. Get a free key at: https://console.groq.com/keys")
            print("  3. Your internet connection is working")
            return False


# Test function
if __name__ == "__main__":
    print("Testing LLM Interface...\n")
    print("="*60)
    
    try:
        llm = LLMInterface()
        
        # Test connection
        print("\n" + "="*60)
        print("Testing Connection...")
        print("="*60)
        
        if llm.test_connection():
            print("\n" + "="*60)
            print("✅ LLM Interface is working!")
            print("="*60)
            
            # Test generation
            print("\n" + "="*60)
            print("Testing Answer Generation...")
            print("="*60)
            
            result = llm.generate_answer(
                query="What is this manual about?",
                context_chunks=["This is a test manual about troubleshooting computer systems. It covers basic diagnostics, hardware issues, and software problems."],
                metadatas=[{"chapter": "Introduction", "page": 1}]
            )
            
            print(f"\n💡 Answer:\n{result['answer']}")
            print(f"\n⚡ Generation time: {result['generation_time']:.2f}s")
            print("="*60)
        else:
            print("\n❌ LLM Interface test failed")
            
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")