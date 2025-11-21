"""
Query System Module
Main query pipeline combining retrieval and generation
"""

from vector_store import VectorStore
from llm_interface import LLMInterface
import re
from typing import Optional, Dict


class QuerySystem:
    """
    Complete query answering system with metadata-aware retrieval
    """
    
    def __init__(self):
        """
        Initialize query system with vector store and LLM
        """
        print("\n🚀 Initializing Query System...")
        self.vector_store = VectorStore()
        self.llm = LLMInterface()
        
        # Test LLM connection
        if not self.llm.test_connection():
            print("\n⚠️  Warning: LLM not available. Retrieval will work but answer generation will fail.")
        
        print("✅ Query System ready!\n")
    
    def answer_question(self, query: str, verbose: bool = True) -> Dict:
        """
        Answer a question using RAG pipeline
        
        Args:
            query: User's question
            verbose: Whether to print detailed logs
            
        Returns:
            Dictionary with answer and performance metrics
        """
        if verbose:
            print("\n" + "="*70)
            print(f"❓ QUERY: {query}")
            print("="*70)
        
        # Detect chapter filter from query
        chapter_filter = self._extract_chapter_mention(query)
        
        if chapter_filter and verbose:
            print(f"\n🔍 Detected chapter filter: '{chapter_filter}'")
            print(f"   → Will search only in '{chapter_filter}' chapter")
        
        # Step 1: Retrieve relevant chunks
        if verbose:
            print(f"\n{'─'*70}")
            print("STEP 1: RETRIEVAL")
            print("─"*70)
        
        search_results = self.vector_store.search(
            query=query,
            chapter_filter=chapter_filter,
            top_k=3
        )
        
        if verbose:
            print(f"✓ Retrieval completed in {search_results['retrieval_time']:.3f}s")
            print(f"✓ Retrieved {len(search_results['documents'])} chunks")
            
            if search_results['documents']:
                print(f"\n📚 Retrieved Sources:")
                for i, meta in enumerate(search_results['metadatas'], 1):
                    chapter = meta.get('chapter', 'Unknown')
                    page = meta.get('page', 'Unknown')
                    print(f"  {i}. Chapter: '{chapter}' (Page {page})")
            else:
                print("\n⚠️  No relevant chunks found!")
        
        # Step 2: Generate answer with LLM
        if verbose:
            print(f"\n{'─'*70}")
            print("STEP 2: GENERATION")
            print("─"*70)
        
        llm_response = self.llm.generate_answer(
            query=query,
            context_chunks=search_results['documents'],
            metadatas=search_results['metadatas']
        )
        
        if verbose:
            print(f"✓ Generation completed in {llm_response['generation_time']:.3f}s")
        
        # Performance metrics
        total_time = (search_results['retrieval_time'] + 
                     llm_response['generation_time'])
        
        if verbose:
            print(f"\n{'─'*70}")
            print("⚡ PERFORMANCE METRICS")
            print("─"*70)
            print(f"  Retrieval Latency:  {search_results['retrieval_time']:.3f}s")
            print(f"  Generation Latency: {llm_response['generation_time']:.3f}s")
            print(f"  Total Time:         {total_time:.3f}s")
            print("─"*70)
        
        # Display answer
        if verbose:
            print(f"\n💡 ANSWER:")
            print("─"*70)
            print(llm_response['answer'])
            print("="*70 + "\n")
        
        # Return complete results
        return {
            "query": query,
            "answer": llm_response['answer'],
            "retrieval_time": search_results['retrieval_time'],
            "generation_time": llm_response['generation_time'],
            "total_time": total_time,
            "sources": search_results['metadatas'],
            "chapter_filter": chapter_filter
        }
    
    def _extract_chapter_mention(self, query: str) -> Optional[str]:
        """
        Detect if user mentions a specific chapter/section
        
        Args:
            query: User's query
            
        Returns:
            Chapter name if detected, None otherwise
        """
        query_lower = query.lower()
        
        # Get all available chapters from vector store
        stats = self.vector_store.get_collection_stats()
        available_chapters = stats.get('chapters', [])
        
        # Check if any chapter name is mentioned in the query
        for chapter in available_chapters:
            if chapter.lower() in query_lower:
                return chapter
        
        # Check for common patterns
        patterns = [
            r'in (?:the )?([A-Z][a-z]+(?: [A-Z][a-z]+)*) (?:chapter|section)',
            r'(?:chapter|section) (?:on |about )?([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_chapter = match.group(1).strip().title()
                # Check if it matches any available chapter (case-insensitive)
                for chapter in available_chapters:
                    if chapter.lower() == potential_chapter.lower():
                        return chapter
        
        return None
    
    def get_stats(self) -> Dict:
        """
        Get system statistics
        
        Returns:
            Dictionary with system stats
        """
        return self.vector_store.get_collection_stats()


# Test function
if __name__ == "__main__":
    print("Testing Query System...\n")
    
    query_system = QuerySystem()
    
    # Display stats
    stats = query_system.get_stats()
    print(f"\n📊 System Statistics:")
    print(f"  • Total chunks: {stats['total_chunks']}")
    print(f"  • Unique chapters: {stats['unique_chapters']}")
    
    if stats['total_chunks'] == 0:
        print("\n⚠️  No documents in vector store!")
        print("Run ingestion first: python src/ingest.py")
    else:
        # Test query
        print("\n" + "="*70)
        print("Testing with sample query...")
        print("="*70)
        
        query_system.answer_question("What is this manual about?")