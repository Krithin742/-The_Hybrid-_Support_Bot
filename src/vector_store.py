"""
Vector Store Module
Handles document embeddings and similarity search with metadata filtering
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import time
from pdf_parser import DocumentChunk


class VectorStore:
    """
    Manages vector embeddings and similarity search with metadata support
    """
    
    def __init__(self, collection_name: str = "manual_chunks"):
        """
        Initialize vector store with ChromaDB
        
        Args:
            collection_name: Name for the document collection
        """
        print("🔧 Initializing Vector Store...")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path="./data/chroma_db"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load embedding model
        print("📦 Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Vector Store initialized")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to vector store with embeddings
        
        Args:
            chunks: List of DocumentChunk objects to add
        """
        if not chunks:
            print("⚠️  No chunks to add")
            return
        
        print(f"\n{'='*60}")
        print(f"Adding {len(chunks)} chunks to vector store...")
        print(f"{'='*60}")
        
        # Extract data from chunks
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        start_time = time.time()
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        embedding_time = time.time() - start_time
        print(f"✓ Embeddings generated in {embedding_time:.2f}s")
        
        # Add to ChromaDB in batches
        print("💾 Storing in vector database...")
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            print(f"  ✓ Batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        print(f"\n✅ Successfully added {len(chunks)} chunks to vector store")
        print(f"{'='*60}\n")
    
    def search(
        self,
        query: str,
        chapter_filter: Optional[str] = None,
        top_k: int = 3
    ) -> Dict:
        """
        Search for relevant chunks with optional metadata filtering
        
        Args:
            query: Search query
            chapter_filter: Optional chapter name to filter by
            top_k: Number of results to return
            
        Returns:
            Dictionary with documents, metadata, and retrieval time
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Build metadata filter
        where_filter = None
        if chapter_filter:
            where_filter = {"chapter": {"$eq": chapter_filter}}
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter
        )
        
        retrieval_time = time.time() - start_time
        
        # Handle case where no results found
        if not results['documents'][0]:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "retrieval_time": retrieval_time
            }
        
        return {
            "documents": results['documents'][0],
            "metadatas": results['metadatas'][0],
            "distances": results.get('distances', [[]])[0],
            "retrieval_time": retrieval_time
        }
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get unique chapters
        all_items = self.collection.get()
        chapters = set()
        if all_items['metadatas']:
            chapters = set(meta.get('chapter', 'Unknown') 
                          for meta in all_items['metadatas'])
        
        return {
            "total_chunks": count,
            "unique_chapters": len(chapters),
            "chapters": sorted(list(chapters))
        }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection
        """
        self.client.delete_collection(self.collection.name)
        print(f"✓ Cleared collection: {self.collection.name}")


# Test function
if __name__ == "__main__":
    # Test vector store
    vector_store = VectorStore()
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"\nCollection Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique chapters: {stats['unique_chapters']}")
    print(f"  Chapters: {stats['chapters']}")