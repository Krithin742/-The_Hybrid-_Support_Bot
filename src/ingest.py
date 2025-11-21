"""
Ingestion Pipeline
One-time script to process PDF and populate vector store
"""

import sys
import os
from pathlib import Path

# Add src directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_parser import PDFParser
from vector_store import VectorStore


def main():
    """
    Main ingestion pipeline
    """
    # Configuration
    pdf_path = "data/manual.pdf"
    
    print("\n" + "="*60)
    print("  📄 PDF INGESTION PIPELINE")
    print("="*60 + "\n")
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF not found at {pdf_path}")
        print(f"\n💡 Please ensure:")
        print(f"   1. Create a 'data' folder in your project root")
        print(f"   2. Place your PDF file as: {pdf_path}")
        print(f"\n📂 Current directory: {os.getcwd()}")
        print(f"📂 Looking for PDF at: {os.path.abspath(pdf_path)}")
        sys.exit(1)
    
    print(f"✅ PDF Location: {pdf_path}")
    file_size_mb = os.path.getsize(pdf_path) / 1024 / 1024
    print(f"📊 PDF Size: {file_size_mb:.2f} MB\n")
    
    # Step 1: Parse PDF
    print("="*60)
    print("STEP 1: PARSING PDF")
    print("="*60 + "\n")
    
    try:
        parser = PDFParser(pdf_path)
        chunks = parser.extract_chunks_with_metadata()
    except Exception as e:
        print(f"❌ ERROR: Failed to parse PDF")
        print(f"   Error details: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Ensure PDF is not corrupted")
        print(f"   2. Check if pdfplumber is installed: pip install pdfplumber")
        print(f"   3. Try opening the PDF manually to verify it's valid")
        sys.exit(1)
    
    if not chunks:
        print("❌ ERROR: No chunks extracted from PDF")
        print("\n💡 Possible reasons:")
        print("   1. PDF might be image-based (scanned) without text")
        print("   2. PDF might be encrypted or password-protected")
        print("   3. PDF format might be unsupported")
        sys.exit(1)
    
    # Display statistics
    try:
        chapters = set(chunk.metadata['chapter'] for chunk in chunks)
        total_chars = sum(len(c.text) for c in chunks)
        avg_chunk_size = total_chars / len(chunks)
        
        print(f"\n📊 Extraction Statistics:")
        print(f"  • Total chunks: {len(chunks)}")
        print(f"  • Unique chapters: {len(chapters)}")
        print(f"  • Average chunk size: {avg_chunk_size:.0f} chars")
        print(f"  • Total text extracted: {total_chars:,} characters")
        
        print(f"\n📚 Chapters found:")
        chapter_counts = {}
        for c in chunks:
            ch = c.metadata['chapter']
            chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
        
        for chapter in sorted(chapter_counts.keys()):
            count = chapter_counts[chapter]
            print(f"  • {chapter}: {count} chunks")
    
    except Exception as e:
        print(f"⚠️  Warning: Could not display statistics: {e}")
    
    # Step 2: Create vector embeddings
    print(f"\n{'='*60}")
    print("STEP 2: CREATING VECTOR EMBEDDINGS")
    print("="*60 + "\n")
    
    try:
        print("🔧 Initializing Vector Store...")
        vector_store = VectorStore()
        
        print("💾 Adding documents to vector store...")
        vector_store.add_documents(chunks)
        
        # Verify storage
        print("\n✅ Verifying storage...")
        stats = vector_store.get_collection_stats()
        
        print(f"\n📊 Vector Store Statistics:")
        print(f"  • Total chunks stored: {stats['total_chunks']}")
        print(f"  • Unique chapters: {stats['unique_chapters']}")
        
        if stats['total_chunks'] != len(chunks):
            print(f"\n⚠️  Warning: Expected {len(chunks)} chunks but stored {stats['total_chunks']}")
        else:
            print(f"  • ✅ All chunks stored successfully!")
    
    except Exception as e:
        print(f"\n❌ ERROR: Failed to create vector embeddings")
        print(f"   Error details: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check if ChromaDB is installed: pip install chromadb")
        print(f"   2. Check if sentence-transformers is installed: pip install sentence-transformers")
        print(f"   3. Ensure you have enough disk space")
        print(f"   4. Try deleting the 'data/chroma_db' folder and running again")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Success
    print(f"\n{'='*60}")
    print("✅ INGESTION COMPLETE!")
    print("="*60)
    print("\n🎉 Your RAG system is ready!")
    print("\n📝 Next steps:")
    print("   1. Run queries: python src/main.py")
    print("   2. Or run demo: python src/demo_queries.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)