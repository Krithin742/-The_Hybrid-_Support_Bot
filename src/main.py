"""
Main Application
Interactive query interface for the RAG system
"""

import sys
from query_system import QuerySystem


def print_banner():
    """Print application banner"""
    print("\n" + "="*70)
    print("  🤖 HYBRID SUPPORT BOT - Advanced RAG System")
    print("="*70)
    print("  Intelligent Q&A with Metadata-Aware Retrieval")
    print("="*70 + "\n")


def print_help():
    """Print help information"""
    print("\nAvailable commands:")
    print("  • Type your question and press Enter")
    print("  • 'stats' - Show system statistics")
    print("  • 'help' - Show this help message")
    print("  • 'quit' or 'exit' - Exit the application")
    print()


def main():
    """
    Main application entry point
    """
    print_banner()
    
    # Initialize query system
    try:
        query_system = QuerySystem()
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("\nMake sure you've run the ingestion pipeline:")
        print("  python src/ingest.py")
        sys.exit(1)
    
    # Check if we have documents
    stats = query_system.get_stats()
    if stats['total_chunks'] == 0:
        print("❌ No documents found in vector store!")
        print("\nPlease run the ingestion pipeline first:")
        print("  python src/ingest.py")
        sys.exit(1)
    
    # Display initial stats
    print("📊 System Status:")
    print(f"  • Documents loaded: {stats['total_chunks']} chunks")
    print(f"  • Chapters available: {stats['unique_chapters']}")
    print(f"  • Chapters: {', '.join(stats['chapters'][:5])}")
    if len(stats['chapters']) > 5:
        print(f"             ... and {len(stats['chapters']) - 5} more")
    
    print("\n💡 Tip: Mention a chapter name in your question for filtered search!")
    print("   Example: 'How do I troubleshoot issues in the Troubleshooting chapter?'")
    
    print_help()
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("❓ Your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Hybrid Support Bot!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'stats':
                stats = query_system.get_stats()
                print(f"\n📊 System Statistics:")
                print(f"  • Total chunks: {stats['total_chunks']}")
                print(f"  • Unique chapters: {stats['unique_chapters']}")
                print(f"  • Chapters: {', '.join(stats['chapters'])}")
                print()
                continue
            
            # Process query
            result = query_system.answer_question(user_input)
            
            # Option to continue
            print("\n" + "─"*70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting...")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for assistance.\n")


def demo_mode():
    """
    Run predefined demo queries
    """
    print_banner()
    print("🎬 DEMO MODE - Running Predefined Queries")
    print("="*70 + "\n")
    
    query_system = QuerySystem()
    
    # Check if we have documents
    stats = query_system.get_stats()
    if stats['total_chunks'] == 0:
        print("❌ No documents in vector store. Run ingestion first:")
        print("  python src/ingest.py")
        return
    
    # Demo queries - customize these based on your PDF
    demo_queries = [
        "What is this manual about?",
        "How do I perform basic maintenance?",
        "What are the safety precautions?",
        "Tell me about troubleshooting steps",
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*70}")
        print(f"DEMO QUERY {i}/{len(demo_queries)}")
        print(f"{'='*70}")
        
        query_system.answer_question(query)
        
        if i < len(demo_queries):
            input("\nPress Enter to continue to next query...")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    # Check for demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    else:
        main()