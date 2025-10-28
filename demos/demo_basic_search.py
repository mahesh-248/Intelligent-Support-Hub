"""
Basic Semantic Search Demo

Demonstrates how to use the Intelligent Support Hub to find
semantically similar support tickets using BigQuery vector search.

This demo shows:
1. Finding similar tickets based on meaning, not keywords
2. Understanding semantic similarity scores
3. Comparing keyword search vs semantic search
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from src.orchestrator import TicketTriageOrchestrator

load_dotenv()


def demo_basic_search():
    """Demonstrate basic semantic search."""
    
    print("\n" + "="*80)
    print("🔍 DEMO: Basic Semantic Search")
    print("="*80)
    
    print("""
This demo shows how BigQuery's vector search finds semantically similar tickets,
even when they use completely different words.

Traditional keyword search would miss these connections!
""")
    
    # Initialize orchestrator
    orchestrator = TicketTriageOrchestrator()
    
    # Test queries
    test_queries = [
        {
            'query': 'my application keeps crashing',
            'description': 'User reporting app crashes'
        },
        {
            'query': 'unable to authenticate',
            'description': 'Login/authentication issues'
        },
        {
            'query': 'extremely slow performance',
            'description': 'Performance problems'
        },
        {
            'query': 'charged incorrectly',
            'description': 'Billing/payment issues'
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}: \"{test['query']}\"")
        print(f"Context: {test['description']}")
        print(f"{'─'*80}\n")
        
        # Find similar tickets
        results = orchestrator.find_similar_tickets(test['query'], top_k=5)
        
        if results:
            print("✨ Most Similar Tickets:\n")
            
            for j, ticket in enumerate(results, 1):
                similarity = ticket['similarity_score']
                
                # Color code by similarity
                if similarity >= 80:
                    marker = "🟢"
                elif similarity >= 60:
                    marker = "🟡"
                else:
                    marker = "🟠"
                
                print(f"{marker} #{j} - {similarity:.1f}% Similar")
                print(f"   Title: {ticket['title']}")
                print(f"   Category: {ticket['category']} | Priority: {ticket['priority']}")
                print(f"   Status: {ticket['status']}")
                
                # Show resolution if available
                if ticket.get('resolution'):
                    resolution_preview = ticket['resolution'][:80]
                    print(f"   Solution: {resolution_preview}...")
                
                print()
        else:
            print("❌ No similar tickets found\n")
        
        input("Press Enter to continue to next query...")
    
    # Semantic vs Keyword comparison
    print("\n" + "="*80)
    print("🆚 SEMANTIC SEARCH vs KEYWORD SEARCH")
    print("="*80)
    
    comparison_query = "app won't launch"
    
    print(f"""
Let's compare semantic search vs traditional keyword search:

Query: "{comparison_query}"

KEYWORD SEARCH would look for:
  - Exact words: "app", "won't", "launch"
  - Would MISS: "application crashes on startup"
  - Would MISS: "program freezes when opening"
  
SEMANTIC SEARCH understands:
  - Intent: User wants to start the application
  - Problem: Application is not starting properly
  - Finds: Any ticket about startup/launch issues
""")
    
    print("\n🔍 Semantic Search Results:\n")
    results = orchestrator.find_similar_tickets(comparison_query, top_k=5)
    
    for i, ticket in enumerate(results, 1):
        print(f"{i}. {ticket['title']} ({ticket['similarity_score']:.1f}% similar)")
        print(f"   Notice: Different words, same meaning! ✨\n")
    
    print("\n" + "="*80)
    print("✅ Demo Complete!")
    print("="*80)
    print("""
Key Takeaways:
• Semantic search finds tickets based on MEANING, not keywords
• Works across different phrasings and synonyms
• More accurate than traditional keyword matching
• Powered by BigQuery ML.GENERATE_EMBEDDING and VECTOR_SEARCH
""")


if __name__ == "__main__":
    try:
        demo_basic_search()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have run: python main.py --setup")
