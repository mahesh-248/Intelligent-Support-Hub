"""
Intelligent Support Hub - Main Application

This is the main entry point for the Intelligent Support Ticket Triage System.
Demonstrates BigQuery's vector search capabilities for semantic ticket matching.

Usage:
    # Full setup (first time)
    python main.py --setup
    
    # Search for similar tickets
    python main.py --search "my app keeps crashing"
    
    # Get solution recommendation
    python main.py --recommend "cannot log into account"
    
    # Auto-categorize a ticket
    python main.py --categorize "payment declined at checkout"
    
    # Run batch triage
    python main.py --batch
    
    # Find trending issues
    python main.py --trending
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from src.utils.bq import BigQueryClient, resolve_embedding_model_identifier
from src.orchestrator import TicketTriageOrchestrator
from src.data_loader import TicketDataGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_bigquery_infrastructure():
    """
    Complete setup of BigQuery infrastructure:
    1. Create dataset
    2. Create tables
    3. Load sample data
    4. Generate embeddings
    5. Optionally create vector index
    """
    print("\n" + "="*70)
    print("🚀 Setting up Intelligent Support Hub Infrastructure")
    print("="*70 + "\n")
    
    try:
        # Initialize BigQuery client
        print("📊 Initializing BigQuery client...")
        bq_client = BigQueryClient()
        print(f"✅ Connected to project: {bq_client.project_id}")
        print(f"✅ Dataset: {bq_client.dataset_id}\n")
        
        # Step 1: Create dataset
        print("📁 Creating BigQuery dataset...")
        bq_client.create_dataset(exists_ok=True)
        print(f"✅ Dataset '{bq_client.dataset_id}' ready\n")
        
        # Step 2: Create tickets table
        print("📋 Creating support tickets table...")
        table_name = os.getenv('BQ_TABLE_TICKETS', 'tickets')
        
        sql_file = 'sql/1_create_enriched_table.sql'
        bq_client.execute_sql_file(sql_file, table_name=table_name)
        print(f"✅ Table '{table_name}' created\n")
        
        # Step 3: Generate and load sample data
        print("🎲 Generating sample support tickets...")
        generator = TicketDataGenerator()
        tickets_df = generator.generate_tickets(count=5000)
        
        print(f"✅ Generated {len(tickets_df)} tickets")
        print(f"   Categories: {tickets_df['category'].nunique()}")
        print(f"   Resolved: {(tickets_df['status'] == 'resolved').sum()}")
        print(f"   With solutions: {tickets_df['resolution'].notna().sum()}\n")
        
        # Step 4: Load data to BigQuery
        print("📤 Loading tickets to BigQuery...")
        bq_client.load_dataframe_to_table(
            tickets_df, 
            table_name, 
            write_disposition="WRITE_TRUNCATE"
        )
        row_count = bq_client.get_table_row_count(table_name)
        print(f"✅ Loaded {row_count} tickets to BigQuery\n")
        
        # Step 4.5: Determine embedding model strategy (remote model optional)
        print("🤖 Preparing embedding model reference...")
        env_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-004').strip()
        embedding_model, requires_remote_model = resolve_embedding_model_identifier(
            bq_client.project_id,
            env_model
        )

        # Check if remote model already exists when required
        if requires_remote_model:
            print("   Remote model pattern detected. Checking existence...")
            check_sql = f"""
            SELECT COUNT(*) AS c
            FROM `{bq_client.project_id}.{bq_client.dataset_id}.INFORMATION_SCHEMA.MODELS`
            WHERE model_name = 'textembedding_model'
            """
            try:
                exists_df = bq_client.query_to_dataframe(check_sql)
                model_exists = int(exists_df['c'].iloc[0]) > 0
            except Exception:
                model_exists = False

            if not model_exists:
                print("   Model not found. Attempting creation via inline SQL...")
                create_sql = f"""
                CREATE OR REPLACE MODEL `{bq_client.project_id}.{bq_client.dataset_id}.textembedding_model`
                REMOTE WITH CONNECTION `{bq_client.project_id}.US.vertex-ai`
                OPTIONS (
                    ENDPOINT = 'text-embedding-004'
                )
                """
                try:
                    bq_client.execute_query(create_sql)
                    print("   ✅ Remote embedding model created\n")
                except Exception as remote_err:
                    print(f"   ⚠️ Remote model creation failed: {remote_err}")
                    print("   → Falling back to public model 'text-embedding-004'\n")
                    embedding_model = 'text-embedding-004'
                    requires_remote_model = False
            else:
                print("   ✅ Remote embedding model already exists\n")
        else:
            print("   Using public embedding model (no dataset required)\n")

        print(f"   Final embedding model: {embedding_model}\n")

        # Step 5: Generate embeddings using resolved embedding_model
        print("🧠 Generating vector embeddings (this may take a few minutes)...")
        embeddings_table = os.getenv('BQ_TABLE_TICKETS_EMBEDDINGS', 'tickets_with_embeddings')
        embedding_sql = f"""
CREATE OR REPLACE TABLE `{bq_client.project_id}.{bq_client.dataset_id}.{embeddings_table}` AS
WITH embeddings AS (
  SELECT
    content,
    ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `{embedding_model}`,
    (
      SELECT ticket_id, combined_text AS content
      FROM `{bq_client.project_id}.{bq_client.dataset_id}.{table_name}`
    )
  )
)
SELECT
  t.*,
  e.embedding
FROM `{bq_client.project_id}.{bq_client.dataset_id}.{table_name}` t
JOIN embeddings e ON t.combined_text = e.content
;
"""
        bq_client.execute_query(embedding_sql)

        embedding_count = bq_client.get_table_row_count(embeddings_table)
        print(f"✅ Generated embeddings for {embedding_count} tickets")
        print(f"   Model: {embedding_model}\n")
        
        # Step 6: Optionally create vector index
        use_index = os.getenv('USE_VECTOR_INDEX', 'false').lower() == 'true'
        
        if use_index:
            print("📊 Creating vector index for optimized search...")
            print("   (This is optional and recommended for datasets > 1M rows)")
            
            sql_file = 'sql/3_create_vector_index.sql'
            try:
                bq_client.execute_sql_file(
                    sql_file,
                    index_name='ticket_embeddings_idx',
                    embeddings_table_name=embeddings_table,
                    embedding_model=embedding_model
                )
                print("✅ Vector index created\n")
            except Exception as e:
                print(f"⚠️  Vector index creation skipped: {e}\n")
        else:
            print("ℹ️  Vector index creation skipped (USE_VECTOR_INDEX=false)")
            print("   For large datasets, enable this for faster searches\n")
        
        # Success summary
        print("="*70)
        print("✅ Setup Complete!")
        print("="*70)
        print(f"""
🎉 Your Intelligent Support Hub is ready to use!

📊 BigQuery Resources Created:
   • Dataset: {bq_client.project_id}.{bq_client.dataset_id}
   • Tickets Table: {table_name} ({row_count} rows)
   • Embeddings Table: {embeddings_table} ({embedding_count} rows)
   • Embedding Model: {embedding_model}

🚀 Next Steps:
   1. Search for similar tickets:
      python main.py --search "my app keeps crashing"
   
   2. Get solution recommendations:
      python main.py --recommend "cannot log into account"
   
   3. Run batch triage:
      python main.py --batch
   
   4. Run smoke tests:
      python tests/smoke.py

📚 For more info: See README.md
""")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("\nPlease check:")
        print("  1. GCP_PROJECT_ID is set correctly in .env")
        print("  2. BigQuery API is enabled in your GCP project")
        print("  3. You have proper authentication configured")
        print("  4. Your service account has BigQuery Admin permissions")
        return False


def search_similar_tickets(query: str, top_k: int = 1):
    """Search for similar tickets."""
    print(f"\n🔍 Searching for tickets similar to: '{query}'")
    print("="*70 + "\n")
    
    try:
        orchestrator = TicketTriageOrchestrator()
        results = orchestrator.find_similar_tickets(query, top_k=top_k)
        
        if not results:
            print("No similar tickets found.")
            return
        
        for i, ticket in enumerate(results, 1):
            print(f"#{i} - Similarity: {ticket['similarity_score']:.1f}%")
            print(f"    Title: {ticket['title']}")
            print(f"    Category: {ticket['category']} | Priority: {ticket['priority']}")
            print(f"    Status: {ticket['status']}")
            
            if ticket.get('resolution'):
                print(f"    Solution: {ticket['resolution'][:100]}...")
            
            print()
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"❌ Search failed: {e}")


def recommend_solution(query: str):
    """Get solution recommendation for a ticket."""
    print(f"\n💡 Getting solution recommendation for: '{query}'")
    print("="*70 + "\n")
    
    try:
        orchestrator = TicketTriageOrchestrator()
        recommendation = orchestrator.recommend_solution(query)
        
        if not recommendation['solution']:
            print("❌ No recommendations found. Not enough similar resolved tickets.")
            return
        
        print(f"🎯 Recommended Solution:")
        print(f"   {recommendation['solution']}\n")
        
        print(f"📊 Confidence Metrics:")
        print(f"   • Confidence Score: {recommendation['confidence']:.2f}/100")
        print(f"   • Based on {recommendation['similar_ticket_count']} similar tickets")
        print(f"   • Average Satisfaction: {recommendation['avg_satisfaction']:.1f}/5 ⭐")
        print(f"   • Average Resolution Time: {recommendation['avg_resolution_hours']:.1f} hours")
        print(f"   • Average Similarity: {recommendation['avg_similarity_pct']:.1f}%")
        
        # Handle example tickets safely
        try:
            examples = recommendation.get('example_tickets')
            if examples is not None and (isinstance(examples, list) and len(examples) > 0 or hasattr(examples, '__len__') and len(examples) > 0):
                print(f"\n📋 Example Similar Tickets:")
                if not isinstance(examples, list):
                    examples = list(examples) if hasattr(examples, '__iter__') else []
                for i, ex in enumerate(examples[:3], 1):
                    if isinstance(ex, dict):
                        print(f"   {i}. {ex.get('title', 'N/A')} ({ex.get('similarity_pct', 0):.1f}% similar)")
                    else:
                        # Handle pandas/numpy struct types
                        title = getattr(ex, 'title', 'N/A') if hasattr(ex, 'title') else 'N/A'
                        sim_pct = getattr(ex, 'similarity_pct', 0) if hasattr(ex, 'similarity_pct') else 0
                        print(f"   {i}. {title} ({sim_pct:.1f}% similar)")
        except Exception as ex_err:
            logger.debug(f"Could not display example tickets: {ex_err}")
        
        
        
        print()
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        print(f"❌ Recommendation failed: {e}")


def categorize_ticket(query: str):
    """Auto-categorize a ticket."""
    print(f"\n🏷️  Auto-categorizing ticket: '{query}'")
    print("="*70 + "\n")
    
    try:
        orchestrator = TicketTriageOrchestrator()
        result = orchestrator.auto_categorize(query)
        
        print(f"📂 Predicted Category: {result['category']}")
        print(f"   • Confidence: {result['category_confidence']:.1f}%")
        print(f"   • Based on {result['category_votes']} similar tickets")
        print(f"   • Average Similarity: {result['category_similarity']:.1f}%")
        
        print(f"\n⚡ Suggested Priority: {result['priority']}")
        print(f"   • Based on {result['priority_votes']} similar tickets")
        print(f"   • Average Similarity: {result['priority_similarity']:.1f}%")
        
        print()
        
    except Exception as e:
        logger.error(f"Categorization failed: {e}")
        print(f"❌ Categorization failed: {e}")


def batch_triage():
    """Run batch triage on sample tickets."""
    print("\n📦 Running Batch Triage on Sample Tickets")
    print("="*70 + "\n")
    
    try:
        # Get sample tickets
        generator = TicketDataGenerator()
        sample_tickets = generator.get_sample_new_tickets()
        
        orchestrator = TicketTriageOrchestrator()
        results = orchestrator.batch_triage(sample_tickets, include_recommendations=True)
        
        for i, result in enumerate(results, 1):
            if 'error' in result:
                print(f"❌ Ticket {i} failed: {result['error']}\n")
                continue
            print(f"📌 Ticket {i}: {result['original_text'][:60]}...")
            print(f"   Predicted Category: {result['category']} ({result['category_confidence']:.1f}% confidence)")
            print(f"   Priority: {result['priority']} ({result['priority_confidence']:.1f}% confidence)")
            if result.get('recommendation') and result['recommendation'].get('solution'):
                rec = result['recommendation']
                print(f"   Recommended Solution (score {rec['confidence']:.1f}): {rec['solution'][:80]}...")
            print()
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        print(f"❌ Trend analysis failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Intelligent Support Hub - Semantic Ticket Triage System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                          # Initial setup
  python main.py --search "app crashing"          # Find similar tickets
  python main.py --recommend "login error"        # Get solution
  python main.py --categorize "payment failed"    # Auto-categorize
  python main.py --batch                          # Batch triage
  python main.py --trending --days 7              # Trending issues
        """
    )
    
    parser.add_argument('--setup', action='store_true', help='Setup BigQuery infrastructure')
    parser.add_argument('--search', type=str, help='Search for similar tickets')
    parser.add_argument('--recommend', type=str, help='Get solution recommendation')
    parser.add_argument('--categorize', type=str, help='Auto-categorize a ticket')
    parser.add_argument('--batch', action='store_true', help='Run batch triage demo')
    parser.add_argument('--trending', action='store_true', help='Find trending issues')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results (default: 5)')
    parser.add_argument('--days', type=int, default=7, help='Days for trending analysis (default: 7)')
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Execute requested action
    if args.setup:
        success = setup_bigquery_infrastructure()
        sys.exit(0 if success else 1)
    
    elif args.search:
        search_similar_tickets(args.search, top_k=args.top_k)
    
    elif args.recommend:
        recommend_solution(args.recommend)
    
    elif args.categorize:
        categorize_ticket(args.categorize)
    
    elif args.batch:
        batch_triage()
    
    elif args.trending:
        print("⚠️ Trending issues feature temporarily unavailable in this revision.")


if __name__ == "__main__":
    main()
