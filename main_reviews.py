"""
Amazon Fine Food Reviews - Semantic Search System

This is the main entry point for the Amazon Fine Food Reviews semantic search system.
Demonstrates BigQuery's vector search capabilities for finding similar product reviews.

Dataset: Amazon Fine Food Reviews (1999-2012)
- 568,454 reviews
- 256,059 users  
- 74,258 products

Usage:
    # Full setup (first time)
    python main.py --setup
    
    # Search for similar reviews
    python main.py --search "delicious chocolate candy"
    
    # Find reviews for a specific product
    python main.py --product B001E4KFG0
    
    # Find helpful reviews
    python main.py --helpful --min-score 4
    
    # Get product recommendations based on review similarity
    python main.py --recommend "healthy organic snacks"
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from src.utils.bq import BigQueryClient, resolve_embedding_model_identifier
from src.data_loader import ReviewDataLoader

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
    2. Create reviews table
    3. Load Amazon Fine Food Reviews data
    4. Generate embeddings
    5. Optionally create vector index
    """
    print("\n" + "="*70)
    print("🚀 Setting up Amazon Fine Food Reviews Semantic Search System")
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
        
        # Step 2: Load Amazon Fine Food Reviews
        print("📥 Loading Amazon Fine Food Reviews from Reviews.csv...")
        sample_size = int(os.getenv('REVIEW_SAMPLE_SIZE', '10000'))  # Default 10k for testing
        print(f"   Sample size: {sample_size if sample_size > 0 else 'ALL reviews'}")
        
        loader = ReviewDataLoader(
            csv_path="Reviews.csv",
            sample_size=sample_size if sample_size > 0 else None
        )
        reviews_df = loader.load_reviews()
        
        stats = loader.get_statistics(reviews_df)
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total Reviews: {stats['total_reviews']}")
        print(f"   Unique Products: {stats['unique_products']}")
        print(f"   Unique Users: {stats['unique_users']}")
        print(f"   Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"   Helpful Reviews: {stats['helpful_reviews']}\n")
        
        # Step 3: Create and load reviews table
        print("📋 Creating reviews table in BigQuery...")
        table_name = os.getenv('BQ_TABLE_REVIEWS', 'reviews')
        
        # Define schema for reviews table
        schema = [
            {"name": "review_id", "type": "STRING"},
            {"name": "product_id", "type": "STRING"},
            {"name": "user_id", "type": "STRING"},
            {"name": "profile_name", "type": "STRING"},
            {"name": "helpfulness_numerator", "type": "INTEGER"},
            {"name": "helpfulness_denominator", "type": "INTEGER"},
            {"name": "score", "type": "INTEGER"},
            {"name": "time_unix", "type": "INTEGER"},
            {"name": "summary", "type": "STRING"},
            {"name": "text", "type": "STRING"},
            {"name": "review_date", "type": "TIMESTAMP"},
            {"name": "helpfulness_ratio", "type": "FLOAT"},
            {"name": "combined_text", "type": "STRING"},
            {"name": "sentiment_category", "type": "STRING"},
            {"name": "is_helpful", "type": "BOOLEAN"},
            {"name": "text_length", "type": "INTEGER"},
            {"name": "summary_length", "type": "INTEGER"},
            {"name": "created_at", "type": "TIMESTAMP"},
            {"name": "updated_at", "type": "TIMESTAMP"},
        ]
        
        # Load data to BigQuery
        print("📤 Uploading reviews to BigQuery...")
        bq_client.load_dataframe_to_table(
            reviews_df,
            table_name,
            write_disposition="WRITE_TRUNCATE"
        )
        row_count = bq_client.get_table_row_count(table_name)
        print(f"✅ Loaded {row_count} reviews to BigQuery\n")
        
        # Step 4: Determine embedding model
        print("🤖 Preparing embedding model reference...")
        env_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-004').strip()
        embedding_model, requires_remote_model = resolve_embedding_model_identifier(
            bq_client.project_id,
            env_model
        )
        
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
                print("   Model not found. Attempting creation...")
                create_sql = f"""
                CREATE OR REPLACE MODEL `{bq_client.project_id}.{bq_client.dataset_id}.textembedding_model`
                REMOTE WITH CONNECTION `{bq_client.project_id}.US.vertex-ai`
                OPTIONS (ENDPOINT = 'text-embedding-004')
                """
                try:
                    bq_client.execute_query(create_sql)
                    print("   ✅ Remote embedding model created\n")
                except Exception as remote_err:
                    print(f"   ⚠️ Remote model creation failed: {remote_err}")
                    print("   → Falling back to public model\n")
                    embedding_model = 'text-embedding-004'
            else:
                print("   ✅ Remote embedding model exists\n")
        else:
            print("   Using public embedding model\n")

        print(f"   Final embedding model: {embedding_model}\n")

        # Step 5: Generate embeddings
        print("🧠 Generating vector embeddings for reviews...")
        print("   This may take several minutes depending on dataset size...")
        embeddings_table = os.getenv('BQ_TABLE_REVIEWS_EMBEDDINGS', 'reviews_with_embeddings')
        
        embedding_sql = f"""
CREATE OR REPLACE TABLE `{bq_client.project_id}.{bq_client.dataset_id}.{embeddings_table}` AS
WITH embeddings AS (
  SELECT
    content,
    ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `{embedding_model}`,
    (
      SELECT review_id, combined_text AS content
      FROM `{bq_client.project_id}.{bq_client.dataset_id}.{table_name}`
    )
  )
)
SELECT
  r.*,
  e.embedding
FROM `{bq_client.project_id}.{bq_client.dataset_id}.{table_name}` r
JOIN embeddings e ON r.combined_text = e.content
;
"""
        bq_client.execute_query(embedding_sql)
        
        embedding_count = bq_client.get_table_row_count(embeddings_table)
        print(f"✅ Generated embeddings for {embedding_count} reviews")
        print(f"   Model: {embedding_model}\n")
        
        # Step 6: Optionally create vector index
        use_index = os.getenv('USE_VECTOR_INDEX', 'false').lower() == 'true'
        
        if use_index:
            print("📊 Creating vector index for optimized search...")
            index_name = 'review_embeddings_idx'
            
            index_sql = f"""
CREATE OR REPLACE VECTOR INDEX `{index_name}`
ON `{bq_client.project_id}.{bq_client.dataset_id}.{embeddings_table}`(embedding)
OPTIONS(
  distance_type = 'COSINE',
  index_type = 'IVF',
  ivf_options = '{{"num_lists": 100}}'
)
"""
            try:
                bq_client.execute_query(index_sql)
                print("✅ Vector index created\n")
            except Exception as e:
                print(f"⚠️  Vector index creation skipped: {e}\n")
        else:
            print("ℹ️  Vector index creation skipped (USE_VECTOR_INDEX=false)\n")
        
        # Success summary
        print("="*70)
        print("✅ Setup Complete!")
        print("="*70)
        print(f"""
🎉 Your Amazon Fine Food Reviews Semantic Search System is ready!

📊 BigQuery Resources Created:
   • Dataset: {bq_client.project_id}.{bq_client.dataset_id}
   • Reviews Table: {table_name} ({row_count} rows)
   • Embeddings Table: {embeddings_table} ({embedding_count} rows)
   • Embedding Model: {embedding_model}

📈 Dataset Overview:
   • Total Reviews: {stats['total_reviews']:,}
   • Products: {stats['unique_products']:,}
   • Users: {stats['unique_users']:,}
   • Helpful Reviews: {stats['helpful_reviews']:,}

🚀 Next Steps:
   1. Search for similar reviews:
      python main.py --search "delicious chocolate candy"
   
   2. Find highly-rated products:
      python main.py --helpful --min-score 5
   
   3. Get product insights:
      python main.py --product B001E4KFG0
   
   4. Run demo searches:
      python demos/demo_basic_search.py

📚 For more info: See README.md
""")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\n📥 Please download the Amazon Fine Food Reviews dataset:")
        print("   URL: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews")
        print("   Place Reviews.csv in the project root directory")
        return False
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("\nPlease check:")
        print("  1. Reviews.csv exists in project root")
        print("  2. GCP_PROJECT_ID is set correctly in .env")
        print("  3. BigQuery API is enabled")
        print("  4. Authentication is configured properly")
        return False


def search_similar_reviews(query: str, top_k: int = 10):
    """Search for similar reviews using semantic search."""
    print(f"\n🔍 Searching for reviews similar to: '{query}'")
    print("="*70 + "\n")
    
    try:
        bq_client = BigQueryClient()
        embeddings_table = os.getenv('BQ_TABLE_REVIEWS_EMBEDDINGS', 'reviews_with_embeddings')
        env_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-004').strip()
        embedding_model, _ = resolve_embedding_model_identifier(bq_client.project_id, env_model)
        
        search_sql = f"""
WITH query_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `{embedding_model}`,
    (SELECT '{query}' AS content)
  )
)
SELECT
  base.review_id,
  base.product_id,
  base.score,
  base.summary,
  base.text,
  base.helpfulness_ratio,
  base.sentiment_category,
  distance AS similarity_score
FROM VECTOR_SEARCH(
  TABLE `{bq_client.project_id}.{bq_client.dataset_id}.{embeddings_table}`,
  'embedding',
  (SELECT embedding FROM query_embedding),
  top_k => {top_k},
  distance_type => 'COSINE'
)
ORDER BY similarity_score ASC
LIMIT {top_k};
"""
        
        results = bq_client.query_to_dataframe(search_sql)
        
        if len(results) == 0:
            print("No similar reviews found.")
            return
        
        print(f"Found {len(results)} similar reviews:\n")
        
        for i, row in results.iterrows():
            similarity_pct = (1 - row['similarity_score']) * 100
            print(f"{i+1}. {'⭐' * int(row['score'])} ({row['score']}/5) - Similarity: {similarity_pct:.1f}%")
            print(f"   Product: {row['product_id']}")
            print(f"   Summary: {row['summary']}")
            print(f"   Review: {row['text'][:200]}..." if len(row['text']) > 200 else f"   Review: {row['text']}")
            print(f"   Helpful: {row['helpfulness_ratio']:.0%} | {row['sentiment_category']}")
            print()
        
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"❌ Search failed: {e}")
        return None


def find_product_reviews(product_id: str, min_score: int = 1):
    """Find all reviews for a specific product."""
    print(f"\n📦 Finding reviews for product: {product_id}")
    print("="*70 + "\n")
    
    try:
        bq_client = BigQueryClient()
        table_name = os.getenv('BQ_TABLE_REVIEWS', 'reviews')
        
        query_sql = f"""
SELECT
  review_id,
  score,
  summary,
  text,
  helpfulness_ratio,
  sentiment_category,
  profile_name,
  review_date
FROM `{bq_client.project_id}.{bq_client.dataset_id}.{table_name}`
WHERE product_id = '{product_id}'
  AND score >= {min_score}
ORDER BY helpfulness_ratio DESC, score DESC
LIMIT 20;
"""
        
        results = bq_client.query_to_dataframe(query_sql)
        
        if len(results) == 0:
            print(f"No reviews found for product {product_id}")
            return
        
        print(f"Found {len(results)} reviews:\n")
        
        for i, row in results.iterrows():
            print(f"{i+1}. {'⭐' * int(row['score'])} ({row['score']}/5)")
            print(f"   By: {row['profile_name']} on {row['review_date'].strftime('%Y-%m-%d')}")
            print(f"   Summary: {row['summary']}")
            print(f"   Helpful: {row['helpfulness_ratio']:.0%} | {row['sentiment_category']}")
            print()
        
        return results
        
    except Exception as e:
        logger.error(f"Product search failed: {e}")
        print(f"❌ Search failed: {e}")
        return None


def find_helpful_reviews(min_score: int = 4, min_helpful_ratio: float = 0.8):
    """Find highly helpful and highly-rated reviews."""
    print(f"\n⭐ Finding helpful reviews (Score >= {min_score}, Helpfulness >= {min_helpful_ratio:.0%})")
    print("="*70 + "\n")
    
    try:
        bq_client = BigQueryClient()
        table_name = os.getenv('BQ_TABLE_REVIEWS', 'reviews')
        
        query_sql = f"""
SELECT
  review_id,
  product_id,
  score,
  summary,
  text,
  helpfulness_ratio,
  helpfulness_numerator,
  helpfulness_denominator,
  profile_name
FROM `{bq_client.project_id}.{bq_client.dataset_id}.{table_name}`
WHERE score >= {min_score}
  AND helpfulness_ratio >= {min_helpful_ratio}
  AND helpfulness_denominator >= 10
ORDER BY helpfulness_numerator DESC, helpfulness_ratio DESC
LIMIT 20;
"""
        
        results = bq_client.query_to_dataframe(query_sql)
        
        if len(results) == 0:
            print("No reviews found matching criteria")
            return
        
        print(f"Found {len(results)} highly helpful reviews:\n")
        
        for i, row in results.iterrows():
            print(f"{i+1}. {'⭐' * int(row['score'])} ({row['score']}/5)")
            print(f"   Product: {row['product_id']}")
            print(f"   Summary: {row['summary']}")
            print(f"   Helpful: {row['helpfulness_numerator']}/{row['helpfulness_denominator']} ({row['helpfulness_ratio']:.0%})")
            print(f"   By: {row['profile_name']}")
            print()
        
        return results
        
    except Exception as e:
        logger.error(f"Helpful reviews search failed: {e}")
        print(f"❌ Search failed: {e}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Amazon Fine Food Reviews - Semantic Search System'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup BigQuery infrastructure and load data'
    )
    
    parser.add_argument(
        '--search',
        type=str,
        help='Search for similar reviews (e.g., "delicious chocolate")'
    )
    
    parser.add_argument(
        '--product',
        type=str,
        help='Find reviews for a specific product ID'
    )
    
    parser.add_argument(
        '--helpful',
        action='store_true',
        help='Find highly helpful and highly-rated reviews'
    )
    
    parser.add_argument(
        '--min-score',
        type=int,
        default=4,
        help='Minimum review score (1-5, default: 4)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n📚 Examples:")
        print("  python main.py --setup")
        print("  python main.py --search 'delicious chocolate candy'")
        print("  python main.py --product B001E4KFG0")
        print("  python main.py --helpful --min-score 5")
        return
    
    # Execute commands
    if args.setup:
        success = setup_bigquery_infrastructure()
        sys.exit(0 if success else 1)
    
    elif args.search:
        search_similar_reviews(args.search, args.top_k)
    
    elif args.product:
        find_product_reviews(args.product, args.min_score)
    
    elif args.helpful:
        find_helpful_reviews(args.min_score, 0.8)


if __name__ == "__main__":
    main()
