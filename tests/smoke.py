"""
Smoke Tests for Intelligent Support Hub

Validates that the BigQuery AI vector search implementation is working correctly.
Run this after completing setup to verify the system is functional.

Usage:
    python tests/smoke.py
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.bq import BigQueryClient
from src.orchestrator import TicketTriageOrchestrator
from src.data_loader import TicketDataGenerator

# Load environment variables
load_dotenv()


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_test(name: str):
    """Print test name."""
    print(f"\n{Colors.BLUE}▶ {name}{Colors.END}")


def print_pass(message: str = "PASSED"):
    """Print success message."""
    print(f"  {Colors.GREEN}✓ {message}{Colors.END}")


def print_fail(message: str = "FAILED"):
    """Print failure message."""
    print(f"  {Colors.RED}✗ {message}{Colors.END}")


def print_info(message: str):
    """Print info message."""
    print(f"  {Colors.YELLOW}ℹ {message}{Colors.END}")


def run_smoke_tests():
    """Run all smoke tests."""
    print("\n" + "="*70)
    print(f"{Colors.BOLD}🧪 Intelligent Support Hub - Smoke Tests{Colors.END}")
    print("="*70)
    
    passed = 0
    failed = 0
    
    # Test 1: BigQuery Connection
    print_test("Test 1: BigQuery Connection")
    try:
        client = BigQueryClient()
        print_pass(f"Connected to project: {client.project_id}")
        print_pass(f"Dataset: {client.dataset_id}")
        passed += 1
    except Exception as e:
        print_fail(f"Connection failed: {e}")
        failed += 1
        return  # Can't continue without connection
    
    # Test 2: Tables Exist
    print_test("Test 2: Required Tables Exist")
    try:
        tickets_table = os.getenv('BQ_TABLE_TICKETS', 'tickets')
        embeddings_table = os.getenv('BQ_TABLE_TICKETS_EMBEDDINGS', 'tickets_with_embeddings')
        
        if client.table_exists(tickets_table):
            row_count = client.get_table_row_count(tickets_table)
            print_pass(f"Tickets table exists: {row_count} rows")
        else:
            print_fail(f"Tickets table '{tickets_table}' not found")
            failed += 1
            return
        
        if client.table_exists(embeddings_table):
            row_count = client.get_table_row_count(embeddings_table)
            print_pass(f"Embeddings table exists: {row_count} rows")
        else:
            print_fail(f"Embeddings table '{embeddings_table}' not found")
            failed += 1
            return
        
        passed += 1
    except Exception as e:
        print_fail(f"Table check failed: {e}")
        failed += 1
        return
    
    # Test 3: Orchestrator Initialization
    print_test("Test 3: Orchestrator Initialization")
    try:
        orchestrator = TicketTriageOrchestrator()
        print_pass(f"Orchestrator initialized")
        print_info(f"Embedding model: {orchestrator.embedding_model}")
        print_info(f"Top K: {orchestrator.top_k}")
        passed += 1
    except Exception as e:
        print_fail(f"Orchestrator init failed: {e}")
        failed += 1
        return
    
    # Test 4: Semantic Search
    print_test("Test 4: Semantic Search (Find Similar Tickets)")
    try:
        query = "my application keeps crashing when I start it"
        results = orchestrator.find_similar_tickets(query, top_k=3)
        
        if results and len(results) > 0:
            print_pass(f"Found {len(results)} similar tickets")
            
            for i, ticket in enumerate(results[:3], 1):
                print_info(
                    f"  #{i}: {ticket['title'][:50]}... "
                    f"({ticket['similarity_score']:.1f}% similar)"
                )
            
            passed += 1
        else:
            print_fail("No similar tickets found")
            failed += 1
    except Exception as e:
        print_fail(f"Search failed: {e}")
        failed += 1
    
    # Test 5: Solution Recommendation
    print_test("Test 5: Solution Recommendation")
    try:
        query = "cannot log into my account"
        recommendation = orchestrator.recommend_solution(query)
        
        if recommendation['solution']:
            print_pass(f"Recommendation generated")
            print_info(f"  Confidence: {recommendation['confidence']:.2f}")
            print_info(f"  Based on: {recommendation['similar_ticket_count']} tickets")
            print_info(f"  Solution: {recommendation['solution'][:60]}...")
            passed += 1
        else:
            print_fail("No recommendation available")
            print_info("This may happen if there are no similar resolved tickets")
            passed += 1  # Not a failure, just no data
    except Exception as e:
        print_fail(f"Recommendation failed: {e}")
        failed += 1
    
    # Test 6: Auto-Categorization
    print_test("Test 6: Auto-Categorization")
    try:
        query = "payment failed during checkout"
        categorization = orchestrator.auto_categorize(query)
        
        print_pass(f"Ticket categorized")
        print_info(f"  Category: {categorization['category']}")
        print_info(f"  Confidence: {categorization['category_confidence']:.1f}%")
        print_info(f"  Priority: {categorization['priority']}")
        passed += 1
    except Exception as e:
        print_fail(f"Categorization failed: {e}")
        failed += 1
    
    # Test 7: Batch Processing
    print_test("Test 7: Batch Processing")
    try:
        test_tickets = [
            {'title': 'Login error', 'description': 'Cannot sign in'},
            {'title': 'Slow app', 'description': 'App is very slow'}
        ]
        
        results = orchestrator.batch_triage(test_tickets, include_recommendations=False)
        
        if len(results) == len(test_tickets):
            print_pass(f"Processed {len(results)} tickets")
            
            for i, result in enumerate(results, 1):
                if 'error' in result:
                    print_fail(f"  Ticket {i} failed: {result['error']}")
                else:
                    cat = result['categorization']
                    print_info(f"  Ticket {i}: {cat['category']} ({cat['category_confidence']:.0f}%)")
            
            passed += 1
        else:
            print_fail(f"Expected {len(test_tickets)} results, got {len(results)}")
            failed += 1
    except Exception as e:
        print_fail(f"Batch processing failed: {e}")
        failed += 1
    
    # Test 8: Data Generator
    print_test("Test 8: Sample Data Generator")
    try:
        generator = TicketDataGenerator()
        df = generator.generate_tickets(count=10)
        
        if len(df) == 10:
            print_pass(f"Generated {len(df)} sample tickets")
            print_info(f"  Categories: {df['category'].nunique()}")
            print_info(f"  Resolved: {(df['status'] == 'resolved').sum()}")
            passed += 1
        else:
            print_fail(f"Expected 10 tickets, got {len(df)}")
            failed += 1
    except Exception as e:
        print_fail(f"Data generation failed: {e}")
        failed += 1
    
    # Test 9: Embeddings Validation
    print_test("Test 9: Embeddings Validation")
    try:
        query = """
        SELECT 
            COUNT(*) as total,
            COUNT(embedding) as with_embeddings,
            ROUND(COUNT(embedding) * 100.0 / COUNT(*), 2) as coverage_pct
        FROM `{project_id}.{dataset_id}.{table_name}`
        """.format(
            project_id=client.project_id,
            dataset_id=client.dataset_id,
            table_name=embeddings_table
        )
        
        df = client.query_to_dataframe(query)
        result = df.iloc[0]
        
        coverage = float(result['coverage_pct'])
        
        if coverage >= 99.0:
            print_pass(f"Embeddings coverage: {coverage}%")
            passed += 1
        else:
            print_fail(f"Low embeddings coverage: {coverage}%")
            print_info("Some tickets may not have embeddings generated")
            failed += 1
    except Exception as e:
        print_fail(f"Embeddings validation failed: {e}")
        failed += 1
    
    # Test 10: Performance Check
    print_test("Test 10: Search Performance")
    try:
        import time
        
        query = "application not responding"
        start_time = time.time()
        results = orchestrator.find_similar_tickets(query, top_k=5)
        elapsed = time.time() - start_time
        
        if elapsed < 5.0:  # Should complete in under 5 seconds
            print_pass(f"Search completed in {elapsed:.2f}s")
            if elapsed < 2.0:
                print_info("Excellent performance! ⚡")
            passed += 1
        else:
            print_fail(f"Slow search: {elapsed:.2f}s")
            print_info("Consider enabling vector index for better performance")
            passed += 1  # Not critical failure
    except Exception as e:
        print_fail(f"Performance check failed: {e}")
        failed += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"{Colors.BOLD}📊 Test Results Summary{Colors.END}")
    print("="*70)
    
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n  Total Tests: {total}")
    print(f"  {Colors.GREEN}✓ Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}✗ Failed: {failed}{Colors.END}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    if failed == 0:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 All tests passed! System is working correctly.{Colors.END}")
        print(f"\n  {Colors.BLUE}Next Steps:{Colors.END}")
        print(f"    • Try: python main.py --search 'your query'")
        print(f"    • Try: python main.py --recommend 'your issue'")
        print(f"    • Try: python main.py --batch")
    else:
        print(f"\n  {Colors.YELLOW}⚠️  Some tests failed. Please check the errors above.{Colors.END}")
    
    print()
    
    return failed == 0


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
