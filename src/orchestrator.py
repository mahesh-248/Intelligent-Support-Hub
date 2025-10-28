"""
Ticket Triage Orchestrator

Core module for the Intelligent Support Ticket Triage System.
Handles semantic search, solution recommendations, and ticket categorization
using BigQuery's vector search capabilities.

Key Features:
- Find semantically similar historical tickets
- Recommend solutions based on past resolutions
- Automatically categorize new tickets
- Batch processing support
- Confidence scoring
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.utils.bq import BigQueryClient, resolve_embedding_model_identifier

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TicketTriageOrchestrator:
    """
    Main orchestrator for intelligent ticket triage using BigQuery vector search.
    
    This class provides methods to:
    1. Find semantically similar tickets
    2. Recommend solutions based on historical data
    3. Auto-categorize tickets
    4. Batch process multiple tickets
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        embeddings_table: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset name
            embeddings_table: Table with embeddings
            embedding_model: Optional model override (defaults to ENV setting)
        """
        self.bq_client = BigQueryClient(project_id=project_id, dataset_id=dataset_id)
        self.project_id = self.bq_client.project_id
        self.dataset_id = self.bq_client.dataset_id
        self.embeddings_table = embeddings_table or os.getenv(
            'BQ_TABLE_TICKETS_EMBEDDINGS', 
            'tickets_with_embeddings'
        )
        model_hint = embedding_model or os.getenv('EMBEDDING_MODEL', 'text-embedding-004')
        resolved_model, _ = resolve_embedding_model_identifier(
            self.project_id,
            model_hint
        )
        self.embedding_model = resolved_model
        
        # Configuration
        self.top_k = int(os.getenv('TOP_K_SIMILAR', 1))
        self.min_similarity = float(os.getenv('MIN_SIMILARITY_SCORE', 0.70))
        
        logger.info(
            f"Initialized TicketTriageOrchestrator for "
            f"{self.project_id}.{self.dataset_id}.{self.embeddings_table}"
        )
    
    def find_similar_tickets(
        self,
        ticket_text: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        distance_type: str = "COSINE"
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar historical tickets.
        
        Args:
            ticket_text: Text of the new ticket (title + description)
            top_k: Number of similar tickets to return
            filters: Optional filters (e.g., {'status': 'resolved'})
            distance_type: Distance metric (COSINE, EUCLIDEAN, DOT_PRODUCT)
            
        Returns:
            List of similar tickets with similarity scores
        """
        top_k = top_k or self.top_k
        
        logger.info(f"Searching for similar tickets: '{ticket_text[:50]}...'")
        
        # Build the query - VECTOR_SEARCH returns base table as a STRUCT, need to flatten
        query = f"""
        SELECT
            base.ticket_id,
            base.title,
            base.description,
            base.category,
            base.priority,
            base.status,
            base.resolution,
            base.satisfaction_score,
            base.resolution_time_hours,
            base.created_at,
            distance,
            ROUND((1 - distance) * 100, 2) AS similarity_score
        FROM VECTOR_SEARCH(
            TABLE `{self.project_id}.{self.dataset_id}.{self.embeddings_table}`,
            'embedding',
            (
                SELECT ml_generate_embedding_result AS embedding
                FROM ML.GENERATE_EMBEDDING(
                    MODEL `{self.embedding_model}`,
                    (SELECT @ticket_text AS content)
                )
            ),
            distance_type => '{distance_type}',
            top_k => {top_k * 2}
        )
        WHERE 1=1
        """
        
        # Add filters
        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    query += f" AND base.{key} = '{value}'"
                elif isinstance(value, list):
                    values_str = "', '".join(str(v) for v in value)
                    query += f" AND base.{key} IN ('{values_str}')"
                else:
                    query += f" AND base.{key} = {value}"
        
        query += f"""
        ORDER BY distance ASC
        LIMIT {top_k}
        """
        
        # Execute query with parameter
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticket_text", "STRING", ticket_text)
            ]
        )
        
        try:
            df = self.bq_client.client.query(query, job_config=job_config).to_dataframe()
            
            # Convert to list of dicts
            results = df.to_dict('records')
            
            logger.info(f"Found {len(results)} similar tickets")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar tickets: {e}")
            raise
    
    def recommend_solution(
        self,
        ticket_text: str,
        min_satisfaction: int = 4,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Recommend a solution based on similar resolved tickets.
        
        Args:
            ticket_text: Text of the new ticket
            min_satisfaction: Minimum satisfaction score to consider
            top_k: Number of similar tickets to analyze
            
        Returns:
            Recommended solution with confidence score and supporting data
        """
        logger.info(f"Generating solution recommendation for: '{ticket_text[:50]}...'")

        query = f"""
        WITH raw AS (
            SELECT
                base.ticket_id,
                base.title,
                base.resolution,
                base.satisfaction_score,
                base.resolution_time_hours,
                base.status,
                distance
            FROM VECTOR_SEARCH(
                TABLE `{self.project_id}.{self.dataset_id}.{self.embeddings_table}`,
                'embedding',
                (
                    SELECT ml_generate_embedding_result AS embedding
                    FROM ML.GENERATE_EMBEDDING(
                        MODEL `{self.embedding_model}`,
                        (SELECT @ticket_text AS content)
                    )
                ),
                distance_type => 'COSINE',
                top_k => {top_k}
            )
        ),
        similar_tickets AS (
            SELECT
                ticket_id,
                title,
                resolution,
                satisfaction_score,
                resolution_time_hours,
                distance,
                ROUND((1 - distance) * 100, 2) AS similarity_pct
            FROM raw
            WHERE status = 'resolved'
                AND resolution IS NOT NULL
                AND satisfaction_score >= {min_satisfaction}
                -- Distance is 1 - cosine_similarity. If min_similarity is in 0-1 range (e.g. 0.70),
                -- we filter where distance < (1 - min_similarity)
                AND distance < {1 - self.min_similarity}
        ),
        resolution_ranking AS (
            SELECT
                resolution,
                COUNT(*) as frequency,
                AVG(satisfaction_score) as avg_satisfaction,
                AVG(resolution_time_hours) as avg_resolution_time,
                AVG(similarity_pct) as avg_similarity,
                ARRAY_AGG(
                    STRUCT(ticket_id, title, similarity_pct) 
                    ORDER BY similarity_pct DESC 
                    LIMIT 3
                ) as example_tickets
            FROM similar_tickets
            GROUP BY resolution
        )
        SELECT
            resolution,
            frequency,
            ROUND(avg_satisfaction, 2) as avg_satisfaction,
            ROUND(avg_resolution_time, 2) as avg_resolution_hours,
            ROUND(avg_similarity, 2) as avg_similarity_pct,
            ROUND(
                (frequency * 0.3 + avg_satisfaction * 15 + avg_similarity * 0.5) / 
                (GREATEST(avg_resolution_time, 1) * 0.1),
                2
            ) as confidence_score,
            example_tickets
        FROM resolution_ranking
        ORDER BY confidence_score DESC
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticket_text", "STRING", ticket_text)
            ]
        )
        
        try:
            df = self.bq_client.client.query(query, job_config=job_config).to_dataframe()
            
            if df.empty:
                logger.warning("No similar resolved tickets found")
                return {
                    'solution': None,
                    'confidence': 0.0,
                    'similar_ticket_count': 0,
                    'example_tickets': []
                }
            
            result = df.iloc[0].to_dict()
            
            recommendation = {
                'solution': result['resolution'],
                'confidence': float(result['confidence_score']),
                'similar_ticket_count': int(result['frequency']),
                'avg_satisfaction': float(result['avg_satisfaction']),
                'avg_resolution_hours': float(result['avg_resolution_hours']),
                'avg_similarity_pct': float(result['avg_similarity_pct']),
                'example_tickets': result['example_tickets']
            }
            
            logger.info(
                f"Recommended solution with {recommendation['confidence']:.2f} confidence "
                f"based on {recommendation['similar_ticket_count']} similar tickets"
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            raise
    
    def auto_categorize(
        self,
        ticket_text: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Automatically categorize a ticket based on similar historical tickets.
        
        Args:
            ticket_text: Text of the new ticket
            top_k: Number of similar tickets to analyze
            
        Returns:
            Predicted category with confidence score
        """
        logger.info(f"Auto-categorizing ticket: '{ticket_text[:50]}...'")
        
        query = f"""
        WITH raw AS (
            SELECT
                base.category,
                base.priority,
                distance
            FROM VECTOR_SEARCH(
                TABLE `{self.project_id}.{self.dataset_id}.{self.embeddings_table}`,
                'embedding',
                (
                    SELECT ml_generate_embedding_result AS embedding
                    FROM ML.GENERATE_EMBEDDING(
                        MODEL `{self.embedding_model}`,
                        (SELECT @ticket_text AS content)
                    )
                ),
                distance_type => 'COSINE',
                top_k => {top_k}
            )
        ),
        similar_tickets AS (
            SELECT
                category,
                priority,
                distance,
                ROUND((1 - distance) * 100, 2) AS similarity_pct
            FROM raw
        ),
        category_votes AS (
            SELECT
                category,
                COUNT(*) as vote_count,
                ROUND(AVG(similarity_pct), 2) as avg_similarity,
                ROUND(COUNT(*) * 100.0 / {top_k}, 2) as confidence_pct
            FROM similar_tickets
            GROUP BY category
        ),
        priority_votes AS (
            SELECT
                priority,
                COUNT(*) as vote_count,
                ROUND(AVG(similarity_pct), 2) as avg_similarity
            FROM similar_tickets
            GROUP BY priority
        )
        SELECT
            c.category,
            c.vote_count as category_votes,
            c.avg_similarity as category_similarity,
            c.confidence_pct as category_confidence,
            p.priority,
            p.vote_count as priority_votes,
            p.avg_similarity as priority_similarity
        FROM category_votes c
        CROSS JOIN priority_votes p
        ORDER BY c.vote_count DESC, c.avg_similarity DESC, p.vote_count DESC
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticket_text", "STRING", ticket_text)
            ]
        )
        
        try:
            df = self.bq_client.client.query(query, job_config=job_config).to_dataframe()
            
            if df.empty:
                logger.warning("No similar tickets found for categorization")
                return {
                    'category': 'Unknown',
                    'category_confidence': 0.0,
                    'priority': 'medium',
                    'priority_confidence': 0.0
                }
            
            result = df.iloc[0].to_dict()
            
            categorization = {
                'category': result['category'],
                'category_confidence': float(result['category_confidence']),
                'category_votes': int(result['category_votes']),
                'category_similarity': float(result['category_similarity']),
                'priority': result['priority'],
                'priority_votes': int(result['priority_votes']),
                'priority_similarity': float(result['priority_similarity'])
            }
            
            logger.info(
                f"Categorized as '{categorization['category']}' "
                f"with {categorization['category_confidence']:.2f}% confidence"
            )
            
            return categorization
            
        except Exception as e:
            logger.error(f"Error auto-categorizing ticket: {e}")
            raise
    
    def batch_triage(
        self,
        tickets: List[Dict[str, str]],
        include_recommendations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple tickets at once.
        
        Args:
            tickets: List of ticket dicts with 'title' and 'description'
            include_recommendations: Whether to include solution recommendations
            
        Returns:
            List of triage results for each ticket
        """
        logger.info(f"Batch triaging {len(tickets)} tickets...")
        
        results = []
        
        for i, ticket in enumerate(tickets):
            logger.info(f"Processing ticket {i+1}/{len(tickets)}")
            
            ticket_text = f"{ticket.get('title', '')} {ticket.get('description', '')}"
            
            try:
                # Find similar tickets
                similar = self.find_similar_tickets(ticket_text, top_k=5)
                
                # Auto-categorize
                categorization = self.auto_categorize(ticket_text)
                
                # Get recommendation if requested
                recommendation = None
                if include_recommendations:
                    recommendation = self.recommend_solution(ticket_text)
                
                result = {
                    'ticket': ticket,
                    'similar_tickets': similar,
                    'categorization': categorization,
                    'recommendation': recommendation
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing ticket {i+1}: {e}")
                results.append({
                    'ticket': ticket,
                    'error': str(e)
                })
        
        logger.info(f"Completed batch triage of {len(tickets)} tickets")
        return results
    
    def get_trending_issues(
        self,
        days: int = 7,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Identify trending issues by clustering recent similar tickets.
        
        Args:
            days: Number of days to look back
            min_cluster_size: Minimum tickets to form a cluster
            similarity_threshold: Maximum distance to consider similar
            
        Returns:
            List of trending issues with cluster information
        """
        logger.info(f"Identifying trending issues from last {days} days...")
        
        query = f"""
        WITH recent_tickets AS (
            SELECT ticket_id, title, created_at, embedding
            FROM `{self.project_id}.{self.dataset_id}.{self.embeddings_table}`
            WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        ),
        ticket_pairs AS (
            SELECT
                t1.ticket_id as ticket_id_1,
                t1.title as title_1,
                t1.created_at as created_1,
                base.ticket_id as ticket_id_2,
                base.title as title_2,
                base.created_at as created_2,
                distance
            FROM recent_tickets t1
            CROSS JOIN LATERAL (
                SELECT ticket_id, title, created_at, distance
                FROM VECTOR_SEARCH(
                    TABLE `{self.project_id}.{self.dataset_id}.{self.embeddings_table}`,
                    'embedding',
                    t1.embedding,
                    distance_type => 'COSINE',
                    top_k => 10
                )
                WHERE distance < {similarity_threshold}
                  AND ticket_id != t1.ticket_id
            ) base
        )
        SELECT
            title_1 as representative_issue,
            COUNT(DISTINCT ticket_id_2) as cluster_size,
            MIN(created_1) as first_occurrence,
            MAX(created_2) as last_occurrence,
            ARRAY_AGG(DISTINCT title_2 ORDER BY created_2 DESC LIMIT 5) as similar_issues
        FROM ticket_pairs
        GROUP BY title_1
        HAVING cluster_size >= {min_cluster_size}
        ORDER BY cluster_size DESC, last_occurrence DESC
        LIMIT 10
        """
        
        try:
            df = self.bq_client.query_to_dataframe(query)
            results = df.to_dict('records')
            
            logger.info(f"Found {len(results)} trending issues")
            return results
            
        except Exception as e:
            logger.error(f"Error identifying trending issues: {e}")
            raise


if __name__ == "__main__":
    # Quick test
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        orchestrator = TicketTriageOrchestrator()
        print(f"✅ Successfully initialized orchestrator")
        print(f"✅ Using table: {orchestrator.embeddings_table}")
        print(f"✅ Embedding model: {orchestrator.embedding_model}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
