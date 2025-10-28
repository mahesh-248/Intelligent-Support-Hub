"""
BigQuery Utility Module

Provides connection management and helper functions for interacting with BigQuery,
specifically for the Intelligent Support Ticket Triage System.

This module handles:
- BigQuery client initialization
- Query execution with error handling
- Table operations
- Embedding generation helpers
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BigQueryClient:
    """
    Wrapper class for BigQuery operations with error handling and logging.
    
    Attributes:
        project_id: GCP project ID
        dataset_id: BigQuery dataset name
        location: BigQuery dataset location (e.g., 'US', 'EU')
        client: Google Cloud BigQuery client instance
    """
    
    def __init__(
        self, 
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        location: str = "US"
    ):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: GCP project ID (defaults to env var GCP_PROJECT_ID)
            dataset_id: BigQuery dataset name (defaults to env var BQ_DATASET)
            location: BigQuery location (defaults to env var BQ_LOCATION or 'US')
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.dataset_id = dataset_id or os.getenv("BQ_DATASET", "support_tickets")
        self.location = location or os.getenv("BQ_LOCATION", "US")
        
        if not self.project_id:
            raise ValueError(
                "GCP_PROJECT_ID must be set in environment variables or passed as parameter"
            )
        
        # Initialize BigQuery client
        try:
            self.client = bigquery.Client(project=self.project_id)
            logger.info(f"Connected to BigQuery project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    def create_dataset(self, exists_ok: bool = True) -> None:
        """
        Create BigQuery dataset if it doesn't exist.
        
        Args:
            exists_ok: If True, don't raise error if dataset exists
        """
        dataset_id = f"{self.project_id}.{self.dataset_id}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = self.location
        
        try:
            dataset = self.client.create_dataset(dataset, exists_ok=exists_ok)
            logger.info(f"Created dataset {dataset_id}")
        except GoogleCloudError as e:
            logger.error(f"Error creating dataset: {e}")
            raise
    
    def execute_query(
        self, 
        query: str, 
        job_config: Optional[bigquery.QueryJobConfig] = None
    ) -> bigquery.table.RowIterator:
        """
        Execute a BigQuery SQL query.
        
        Args:
            query: SQL query string
            job_config: Optional query job configuration
            
        Returns:
            Query results as RowIterator
        """
        try:
            logger.info(f"Executing query (first 100 chars): {query[:100]}...")
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Log query statistics (handle DDL statements that don't have total_rows)
            bytes_processed = getattr(query_job, 'total_bytes_processed', 0) or 0
            total_rows = getattr(query_job, 'total_rows', None)
            
            if total_rows is not None:
                logger.info(
                    f"Query completed. "
                    f"Bytes processed: {bytes_processed:,}, "
                    f"Rows: {total_rows}"
                )
            else:
                logger.info(
                    f"Query completed. "
                    f"Statement executed successfully (DDL/DML operation)"
                )
            
            return results
        except GoogleCloudError as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def query_to_dataframe(self, query: str) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"Retrieved {len(df)} rows as DataFrame")
            return df
        except Exception as e:
            logger.error(f"Failed to convert query results to DataFrame: {e}")
            raise
    
    def load_dataframe_to_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        write_disposition: str = "WRITE_APPEND"
    ) -> None:
        """
        Load pandas DataFrame to BigQuery table.
        
        Args:
            df: pandas DataFrame to load
            table_name: Target table name (without dataset prefix)
            write_disposition: How to handle existing data
                - WRITE_APPEND: Append to existing data
                - WRITE_TRUNCATE: Overwrite existing data
                - WRITE_EMPTY: Only write if table is empty
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
        )
        
        try:
            logger.info(f"Loading {len(df)} rows to {table_id}...")
            job = self.client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            job.result()  # Wait for job to complete
            
            logger.info(f"Successfully loaded {len(df)} rows to {table_id}")
        except GoogleCloudError as e:
            logger.error(f"Failed to load data to BigQuery: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the dataset.
        
        Args:
            table_name: Table name to check
            
        Returns:
            True if table exists, False otherwise
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        try:
            self.client.get_table(table_id)
            return True
        except Exception:
            return False
    
    def execute_sql_file(self, file_path: str, **kwargs) -> bigquery.table.RowIterator:
        """
        Execute SQL from a file with optional parameter substitution.
        
        Args:
            file_path: Path to SQL file
            **kwargs: Parameters to substitute in SQL (e.g., {table_name})
            
        Returns:
            Query results
        """
        try:
            with open(file_path, 'r') as f:
                query = f.read()
            
            # Substitute parameters
            if kwargs:
                query = query.format(
                    project_id=self.project_id,
                    dataset_id=self.dataset_id,
                    **kwargs
                )
            
            return self.execute_query(query)
        except FileNotFoundError:
            logger.error(f"SQL file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error executing SQL file {file_path}: {e}")
            raise
    
    def get_table_schema(self, table_name: str) -> List[bigquery.SchemaField]:
        """
        Get schema of a BigQuery table.
        
        Args:
            table_name: Table name
            
        Returns:
            List of schema fields
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        table = self.client.get_table(table_id)
        return table.schema
    
    def get_table_row_count(self, table_name: str) -> int:
        """
        Get number of rows in a table.
        
        Args:
            table_name: Table name
            
        Returns:
            Number of rows
        """
        query = f"""
        SELECT COUNT(*) as row_count
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        """
        result = self.query_to_dataframe(query)
        return int(result['row_count'].iloc[0])
    
    def delete_table(self, table_name: str, not_found_ok: bool = True) -> None:
        """
        Delete a BigQuery table.
        
        Args:
            table_name: Table name to delete
            not_found_ok: If True, don't raise error if table doesn't exist
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        try:
            self.client.delete_table(table_id, not_found_ok=not_found_ok)
            logger.info(f"Deleted table {table_id}")
        except GoogleCloudError as e:
            logger.error(f"Error deleting table: {e}")
            raise


def get_bq_client() -> BigQueryClient:
    """
    Factory function to create a BigQueryClient instance.
    
    Returns:
        Configured BigQueryClient instance
    """
    return BigQueryClient()


# Helper functions for common operations
def generate_embedding_query(
    text_column: str,
    source_table: str,
    model: str = "text-embedding-004"
) -> str:
    """
    Generate SQL query for creating embeddings using ML.GENERATE_EMBEDDING.
    
    Args:
        text_column: Column containing text to embed
        source_table: Fully qualified source table name
        model: Embedding model to use
        
    Returns:
        SQL query string
    """
    return f"""
    SELECT
        *,
        ml_generate_embedding_result AS embedding
    FROM
        ML.GENERATE_EMBEDDING(
            MODEL `{model}`,
            (SELECT * FROM `{source_table}`),
            STRUCT('{text_column}' AS content_column, TRUE AS flatten_json_output)
        )
    """


def resolve_embedding_model_identifier(
    project_id: str,
    raw_model: Optional[str]
) -> Tuple[str, bool]:
    """Resolve embedding model name to a BigQuery-friendly identifier."""
    model_hint = (raw_model or os.getenv('EMBEDDING_MODEL') or 'text-embedding-004').strip()

    if model_hint.count('.') == 2:
        return model_hint, True

    if model_hint.count('.') == 1:
        return f"{project_id}.{model_hint}", True

    return model_hint, False


def vector_search_query(
    query_text: str,
    embeddings_table: str,
    embedding_column: str = "embedding",
    top_k: int = 5,
    distance_type: str = "COSINE",
    model: str = "text-embedding-004"
) -> str:
    """
    Generate SQL query for vector search using VECTOR_SEARCH.
    
    Args:
        query_text: Text to search for
        embeddings_table: Table containing embeddings
        embedding_column: Column name containing embeddings
        top_k: Number of similar results to return
        distance_type: Distance metric (COSINE, EUCLIDEAN, DOT_PRODUCT)
        model: Embedding model to use (must match table embeddings)
        
    Returns:
        SQL query string
    """
    return f"""
    SELECT
        base.*,
        distance
    FROM
        VECTOR_SEARCH(
            TABLE `{embeddings_table}`,
            '{embedding_column}',
            (
                SELECT ml_generate_embedding_result AS embedding
                FROM ML.GENERATE_EMBEDDING(
                    MODEL `{model}`,
                    (SELECT '{query_text}' AS content)
                )
            ),
            distance_type => '{distance_type}',
            top_k => {top_k}
        )
    """


if __name__ == "__main__":
    # Quick test
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        client = get_bq_client()
        print(f"✅ Successfully connected to BigQuery project: {client.project_id}")
        print(f"✅ Dataset: {client.dataset_id}")
        print(f"✅ Location: {client.location}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
