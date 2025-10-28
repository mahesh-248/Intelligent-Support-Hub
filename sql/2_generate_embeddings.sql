-- ============================================================================
-- Step 2: Generate Embeddings for Support Tickets
-- ============================================================================
-- This script creates a new table with vector embeddings for semantic search
-- Uses BigQuery ML.GENERATE_EMBEDDING with Vertex AI embedding model
-- ============================================================================

-- ============================================================================
-- Option A: Create new table with embeddings (recommended for production)
-- ============================================================================

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{embeddings_table_name}` AS
SELECT
    *
FROM
    ML.GENERATE_EMBEDDING(
        MODEL `{embedding_model}`,
        TABLE `{project_id}.{dataset_id}.{source_table_name}`,
        STRUCT('combined_text' AS content_column, TRUE AS flatten_json_output)
    );

-- ============================================================================
-- Add table description and optimize
-- ============================================================================

ALTER TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`
SET OPTIONS(
    description="Support tickets with vector embeddings for semantic search using {embedding_model}",
    labels=[("purpose", "vector_search"), ("model", "text_embedding_004")]
);

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Check embedding generation success
-- SELECT 
--     COUNT(*) as total_rows,
--     COUNT(embedding) as rows_with_embeddings,
--     ROUND(COUNT(embedding) * 100.0 / COUNT(*), 2) as embedding_coverage_pct
-- FROM `{project_id}.{dataset_id}.{embeddings_table_name}`;

-- Sample embeddings (first 5 dimensions of first 3 tickets)
-- SELECT 
--     ticket_id,
--     title,
--     ARRAY_LENGTH(embedding) as embedding_dimensions,
--     ARRAY(SELECT CAST(dim AS STRING) FROM UNNEST(embedding) AS dim LIMIT 5) as sample_dimensions
-- FROM `{project_id}.{dataset_id}.{embeddings_table_name}`
-- LIMIT 3;

-- ============================================================================
-- Option B: Add embedding column to existing table (alternative approach)
-- ============================================================================

-- Uncomment to add embeddings to existing table instead of creating new table:
/*
ALTER TABLE `{project_id}.{dataset_id}.{source_table_name}`
ADD COLUMN IF NOT EXISTS embedding ARRAY<FLOAT64>;

MERGE `{project_id}.{dataset_id}.{source_table_name}` T
USING (
    SELECT
        *
    FROM ML.GENERATE_EMBEDDING(
        MODEL `{embedding_model}`,
        TABLE `{project_id}.{dataset_id}.{source_table_name}`,
        STRUCT('combined_text' AS content_column, TRUE AS flatten_json_output)
    )
) S
ON T.ticket_id = S.ticket_id
WHEN MATCHED THEN
    UPDATE SET T.ml_generate_embedding_result = S.ml_generate_embedding_result;
*/

-- ============================================================================
-- Notes on Embedding Models
-- ============================================================================

-- Available models:
--   - text-embedding-004: Latest model, best quality (768 dimensions)
--   - text-multilingual-embedding-002: For multilingual support (768 dimensions)
--   - textembedding-gecko@003: Older model (768 dimensions)

-- Performance:
--   - Embedding generation: ~200ms per ticket
--   - Batch processing: Can process thousands of tickets in minutes
--   - Cost: Based on input characters processed

-- Best Practices:
--   1. Use combined_text field (title + description) for best semantic understanding
--   2. Batch process embeddings for large datasets
--   3. Regenerate embeddings when ticket content is updated
--   4. Consider incremental updates for new tickets only
