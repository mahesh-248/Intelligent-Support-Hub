-- ============================================================================
-- Step 4: Example Semantic Search Queries
-- ============================================================================
-- This file contains example queries demonstrating vector search capabilities
-- Use these as templates for building your application queries
-- ============================================================================

-- ============================================================================
-- Example 1: Basic Semantic Search
-- ============================================================================
-- Find tickets similar to a query text

SELECT
    ticket_id,
    title,
    description,
    category,
    status,
    resolution,
    distance,
    ROUND((1 - distance) * 100, 2) as similarity_score_pct
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
    'embedding',
    (
        SELECT ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `{embedding_model}`,
            (SELECT 'my application keeps crashing when I start it' AS content)
        )
    ),
    distance_type => 'COSINE',
    top_k => 5
)
ORDER BY distance ASC;

-- ============================================================================
-- Example 2: Search with Filters
-- ============================================================================
-- Find similar resolved tickets only (for solution recommendations)

SELECT
    base.ticket_id,
    base.title,
    base.category,
    base.resolution,
    base.satisfaction_score,
    distance,
    ROUND((1 - distance) * 100, 2) as similarity_score_pct
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
    'embedding',
    (
        SELECT ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `{embedding_model}`,
            (SELECT 'cannot log into my account' AS content)
        )
    ),
    distance_type => 'COSINE',
    top_k => 10,
    -- Filter for resolved tickets only
    options => '{"fraction_lists_to_search": 0.1}'
) base
WHERE base.status = 'resolved'
  AND base.resolution IS NOT NULL
  AND base.satisfaction_score >= 4  -- High satisfaction only
ORDER BY distance ASC
LIMIT 5;

-- ============================================================================
-- Example 3: Category Detection
-- ============================================================================
-- Automatically categorize a new ticket based on similar historical tickets

WITH similar_tickets AS (
    SELECT
        base.category,
        distance
    FROM VECTOR_SEARCH(
        TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
        'embedding',
        (
            SELECT ml_generate_embedding_result AS embedding
            FROM ML.GENERATE_EMBEDDING(
                MODEL `{embedding_model}`,
                (SELECT 'payment failed during checkout' AS content)
            )
        ),
        distance_type => 'COSINE',
        top_k => 10
    ) base
)
SELECT
    category,
    COUNT(*) as vote_count,
    ROUND(AVG(1 - distance) * 100, 2) as avg_similarity_pct,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as confidence_pct
FROM similar_tickets
GROUP BY category
ORDER BY vote_count DESC, avg_similarity_pct DESC
LIMIT 3;

-- ============================================================================
-- Example 4: Batch Search
-- ============================================================================
-- Search for multiple queries at once

WITH queries AS (
    SELECT 'app crashes on startup' AS query_text, 'Q1' AS query_id
    UNION ALL
    SELECT 'cannot reset my password', 'Q2'
    UNION ALL
    SELECT 'slow performance', 'Q3'
),
query_embeddings AS (
    SELECT
        query_id,
        query_text,
        ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
        MODEL `{embedding_model}`,
        (SELECT query_id, query_text AS content FROM queries)
    )
)
SELECT
    qe.query_id,
    qe.query_text,
    vs.ticket_id,
    vs.title,
    vs.category,
    vs.distance,
    ROUND((1 - vs.distance) * 100, 2) as similarity_pct
FROM query_embeddings qe
CROSS JOIN LATERAL (
    SELECT *
    FROM VECTOR_SEARCH(
        TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
        'embedding',
        qe.embedding,
        distance_type => 'COSINE',
        top_k => 3
    )
) vs
ORDER BY qe.query_id, vs.distance;

-- ============================================================================
-- Example 5: Solution Recommendation with Aggregation
-- ============================================================================
-- Find best solution based on similar tickets

WITH similar_tickets AS (
    SELECT
        base.ticket_id,
        base.title,
        base.resolution,
        base.satisfaction_score,
        base.resolution_time_hours,
        distance,
        ROUND((1 - distance) * 100, 2) as similarity_pct
    FROM VECTOR_SEARCH(
        TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
        'embedding',
        (
            SELECT ml_generate_embedding_result AS embedding
            FROM ML.GENERATE_EMBEDDING(
                MODEL `{embedding_model}`,
                (SELECT 'login error invalid credentials' AS content)
            )
        ),
        distance_type => 'COSINE',
        top_k => 20
    ) base
    WHERE base.status = 'resolved'
      AND base.resolution IS NOT NULL
      AND base.satisfaction_score >= 4
),
resolution_ranking AS (
    SELECT
        resolution,
        COUNT(*) as frequency,
        AVG(satisfaction_score) as avg_satisfaction,
        AVG(resolution_time_hours) as avg_resolution_time,
        AVG(similarity_pct) as avg_similarity,
        ARRAY_AGG(STRUCT(ticket_id, title, similarity_pct) ORDER BY similarity_pct DESC LIMIT 3) as example_tickets
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
        (frequency * 0.4 + avg_satisfaction * 20 + avg_similarity * 0.4) / 
        (GREATEST(avg_resolution_time, 1) * 0.1),
        2
    ) as confidence_score,
    example_tickets
FROM resolution_ranking
ORDER BY confidence_score DESC
LIMIT 3;

-- ============================================================================
-- Example 6: Semantic Clustering
-- ============================================================================
-- Find clusters of similar tickets (e.g., identify trending issues)

WITH ticket_pairs AS (
    SELECT
        t1.ticket_id as ticket_id_1,
        t1.title as title_1,
        t1.created_at as created_1,
        t2.ticket_id as ticket_id_2,
        t2.title as title_2,
        t2.created_at as created_2,
        distance
    FROM VECTOR_SEARCH(
        TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
        'embedding',
        (SELECT ticket_id, embedding FROM `{project_id}.{dataset_id}.{embeddings_table_name}` WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)),
        distance_type => 'COSINE',
        top_k => 5
    ) t2
    JOIN `{project_id}.{dataset_id}.{embeddings_table_name}` t1
    ON t2.ticket_id = t1.ticket_id
    WHERE distance < 0.2  -- High similarity threshold
      AND t1.ticket_id != t2.ticket_id_2
)
SELECT
    title_1 as representative_issue,
    COUNT(DISTINCT ticket_id_2) as cluster_size,
    MIN(created_1) as first_occurrence,
    MAX(created_2) as last_occurrence,
    ARRAY_AGG(DISTINCT title_2 LIMIT 5) as similar_issues
FROM ticket_pairs
GROUP BY title_1
HAVING cluster_size >= 3  -- At least 3 similar tickets
ORDER BY cluster_size DESC, last_occurrence DESC
LIMIT 10;

-- ============================================================================
-- Example 7: Multi-Field Semantic Search
-- ============================================================================
-- Search across multiple text fields with different weights

WITH query_embedding AS (
    SELECT ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
        MODEL `{embedding_model}`,
        (SELECT 'refund request for cancelled order' AS content)
    )
)
SELECT
    base.ticket_id,
    base.title,
    base.description,
    base.category,
    base.tags,
    base.distance as semantic_distance,
    -- Additional scoring based on metadata
    CASE
        WHEN base.category IN ('Billing', 'Payment') THEN 0.1
        WHEN 'refund' IN UNNEST(base.tags) THEN 0.15
        ELSE 0
    END as category_boost,
    ROUND(
        (1 - base.distance) * 100 + 
        CASE WHEN base.category IN ('Billing', 'Payment') THEN 10 ELSE 0 END +
        CASE WHEN 'refund' IN UNNEST(base.tags) THEN 15 ELSE 0 END,
        2
    ) as final_score
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
    'embedding',
    (SELECT embedding FROM query_embedding),
    distance_type => 'COSINE',
    top_k => 20
) base
ORDER BY final_score DESC
LIMIT 10;

-- ============================================================================
-- Example 8: Time-Weighted Semantic Search
-- ============================================================================
-- Prioritize more recent similar tickets

SELECT
    base.ticket_id,
    base.title,
    base.created_at,
    base.distance,
    ROUND((1 - base.distance) * 100, 2) as similarity_pct,
    TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), base.created_at, DAY) as age_days,
    ROUND(
        (1 - base.distance) * 100 * 
        EXP(-TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), base.created_at, DAY) / 30.0),
        2
    ) as time_weighted_score
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
    'embedding',
    (
        SELECT ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `{embedding_model}`,
            (SELECT 'app not responding' AS content)
        )
    ),
    distance_type => 'COSINE',
    top_k => 20
) base
ORDER BY time_weighted_score DESC
LIMIT 10;

-- ============================================================================
-- Notes on Query Optimization
-- ============================================================================

-- Distance Types:
--   - COSINE: Best for text similarity (normalized, -1 to 1)
--   - EUCLIDEAN: Considers magnitude (0 to infinity)
--   - DOT_PRODUCT: Unnormalized similarity (can be negative)

-- top_k recommendations:
--   - 5-10: Good for end-user results
--   - 20-50: Good for aggregation/recommendation
--   - 100+: For analytics and clustering

-- Performance tips:
--   1. Use vector index for large datasets
--   2. Filter results in outer query (not in VECTOR_SEARCH)
--   3. Batch multiple searches when possible
--   4. Cache embeddings for common queries
--   5. Use appropriate fraction_lists_to_search (0.01-0.1)

-- Accuracy vs Speed tradeoff:
--   - fraction_lists_to_search: 0.01 = fast, lower recall
--   - fraction_lists_to_search: 0.1 = slower, higher recall
--   - use_brute_force: true = slowest, 100% accuracy
