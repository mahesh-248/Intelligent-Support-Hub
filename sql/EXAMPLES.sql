-- ============================================================================
-- BigQuery AI Showcase - Complete Examples
-- ============================================================================
-- This file demonstrates the full power of BigQuery's vector search
-- capabilities through practical examples you can run directly.
-- ============================================================================

-- ============================================================================
-- EXAMPLE 1: Basic Semantic Search
-- ============================================================================
-- Find tickets similar to a new support request
-- This is the foundation of the intelligent triage system

SELECT
    ticket_id,
    title,
    category,
    ROUND((1 - distance) * 100, 1) as similarity_percentage,
    status,
    resolution
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
    'embedding',
    (
        -- Generate embedding for the query text
        SELECT ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `text-embedding-004`,
            (SELECT 'my application keeps freezing and crashing' AS content)
        )
    ),
    distance_type => 'COSINE',
    top_k => 5
)
ORDER BY distance ASC;

-- Expected Output:
-- High similarity (80-95%) to tickets about:
-- • App crashes
-- • Freezing issues
-- • Performance problems
-- Even if they use different words!

-- ============================================================================
-- EXAMPLE 2: Intelligent Solution Recommendation
-- ============================================================================
-- This is the complete recommendation engine in action
-- Finds similar resolved tickets and ranks their solutions

WITH query_embedding AS (
    -- Step 1: Generate embedding for new ticket
    SELECT ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
        MODEL `text-embedding-004`,
        (SELECT 'cannot log into my account password not working' AS content)
    )
),
similar_resolved_tickets AS (
    -- Step 2: Find semantically similar resolved tickets
    SELECT
        base.ticket_id,
        base.title,
        base.resolution,
        base.satisfaction_score,
        base.resolution_time_hours,
        distance,
        ROUND((1 - distance) * 100, 1) as similarity_pct
    FROM VECTOR_SEARCH(
        TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
        'embedding',
        (SELECT embedding FROM query_embedding),
        distance_type => 'COSINE',
        top_k => 20
    ) base
    WHERE base.status = 'resolved'
      AND base.resolution IS NOT NULL
      AND base.satisfaction_score >= 4  -- Only high-satisfaction resolutions
      AND distance < 0.3  -- High similarity threshold
),
solution_ranking AS (
    -- Step 3: Aggregate and rank solutions
    SELECT
        resolution,
        COUNT(*) as frequency,
        AVG(satisfaction_score) as avg_satisfaction,
        AVG(resolution_time_hours) as avg_resolution_time,
        AVG(similarity_pct) as avg_similarity,
        -- Calculate confidence score
        ROUND(
            (COUNT(*) * 0.3 +              -- Frequency weight
             AVG(satisfaction_score) * 15 + -- Satisfaction weight
             AVG(similarity_pct) * 0.5) /   -- Similarity weight
            (GREATEST(AVG(resolution_time_hours), 1) * 0.1), -- Time penalty
            2
        ) as confidence_score,
        ARRAY_AGG(
            STRUCT(ticket_id, title, similarity_pct) 
            ORDER BY similarity_pct DESC 
            LIMIT 3
        ) as supporting_tickets
    FROM similar_resolved_tickets
    GROUP BY resolution
)
-- Step 4: Return top recommendation
SELECT
    resolution as recommended_solution,
    confidence_score,
    frequency as based_on_tickets,
    ROUND(avg_satisfaction, 1) as avg_customer_satisfaction,
    ROUND(avg_resolution_time, 1) as avg_hours_to_resolve,
    ROUND(avg_similarity, 1) as avg_similarity_pct,
    supporting_tickets
FROM solution_ranking
ORDER BY confidence_score DESC
LIMIT 1;

-- Expected Output:
-- • Recommended solution (e.g., "Reset password and clear cache...")
-- • Confidence: 75-95 (out of 100)
-- • Based on: 5-15 similar tickets
-- • Avg satisfaction: 4.5-5.0
-- • Supporting evidence: 3 example tickets

-- ============================================================================
-- EXAMPLE 3: Auto-Categorization Using Semantic Voting
-- ============================================================================
-- Automatically categorize tickets based on similar historical tickets

WITH query_embedding AS (
    SELECT ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
        MODEL `text-embedding-004`,
        (SELECT 'payment declined during checkout process' AS content)
    )
),
similar_tickets AS (
    SELECT
        base.category,
        base.priority,
        distance,
        ROUND((1 - distance) * 100, 1) as similarity_pct
    FROM VECTOR_SEARCH(
        TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
        'embedding',
        (SELECT embedding FROM query_embedding),
        distance_type => 'COSINE',
        top_k => 10
    ) base
)
-- Vote-based classification
SELECT
    category as predicted_category,
    COUNT(*) as votes,
    ROUND(AVG(similarity_pct), 1) as avg_similarity,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as confidence_percentage
FROM similar_tickets
GROUP BY category
ORDER BY votes DESC, avg_similarity DESC
LIMIT 1;

-- Expected Output:
-- • Predicted Category: "Payment" or "Billing"
-- • Votes: 7-10 (out of 10 similar tickets)
-- • Confidence: 70-100%

-- ============================================================================
-- EXAMPLE 4: Duplicate Detection
-- ============================================================================
-- Find potential duplicate tickets using semantic similarity

DECLARE new_ticket_text STRING DEFAULT 'app crashing on startup';

WITH query_embedding AS (
    SELECT ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
        MODEL `text-embedding-004`,
        (SELECT new_ticket_text AS content)
    )
)
SELECT
    base.ticket_id,
    base.title,
    base.status,
    base.created_at,
    ROUND((1 - distance) * 100, 1) as similarity_pct,
    -- Duplicate likelihood
    CASE
        WHEN distance < 0.05 THEN 'Very Likely Duplicate'
        WHEN distance < 0.10 THEN 'Likely Duplicate'
        WHEN distance < 0.20 THEN 'Possibly Related'
        ELSE 'Different Issue'
    END as duplicate_assessment
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
    'embedding',
    (SELECT embedding FROM query_embedding),
    distance_type => 'COSINE',
    top_k => 10
) base
WHERE distance < 0.20  -- Only show potential matches
ORDER BY distance ASC;

-- Expected Output:
-- High-similarity tickets that likely describe the same issue
-- Can be used to link related tickets or mark duplicates

-- ============================================================================
-- EXAMPLE 5: Trend Detection - Clustering Similar Issues
-- ============================================================================
-- Identify trending issues by finding clusters of similar recent tickets

WITH recent_tickets AS (
    SELECT ticket_id, title, created_at, embedding
    FROM `{project_id}.{dataset_id}.tickets_with_embeddings`
    WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
),
similarity_pairs AS (
    SELECT
        t1.ticket_id as anchor_ticket,
        t1.title as anchor_title,
        t1.created_at as anchor_date,
        base.ticket_id as similar_ticket,
        base.title as similar_title,
        distance
    FROM recent_tickets t1
    CROSS JOIN LATERAL (
        SELECT ticket_id, title, distance
        FROM VECTOR_SEARCH(
            TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
            'embedding',
            t1.embedding,
            distance_type => 'COSINE',
            top_k => 5
        )
        WHERE distance < 0.15  -- Very similar only
          AND ticket_id != t1.ticket_id
    ) base
)
-- Find trending clusters
SELECT
    anchor_title as representative_issue,
    COUNT(DISTINCT similar_ticket) as cluster_size,
    MIN(anchor_date) as first_seen,
    ARRAY_AGG(DISTINCT similar_title LIMIT 5) as related_issues
FROM similarity_pairs
GROUP BY anchor_title
HAVING cluster_size >= 3  -- At least 3 similar tickets
ORDER BY cluster_size DESC, first_seen DESC
LIMIT 10;

-- Expected Output:
-- Trending issues in the last 7 days
-- Example: "Login errors" affecting multiple users
-- Helps prioritize systemic issues

-- ============================================================================
-- EXAMPLE 6: Multi-Factor Semantic Search
-- ============================================================================
-- Combine semantic search with metadata filters for precise results

WITH query_embedding AS (
    SELECT ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
        MODEL `text-embedding-004`,
        (SELECT 'slow performance issues' AS content)
    )
)
SELECT
    base.ticket_id,
    base.title,
    base.category,
    base.priority,
    base.created_at,
    ROUND((1 - distance) * 100, 1) as semantic_similarity,
    base.satisfaction_score,
    -- Composite score combining semantic + metadata
    ROUND(
        (1 - distance) * 80 +                    -- Semantic weight: 80%
        CASE base.priority
            WHEN 'high' THEN 15
            WHEN 'medium' THEN 10
            ELSE 5
        END +                                     -- Priority boost
        COALESCE(base.satisfaction_score * 1, 0) -- Satisfaction boost
    , 1) as composite_score
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
    'embedding',
    (SELECT embedding FROM query_embedding),
    distance_type => 'COSINE',
    top_k => 20
) base
WHERE base.category = 'Performance'  -- Category filter
  AND base.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)  -- Recency
ORDER BY composite_score DESC
LIMIT 10;

-- Expected Output:
-- Best matches considering:
-- • Semantic similarity (most important)
-- • Priority level (high priority preferred)
-- • Customer satisfaction (successful resolutions)
-- • Recency (recent tickets more relevant)

-- ============================================================================
-- EXAMPLE 7: Batch Embedding Generation
-- ============================================================================
-- Generate embeddings for multiple tickets at once (efficient!)

-- This shows how to batch process for better performance

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.new_tickets_with_embeddings` AS
SELECT
    ticket_id,
    title,
    description,
    combined_text,
    ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
    MODEL `text-embedding-004`,
    (
        SELECT 
            ticket_id,
            title,
            description,
            CONCAT(title, ' ', description) as combined_text,
            combined_text AS content  -- Field to embed
        FROM `{project_id}.{dataset_id}.tickets`
        -- Can process thousands of tickets in one query!
    ),
    STRUCT('content' AS content_column, TRUE AS flatten_json_output)
);

-- Verify embeddings
SELECT
    COUNT(*) as total_tickets,
    COUNT(embedding) as tickets_with_embeddings,
    ROUND(COUNT(embedding) * 100.0 / COUNT(*), 2) as coverage_percentage,
    AVG(ARRAY_LENGTH(embedding)) as avg_embedding_dimensions
FROM `{project_id}.{dataset_id}.new_tickets_with_embeddings`;

-- Expected Output:
-- • Total: 500 tickets
-- • With embeddings: 500 (100%)
-- • Dimensions: 768 (text-embedding-004)

-- ============================================================================
-- EXAMPLE 8: Performance Comparison - Brute Force vs Index
-- ============================================================================
-- Compare search performance with and without vector index

-- WITHOUT INDEX (Brute Force - Exact Search)
SELECT
    'Brute Force' as search_type,
    COUNT(*) as results_found,
    -- Note: Timing must be measured externally
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
    'embedding',
    (
        SELECT ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `text-embedding-004`,
            (SELECT 'test query' AS content)
        )
    ),
    distance_type => 'COSINE',
    top_k => 10,
    options => '{"use_brute_force": true}'  -- Force exact search
);

-- WITH INDEX (Approximate Nearest Neighbor - Fast Search)
-- Requires vector index to be created first (see 3_create_vector_index.sql)
SELECT
    'With Index (ANN)' as search_type,
    COUNT(*) as results_found
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
    'embedding',
    (
        SELECT ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `text-embedding-004`,
            (SELECT 'test query' AS content)
        )
    ),
    distance_type => 'COSINE',
    top_k => 10,
    options => '{"use_brute_force": false}'  -- Use index
);

-- Performance comparison:
-- Small dataset (< 10K): Similar performance
-- Medium dataset (10K-1M): 2-5x faster with index
-- Large dataset (> 1M): 10-100x faster with index

-- ============================================================================
-- EXAMPLE 9: Embedding Quality Analysis
-- ============================================================================
-- Analyze the quality and distribution of embeddings

SELECT
    'Embedding Statistics' as metric_type,
    COUNT(*) as total_embeddings,
    AVG(ARRAY_LENGTH(embedding)) as avg_dimensions,
    MIN(ARRAY_LENGTH(embedding)) as min_dimensions,
    MAX(ARRAY_LENGTH(embedding)) as max_dimensions,
    -- Sample statistics from first dimension
    AVG(embedding[OFFSET(0)]) as avg_first_dim,
    STDDEV(embedding[OFFSET(0)]) as stddev_first_dim
FROM `{project_id}.{dataset_id}.tickets_with_embeddings`
WHERE embedding IS NOT NULL;

-- Check for potential issues
SELECT
    ticket_id,
    title,
    ARRAY_LENGTH(embedding) as embedding_dims,
    CASE
        WHEN embedding IS NULL THEN 'Missing embedding'
        WHEN ARRAY_LENGTH(embedding) != 768 THEN 'Wrong dimensions'
        ELSE 'OK'
    END as embedding_status
FROM `{project_id}.{dataset_id}.tickets_with_embeddings`
WHERE embedding IS NULL 
   OR ARRAY_LENGTH(embedding) != 768
LIMIT 10;

-- Expected Output:
-- • All embeddings should be 768 dimensions
-- • No NULL embeddings
-- • Normally distributed values

-- ============================================================================
-- EXAMPLE 10: Real-World Production Query
-- ============================================================================
-- Complete production-ready query with all best practices

-- Input parameters (in production, these come from application)
DECLARE new_ticket_title STRING DEFAULT 'Application not responding';
DECLARE new_ticket_description STRING DEFAULT 'The app freezes when I try to save my work. This happens every time.';
DECLARE similarity_threshold FLOAT64 DEFAULT 0.70;
DECLARE min_satisfaction INT64 DEFAULT 4;

WITH
-- Step 1: Generate embedding for new ticket
query_embedding AS (
    SELECT ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
        MODEL `text-embedding-004`,
        (SELECT CONCAT(new_ticket_title, ' ', new_ticket_description) AS content)
    )
),
-- Step 2: Find similar tickets
similar_tickets AS (
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
        distance,
        ROUND((1 - distance) * 100, 1) as similarity_score
    FROM VECTOR_SEARCH(
        TABLE `{project_id}.{dataset_id}.tickets_with_embeddings`,
        'embedding',
        (SELECT embedding FROM query_embedding),
        distance_type => 'COSINE',
        top_k => 20
    ) base
    WHERE (1 - distance) >= similarity_threshold  -- Similarity filter
),
-- Step 3: Auto-categorize
category_votes AS (
    SELECT
        category,
        COUNT(*) as votes,
        ROUND(AVG(similarity_score), 1) as avg_similarity
    FROM similar_tickets
    GROUP BY category
    ORDER BY votes DESC, avg_similarity DESC
    LIMIT 1
),
-- Step 4: Get best solution from resolved tickets
best_solution AS (
    SELECT
        resolution,
        COUNT(*) as frequency,
        AVG(satisfaction_score) as avg_satisfaction,
        AVG(resolution_time_hours) as avg_time,
        AVG(similarity_score) as avg_similarity
    FROM similar_tickets
    WHERE status = 'resolved'
      AND resolution IS NOT NULL
      AND satisfaction_score >= min_satisfaction
    GROUP BY resolution
    ORDER BY 
        frequency DESC,
        avg_satisfaction DESC,
        avg_time ASC
    LIMIT 1
)
-- Step 5: Return comprehensive triage result
SELECT
    -- Input
    new_ticket_title as ticket_title,
    new_ticket_description as ticket_description,
    
    -- Categorization
    (SELECT category FROM category_votes) as suggested_category,
    (SELECT votes FROM category_votes) as category_confidence_votes,
    
    -- Recommendation
    (SELECT resolution FROM best_solution) as recommended_solution,
    (SELECT frequency FROM best_solution) as solution_frequency,
    (SELECT ROUND(avg_satisfaction, 1) FROM best_solution) as solution_satisfaction,
    (SELECT ROUND(avg_time, 1) FROM best_solution) as solution_avg_hours,
    
    -- Similar tickets for reference
    ARRAY(
        SELECT AS STRUCT ticket_id, title, similarity_score
        FROM similar_tickets
        ORDER BY similarity_score DESC
        LIMIT 5
    ) as similar_historical_tickets,
    
    -- Metadata
    CURRENT_TIMESTAMP() as triage_timestamp;

-- Expected Output:
-- Complete triage result ready for production use:
-- • Suggested category and confidence
-- • Recommended solution with metrics
-- • Supporting evidence (similar tickets)
-- • All in one query!

-- ============================================================================
-- Performance Notes
-- ============================================================================

-- Embedding Generation:
-- • Single ticket: ~200ms
-- • Batch (100): ~10 seconds
-- • Batch (1000): ~90 seconds

-- Vector Search:
-- • No index, 1K rows: ~300ms
-- • No index, 10K rows: ~1s
-- • With index, 1M rows: ~300ms
-- • With index, 10M rows: ~500ms

-- Best Practices:
-- 1. Batch embed when possible
-- 2. Use vector index for > 1M rows
-- 3. Filter results early (WHERE clauses)
-- 4. Cache common queries
-- 5. Monitor query costs

-- ============================================================================
-- END OF EXAMPLES
-- ============================================================================
