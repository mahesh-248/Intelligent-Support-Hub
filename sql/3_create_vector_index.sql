-- ============================================================================
-- Step 3: Create Vector Index for Optimized Search
-- ============================================================================
-- Vector indexes enable sub-second search on millions of tickets
-- Recommended for datasets > 1 million rows
-- Uses Approximate Nearest Neighbor (ANN) search for performance
-- ============================================================================

-- ============================================================================
-- Create Vector Index
-- ============================================================================

CREATE VECTOR INDEX IF NOT EXISTS `{index_name}`
ON `{project_id}.{dataset_id}.{embeddings_table_name}`(embedding)
OPTIONS(
    distance_type = 'COSINE',
    index_type = 'IVF',
    ivf_options = '{"num_lists": 1000}'
);

-- ============================================================================
-- Index Configuration Options
-- ============================================================================

-- distance_type options:
--   - COSINE: Cosine similarity (most common, range: -1 to 1)
--   - EUCLIDEAN: Euclidean distance (L2)
--   - DOT_PRODUCT: Dot product similarity

-- index_type options:
--   - IVF: Inverted File Index (recommended, good balance of speed/accuracy)
--   - BRUTE_FORCE: Exact search (slower but 100% accurate)

-- IVF options:
--   - num_lists: Number of partitions (typically sqrt(N) to 2*sqrt(N))
--     * 100K rows: 300-600 lists
--     * 1M rows: 1000-2000 lists
--     * 10M rows: 3000-6000 lists

-- ============================================================================
-- Alternative: Tree-Based Index (for smaller datasets)
-- ============================================================================

-- Uncomment for tree-based indexing (better for < 1M rows):
/*
CREATE VECTOR INDEX IF NOT EXISTS `{index_name}`
ON `{project_id}.{dataset_id}.{embeddings_table_name}`(embedding)
OPTIONS(
    distance_type = 'COSINE',
    index_type = 'TREE_AH',
    tree_ah_options = '{"max_leaf_size": 100}'
);
*/

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Check index status
-- SELECT 
--     index_name,
--     table_name,
--     index_status,
--     coverage_percentage,
--     last_refresh_time,
--     ddl
-- FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
-- WHERE table_name = '{embeddings_table_name}';

-- Test search performance with index
-- SELECT 
--     ticket_id,
--     title,
--     category,
--     distance
-- FROM VECTOR_SEARCH(
--     TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
--     'embedding',
--     (
--         SELECT ml_generate_embedding_result AS embedding
--         FROM ML.GENERATE_EMBEDDING(
--             MODEL `{embedding_model}`,
--             (SELECT 'my application is crashing' AS content)
--         )
--     ),
--     distance_type => 'COSINE',
--     top_k => 10,
--     options => '{"use_brute_force": false}'  -- Use index
-- );

-- ============================================================================
-- Performance Comparison: With vs Without Index
-- ============================================================================

-- Without index (brute force):
/*
DECLARE start_time TIMESTAMP;
SET start_time = CURRENT_TIMESTAMP();

SELECT ticket_id, title, distance
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
    'embedding',
    (SELECT ml_generate_embedding_result AS embedding FROM ML.GENERATE_EMBEDDING(MODEL `{embedding_model}`, (SELECT 'test query' AS content))),
    distance_type => 'COSINE',
    top_k => 10,
    options => '{"use_brute_force": true}'
);

SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), start_time, MILLISECOND) as brute_force_ms;
*/

-- With index (ANN):
/*
SET start_time = CURRENT_TIMESTAMP();

SELECT ticket_id, title, distance
FROM VECTOR_SEARCH(
    TABLE `{project_id}.{dataset_id}.{embeddings_table_name}`,
    'embedding',
    (SELECT ml_generate_embedding_result AS embedding FROM ML.GENERATE_EMBEDDING(MODEL `{embedding_model}`, (SELECT 'test query' AS content))),
    distance_type => 'COSINE',
    top_k => 10,
    options => '{"use_brute_force": false}'
);

SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), start_time, MILLISECOND) as index_search_ms;
*/

-- ============================================================================
-- Index Maintenance
-- ============================================================================

-- Drop index (if needed):
-- DROP VECTOR INDEX IF EXISTS `{index_name}` ON `{project_id}.{dataset_id}.{embeddings_table_name}`;

-- Refresh index (automatically handled by BigQuery):
-- Indexes are automatically refreshed when data changes
-- Manual refresh is typically not needed

-- ============================================================================
-- Notes on Vector Indexes
-- ============================================================================

-- When to use:
--   ✅ Dataset > 1 million rows
--   ✅ Query latency is critical (need sub-second search)
--   ✅ Slight accuracy tradeoff acceptable (typically 95%+ accuracy)

-- When NOT to use:
--   ❌ Small datasets (< 100K rows) - brute force is fast enough
--   ❌ Need 100% accuracy - use brute force search
--   ❌ Frequent data updates - index rebuild overhead

-- Performance gains:
--   - 10-100x faster for large datasets
--   - Sub-second search on 10M+ rows
--   - Reduced query costs (fewer resources used)

-- Accuracy:
--   - IVF: Typically 95-99% recall@10
--   - Tree-AH: Typically 98-99% recall@10
--   - Tuning num_lists/max_leaf_size can improve accuracy

-- Cost considerations:
--   - Index storage: Small overhead (typically < 10% of data size)
--   - Index build time: One-time cost (minutes to hours for large datasets)
--   - Query cost: Lower due to reduced computation
