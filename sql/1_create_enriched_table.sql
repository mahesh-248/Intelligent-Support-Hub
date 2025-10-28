-- ============================================================================
-- Step 1: Create Support Tickets Table
-- ============================================================================
-- This table stores historical support tickets with their metadata and resolutions
-- It includes a combined_text field for embedding generation
-- ============================================================================

-- Create or replace the main support tickets table
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_name}` (
    -- Primary identifiers
    ticket_id STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    
    -- Ticket content
    title STRING NOT NULL,
    description STRING NOT NULL,
    category STRING NOT NULL,
    priority STRING NOT NULL,
    
    -- Status and resolution
    status STRING NOT NULL,  -- open, in_progress, resolved, closed
    resolution STRING,        -- Solution or fix applied (null if not resolved)
    resolved_at TIMESTAMP,
    resolution_time_hours FLOAT64,
    
    -- Customer information
    customer_id STRING,
    customer_tier STRING,  -- basic, premium, enterprise
    
    -- Agent information
    assigned_agent_id STRING,
    assigned_team STRING,
    
    -- Metrics
    satisfaction_score INT64,  -- 1-5 rating
    reopened_count INT64 DEFAULT 0,
    
    -- Combined text for embedding (title + description for semantic search)
    combined_text STRING NOT NULL,
    
    -- Tags for categorization
    tags ARRAY<STRING>,
    
    -- Metadata
    source_channel STRING,  -- email, chat, phone, portal
    language STRING DEFAULT 'en'
)
PARTITION BY DATE(created_at)
CLUSTER BY category, status, priority
OPTIONS(
    description="Historical support tickets with resolutions for semantic search and triage",
    labels=[("purpose", "ml_training"), ("type", "support_data")]
);

-- ============================================================================
-- Create indexes for common queries
-- ============================================================================

-- Note: BigQuery automatically optimizes partitioned and clustered tables
-- Additional indexes are typically not needed unless using specific features

-- ============================================================================
-- Sample Data Insertion (for testing - will be replaced by data_loader.py)
-- ============================================================================

-- This is a placeholder - actual data will be loaded programmatically
-- Uncomment below to insert a few sample records for testing:

/*
INSERT INTO `{project_id}.{dataset_id}.{table_name}` VALUES
(
    'TKT-001',
    TIMESTAMP('2024-01-15 10:30:00'),
    TIMESTAMP('2024-01-15 14:45:00'),
    'Cannot log into my account',
    'I have been trying to log into my account for the past hour but keep getting an "invalid credentials" error. I am sure my password is correct as I saved it in my password manager.',
    'Authentication',
    'high',
    'resolved',
    'Reset password and cleared browser cache. Issue was caused by expired session tokens. Advised customer to use incognito mode if issue persists.',
    TIMESTAMP('2024-01-15 14:45:00'),
    4.25,
    'CUST-12345',
    'premium',
    'AGENT-101',
    'authentication_team',
    5,
    0,
    'Cannot log into my account I have been trying to log into my account for the past hour but keep getting an "invalid credentials" error. I am sure my password is correct as I saved it in my password manager.',
    ['login', 'authentication', 'credentials'],
    'email',
    'en'
);
*/

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Count total tickets
-- SELECT COUNT(*) as total_tickets FROM `{project_id}.{dataset_id}.{table_name}`;

-- Check tickets by category
-- SELECT category, COUNT(*) as count 
-- FROM `{project_id}.{dataset_id}.{table_name}` 
-- GROUP BY category 
-- ORDER BY count DESC;

-- Check resolution rate
-- SELECT 
--   status,
--   COUNT(*) as count,
--   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
-- FROM `{project_id}.{dataset_id}.{table_name}`
-- GROUP BY status;
