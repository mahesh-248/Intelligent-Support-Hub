-- ============================================================================
-- Step 0: Create Cloud Resource Connection and Remote Model
-- ============================================================================
-- This script:
-- 1. Creates a Cloud Resource Connection to Vertex AI (if not exists)
-- 2. Creates a remote model for text embeddings
-- 
-- Prerequisites:
-- - Vertex AI API must be enabled
-- - User must have permissions to create connections
-- ============================================================================

-- Note: Connection creation is done via command line or Console
-- Run this command first if connection doesn't exist:
-- Example (replace <PROJECT_ID>):
-- bq mk --connection --location=US --project_id=<PROJECT_ID> --connection_type=CLOUD_RESOURCE vertex-ai

-- Create remote model for text embeddings using Vertex AI
-- This uses the textembedding-gecko model (768 dimensions)
-- Inline creation now handled programmatically in main.py; keep for reference only.
-- To run manually after replacing placeholders:
-- CREATE OR REPLACE MODEL `<PROJECT_ID>.<DATASET_ID>.textembedding_model`
-- REMOTE WITH CONNECTION `<PROJECT_ID>.US.vertex-ai`
OPTIONS (
    ENDPOINT = 'text-embedding-004'
);
