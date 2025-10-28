#!/usr/bin/env python3
"""
Setup script for creating Vertex AI connection and verifying permissions
"""
import os
import subprocess
import sys
from dotenv import load_dotenv

def run_command(cmd, check=True):
    """Run a shell command and return output"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(result.stdout)
        if result.stderr and check:
            print(f"Warning: {result.stderr}")
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False, e.stderr

def main():
    # Load environment variables
    load_dotenv()
    project_id = os.getenv('GCP_PROJECT_ID', 'big-query-project-476503')
    
    print(f"\n🔧 Setting up Vertex AI connection for project: {project_id}")
    print("="*70)
    
    # Step 1: Set the project
    print("\n1️⃣ Setting active project...")
    success, _ = run_command(f"gcloud config set project {project_id}")
    if not success:
        print("❌ Failed to set project. Make sure you're authenticated and have access.")
        sys.exit(1)
    
    # Step 2: Enable required APIs
    print("\n2️⃣ Enabling required APIs...")
    apis = [
        "bigquery.googleapis.com",
        "aiplatform.googleapis.com", 
        "bigqueryconnection.googleapis.com",
        "cloudresourcemanager.googleapis.com"
    ]
    
    for api in apis:
        print(f"   Enabling {api}...")
        run_command(f"gcloud services enable {api} --project {project_id}", check=False)
    
    # Step 3: Create the Vertex AI connection
    print("\n3️⃣ Creating Vertex AI connection...")
    connection_cmd = f'bq mk --connection --location=US --connection_type=CLOUD_RESOURCE --project_id={project_id} vertex-ai'
    success, output = run_command(connection_cmd, check=False)
    
    if "already exists" in output.lower() or success:
        print("   ✅ Connection 'vertex-ai' is ready")
    else:
        print("   ⚠️ Could not create connection. It may already exist or you need permissions.")
    
    # Step 4: List connections to verify
    print("\n4️⃣ Verifying connections...")
    run_command(f"bq ls --connection --project_id={project_id} --location=US", check=False)
    
    # Step 5: Grant service account permissions (if connection was created)
    print("\n5️⃣ Setting up service account permissions...")
    print("   Note: The connection service account needs Vertex AI User role")
    print("   You may need to manually grant this in Cloud Console")
    
    # Get the service account from the connection
    get_sa_cmd = f"bq show --connection --project_id={project_id} --location=US vertex-ai --format=json"
    success, output = run_command(get_sa_cmd, check=False)
    
    if success and "serviceAccountId" in output:
        import json
        try:
            conn_info = json.loads(output)
            service_account = conn_info.get('cloudResource', {}).get('serviceAccountId', '')
            if service_account:
                print(f"   Connection service account: {service_account}")
                print(f"   Granting Vertex AI User role...")
                grant_cmd = f'gcloud projects add-iam-policy-binding {project_id} --member="serviceAccount:{service_account}" --role="roles/aiplatform.user"'
                run_command(grant_cmd, check=False)
        except:
            pass
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Run: python main.py --setup")
    print("2. If embeddings fail, try setting EMBEDDING_MODEL='text-embedding-004' in .env")
    
if __name__ == "__main__":
    main()