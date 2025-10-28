# Quick Start Guide - Intelligent Support Hub

## 🚀 Get Started in 5 Minutes

### Step 1: Prerequisites (2 minutes)

1. **Google Cloud Project**

   - Create or use existing GCP project
   - Enable BigQuery API: https://console.cloud.google.com/apis/library/bigquery.googleapis.com

2. **Service Account** (Recommended)

   ```bash
   # Create service account
   gcloud iam service-accounts create bigquery-ai-demo \
       --display-name="BigQuery AI Demo"

   # Grant BigQuery Admin role
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
       --member="serviceAccount:bigquery-ai-demo@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
       --role="roles/bigquery.admin"

   # Download key
   gcloud iam service-accounts keys create service-account-key.json \
       --iam-account=bigquery-ai-demo@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

3. **Python 3.9+**
   ```bash
   python --version  # Should be 3.9 or higher
   ```

### Step 2: Installation (1 minute)

```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configuration (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your project ID
# Set GOOGLE_APPLICATION_CREDENTIALS to your service account key path
notepad .env  # or use your favorite editor
```

Example `.env`:

```bash
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json
BQ_DATASET=support_tickets
BQ_LOCATION=US
```

### Step 4: Setup (2 minutes)

```bash
# Run complete setup
python main.py --setup
```

This will:

- ✅ Create BigQuery dataset
- ✅ Create tables
- ✅ Generate 500 sample tickets
- ✅ Create embeddings using Gemini
- ✅ Ready to search!

### Step 5: Try It! (1 minute)

```bash
# Find similar tickets
python main.py --search "my app keeps crashing"

# Get solution recommendation
python main.py --recommend "cannot log into account"

# Run tests
python tests/smoke.py
```

---

## 🎯 Example Queries to Try

### Authentication Issues

```bash
python main.py --search "login error invalid password"
python main.py --recommend "cannot sign in to my account"
```

### Performance Problems

```bash
python main.py --search "application running very slow"
python main.py --recommend "app freezes constantly"
```

### Payment Issues

```bash
python main.py --search "payment declined at checkout"
python main.py --recommend "charged twice for subscription"
```

### General Issues

```bash
python main.py --search "pictures not loading"
python main.py --categorize "need help exporting my data"
```

---

## 📊 Understanding Results

### Similarity Scores

- **90-100%**: Nearly identical issues (likely duplicates)
- **80-89%**: Very similar, same core problem
- **70-79%**: Similar category, related issues
- **60-69%**: Somewhat related
- **< 60%**: Different issues, may not be relevant

### Confidence Scores (Recommendations)

- **80-100**: High confidence, safe to auto-apply
- **60-79**: Medium confidence, agent should review
- **40-59**: Low confidence, use as suggestion only
- **< 40**: Very low confidence, manual triage needed

---

## 🎨 Interactive Demos

### Basic Search Demo

```bash
python demos/demo_basic_search.py
```

Shows how semantic search works vs keyword matching

### Recommendation Demo

```bash
python demos/demo_recommendation.py
```

Full workflow: search → analyze → recommend → categorize

---

## 🔧 Troubleshooting

### "Permission Denied" Error

**Solution**: Make sure service account has `BigQuery Admin` role

### "Authentication Failed"

**Solution**:

```bash
# Option 1: Use service account
export GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json

# Option 2: Use gcloud auth
gcloud auth application-default login
```

### "Table Not Found"

**Solution**: Run setup first

```bash
python main.py --setup
```

### "Module Not Found"

**Solution**: Install dependencies

```bash
pip install -r requirements.txt
```

### Slow Searches (> 5 seconds)

**Solution**: Enable vector index

```bash
# Edit .env
USE_VECTOR_INDEX=true

# Re-run setup
python main.py --setup
```

---

## 📚 What's Next?

### Customize

1. **Add our Own Data**

   ```python
   from src.utils.bq import BigQueryClient
   import pandas as pd

   # Load your tickets
   df = pd.read_csv('your_tickets.csv')

   # Upload to BigQuery
   client = BigQueryClient()
   client.load_dataframe_to_table(df, 'tickets')

   # Generate embeddings
   # (run sql/2_generate_embeddings.sql)
   ```

2. **Adjust Parameters**
   Edit `.env`:

   ```bash
   TOP_K_SIMILAR=10          # More results
   MIN_SIMILARITY_SCORE=0.80  # Higher threshold
   ```

3. **Change Model**
   ```bash
   # For multilingual support
   EMBEDDING_MODEL=text-multilingual-embedding-002
   ```

### Integrate

- Connect to your ticketing system (Zendesk, ServiceNow, etc.)
- Build REST API endpoints
- Create web dashboard
- Add Slack bot integration

### Common Issues

- Check [Troubleshooting](#troubleshooting) section
- Review logs in `app.log`
- Run smoke tests: `python tests/smoke.py`

### Resources

- [BigQuery Vector Search Docs](https://cloud.google.com/bigquery/docs/vector-search)
- [Gemini Embedding Models](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
- [BigQuery ML Docs](https://cloud.google.com/bigquery-ml/docs)
