# Amazon Fine Food Reviews - Semantic Search System 🍎🔍

A production-ready semantic search system leveraging **BigQuery's native vector search capabilities** to find similar product reviews and extract insights from the Amazon Fine Food Reviews dataset.

## 🎯 Project Overview

This system demonstrates advanced semantic search capabilities using BigQuery ML and Gemini embeddings to analyze over **568,000 Amazon food product reviews** spanning 10+ years (1999-2012).

### What Makes This Different?

Traditional review search relies on keyword matching. This system understands **meaning and context**:

**Traditional Search:** "chocolate" only finds reviews with that exact word  
**Semantic Search:** Finds reviews about "cocoa", "sweet candy", "dark confection", etc.

### Key Capabilities

- 🔍 **Semantic Review Search**: Find similar reviews based on meaning, not just keywords
- 📊 **Product Analysis**: Aggregate insights across all reviews for a product
- ⭐ **Quality Filtering**: Find the most helpful and highly-rated reviews
- 🎯 **Smart Recommendations**: Suggest products based on review similarity
- 📈 **Sentiment Analysis**: Understand review sentiment patterns

---

## 📦 Dataset: Amazon Fine Food Reviews

**Source:** [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

### Dataset Statistics

- **568,454 reviews** from October 1999 - October 2012
- **256,059 unique users**
- **74,258 unique products**
- **260 users** with > 50 reviews each

### Dataset Columns

| Column | Description |
|--------|-------------|
| `Id` | Review ID (1-568,454) |
| `ProductId` | Unique product identifier (e.g., B001E4KFG0) |
| `UserId` | Unique user identifier |
| `ProfileName` | User's profile name |
| `HelpfulnessNumerator` | # users who found review helpful |
| `HelpfulnessDenominator` | # users who voted on helpfulness |
| `Score` | Rating between 1-5 stars |
| `Time` | Unix timestamp of review |
| `Summary` | Brief review summary |
| `Text` | Full review text |

### Sample Reviews

```
⭐⭐⭐⭐⭐ (5/5) - "Good Quality Dog Food"
I have bought several of the Vitality canned dog food products and have found them all to be 
of good quality. The product looks more like a stew than a processed meat and it smells better.

⭐ (1/5) - "Not as Advertised"
Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted.

⭐⭐⭐⭐ (4/5) - "Delight says it all"
This is a confection that has been around a few centuries. It is a light, pillowy citrus gelatin
with nuts - in this case Filberts. And it is cut into tiny squares...
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          Amazon Fine Food Reviews Dataset (CSV)             │
│                    568,454 Reviews                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Load into BigQuery Table                        │
│      (reviews table with all metadata & text)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         BigQuery ML.GENERATE_EMBEDDING (Gemini)             │
│       Convert "summary + text" to vector embeddings          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Store Embeddings in BigQuery Table                  │
│         (reviews_with_embeddings table)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              BigQuery VECTOR_SEARCH                          │
│         Find semantically similar reviews                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Review Analysis & Insights                         │
│  • Product recommendations                                    │
│  • Helpful review discovery                                   │
│  • Sentiment patterns                                         │
│  • Quality filtering                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

1. **Google Cloud Project** with BigQuery API enabled
2. **Service Account** with BigQuery Admin permissions
3. **Python 3.9-3.12** (not 3.13+, see troubleshooting)
4. **Amazon Fine Food Reviews dataset** (Reviews.csv)

### Step 1: Download Dataset

```bash
# Download from Kaggle:
# https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

# Place Reviews.csv in the project root directory
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your GCP project details
notepad .env  # Windows
# or
nano .env     # Linux/Mac
```

Example `.env`:
```bash
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json
BQ_DATASET=amazon_food_reviews
REVIEW_SAMPLE_SIZE=10000  # Start with 10k for testing
```

### Step 4: Setup BigQuery Infrastructure

```bash
python main.py --setup
```

This will:
- ✅ Create BigQuery dataset
- ✅ Load reviews from CSV
- ✅ Generate embeddings using Gemini
- ✅ Create optimized tables
- ⏱️ Takes 5-15 minutes depending on sample size

### Step 5: Run Searches!

```bash
# Search for similar reviews
python main.py --search "delicious chocolate candy"

# Find reviews for a specific product
python main.py --product B001E4KFG0

# Find highly helpful reviews
python main.py --helpful --min-score 5

# Get more results
python main.py --search "healthy organic snacks" --top-k 20
```

---

## 💡 Usage Examples

### Example 1: Semantic Review Search

Find reviews similar to a query:

```bash
python main.py --search "healthy organic natural ingredients"
```

**Output:**
```
🔍 Searching for reviews similar to: 'healthy organic natural ingredients'
======================================================================

Found 10 similar reviews:

1. ⭐⭐⭐⭐⭐ (5/5) - Similarity: 94.2%
   Product: B000LKTTTW
   Summary: Great organic snack
   Review: These organic fruit bars are made with natural ingredients...
   Helpful: 92% | Very Positive

2. ⭐⭐⭐⭐⭐ (5/5) - Similarity: 91.8%
   Product: B001EO5Q64
   Summary: Healthy and delicious
   Review: Finally found a healthy snack that tastes great...
```

### Example 2: Product Analysis

```bash
python main.py --product B001E4KFG0
```

Find all reviews for a specific product with quality filtering.

### Example 3: Find Helpful Reviews

```bash
python main.py --helpful --min-score 5
```

Discover the most helpful 5-star reviews across all products.

### Example 4: Programmatic Usage

```python
from src.data_loader import ReviewDataLoader
from src.utils.bq import BigQueryClient

# Load and analyze reviews
loader = ReviewDataLoader("Reviews.csv", sample_size=10000)
reviews_df = loader.load_reviews()

stats = loader.get_statistics(reviews_df)
print(f"Loaded {stats['total_reviews']} reviews")
print(f"Score distribution: {stats['score_distribution']}")

# Query BigQuery for semantic search
bq = BigQueryClient()
results = bq.query_to_dataframe("""
    SELECT * FROM `project.dataset.reviews_with_embeddings`
    WHERE score >= 4 AND is_helpful = TRUE
    LIMIT 10
""")
```

---

## 🎯 Use Cases

### 1. **Product Recommendation Engine**
- Find products with similar positive reviews
- Suggest alternatives based on review content

### 2. **Review Quality Analysis**
- Identify most helpful reviews
- Filter by helpfulness ratio and vote count
- Discover detailed, informative reviews

### 3. **Sentiment & Trend Analysis**
- Track sentiment patterns over time
- Identify common praise/complaint themes
- Analyze product category trends

### 4. **Customer Insights**
- Understand what customers value
- Find feature requests and pain points
- Discover unexpected use cases

### 5. **Competitive Analysis**
- Compare similar products
- Identify competitive advantages
- Understand market positioning

---

## 📁 Project Structure

```
Intelligent Support Hub/
├── README.md                          # This file
├── QUICKSTART.md                      # Quick start guide
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
├── .gitignore                         # Git ignore rules
├── Reviews.csv                        # Amazon dataset (download separately)
├── main.py                            # Main application
├── main_tickets_backup.py             # Original ticket system (backup)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # ReviewDataLoader class
│   ├── orchestrator.py                # Review search orchestrator
│   └── utils/
│       ├── __init__.py
│       └── bq.py                      # BigQuery utilities
│
├── sql/
│   ├── 1_create_enriched_table.sql    # Reviews table schema
│   ├── 2_generate_embeddings.sql      # Embedding generation
│   ├── 3_create_vector_index.sql      # Vector index (optional)
│   └── 4_semantic_search.sql          # Search query examples
│
├── demos/
│   ├── demo_basic_search.py           # Basic semantic search demo
│   └── demo_recommendation.py         # Product recommendation demo
│
└── tests/
    └── smoke.py                       # Smoke tests
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GCP_PROJECT_ID` | Your Google Cloud project ID | *Required* |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key | service-account-key.json |
| `BQ_DATASET` | BigQuery dataset name | amazon_food_reviews |
| `BQ_LOCATION` | BigQuery location | US |
| `REVIEW_SAMPLE_SIZE` | Number of reviews to load (0 = all) | 10000 |
| `EMBEDDING_MODEL` | Gemini embedding model | text-embedding-004 |
| `USE_VECTOR_INDEX` | Enable vector index for speed | false |
| `TOP_K_SIMILAR` | Default number of results | 10 |

### Sample Sizes

| Sample Size | Processing Time | Use Case |
|-------------|----------------|----------|
| 1,000 | 1-2 min | Quick testing |
| 10,000 | 5-10 min | Development |
| 50,000 | 15-30 min | Testing at scale |
| 568,454 (all) | 1-2 hours | Production |

---

## 🧪 Advanced Usage

### Batch Processing

Process multiple queries at once:

```python
queries = [
    "delicious chocolate candy",
    "healthy organic snacks",
    "coffee beans strong flavor"
]

for query in queries:
    results = search_similar_reviews(query, top_k=5)
    # Analyze results...
```

### Custom Filtering

```sql
-- Find helpful reviews for high-rated products
SELECT *
FROM `project.dataset.reviews_with_embeddings`
WHERE score >= 4
  AND helpfulness_ratio >= 0.8
  AND helpfulness_denominator >= 10
ORDER BY helpfulness_numerator DESC
LIMIT 100;
```

### Product Clustering

Group similar products based on review embeddings:

```sql
-- Coming soon: K-means clustering on review embeddings
```

---

## 🐛 Troubleshooting

### "Reviews.csv not found"

**Solution:** Download the dataset from Kaggle and place it in the project root.

### "TypeError: Metaclasses with custom tp_new are not supported"

**Problem:** Using Python 3.14 (unsupported)  
**Solution:** Use Python 3.9-3.12

```bash
# Check version
python --version

# Create venv with correct Python
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### "Permission Denied" / "Authentication Failed"

**Solution:** Check GCP authentication

```bash
# Option 1: Use service account
export GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json

# Option 2: Use gcloud auth
gcloud auth application-default login
```

### Slow Embedding Generation

**Solution:** Reduce sample size or enable batch processing

```bash
# In .env
REVIEW_SAMPLE_SIZE=5000  # Start smaller
```

### Out of Memory

**Solution:** Process in smaller batches

```python
# Process reviews in chunks
for chunk in pd.read_csv('Reviews.csv', chunksize=10000):
    # Process chunk...
```

---

## 📊 Performance

### Benchmarks (10,000 reviews)

| Operation | Time | Notes |
|-----------|------|-------|
| Setup (first time) | 5-10 min | One-time cost |
| Embedding generation | 3-7 min | Depends on batch size |
| Semantic search | < 2 sec | Per query |
| Vector index creation | 2-5 min | Optional, for scale |

### Scaling to Full Dataset (568K reviews)

- **Initial setup**: 1-2 hours
- **With vector index**: Searches remain < 2 sec
- **Storage**: ~500 MB for embeddings

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add product clustering
- [ ] Implement review summarization
- [ ] Build Streamlit dashboard
- [ ] Add sentiment timeline analysis
- [ ] Create REST API endpoints

---

## 📚 Resources

- [BigQuery Vector Search Docs](https://cloud.google.com/bigquery/docs/vector-search)
- [Gemini Embedding Models](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
- [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- [BigQuery ML Documentation](https://cloud.google.com/bigquery-ml/docs)

---

## 📄 License

This project is for educational purposes. The Amazon Fine Food Reviews dataset has its own license terms on Kaggle.

---

## 🎓 Learning Outcomes

By exploring this project, you'll learn:

- ✅ How to implement semantic search using BigQuery
- ✅ Working with vector embeddings (Gemini)
- ✅ BigQuery ML capabilities
- ✅ Processing large-scale review datasets
- ✅ Building production-ready data pipelines
- ✅ GCP authentication and service accounts

---

**Built with** ❤️ **using BigQuery ML & Gemini Embeddings**

*Transform how you search and analyze product reviews!*
