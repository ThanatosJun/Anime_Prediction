# Text Processing Branch

This module handles text preprocessing, embedding generation, and regression modeling for anime synopsis data.

## Directory Structure

```
src/text_branch/
├── __init__.py                     # Package initialization
├── README.md                       # This file
├── text_preprocessor.py            # Text cleaning and normalization
├── embedding_generator.py          # Sentence embedding generation
├── baseline_model.py               # Text-only regression baseline
├── configs/
│   └── embedding_config.yaml       # Configuration file
└── tests/
    ├── test_preprocessor.py        # Tests for text preprocessing
    ├── test_embeddings.py          # Tests for embedding generation
    └── test_baseline.py            # Tests for baseline model
```

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers pandas numpy scikit-learn scipy pyyaml
```

### 2. Prepare Data

Input file: `data/processed/anilist_anime_multimodal_input_v1.csv`

Split files:
- `data/processed/anilist_anime_multimodal_input_train.csv`
- `data/processed/anilist_anime_multimodal_input_val.csv`
- `data/processed/anilist_anime_multimodal_input_test.csv`
- `data/processed/anilist_anime_multimodal_input_holdout_unknown.csv`

### 3. Run Pipeline

```python
from src.text_branch.text_preprocessor import TextPreprocessor
from src.text_branch.embedding_generator import EmbeddingGenerator

# Preprocess
preprocessor = TextPreprocessor()
df = pd.read_csv("data/processed/anilist_anime_multimodal_input_v1.csv")
df_clean = preprocessor.process_dataframe(df, text_column="description")

# Generate embeddings
generator = EmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",  # or "cpu"
)
df_embeddings = generator.encode_dataframe(df_clean, text_column="description")

# Save
df_embeddings.to_parquet("artifacts/text_embeddings_v1.parquet")
```

## Configuration

Edit `configs/embedding_config.yaml` to customize:
- Model selection (Sentence-Transformers model)
- Text preprocessing settings
- Batch size and device
- Random seed (for reproducibility)

## Key Modules

### TextPreprocessor
- Cleans anime synopsis text
- Removes URLs, normalizes whitespace
- Enforces length constraints
- Returns preprocessing statistics

### EmbeddingGenerator
- Uses Sentence-Transformers for embedding
- Handles batch processing
- Manages device (GPU/CPU)
- Tracks model metadata for reproducibility

## Output

Embeddings are saved as:
- Format: `.parquet` (PyArrow format for efficient storage)
- Location: `artifacts/text_embeddings_*.parquet`
- Contains embedding vectors for each anime synopsis

Metrics are saved as:
- Format: `.json`
- Location: `reports/text_branch_metrics.json`
- Includes: MAE, RMSE, Spearman correlation for both targets

## Next Steps

1. ✅ Set up folder structure (done)
2. Run text preprocessing on all splits
3. Generate embeddings for all splits
4. Train baseline regression models (popularity + meanScore)
5. Generate evaluation metrics
6. Document model versions and reproducibility info
