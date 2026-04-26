# CLAUDE.md — Anime Prediction

## Project Overview
Pre-release anime popularity and score prediction using multimodal fusion (text + image + metadata + RAG).
Two regression targets: `popularity` and `meanScore`. Models are trained independently per target.
Environment: `conda activate animeprediction` (Python, PyTorch CUDA 12.8, RTX 5070 Ti).

---

## Commands

```bash
# Environment
conda activate animeprediction

# Data pipeline (run in order)
python scripts/build_interim_dataset.py
python scripts/build_processed_dataset.py
python scripts/export_multimodal_inputs.py

# Text embeddings
python -m src.text_branch.run_text_embedding_pipeline

# Image embeddings (requires images in data/image/ and checkpoint in results/01/best/)
python -m src.image_branch.run_fetch       # download cover images
python -m src.image_branch.run_train       # fine-tune Swin Transformer
python -m src.image_branch.run_predict     # inference → parquet

# Fusion RAG pipeline
python -m src.fussion_branch.run_rag       # build Qdrant + query all splits

# Fusion MLP training
python -m src.fussion_branch.run_train                     # train both targets
python -m src.fussion_branch.run_train --target popularity # train only popularity
python -m src.fussion_branch.run_train --target meanScore  # train only meanScore
```

---

## Architecture

```
src/
  text_branch/        all-MiniLM-L6-v2, outputs artifacts/text_embeddings_{split}.parquet (384-dim)
  image_branch/       Swin-base contrastive fine-tuning, outputs data/processed/image_embeddings.parquet (1024-dim)
  fussion_branch/     RAG (Qdrant sparse) + MLP fusion model

data/
  interim/            cleaned intermediate CSV
  processed/          feature-engineered CSV + multimodal input splits
  fussion/            fusion_meta_{train,val,test}.csv — 28 pre-release cols (no split flags), split counts: 13376/2918/3087
  image/              downloaded cover images (gitignored)

artifacts/
  text_embeddings_{split}.parquet   cols: id, emb_0..emb_383, popularity, meanScore, split
  rag_features_{split}.parquet      cols: id, rag_title_romaji, rag_popularity, rag_score, rag_release_year, rag_studios, rag_found
  sparse_encoder.json               genre+studio sparse vocab (1483 tokens: 19 genres + 1464 studios)

qdrant_db/            local Qdrant vector store (gitignored, rebuilt by run_rag)
results/              image branch checkpoints (gitignored)
```

---

## Code Conventions

- All modules run as `python -m src.<branch>.run_*` from the project root
- Imports use `from src.<branch>.<module> import ...` — no relative imports
- Configs are YAML files under `src/<branch>/configs/`
- Parquet is the standard format for embedding outputs; CSV for metadata
- `id` (AniList integer) is the join key across all datasets
- `studios` column in CSVs is a JSON string — parse with `json.loads()`, not `ast.literal_eval()` (contains JSON booleans)
- `genres` column is a Python list string — parse with `ast.literal_eval()`

---

## Data Rules

- **Temporal split**: train = earlier quarters, val/test = later quarters (chronological, not random)
- **holdout_unknown** (943 rows, missing `release_quarter`) is excluded from all model training and evaluation
- **Pre-release constraint**: only features knowable before airing are allowed as model inputs; drop `averageScore`, `favourites`, `trending`, `status`
- **RAG leakage prevention**: Qdrant stores only training set; Python post-filter removes `release_year >= target`

---

## Common Workflows

**Add a new feature to fusion metadata:**
1. Edit `KEEP_COLS` in `data/fussion/fusion_data_exploration.ipynb` Section 8
2. Regenerate CSVs by rerunning the generation cell
3. Rerun `python -m src.fussion_branch.run_rag` to rebuild Qdrant with new payload fields

**Rebuild text embeddings:**
```bash
python -m src.text_branch.run_text_embedding_pipeline
```
Overwrites `artifacts/text_embeddings_{train,val,test}.parquet`.

**Reset Qdrant collection:**
`qdrant_db/` is rebuilt automatically each time `run_rag.py` runs. Just delete the folder or rerun the script.

---

## Important Notes

- Do NOT run training commands (image branch, fusion MLP) without confirming GPU is available (`torch.cuda.is_available()`)
- Do NOT add post-release features (`averageScore`, `favourites`, `trending`) to fusion_meta — they cause data leakage
- `rag_title_romaji` in rag_features is for interpretability only — do NOT feed it into the MLP
- `rag_studios` in rag_features is a JSON string of studio names; multi-hot encoding happens at model training time, not at RAG query time
- Image embeddings (`image_embeddings.parquet`) are not yet generated — fusion model currently runs without the image branch
