"""
RAG Pipeline for Fusion Branch.

Pre-requisite (for hybrid search):
  python -m src.fussion_branch.run_text_embedding
  → generates src/fussion_branch/RAG/text_embeddings_{split}.parquet

If text embeddings are absent, falls back to sparse-only mode automatically.

Usage:
  conda activate animeprediction
  python -m src.fussion_branch.run_rag
"""
from src.fussion_branch.RAG.rag_builder import build_collection
from src.fussion_branch.RAG.rag_query import query_all_splits


def main():
    print("=== Step 1: Build Qdrant collection from training set ===")
    build_collection()

    print()
    print("=== Step 2: RAG query for train / val / test ===")
    query_all_splits()

    print()
    print("Done. Outputs:")
    print("  src/fussion_branch/RAG/sparse_encoder.json")
    print("  src/fussion_branch/RAG/rag_features_{train,val,test}.parquet")


if __name__ == "__main__":
    main()
