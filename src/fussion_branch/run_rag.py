from src.fussion_branch.rag_builder import build_collection
from src.fussion_branch.rag_query import query_all_splits


def main():
    print("=== Step 1: Build Qdrant collection from training set ===")
    build_collection()

    print()
    print("=== Step 2: RAG query for train / val / test ===")
    query_all_splits()

    print()
    print("Done. Outputs:")
    print("  artifacts/sparse_encoder.json")
    print("  artifacts/rag_features_{train,val,test}.parquet")


if __name__ == "__main__":
    main()
