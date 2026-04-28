"""
Merge supplemented descriptions into fusion_meta_clean CSVs by id.

Reads supplemented_descriptions.csv (from run_supplement_descriptions.py),
joins to original fusion_meta_clean_{split}.csv by id to fill null descriptions,
and saves the result to data/fussion/supplemented/ (original files untouched).

Input:
    data/fussion/supplemented_descriptions.csv
    data/fussion/fusion_meta_clean_{split}.csv

Output:
    data/fussion/supplemented/fusion_meta_clean_{split}.csv

Usage:
    python -m src.fussion_branch.run_supplement_merge
    python -m src.fussion_branch.run_supplement_merge --source-dir data/fussion
"""
import argparse
from pathlib import Path

import pandas as pd

SUPP_PATH = Path("data/fussion/supplemented_descriptions.csv")
OUT_DIR   = Path("data/fussion/supplemented")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="data/fussion",
        help="Directory containing the original fusion_meta_clean_{split}.csv files",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SUPP_PATH.exists():
        raise FileNotFoundError(
            f"{SUPP_PATH} not found. Run run_supplement_descriptions.py first."
        )

    supp = pd.read_csv(SUPP_PATH)
    supp = (
        supp[supp["description_source"] == "jikan_mal"][["id", "description"]]
        .rename(columns={"description": "description_supp"})
    )
    print(f"Supplemented descriptions available: {len(supp)}")

    for split in ["train", "val", "test", "holdout_unknown"]:
        src_path = source_dir / f"fusion_meta_clean_{split}.csv"
        if not src_path.exists():
            print(f"  [{split}] not found — skipping")
            continue

        meta = pd.read_csv(src_path)
        null_before = meta["description"].isna().sum()

        meta = meta.merge(supp, on="id", how="left")
        meta["description"] = meta["description"].fillna(meta["description_supp"])
        meta = meta.drop(columns=["description_supp"])

        null_after = meta["description"].isna().sum()
        out_path = OUT_DIR / f"fusion_meta_clean_{split}.csv"
        meta.to_csv(out_path, index=False)

        print(f"  [{split}] {len(meta)} rows  "
              f"null description: {null_before} → {null_after}  "
              f"(filled {null_before - null_after})  → {out_path}")


if __name__ == "__main__":
    main()
