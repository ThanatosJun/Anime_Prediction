from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.image_branch.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize downloaded image coverage by split and image column.")
    parser.add_argument(
        "--config",
        default="src/image_branch/configs/image_process_config.yaml",
        help="Image process config YAML.",
    )
    parser.add_argument(
        "--output",
        default=".exp/image_fetch/fetch_coverage_summary.csv",
        help="CSV path for the coverage summary.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rows = summarize_coverage(config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n[done] coverage summary -> {out_path}")


def summarize_coverage(config: dict) -> list[dict]:
    data_cfg = config["data"]
    image_dir = Path(data_cfg["image_dir"])
    log_path = Path(data_cfg["log_path"])
    image_cols = list(data_cfg["image_columns"])
    split_csv = data_cfg.get("split_csv", {})
    log_counts = _read_log_counts(log_path)

    rows: list[dict] = []
    for split, csv_path in split_csv.items():
        df = pd.read_csv(csv_path, usecols=["id", *image_cols])
        for col in image_cols:
            url_available = df[col].fillna("").astype(str).str.startswith("http")
            expected = int(url_available.sum())
            downloaded = 0
            for anime_id in df.loc[url_available, "id"].astype(int):
                if (image_dir / f"{anime_id}_{col}.jpg").exists():
                    downloaded += 1
            rows.append(
                {
                    "split": split,
                    "image_col": col,
                    "rows": int(len(df)),
                    "url_available": expected,
                    "downloaded_files": int(downloaded),
                    "file_coverage": round(downloaded / expected, 6) if expected else 0.0,
                    "logged_success": log_counts[(col, "success")],
                    "logged_error": log_counts[(col, "error")],
                    "logged_skip": log_counts[(col, "skip")],
                }
            )
    return rows


def _read_log_counts(log_path: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    if not log_path.exists():
        return counts
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            _, col, _, status = row[:4]
            counts[(col, status)] += 1
    return counts


if __name__ == "__main__":
    main()
