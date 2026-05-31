"""
Filter image-ready external exams to rows with local cover images present.

This script is intentionally separate from prepare_external_evaluation_assets.py:
the base exams are deterministic CSV transforms, while local readiness depends on
whether image downloads have completed on the current machine.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "external_transformed"
DEFAULT_EXAMS = [
    OUT_DIR / "mal2025_image_mal_only_popularity_exam.csv",
    OUT_DIR / "mal2025_image_mal_only_dual_target_exam.csv",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local-ready external image exams.")
    parser.add_argument(
        "--exam-csv",
        action="append",
        default=None,
        help="Exam CSV to filter. Can be passed multiple times. Defaults to MAL 2025 popularity and dual exams.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(OUT_DIR / "mal2025_image_local_ready_summary.json"),
        help="Summary JSON output path.",
    )
    return parser.parse_args()


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _prepare_one(exam_csv: Path) -> dict:
    df = pd.read_csv(exam_csv)
    required = {"external_cover_image_path", "external_exam_id", "mal_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{exam_csv} missing required columns: {sorted(missing)}")

    image_paths = df["external_cover_image_path"].map(_resolve)
    exists = image_paths.map(lambda path: path.exists() and path.is_file() and path.stat().st_size > 0)
    out = df.copy()
    out["local_cover_image_exists"] = exists
    out["local_cover_image_path"] = image_paths.map(lambda path: path.as_posix())

    stem = exam_csv.stem
    local_ready_path = exam_csv.with_name(stem + "_local_ready.csv")
    missing_path = exam_csv.with_name(stem + "_missing_local_images.csv")

    local_ready = out[out["local_cover_image_exists"]].copy()
    missing_images = out[~out["local_cover_image_exists"]].copy()
    local_ready.to_csv(local_ready_path, index=False)
    missing_images.to_csv(missing_path, index=False)

    return {
        "exam_csv": _display(exam_csv),
        "rows": int(len(out)),
        "local_ready_rows": int(len(local_ready)),
        "missing_local_image_rows": int(len(missing_images)),
        "local_ready_csv": _display(local_ready_path),
        "missing_local_images_csv": _display(missing_path),
    }


def main() -> None:
    args = _parse_args()
    exam_paths = [_resolve(path) for path in (args.exam_csv or DEFAULT_EXAMS)]
    summaries = [_prepare_one(path) for path in exam_paths]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [_display(path) for path in exam_paths],
        "outputs": summaries,
    }
    summary_path = _resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
