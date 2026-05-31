"""
Convert local-ready external exams into src_2-compatible model input CSVs.

The current AnimeDataset expects integer ids and files named:
  src_2/data/dataset/fusion_meta_clean_{split}_v2.csv

External MAL rows use stable surrogate ids so they cannot collide with AniList
ids. A sidecar map keeps the original MAL ids, external labels, and local image
paths for evaluation and traceability.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DATASET_DIR = ROOT / "src_2" / "data" / "dataset"
OUT_TRANSFORMED_DIR = ROOT / "data" / "external_transformed"
DEFAULT_INPUTS = {
    "mal2025_popularity_local_ready": OUT_TRANSFORMED_DIR
    / "mal2025_image_mal_only_popularity_exam_local_ready.csv",
    "mal2025_dual_local_ready": OUT_TRANSFORMED_DIR
    / "mal2025_image_mal_only_dual_target_exam_local_ready.csv",
}
SURROGATE_ID_BASE = 900_000_000

MODEL_COLUMNS = [
    "id",
    "title_romaji",
    "title_english",
    "title_native",
    "description",
    "voice_actor_names",
    "format",
    "episodes",
    "duration",
    "meanScore",
    "popularity",
    "source",
    "countryOfOrigin",
    "isAdult",
    "startDate_month",
    "startDate_day",
    "genres",
    "studios",
    "is_sequel",
    "has_sequel",
    "prequel_count",
    "prequel_popularity_mean",
    "prequel_meanScore_mean",
    "release_year",
    "release_quarter",
]

SIDECAR_COLUMNS = [
    "id",
    "external_exam_id",
    "mal_id",
    "resolved_anilist_id",
    "title_romaji",
    "external_popularity_members",
    "external_popularity_rank",
    "external_score_0_10",
    "external_score_0_100",
    "external_cover_image_url",
    "local_cover_image_path",
    "local_cover_image_exists",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare src_2 external model input CSVs.")
    parser.add_argument(
        "--exam",
        action="append",
        default=None,
        help=(
            "External split spec in the form split_name=path/to/local_ready.csv. "
            "Can be passed multiple times. Defaults to MAL 2025 popularity and dual local-ready exams."
        ),
    )
    parser.add_argument("--id-base", type=int, default=SURROGATE_ID_BASE)
    parser.add_argument("--output-dataset-dir", default=str(OUT_DATASET_DIR))
    parser.add_argument(
        "--summary-json",
        default=str(OUT_TRANSFORMED_DIR / "mal2025_external_model_input_summary.json"),
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


def _load_specs(specs: list[str] | None) -> dict[str, Path]:
    if not specs:
        return DEFAULT_INPUTS.copy()
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --exam spec, expected split_name=csv_path: {spec}")
        split_name, path_text = spec.split("=", 1)
        split_name = split_name.strip()
        if not split_name:
            raise ValueError(f"Empty split name in --exam spec: {spec}")
        parsed[split_name] = _resolve(path_text.strip())
    return parsed


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _prepare_one(split_name: str, exam_csv: Path, output_dataset_dir: Path, id_base: int) -> dict:
    df = pd.read_csv(exam_csv)
    required = {
        "mal_id",
        "external_exam_id",
        "title_romaji",
        "description",
        "format",
        "release_year",
        "release_quarter",
        "genres",
        "studios",
        "source",
        "local_cover_image_path",
        "local_cover_image_exists",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{exam_csv} missing required columns: {sorted(missing)}")

    ids = id_base + pd.to_numeric(df["mal_id"], errors="raise").astype(int)
    if ids.duplicated().any():
        dupes = ids[ids.duplicated()].head().tolist()
        raise ValueError(f"Surrogate ids are not unique for {split_name}: {dupes}")

    model = pd.DataFrame(index=df.index)
    model["id"] = ids
    model["title_romaji"] = df["title_romaji"]
    model["title_english"] = df.get("title_english", df["title_romaji"]).fillna(df["title_romaji"])
    model["title_native"] = df.get("title_native", model["title_english"])
    model["description"] = df["description"].fillna("")
    model["voice_actor_names"] = df.get("voice_actor_names", "").fillna("")
    model["format"] = df["format"]
    model["episodes"] = pd.to_numeric(df.get("episodes", df.get("episodes_numeric")), errors="coerce").fillna(0)
    model["duration"] = pd.to_numeric(df.get("duration", df.get("duration_minutes")), errors="coerce").fillna(0)
    model["meanScore"] = pd.to_numeric(df.get("external_score_0_100"), errors="coerce").fillna(0)
    model["popularity"] = pd.to_numeric(df["external_popularity_members"], errors="coerce").fillna(0)
    model["source"] = df["source"].fillna("UNKNOWN_SOURCE")
    model["countryOfOrigin"] = df.get("countryOfOrigin", "JP").fillna("JP")
    model["isAdult"] = _bool_series(df.get("isAdult", pd.Series(False, index=df.index)))
    model["startDate_month"] = pd.to_numeric(df.get("startDate_month"), errors="coerce").fillna(1).astype(int)
    model["startDate_day"] = pd.to_numeric(df.get("startDate_day"), errors="coerce").fillna(1).astype(int)
    model["genres"] = df["genres"].fillna("[]")
    model["studios"] = df["studios"].fillna("[]")
    model["is_sequel"] = _bool_series(df.get("is_sequel", pd.Series(False, index=df.index)))
    model["has_sequel"] = _bool_series(df.get("has_sequel", pd.Series(False, index=df.index)))
    model["prequel_count"] = pd.to_numeric(df.get("prequel_count"), errors="coerce").fillna(0)
    model["prequel_popularity_mean"] = pd.to_numeric(
        df.get("prequel_popularity_mean"), errors="coerce"
    ).fillna(0)
    model["prequel_meanScore_mean"] = pd.to_numeric(
        df.get("prequel_meanScore_mean"), errors="coerce"
    ).fillna(0)
    model["release_year"] = pd.to_numeric(df["release_year"], errors="raise").astype(int)
    model["release_quarter"] = pd.to_numeric(df["release_quarter"], errors="raise").astype(int)
    model = model[MODEL_COLUMNS]

    sidecar = pd.DataFrame(index=df.index)
    for col in SIDECAR_COLUMNS:
        if col == "id":
            sidecar[col] = ids
        elif col in df.columns:
            sidecar[col] = df[col]
        else:
            sidecar[col] = pd.NA

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    OUT_TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = output_dataset_dir / f"fusion_meta_clean_{split_name}_v2.csv"
    sidecar_path = OUT_TRANSFORMED_DIR / f"{split_name}_id_map.csv"
    model.to_csv(model_path, index=False)
    sidecar.to_csv(sidecar_path, index=False)

    return {
        "split_name": split_name,
        "exam_csv": _display(exam_csv),
        "model_input_csv": _display(model_path),
        "id_map_csv": _display(sidecar_path),
        "rows": int(len(model)),
        "surrogate_id_base": id_base,
        "min_id": int(model["id"].min()) if len(model) else None,
        "max_id": int(model["id"].max()) if len(model) else None,
    }


def main() -> None:
    args = _parse_args()
    output_dataset_dir = _resolve(args.output_dataset_dir)
    specs = _load_specs(args.exam)
    outputs = [
        _prepare_one(split_name, _resolve(path), output_dataset_dir, args.id_base)
        for split_name, path in specs.items()
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
        "next_steps": [
            "Generate text embeddings for each split into src_2/embedding/text/text_embeddings_<split>.parquet.",
            "Generate image embeddings for each split into src_2/embedding/image/image_embeddings_<split>.parquet.",
            "Generate RAG returns for each split into src_2/RAG/return/rag_features_<split>.parquet.",
            "Run an external inference helper that preserves id_map labels for MAL metrics.",
        ],
    }
    summary_path = _resolve(args.summary_json)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
