"""
Prepare external evaluation assets without mutating the official processed files.

Outputs:
- data/external_transformed/aodb_id_crosswalk.csv
- data/external_transformed/aodb_holdout_unknown_recovered_rows.csv
- data/external_transformed/anilist_anime_multimodal_input_v1_aodb_recovered_future.csv
- data/external_transformed/mal_july2025_external_eval_contract.csv
- data/external_transformed/external_evaluation_assets_summary.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "external_transformed"

RAW_PICKLE = ROOT / "data" / "raw" / "anilist_anime_data_complete.pkl"
RAW_CSV = ROOT / "data" / "raw" / "anilist_anime_data_complete.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "anilist_anime_data_processed_v1.csv"
MULTIMODAL_CSV = ROOT / "data" / "processed" / "anilist_anime_multimodal_input_v1.csv"
HOLDOUT_CSV = ROOT / "data" / "processed" / "anilist_anime_multimodal_input_holdout_unknown.csv"

AODB_CSV = ROOT / "outtestdataset" / "Anime Offline Database" / "anime_database.csv"
MAL_JULY_ANIME_CSV = (
    ROOT
    / "outtestdataset"
    / "MyAnimeList Anime & Manga Dataset (July 2025)"
    / "anime_entries.csv"
)
MAL_2025_IMAGE_ANIME_CSV = ROOT / "outtestdataset" / "MyAnimeList 2025" / "mal_anime.csv"

SEASON_TO_QUARTER = {"WINTER": 1, "SPRING": 2, "SUMMER": 3, "FALL": 4}
SEASON_WORD_TO_PROJECT = {
    "winter": "WINTER",
    "spring": "SPRING",
    "summer": "SUMMER",
    "fall": "FALL",
}
MONTH_TO_QUARTER = {
    "jan": 1,
    "feb": 1,
    "mar": 1,
    "apr": 2,
    "may": 2,
    "jun": 2,
    "jul": 3,
    "aug": 3,
    "sep": 3,
    "oct": 4,
    "nov": 4,
    "dec": 4,
}
BUCKET_BINS = [-0.000001, 0.25, 0.50, 0.75, 1.0]
BUCKET_LABELS = ["cold_0_25", "warm_25_50", "hot_50_75", "top_75_100"]


def _load_raw() -> pd.DataFrame:
    if RAW_PICKLE.exists():
        return pd.read_pickle(RAW_PICKLE)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError("Missing raw AniList dataset.")


def _extract_id(series: pd.Series, pattern: str) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.extract(pattern, expand=False),
        errors="coerce",
    ).astype("Int64")


def _clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("#", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False),
        errors="coerce",
    )


def _derive_year_from_text(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.extract(r"(19\d{2}|20\d{2}|21\d{2})", expand=False),
        errors="coerce",
    ).astype("Int64")


def _derive_quarter_from_month_text(series: pd.Series) -> pd.Series:
    month_key = series.astype("string").str.extract(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
        expand=False,
        flags=re.IGNORECASE,
    )
    return month_key.str.lower().str[:3].map(MONTH_TO_QUARTER).astype("Int64")


def _derive_season_from_premier(series: pd.Series) -> pd.Series:
    season = series.astype("string").str.extract(
        r"\b(Winter|Spring|Summer|Fall)\b",
        expand=False,
        flags=re.IGNORECASE,
    )
    return season.str.lower().map(SEASON_WORD_TO_PROJECT).astype("string")


def _parse_duration_minutes(series: pd.Series) -> pd.Series:
    def parse_one(value: object) -> float | None:
        if pd.isna(value):
            return None
        text = str(value).lower()
        hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:hr|hour)", text)
        minutes = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minute)", text)
        seconds = re.search(r"(\d+(?:\.\d+)?)\s*(?:sec|second)", text)
        total = 0.0
        matched = False
        if hours:
            total += float(hours.group(1)) * 60.0
            matched = True
        if minutes:
            total += float(minutes.group(1))
            matched = True
        if seconds:
            total += float(seconds.group(1)) / 60.0
            matched = True
        if matched:
            return total
        number = re.search(r"([-+]?\d*\.?\d+)", text)
        return float(number.group(1)) if number else None

    return series.map(parse_one)


def _split_comma_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip() and item.strip().lower() not in {"nan", "none", "unknown"}
    ]


def _json_list_from_comma_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: json.dumps(_split_comma_list(value), ensure_ascii=False))


def _json_studios_from_comma_series(series: pd.Series) -> pd.Series:
    def encode(value: object) -> str:
        studios = []
        for idx, name in enumerate(_split_comma_list(value), start=1):
            studios.append(
                {
                    "id": None,
                    "isMain": idx == 1,
                    "node": {
                        "id": None,
                        "name": name,
                        "isAnimationStudio": True,
                    },
                }
            )
        return json.dumps(studios, ensure_ascii=False)

    return series.map(encode)


def _normalize_enum_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_")
        .replace("", pd.NA)
    )


def _format_to_project(series: pd.Series) -> pd.Series:
    normalized = _normalize_enum_series(series)
    mapping = {
        "MOVIE": "MOVIE",
        "TV": "TV",
        "TV_SPECIAL": "SPECIAL",
        "SPECIAL": "SPECIAL",
        "OVA": "OVA",
        "ONA": "ONA",
        "MUSIC": "MUSIC",
    }
    return normalized.map(lambda value: mapping.get(value, value if pd.notna(value) else pd.NA))


def _source_to_project(series: pd.Series) -> pd.Series:
    normalized = _normalize_enum_series(series)
    mapping = {
        "LIGHT_NOVEL": "LIGHT_NOVEL",
        "VISUAL_NOVEL": "VISUAL_NOVEL",
        "VIDEO_GAME": "VIDEO_GAME",
        "WEB_MANGA": "MANGA",
        "WEB_NOVEL": "OTHER",
        "4_KOMA_MANGA": "MANGA",
        "MANGA": "MANGA",
        "ORIGINAL": "ORIGINAL",
        "NOVEL": "LIGHT_NOVEL",
        "GAME": "VIDEO_GAME",
    }
    return normalized.map(lambda value: mapping.get(value, value if pd.notna(value) else "UNKNOWN_SOURCE"))


def _quarter_to_start_month(quarter: pd.Series) -> pd.Series:
    return quarter.map({1: 1, 2: 4, 3: 7, 4: 10}).astype("Int64")


def _valid_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return series.notna() & text.notna() & ~text.str.lower().isin(
        ["", "nan", "none", "null", "unknown", "undefined"]
    )


def _read_aodb_crosswalk() -> pd.DataFrame:
    aodb = pd.read_csv(AODB_CSV, low_memory=False)
    crosswalk = pd.DataFrame(
        {
            "aodb_title": aodb["title"],
            "aodb_sources": aodb["sources"],
            "anilist_id": _extract_id(aodb["sources"], r"anilist\.co/anime/(\d+)"),
            "mal_id": _extract_id(aodb["sources"], r"myanimelist\.net/anime/(\d+)"),
            "aodb_type": aodb.get("type"),
            "aodb_status": aodb.get("status"),
            "aodb_season": aodb["animeSeason.season"].astype("string").str.upper().str.strip(),
            "aodb_season_year": pd.to_numeric(aodb["animeSeason.year"], errors="coerce").astype("Int64"),
            "aodb_episodes": pd.to_numeric(aodb.get("episodes"), errors="coerce"),
            "aodb_duration_value": pd.to_numeric(aodb.get("duration.value"), errors="coerce"),
            "aodb_duration_unit": aodb.get("duration.unit"),
            "aodb_score_agm": pd.to_numeric(aodb.get("score.arithmeticGeometricMean"), errors="coerce"),
            "aodb_score_mean": pd.to_numeric(aodb.get("score.arithmeticMean"), errors="coerce"),
            "aodb_score_median": pd.to_numeric(aodb.get("score.median"), errors="coerce"),
        }
    )
    crosswalk["aodb_release_quarter"] = crosswalk["aodb_season"].map(SEASON_TO_QUARTER).astype("Int64")
    return crosswalk


def _write_crosswalk(crosswalk: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = crosswalk.copy()
    out.to_csv(OUT_DIR / "aodb_id_crosswalk.csv", index=False)
    return OUT_DIR / "aodb_id_crosswalk.csv"


def _official_quarter_split_map(processed: pd.DataFrame) -> tuple[dict[int, str], dict[str, dict[str, int]]]:
    known = processed[
        processed["release_year"].notna()
        & processed["release_quarter"].notna()
        & processed["split_pre_release_effective"].isin(["train", "val", "test"])
    ].copy()
    known["quarter_index"] = known["release_year"].astype(int) * 10 + known["release_quarter"].astype(int)

    quarter_split = (
        known[["quarter_index", "split_pre_release_effective"]]
        .drop_duplicates()
        .groupby("quarter_index")["split_pre_release_effective"]
        .first()
        .to_dict()
    )
    ranges = {}
    for split, group in known.groupby("split_pre_release_effective"):
        ranges[str(split)] = {
            "min": int(group["quarter_index"].min()),
            "max": int(group["quarter_index"].max()),
            "count": int(len(group)),
        }
    return {int(k): str(v) for k, v in quarter_split.items()}, ranges


def _assign_split(quarter_index: object, quarter_split: dict[int, str], ranges: dict[str, dict[str, int]]) -> str:
    if pd.isna(quarter_index):
        return "holdout_unknown"
    q = int(quarter_index)
    if q in quarter_split:
        return quarter_split[q]
    for split in ("train", "val", "test"):
        info = ranges.get(split)
        if info and info["min"] <= q <= info["max"]:
            return split
    return "holdout_unknown"


def _recompute_quarter_popularity_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid = out["release_year"].notna() & out["release_quarter"].notna()
    out["release_quarter_key"] = pd.NA
    out.loc[valid, "release_quarter_key"] = (
        out.loc[valid, "release_year"].astype(int).astype(str)
        + "Q"
        + out.loc[valid, "release_quarter"].astype(int).astype(str)
    )
    pct = pd.Series(pd.NA, index=out.index, dtype="Float64")
    pct.loc[valid] = (
        pd.to_numeric(out.loc[valid, "popularity"], errors="coerce")
        .groupby(out.loc[valid, "release_quarter_key"], dropna=False)
        .rank(pct=True, ascending=True)
        .astype("Float64")
    )
    out["popularity_quarter_pct"] = pct
    out["popularity_quarter_bucket"] = pd.cut(pct, bins=BUCKET_BINS, labels=BUCKET_LABELS)
    out.loc[pct.isna(), "popularity_quarter_bucket"] = pd.NA
    return out


def _prepare_holdout_recovery(raw: pd.DataFrame, crosswalk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    processed = pd.read_csv(PROCESSED_CSV)
    multimodal = pd.read_csv(MULTIMODAL_CSV)
    holdout = pd.read_csv(HOLDOUT_CSV)

    raw_keys = raw[["id", "idMal"]].copy()
    raw_keys["id"] = pd.to_numeric(raw_keys["id"], errors="coerce").astype("Int64")
    raw_keys["idMal"] = pd.to_numeric(raw_keys["idMal"], errors="coerce").astype("Int64")

    ani = (
        crosswalk[crosswalk["anilist_id"].notna()]
        .drop_duplicates("anilist_id", keep="first")
        [["anilist_id", "aodb_season", "aodb_season_year", "aodb_release_quarter"]]
    )
    mal = (
        crosswalk[crosswalk["mal_id"].notna()]
        .drop_duplicates("mal_id", keep="first")
        [["mal_id", "aodb_season", "aodb_season_year", "aodb_release_quarter"]]
    )

    recovered = holdout.merge(raw_keys, on="id", how="left")
    recovered = recovered.merge(ani, left_on="id", right_on="anilist_id", how="left").rename(
        columns={
            "aodb_season": "aodb_anilist_season",
            "aodb_season_year": "aodb_anilist_year",
            "aodb_release_quarter": "aodb_anilist_quarter",
        }
    )
    recovered = recovered.merge(mal, left_on="idMal", right_on="mal_id", how="left").rename(
        columns={
            "aodb_season": "aodb_mal_season",
            "aodb_season_year": "aodb_mal_year",
            "aodb_release_quarter": "aodb_mal_quarter",
        }
    )
    recovered["aodb_fill_year"] = recovered["aodb_anilist_year"].fillna(recovered["aodb_mal_year"])
    recovered["aodb_fill_quarter"] = recovered["aodb_anilist_quarter"].fillna(recovered["aodb_mal_quarter"])
    recovered["aodb_fill_season"] = recovered["aodb_anilist_season"].where(
        recovered["aodb_anilist_quarter"].notna(),
        recovered["aodb_mal_season"],
    )
    recoverable = recovered["aodb_fill_year"].notna() & recovered["aodb_fill_quarter"].notna()
    recovered_rows = recovered.loc[recoverable].copy()

    quarter_split, ranges = _official_quarter_split_map(processed)
    recovered_rows["release_year_original"] = recovered_rows["release_year"]
    recovered_rows["release_quarter_original"] = recovered_rows["release_quarter"]
    recovered_rows["release_year"] = recovered_rows["release_year"].fillna(recovered_rows["aodb_fill_year"]).astype("Int64")
    recovered_rows["release_quarter"] = recovered_rows["aodb_fill_quarter"].astype("Int64")
    recovered_rows["release_quarter_key"] = (
        recovered_rows["release_year"].astype(int).astype(str)
        + "Q"
        + recovered_rows["release_quarter"].astype(int).astype(str)
    )
    recovered_rows["quarter_index"] = (
        recovered_rows["release_year"].astype(int) * 10 + recovered_rows["release_quarter"].astype(int)
    )
    recovered_rows["split_pre_release_effective"] = recovered_rows["quarter_index"].map(
        lambda q: _assign_split(q, quarter_split, ranges)
    )
    recovered_rows["is_model_split"] = recovered_rows["split_pre_release_effective"].isin(["train", "val", "test"])
    recovered_rows["recovery_source"] = "anime_offline_database"
    recovered_rows["recovery_note"] = "AniList ID first; MAL ID fallback; official split quarter map preserved."

    future = multimodal.copy()
    future["recovery_source"] = pd.NA
    future["recovery_note"] = pd.NA
    future_idx = future.set_index("id", drop=False)
    recovered_idx = recovered_rows.set_index("id", drop=False)
    update_cols = [
        "release_year",
        "release_quarter",
        "split_pre_release_effective",
        "is_model_split",
        "recovery_source",
        "recovery_note",
    ]
    for col in update_cols:
        future_idx.loc[recovered_idx.index, col] = recovered_idx[col]
    future = future_idx.reset_index(drop=True)
    future = _recompute_quarter_popularity_targets(future)

    summary = {
        "holdout_rows": int(len(holdout)),
        "recovered_rows": int(len(recovered_rows)),
        "remaining_holdout_unknown_rows": int(len(holdout) - len(recovered_rows)),
        "recovered_split_counts": recovered_rows["split_pre_release_effective"].value_counts(dropna=False).to_dict(),
        "official_quarter_ranges_preserved": ranges,
    }
    return recovered_rows, future, summary


def _prepare_mal_only_exams(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mal_only = joined[joined["anilist_id"].isna()].copy()

    premier_year = _derive_year_from_text(mal_only["premier_date"])
    airing_year = _derive_year_from_text(mal_only["airing_date"])
    premier_season = _derive_season_from_premier(mal_only["premier_date"])
    premier_quarter = premier_season.map(SEASON_TO_QUARTER).astype("Int64")
    airing_quarter = _derive_quarter_from_month_text(mal_only["airing_date"])

    mal_only["external_exam_id"] = "mal_" + mal_only["mal_id"].astype("Int64").astype(str)
    mal_only["aodb_anilist_id"] = mal_only["aodb_anilist_id"].astype("Int64")
    mal_only["title_romaji"] = mal_only["title_name"]
    mal_only["title_english"] = mal_only["english_name"].fillna(mal_only["title_name"])
    mal_only["description"] = mal_only["description_external"]
    mal_only["format"] = mal_only["item_type"]
    mal_only["season"] = mal_only["aodb_season"].where(
        _valid_text(mal_only["aodb_season"]),
        premier_season,
    )
    mal_only["release_year"] = (
        mal_only["aodb_season_year"]
        .fillna(premier_year)
        .fillna(airing_year)
        .astype("Int64")
    )
    mal_only["release_quarter"] = (
        mal_only["aodb_release_quarter"]
        .fillna(premier_quarter)
        .fillna(airing_quarter)
        .astype("Int64")
    )
    valid_quarter = mal_only["release_year"].notna() & mal_only["release_quarter"].notna()
    mal_only["release_quarter_key"] = pd.NA
    mal_only.loc[valid_quarter, "release_quarter_key"] = (
        mal_only.loc[valid_quarter, "release_year"].astype(int).astype(str)
        + "Q"
        + mal_only.loc[valid_quarter, "release_quarter"].astype(int).astype(str)
    )
    mal_only["episodes_numeric"] = _clean_number(mal_only["episodes"])
    mal_only["duration_minutes"] = _parse_duration_minutes(mal_only["duration"])
    mal_only["has_text_description"] = _valid_text(mal_only["description"])
    mal_only["has_cover_image"] = False
    mal_only["has_banner_image"] = False
    mal_only["has_trailer"] = False
    mal_only["has_external_popularity_target"] = mal_only["external_popularity_members"].notna()
    mal_only["has_external_score_target"] = mal_only["external_score_0_100"].notna()
    mal_only["has_dual_targets"] = (
        mal_only["has_external_popularity_target"] & mal_only["has_external_score_target"]
    )
    mal_only["has_release_year_quarter"] = valid_quarter
    mal_only["has_aodb_anilist_id"] = mal_only["aodb_anilist_id"].notna()
    mal_only["can_prepare_text_metadata_inference"] = (
        mal_only["has_release_year_quarter"] & mal_only["has_text_description"]
    )
    mal_only["can_run_current_full_multimodal_without_new_assets"] = False
    mal_only["readiness_note"] = (
        "MAL-only exam row. Needs fresh text/image feature generation before current full model inference; "
        "no title matching was used."
    )

    exam_cols = [
        "external_exam_id",
        "mal_id",
        "aodb_anilist_id",
        "title_romaji",
        "title_english",
        "japanese_name",
        "format",
        "status",
        "season",
        "release_year",
        "release_quarter",
        "release_quarter_key",
        "episodes_numeric",
        "duration_minutes",
        "source",
        "genres",
        "themes",
        "demographic",
        "studios",
        "description",
        "external_popularity_members",
        "external_popularity_rank",
        "external_score_rank",
        "external_score_0_10",
        "external_score_0_100",
        "external_scored_by",
        "has_external_popularity_target",
        "has_external_score_target",
        "has_dual_targets",
        "has_release_year_quarter",
        "has_text_description",
        "has_cover_image",
        "has_banner_image",
        "has_trailer",
        "has_aodb_anilist_id",
        "can_prepare_text_metadata_inference",
        "can_run_current_full_multimodal_without_new_assets",
        "readiness_note",
    ]
    existing_cols = [col for col in exam_cols if col in mal_only.columns]
    popularity_exam = mal_only[existing_cols].copy()
    dual_exam = popularity_exam[popularity_exam["has_dual_targets"]].copy()

    summary = {
        "mal_only_rows": int(len(popularity_exam)),
        "dual_target_rows": int(len(dual_exam)),
        "popularity_only_rows": int(popularity_exam["has_external_popularity_target"].sum()),
        "mal_only_with_release_year_quarter": int(popularity_exam["has_release_year_quarter"].sum()),
        "dual_with_release_year_quarter": int(dual_exam["has_release_year_quarter"].sum()),
        "mal_only_with_aodb_anilist_id": int(popularity_exam["has_aodb_anilist_id"].sum()),
        "dual_with_aodb_anilist_id": int(dual_exam["has_aodb_anilist_id"].sum()),
        "mal_only_text_metadata_inference_candidates": int(
            popularity_exam["can_prepare_text_metadata_inference"].sum()
        ),
        "dual_text_metadata_inference_candidates": int(
            dual_exam["can_prepare_text_metadata_inference"].sum()
        ),
        "current_full_multimodal_ready_rows": int(
            popularity_exam["can_run_current_full_multimodal_without_new_assets"].sum()
        ),
    }
    return dual_exam, popularity_exam, summary


def _prepare_mal_july_external_eval(
    raw: pd.DataFrame, crosswalk: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    mal = pd.read_csv(MAL_JULY_ANIME_CSV, low_memory=False)
    multimodal = pd.read_csv(MULTIMODAL_CSV)

    raw_keys = raw[["id", "idMal"]].copy()
    raw_keys["anilist_id"] = pd.to_numeric(raw_keys["id"], errors="coerce").astype("Int64")
    raw_keys["mal_id"] = pd.to_numeric(raw_keys["idMal"], errors="coerce").astype("Int64")
    raw_keys = raw_keys.drop(columns=["id", "idMal"]).dropna(subset=["mal_id"]).drop_duplicates("mal_id")

    mal_contract = mal.copy()
    mal_contract["mal_id"] = _clean_number(mal_contract["id"]).astype("Int64")
    mal_contract["external_popularity_members"] = _clean_number(mal_contract["members"])
    mal_contract["external_popularity_rank"] = _clean_number(mal_contract["popularity"])
    mal_contract["external_score_0_10"] = pd.to_numeric(mal_contract["score"], errors="coerce")
    mal_contract["external_score_0_100"] = mal_contract["external_score_0_10"] * 10
    mal_contract["external_scored_by"] = _clean_number(mal_contract["scored_by"])
    mal_contract = mal_contract.rename(columns={"description": "description_external"})

    joined = mal_contract.merge(raw_keys, on="mal_id", how="left")
    aodb_by_mal = (
        crosswalk[crosswalk["mal_id"].notna()]
        .drop_duplicates("mal_id", keep="first")
        [[
            "mal_id",
            "anilist_id",
            "aodb_season",
            "aodb_season_year",
            "aodb_release_quarter",
        ]]
        .rename(columns={"anilist_id": "aodb_anilist_id"})
    )
    joined = joined.merge(aodb_by_mal, on="mal_id", how="left")
    dual_exam, popularity_exam, mal_only_summary = _prepare_mal_only_exams(joined)

    model_inputs = multimodal.rename(columns={"id": "anilist_id"})
    model_inputs["anilist_id"] = pd.to_numeric(model_inputs["anilist_id"], errors="coerce").astype("Int64")
    joined = joined.merge(
        model_inputs,
        on="anilist_id",
        how="left",
        suffixes=("_external", "_internal"),
    )
    joined = joined.rename(
        columns={
            "popularity_internal": "anilist_popularity",
            "meanScore": "anilist_meanScore",
        }
    )

    out_cols = [
        "mal_id",
        "anilist_id",
        "title_name",
        "english_name",
        "japanese_name",
        "item_type",
        "external_popularity_members",
        "external_popularity_rank",
        "external_score_rank",
        "external_score_0_10",
        "external_score_0_100",
        "external_scored_by",
        "ranked",
        "members",
        "score",
        "description_external",
        "release_year",
        "release_quarter",
        "split_pre_release_effective",
        "is_model_split",
        "anilist_popularity",
        "anilist_meanScore",
        "title_romaji",
        "title_english",
        "description_internal",
        "coverImage_extraLarge",
        "coverImage_large",
        "coverImage_medium",
        "bannerImage",
        "trailer_id",
        "trailer_site",
        "trailer_thumbnail",
        "has_text_description",
        "has_cover_image",
        "has_banner_image",
        "has_trailer",
    ]
    existing_cols = [col for col in out_cols if col in joined.columns]
    joined = joined[existing_cols].copy()
    joined["external_eval_ready"] = (
        joined["anilist_id"].notna()
        & joined["external_popularity_members"].notna()
        & joined["external_score_0_100"].notna()
        & joined["release_year"].notna()
        & joined["release_quarter"].notna()
    )
    joined["external_label_note"] = (
        "MAL members is the external popularity count proxy; MAL score is multiplied by 10."
    )

    summary = {
        "source_rows": int(len(mal)),
        "rows_with_mal_id": int(mal_contract["mal_id"].notna().sum()),
        "rows_mapped_to_internal_anilist_id": int(joined["anilist_id"].notna().sum()),
        "rows_with_external_members": int(joined["external_popularity_members"].notna().sum()),
        "rows_with_external_score_0_100": int(joined["external_score_0_100"].notna().sum()),
        "external_eval_ready_rows": int(joined["external_eval_ready"].sum()),
        "external_eval_ready_by_internal_split": (
            joined.loc[joined["external_eval_ready"], "split_pre_release_effective"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "label_mapping": {
            "external_popularity_members": "MAL members; count-like, higher means more popular.",
            "external_score_0_100": "MAL score * 10; score-like, roughly aligned to AniList 0-100 scale.",
            "external_popularity_rank": "MAL popularity rank; lower means more popular; keep for rank metrics only.",
        },
        "mal_only_exams": mal_only_summary,
    }
    return joined, dual_exam, popularity_exam, summary


def _prepare_mal2025_image_external_eval(
    raw: pd.DataFrame, crosswalk: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    mal = pd.read_csv(MAL_2025_IMAGE_ANIME_CSV, low_memory=False)
    multimodal = pd.read_csv(MULTIMODAL_CSV)

    internal_ids = set(pd.to_numeric(multimodal["id"], errors="coerce").dropna().astype(int))
    raw_keys = raw[["id", "idMal"]].copy()
    raw_keys["internal_anilist_id"] = pd.to_numeric(raw_keys["id"], errors="coerce").astype("Int64")
    raw_keys["mal_id"] = pd.to_numeric(raw_keys["idMal"], errors="coerce").astype("Int64")
    raw_keys = raw_keys.drop(columns=["id", "idMal"]).dropna(subset=["mal_id"]).drop_duplicates("mal_id")

    aodb_by_mal = (
        crosswalk[crosswalk["mal_id"].notna()]
        .drop_duplicates("mal_id", keep="first")
        [[
            "mal_id",
            "anilist_id",
            "aodb_season",
            "aodb_season_year",
            "aodb_release_quarter",
        ]]
        .rename(columns={"anilist_id": "aodb_anilist_id"})
    )

    contract = mal.copy()
    contract["mal_id"] = pd.to_numeric(contract["myanimelist_id"], errors="coerce").astype("Int64")
    contract["external_popularity_members"] = _clean_number(contract["Members"])
    contract["external_popularity_rank"] = _clean_number(contract["Popularity"])
    contract["external_score_0_10"] = pd.to_numeric(contract["Score"], errors="coerce")
    contract["external_score_0_100"] = contract["external_score_0_10"] * 10
    contract["external_score_rank"] = _clean_number(contract["Ranked"])
    contract["external_scored_by"] = pd.NA
    contract = contract.merge(raw_keys, on="mal_id", how="left")
    contract = contract.merge(aodb_by_mal, on="mal_id", how="left")

    released_year = pd.to_numeric(contract["Released_Year"], errors="coerce").astype("Int64")
    released_quarter = (
        contract["Released_Season"].astype("string").str.upper().str.strip().map(SEASON_TO_QUARTER).astype("Int64")
    )
    contract["release_year"] = released_year.fillna(contract["aodb_season_year"]).astype("Int64")
    contract["release_quarter"] = released_quarter.fillna(contract["aodb_release_quarter"]).astype("Int64")
    valid_quarter = contract["release_year"].notna() & contract["release_quarter"].notna()
    contract["release_quarter_key"] = pd.NA
    contract.loc[valid_quarter, "release_quarter_key"] = (
        contract.loc[valid_quarter, "release_year"].astype(int).astype(str)
        + "Q"
        + contract.loc[valid_quarter, "release_quarter"].astype(int).astype(str)
    )

    contract["resolved_anilist_id"] = contract["internal_anilist_id"].fillna(contract["aodb_anilist_id"]).astype("Int64")
    contract["is_internal_anilist_row"] = contract["resolved_anilist_id"].isin(internal_ids)
    contract["external_exam_id"] = "mal2025_" + contract["mal_id"].astype("Int64").astype(str)
    contract["title_romaji"] = contract["title"]
    contract["title_english"] = contract["title"]
    contract["description"] = contract["description"]
    contract["format"] = _format_to_project(contract["Type"])
    contract["status"] = contract["Status"]
    contract["season"] = contract["Released_Season"].astype("string").str.upper().str.strip()
    contract["season"] = contract["season"].where(contract["season"].isin(SEASON_TO_QUARTER), contract["aodb_season"])
    contract["episodes_numeric"] = pd.to_numeric(contract["Episodes"], errors="coerce")
    contract["episodes"] = contract["episodes_numeric"]
    contract["duration_minutes"] = _parse_duration_minutes(contract["Duration"])
    contract["duration"] = contract["duration_minutes"]
    contract["source"] = _source_to_project(contract["Source"])
    contract["genres"] = _json_list_from_comma_series(contract["Genres"])
    contract["themes"] = _json_list_from_comma_series(contract["Themes"])
    contract["studios"] = _json_studios_from_comma_series(contract["Studios"])
    contract["countryOfOrigin"] = "JP"
    contract["isAdult"] = contract["Rating"].astype("string").str.contains("Hentai", case=False, na=False)
    contract["is_sequel"] = False
    contract["has_sequel"] = False
    contract["prequel_count"] = 0
    contract["prequel_popularity_mean"] = 0.0
    contract["prequel_meanScore_mean"] = 0.0
    contract["startDate_month"] = _quarter_to_start_month(contract["release_quarter"])
    contract["startDate_day"] = 1
    contract["voice_actor_names"] = ""
    contract["popularity"] = pd.NA
    contract["meanScore"] = pd.NA
    contract["coverImage_extraLarge"] = contract["image"]
    contract["coverImage_large"] = contract["image"]
    contract["coverImage_medium"] = contract["image"]
    contract["bannerImage"] = pd.NA
    contract["external_cover_image_url"] = contract["image"]
    contract["external_cover_image_path"] = (
        "data/external_assets/mal2025_image/cover/"
        + contract["external_exam_id"].astype(str)
        + "_coverImage_extraLarge.jpg"
    )
    contract["external_banner_image_path"] = pd.NA
    contract["has_text_description"] = _valid_text(contract["description"])
    contract["has_cover_image"] = contract["external_cover_image_url"].astype("string").str.startswith("http")
    contract["has_banner_image"] = False
    contract["has_external_popularity_target"] = contract["external_popularity_members"].notna()
    contract["has_external_score_target"] = contract["external_score_0_100"].notna()
    contract["has_dual_targets"] = (
        contract["has_external_popularity_target"] & contract["has_external_score_target"]
    )
    contract["has_release_year_quarter"] = valid_quarter
    contract["has_aodb_anilist_id"] = contract["aodb_anilist_id"].notna()
    contract["can_prepare_image_text_metadata_inference"] = (
        contract["has_cover_image"]
        & contract["has_text_description"]
        & contract["has_release_year_quarter"]
    )
    contract["can_run_current_full_multimodal_without_new_assets"] = False
    contract["readiness_note"] = (
        "MAL 2025 image-ready row. It has cover URL, text, metadata, and labels; "
        "download images and generate fresh embeddings before current full model inference."
    )

    model_inputs = multimodal.rename(columns={"id": "resolved_anilist_id"})
    model_inputs["resolved_anilist_id"] = pd.to_numeric(
        model_inputs["resolved_anilist_id"], errors="coerce"
    ).astype("Int64")
    aligned = contract.merge(
        model_inputs,
        on="resolved_anilist_id",
        how="left",
        suffixes=("_external", "_internal"),
    )
    aligned = aligned.rename(
        columns={
            "popularity_internal": "anilist_popularity",
            "meanScore_internal": "anilist_meanScore",
            "split_pre_release_effective_internal": "split_pre_release_effective",
        }
    )

    exam_cols = [
        "external_exam_id",
        "mal_id",
        "internal_anilist_id",
        "aodb_anilist_id",
        "resolved_anilist_id",
        "is_internal_anilist_row",
        "title_romaji",
        "title_english",
        "format",
        "status",
        "season",
        "release_year",
        "release_quarter",
        "release_quarter_key",
        "episodes",
        "episodes_numeric",
        "duration",
        "duration_minutes",
        "source",
        "genres",
        "themes",
        "Studios",
        "studios",
        "description",
        "countryOfOrigin",
        "isAdult",
        "startDate_month",
        "startDate_day",
        "voice_actor_names",
        "is_sequel",
        "has_sequel",
        "prequel_count",
        "prequel_popularity_mean",
        "prequel_meanScore_mean",
        "coverImage_extraLarge",
        "coverImage_large",
        "coverImage_medium",
        "bannerImage",
        "external_cover_image_url",
        "external_cover_image_path",
        "external_banner_image_path",
        "external_popularity_members",
        "external_popularity_rank",
        "external_score_0_10",
        "external_score_0_100",
        "external_scored_by",
        "has_external_popularity_target",
        "has_external_score_target",
        "has_dual_targets",
        "has_release_year_quarter",
        "has_text_description",
        "has_cover_image",
        "has_banner_image",
        "has_aodb_anilist_id",
        "can_prepare_image_text_metadata_inference",
        "can_run_current_full_multimodal_without_new_assets",
        "readiness_note",
    ]
    existing_exam_cols = [col for col in exam_cols if col in contract.columns]
    image_ready = contract[contract["can_prepare_image_text_metadata_inference"]].copy()
    mal_only_image_ready = image_ready[~image_ready["is_internal_anilist_row"]].copy()
    popularity_exam = mal_only_image_ready[
        mal_only_image_ready["has_external_popularity_target"]
    ][existing_exam_cols].copy()
    dual_exam = mal_only_image_ready[mal_only_image_ready["has_dual_targets"]][existing_exam_cols].copy()

    aligned_cols = [
        "mal_id",
        "resolved_anilist_id",
        "title",
        "external_popularity_members",
        "external_popularity_rank",
        "external_score_0_10",
        "external_score_0_100",
        "external_cover_image_url",
        "release_year_external",
        "release_quarter_external",
        "split_pre_release_effective",
        "anilist_popularity",
        "anilist_meanScore",
        "can_prepare_image_text_metadata_inference",
    ]
    existing_aligned_cols = [col for col in aligned_cols if col in aligned.columns]
    aligned_contract = aligned[aligned["is_internal_anilist_row"]][existing_aligned_cols].copy()
    aligned_contract["external_eval_ready"] = (
        aligned_contract["external_popularity_members"].notna()
        & aligned_contract["external_score_0_100"].notna()
        & aligned_contract["can_prepare_image_text_metadata_inference"]
    )
    aligned_contract["external_label_note"] = (
        "MyAnimeList 2025 provides MAL members, score, and cover image URL; no title matching was used."
    )

    summary = {
        "source_rows": int(len(mal)),
        "rows_with_mal_id": int(contract["mal_id"].notna().sum()),
        "rows_with_cover_image_url": int(contract["has_cover_image"].sum()),
        "rows_with_text_description": int(contract["has_text_description"].sum()),
        "rows_with_release_year_quarter": int(contract["has_release_year_quarter"].sum()),
        "rows_with_external_members": int(contract["has_external_popularity_target"].sum()),
        "rows_with_external_score_0_100": int(contract["has_external_score_target"].sum()),
        "rows_mapped_to_internal_anilist_id": int(contract["is_internal_anilist_row"].sum()),
        "image_ready_rows": int(contract["can_prepare_image_text_metadata_inference"].sum()),
        "aligned_internal_image_ready_rows": int(
            (contract["is_internal_anilist_row"] & contract["can_prepare_image_text_metadata_inference"]).sum()
        ),
        "mal_only_rows_by_id": int((~contract["is_internal_anilist_row"]).sum()),
        "mal_only_image_ready_popularity_rows": int(len(popularity_exam)),
        "mal_only_image_ready_dual_target_rows": int(len(dual_exam)),
        "image_asset_note": "Only cover image URLs are available; banner and YOLO crops must be treated as missing/generated assets.",
    }
    return aligned_contract, dual_exam, popularity_exam, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _load_raw()
    crosswalk = _read_aodb_crosswalk()
    crosswalk_path = _write_crosswalk(crosswalk)

    recovered_rows, future_multimodal, holdout_summary = _prepare_holdout_recovery(raw, crosswalk)
    recovered_path = OUT_DIR / "aodb_holdout_unknown_recovered_rows.csv"
    future_path = OUT_DIR / "anilist_anime_multimodal_input_v1_aodb_recovered_future.csv"
    recovered_rows.to_csv(recovered_path, index=False)
    future_multimodal.to_csv(future_path, index=False)

    external_eval, mal_only_dual_exam, mal_only_popularity_exam, external_summary = _prepare_mal_july_external_eval(
        raw, crosswalk
    )
    external_eval_path = OUT_DIR / "mal_july2025_external_eval_contract.csv"
    mal_only_dual_path = OUT_DIR / "mal_july2025_mal_only_dual_target_exam.csv"
    mal_only_popularity_path = OUT_DIR / "mal_july2025_mal_only_popularity_exam.csv"
    external_eval.to_csv(external_eval_path, index=False)
    mal_only_dual_exam.to_csv(mal_only_dual_path, index=False)
    mal_only_popularity_exam.to_csv(mal_only_popularity_path, index=False)

    (
        mal2025_image_external_eval,
        mal2025_image_dual_exam,
        mal2025_image_popularity_exam,
        mal2025_image_summary,
    ) = _prepare_mal2025_image_external_eval(raw, crosswalk)
    mal2025_image_external_eval_path = OUT_DIR / "mal2025_image_external_eval_contract.csv"
    mal2025_image_dual_path = OUT_DIR / "mal2025_image_mal_only_dual_target_exam.csv"
    mal2025_image_popularity_path = OUT_DIR / "mal2025_image_mal_only_popularity_exam.csv"
    mal2025_image_external_eval.to_csv(mal2025_image_external_eval_path, index=False)
    mal2025_image_dual_exam.to_csv(mal2025_image_dual_path, index=False)
    mal2025_image_popularity_exam.to_csv(mal2025_image_popularity_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "aodb_csv": AODB_CSV.as_posix(),
            "mal_july_anime_csv": MAL_JULY_ANIME_CSV.as_posix(),
            "mal2025_image_anime_csv": MAL_2025_IMAGE_ANIME_CSV.as_posix(),
            "processed_csv": PROCESSED_CSV.as_posix(),
            "multimodal_csv": MULTIMODAL_CSV.as_posix(),
            "holdout_csv": HOLDOUT_CSV.as_posix(),
        },
        "outputs": {
            "aodb_id_crosswalk": crosswalk_path.as_posix(),
            "aodb_holdout_unknown_recovered_rows": recovered_path.as_posix(),
            "future_multimodal_with_aodb_recovery": future_path.as_posix(),
            "mal_july2025_external_eval_contract": external_eval_path.as_posix(),
            "mal_july2025_mal_only_dual_target_exam": mal_only_dual_path.as_posix(),
            "mal_july2025_mal_only_popularity_exam": mal_only_popularity_path.as_posix(),
            "mal2025_image_external_eval_contract": mal2025_image_external_eval_path.as_posix(),
            "mal2025_image_mal_only_dual_target_exam": mal2025_image_dual_path.as_posix(),
            "mal2025_image_mal_only_popularity_exam": mal2025_image_popularity_path.as_posix(),
        },
        "crosswalk": {
            "rows": int(len(crosswalk)),
            "unique_anilist_ids": int(crosswalk["anilist_id"].nunique(dropna=True)),
            "unique_mal_ids": int(crosswalk["mal_id"].nunique(dropna=True)),
            "rows_with_valid_quarter": int(crosswalk["aodb_release_quarter"].notna().sum()),
            "rows_with_season_year": int(crosswalk["aodb_season_year"].notna().sum()),
        },
        "holdout_recovery": holdout_summary,
        "external_eval": external_summary,
        "mal2025_image_external_eval": mal2025_image_summary,
    }
    summary_path = OUT_DIR / "external_evaluation_assets_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
