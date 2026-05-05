"""
Fetch missing anime descriptions from Jikan (MAL unofficial API, no key needed).

Input:  data/fussion/missing_descriptions.csv
Output: data/fussion/supplemented_descriptions.csv
        columns: id, title_romaji, description, description_source

Usage:
    python -m src.fussion_branch.run_supplement_descriptions
    python -m src.fussion_branch.run_supplement_descriptions --skip-adult
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

JIKAN_SEARCH = "https://api.jikan.moe/v4/anime"
JIKAN_DELAY  = 0.5   # seconds between requests (rate limit: 3/s, 60/min)
OUT_PATH     = Path("data/fussion/supplemented_descriptions.csv")


def search_jikan(title: str, session: requests.Session) -> str | None:
    """Search MAL via Jikan and return synopsis if found."""
    try:
        resp = session.get(JIKAN_SEARCH, params={"q": title, "limit": 1}, timeout=10)
        if resp.status_code == 429:
            time.sleep(5)
            resp = session.get(JIKAN_SEARCH, params={"q": title, "limit": 1}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        synopsis = data[0].get("synopsis", "").strip()
        if synopsis and synopsis != "No synopsis information has been added to this title.":
            return synopsis
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-adult", action="store_true",
                        help="Skip isAdult=True anime")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    missing_df = pd.read_csv("data/fussion/missing_descriptions.csv")

    if args.skip_adult:
        missing_df = missing_df[~missing_df["isAdult"].astype(bool)]
        print(f"Skipping adult content → {len(missing_df)} remaining")

    # resume: skip already-fetched IDs
    done_ids = set()
    results = []
    if args.resume and OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        done_ids = set(existing["id"])
        results = existing.to_dict("records")
        print(f"Resuming — {len(done_ids)} already done")

    todo = missing_df[~missing_df["id"].isin(done_ids)]
    print(f"Fetching {len(todo)} anime from Jikan API…")

    session = requests.Session()
    found = 0

    for i, row in enumerate(todo.itertuples(), 1):
        title = row.title_english if pd.notna(row.title_english) and str(row.title_english).strip() else row.title_romaji
        synopsis = search_jikan(title, session)

        if synopsis:
            results.append({
                "id": row.id,
                "title_romaji": row.title_romaji,
                "description": synopsis,
                "description_source": "jikan_mal",
            })
            found += 1
        else:
            results.append({
                "id": row.id,
                "title_romaji": row.title_romaji,
                "description": None,
                "description_source": "not_found",
            })

        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(results).to_csv(OUT_PATH, index=False)
            print(f"  [{i}/{len(todo)}] found={found}  saved → {OUT_PATH}")

        time.sleep(JIKAN_DELAY)

    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)

    total = len(df)
    retrieved = (df["description_source"] == "jikan_mal").sum()
    print(f"\nDone: {retrieved}/{total} descriptions retrieved ({100*retrieved/total:.1f}%)")
    print(f"Saved → {OUT_PATH}")

    print(f"\nNext step: run python -m src.fussion_branch.run_supplement_merge")


if __name__ == "__main__":
    main()
