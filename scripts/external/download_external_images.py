"""
Download external exam cover images declared by an image-ready exam CSV.

The CSV is expected to contain:
- external_cover_image_url
- external_cover_image_path

Generated images are written under data/external_assets/, which is ignored by git.
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAM = ROOT / "data" / "external_transformed" / "mal2025_image_mal_only_dual_target_exam.csv"
USER_AGENT = "AnimePredictionExternalEval/1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download external exam cover images.")
    parser.add_argument("--exam-csv", default=str(DEFAULT_EXAM), help="Image-ready external exam CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process.")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between downloads.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds.")
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _download(url: str, out_path: Path, timeout: float) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                return False, f"non-image content-type: {content_type}"
            data = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return True, "downloaded"


def main() -> None:
    args = _parse_args()
    exam_csv = _resolve(args.exam_csv)
    df = pd.read_csv(exam_csv)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    required = {"external_cover_image_url", "external_cover_image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{exam_csv} missing required columns: {sorted(missing)}")

    rows = []
    for row in df.itertuples(index=False):
        url = str(getattr(row, "external_cover_image_url", "") or "").strip()
        out_path = _resolve(str(getattr(row, "external_cover_image_path")))
        status = "skipped"
        message = ""
        if not url.startswith("http"):
            status = "failed"
            message = "missing url"
        elif out_path.exists() and not args.force:
            status = "exists"
            message = "already exists"
        else:
            ok, message = _download(url, out_path, args.timeout)
            status = "downloaded" if ok else "failed"
            if args.sleep > 0:
                time.sleep(args.sleep)

        rows.append(
            {
                "external_exam_id": getattr(row, "external_exam_id", None),
                "mal_id": getattr(row, "mal_id", None),
                "url": url,
                "path": out_path.as_posix(),
                "status": status,
                "message": message,
            }
        )

    log_path = exam_csv.with_name(exam_csv.stem + "_image_download_log.csv")
    log = pd.DataFrame(rows)
    log.to_csv(log_path, index=False)
    print(log["status"].value_counts(dropna=False).to_string())
    print(f"Log -> {log_path}")


if __name__ == "__main__":
    main()
