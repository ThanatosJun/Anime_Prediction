"""
Generate a frozen manifest for raw dataset snapshots.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "raw_manifest.json"
RAW_FILES = [
    RAW_DIR / "anilist_anime_data_complete.pkl",
    RAW_DIR / "anilist_anime_data_complete.csv",
]


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    files = []
    for path in RAW_FILES:
        if not path.exists():
            continue
        files.append(
            {
                "path": str(path.as_posix()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "snapshot_name": "anilist_raw_frozen_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
