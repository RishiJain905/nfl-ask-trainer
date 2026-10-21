"""Download the benchmark data (approved by Rishi 2026-10-21) to <data root>/ask/benchmarks/.

    python scripts/download_benchmarks.py [--only cypherbench,sqale,bird]

- cypherbench: megagonlabs/cypherbench (Apache-2.0): the 11 graphs (full ~5.2 GB, sampled ~56 MB),
  their schemas, train.json / test.json.
- sqale: trl-lab/SQaLe-2-text-to-SQL-Schemas (MIT): the populated schemas (~4.0 GB of parquet).
- bird: BIRD dev (bird-bench.github.io; CC BY-SA 4.0): dev.zip (346 MB), the 1,534 dev questions
  with their 11 SQLite databases.

Each source goes to its own folder with a `manifest.json` (files, sizes, sha256, source, revision,
date). Downloaded files are data only: nothing in them is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/mnt/d/nfl-ml-data/ask/benchmarks")
BIRD_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(dirpath: Path, source: dict) -> None:
    files = sorted(
        p
        for p in dirpath.rglob("*")
        if p.is_file() and p.name != "manifest.json" and ".cache" not in p.parts
    )
    doc = {
        "source": source,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [
            {"path": str(p.relative_to(dirpath)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in files
        ],
    }
    doc["total_bytes"] = sum(f["bytes"] for f in doc["files"])
    (dirpath / "manifest.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"{dirpath.name}: {len(files)} files, {doc['total_bytes'] / 1e9:.2f} GB", flush=True)


def hf(repo: str, dest: Path, patterns: list[str] | None, license_: str) -> None:
    from huggingface_hub import HfApi, snapshot_download

    rev = HfApi().dataset_info(repo).sha
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo, repo_type="dataset", revision=rev, local_dir=str(dest), allow_patterns=patterns
    )
    manifest(dest, {"repo": repo, "revision": rev, "license": license_})


def bird(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "dev.zip"
    if not out.exists():
        tmp = out.with_suffix(".part")
        with urllib.request.urlopen(BIRD_URL, timeout=120) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        tmp.replace(out)
    manifest(
        dest, {"url": BIRD_URL, "license": "CC BY-SA 4.0", "site": "https://bird-bench.github.io/"}
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="cypherbench,sqale,bird")
    want = set(ap.parse_args().only.split(","))
    if "bird" in want:
        bird(ROOT / "bird")
    if "cypherbench" in want:
        hf("megagonlabs/cypherbench", ROOT / "cypherbench", None, "Apache-2.0")
    if "sqale" in want:
        hf(
            "trl-lab/SQaLe-2-text-to-SQL-Schemas",
            ROOT / "sqale2-schemas",
            ["data/*", "README.md"],
            "MIT",
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
