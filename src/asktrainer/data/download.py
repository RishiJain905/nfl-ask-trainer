"""Download the four public text-to-query datasets into `public/<name>/raw/`.

Every download is pinned to the dataset's current commit (the `revision` in the manifest), only
the files the converters read are fetched (no populated databases, images or graph dumps), and
`raw/manifest.json` records what came down: repo id, revision, files with sizes and sha256, the
license and the date.

SQaLe's DDL is the one special case. The questions live in `SQaLe-2-text-to-SQL-Queries` (about
290 MB), but the schemas of its 9,259 databases live in a companion dataset of 4 GB where 98% of
the bytes are the generated table rows. The DDL is a plain string column of the same parquet
files, so it is read with HTTP range requests (only the `schema_id` / `full_schema` /
`number_of_tables` column chunks are transferred) and written to one small file,
`raw/schemas_ddl.parquet`. The populated rows are never downloaded.

Guard: a plan over 2 GB (`MAX_BYTES`) is refused unless `allow_large=True`; ask the lead first.

    python -m asktrainer.data.download --dataset all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asktrainer.settings import get_paths

MAX_BYTES = 2 * 1024**3  # a single dataset download above this needs the lead's OK first

SQALE_SCHEMAS_REPO = "trl-lab/SQaLe-2-text-to-SQL-Schemas"
SQALE_DDL_FILE = "schemas_ddl.parquet"
SQALE_DDL_COLUMNS = ("schema_id", "full_schema", "number_of_tables")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    license: str
    allow_patterns: tuple[str, ...]
    source_url: str


SPECS: dict[str, DatasetSpec] = {
    "sqale": DatasetSpec(
        name="sqale",
        repo_id="trl-lab/SQaLe-2-text-to-SQL-Queries",
        license="MIT",
        allow_patterns=("README.md", "data/*.parquet"),
        source_url="https://huggingface.co/datasets/trl-lab/SQaLe-2-text-to-SQL-Queries",
    ),
    "gretel": DatasetSpec(
        name="gretel",
        repo_id="gretelai/synthetic_text_to_sql",
        license="Apache-2.0",
        allow_patterns=("README.md", "*.parquet"),
        source_url="https://huggingface.co/datasets/gretelai/synthetic_text_to_sql",
    ),
    "text2cypher": DatasetSpec(
        name="text2cypher",
        repo_id="neo4j/text2cypher-2025v1",
        license="Apache-2.0",
        allow_patterns=("README.md", "data/*.parquet"),
        source_url="https://huggingface.co/datasets/neo4j/text2cypher-2025v1",
    ),
    "cypherbench": DatasetSpec(
        name="cypherbench",
        repo_id="megagonlabs/cypherbench",
        license="Apache-2.0",
        allow_patterns=("README.md", "graphs/schemas/*.json", "train.json", "test.json"),
        source_url="https://huggingface.co/datasets/megagonlabs/cypherbench",
    ),
}


def raw_dir(name: str) -> Path:
    return get_paths().public / name / "raw"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _retry(fn, what: str, attempts: int = 4):  # noqa: ANN001, ANN202
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # network hiccups: back off and retry, then give up loudly
            if i == attempts:
                raise
            wait = 3 * i
            print(
                f"  {what}: {type(e).__name__}: {str(e)[:120]}; retry {i}/{attempts - 1} in {wait}s"
            )
            time.sleep(wait)


def _list_files(repo_id: str, revision: str, patterns: tuple[str, ...]) -> list[Any]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    return list(
        filter_repo_objects(info.siblings, allow_patterns=list(patterns), key=lambda s: s.rfilename)
    )


def _head_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo_id).sha


def plan(name: str, revision: str | None = None) -> dict[str, Any]:
    """What `download(name)` would fetch: revision, files, bytes (no network transfer of data)."""
    spec = SPECS[name]
    rev = revision or _head_revision(spec.repo_id)
    files = [
        {"path": s.rfilename, "size": s.size or 0}
        for s in _list_files(spec.repo_id, rev, spec.allow_patterns)
    ]
    out: dict[str, Any] = {
        "dataset": name,
        "repo_id": spec.repo_id,
        "revision": rev,
        "files": files,
        "bytes": sum(f["size"] for f in files),
    }
    if name == "sqale":
        ddl = _sqale_ddl_plan()
        out["schemas"] = ddl
        out["bytes"] += ddl["bytes"]
    return out


# ---- SQaLe DDL (column-projected range reads) -------------------------------------------------


def _sqale_ddl_plan(revision: str | None = None) -> dict[str, Any]:
    """The companion shards and how many bytes of them the three DDL columns take."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    rev = revision or _head_revision(SQALE_SCHEMAS_REPO)
    shards = [s for s in _list_files(SQALE_SCHEMAS_REPO, rev, ("data/*.parquet",))]
    fs = HfFileSystem()
    plan_files = []
    total = 0
    for s in sorted(shards, key=lambda s: s.rfilename):
        with fs.open(f"datasets/{SQALE_SCHEMAS_REPO}@{rev}/{s.rfilename}", "rb") as f:
            md = pq.ParquetFile(f).metadata
        nbytes = 0
        for rg in range(md.num_row_groups):
            for ci in range(md.num_columns):
                col = md.row_group(rg).column(ci)
                if col.path_in_schema in SQALE_DDL_COLUMNS:
                    nbytes += col.total_compressed_size
        total += nbytes
        plan_files.append(
            {
                "path": s.rfilename,
                "size": s.size or 0,
                "sha256": getattr(getattr(s, "lfs", None), "sha256", None),
                "rows": md.num_rows,
                "column_bytes": nbytes,
            }
        )
    return {
        "repo_id": SQALE_SCHEMAS_REPO,
        "revision": rev,
        "columns": list(SQALE_DDL_COLUMNS),
        "shards": plan_files,
        "bytes": total,
        "full_size": sum(f["size"] for f in plan_files),
    }


def _download_sqale_ddl(raw: Path, ddl_plan: dict[str, Any]) -> dict[str, Any]:
    """Read the DDL columns of every companion shard (range requests) into one parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    out = raw / SQALE_DDL_FILE
    tmp = out.with_suffix(".parquet.tmp")
    rev = ddl_plan["revision"]
    writer = None
    n_rows = 0
    try:
        for shard in ddl_plan["shards"]:
            split = "test" if Path(shard["path"]).name.startswith("test") else "train"

            def read_one(path: str = shard["path"]) -> pa.Table:
                with fs.open(f"datasets/{SQALE_SCHEMAS_REPO}@{rev}/{path}", "rb") as f:
                    return pq.ParquetFile(f).read(columns=list(SQALE_DDL_COLUMNS))

            tab = _retry(read_one, shard["path"])
            tab = tab.append_column("split", pa.array([split] * tab.num_rows, pa.string()))
            if writer is None:
                writer = pq.ParquetWriter(tmp, tab.schema, compression="zstd")
            writer.write_table(tab, row_group_size=1024)
            n_rows += tab.num_rows
            print(f"  ddl {shard['path']}: {tab.num_rows} schemas")
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    return {
        "file": SQALE_DDL_FILE,
        "rows": n_rows,
        "size": out.stat().st_size,
        "sha256": _sha256(out),
    }


# ---- the download -------------------------------------------------------------------------------


def _manifest_ok(raw: Path) -> bool:
    mf = raw / "manifest.json"
    if not mf.exists():
        return False
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
        return all(
            (raw / f["path"]).exists() and (raw / f["path"]).stat().st_size == f["size"]
            for f in data["files"]
        )
    except (OSError, ValueError, KeyError):
        return False


def download(name: str, *, force: bool = False, allow_large: bool = False) -> dict[str, Any]:
    """Fetch one dataset into `public/<name>/raw/` and write its manifest; returns the manifest."""
    from huggingface_hub import snapshot_download

    spec = SPECS[name]
    raw = raw_dir(name)
    if _manifest_ok(raw) and not force:
        print(f"[{name}] already downloaded ({raw / 'manifest.json'})")
        return json.loads((raw / "manifest.json").read_text(encoding="utf-8"))

    p = plan(name)
    if p["bytes"] > MAX_BYTES and not allow_large:
        raise RuntimeError(
            f"{name}: the plan is {p['bytes'] / 1e9:.2f} GB, over the {MAX_BYTES / 1e9:.1f} GB "
            "limit. Ask the lead before downloading this."
        )
    mb = p["bytes"] / 1e6
    print(f"[{name}] {spec.repo_id}@{p['revision'][:10]}: {len(p['files'])} files, {mb:.1f} MB")
    raw.mkdir(parents=True, exist_ok=True)
    _retry(
        lambda: snapshot_download(
            spec.repo_id,
            repo_type="dataset",
            revision=p["revision"],
            allow_patterns=list(spec.allow_patterns),
            local_dir=raw,
        ),
        f"snapshot_download {spec.repo_id}",
    )
    shutil.rmtree(raw / ".cache", ignore_errors=True)  # the hub client's resume bookkeeping

    manifest: dict[str, Any] = {
        "dataset": name,
        "repo_id": spec.repo_id,
        "revision": p["revision"],
        "license": spec.license,
        "source_url": spec.source_url,
        "downloaded": datetime.now(UTC).date().isoformat(),
        "files": [],
    }
    if name == "sqale":
        info = _download_sqale_ddl(raw, p["schemas"])
        manifest["schemas"] = {
            "repo_id": SQALE_SCHEMAS_REPO,
            "revision": p["schemas"]["revision"],
            "license": "MIT",
            "columns_read": p["schemas"]["columns"],
            "bytes_transferred_estimate": p["schemas"]["bytes"],
            "full_dataset_size": p["schemas"]["full_size"],
            "source_shards": [
                {k: sh[k] for k in ("path", "size", "sha256", "rows")}
                for sh in p["schemas"]["shards"]
            ],
            "written_to": info,
        }
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp"):
            manifest["files"].append(
                {
                    "path": path.relative_to(raw).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest["total_bytes"] = sum(f["size"] for f in manifest["files"])
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[{name}] done: {manifest['total_bytes'] / 1e6:.1f} MB in {raw}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="all", choices=[*SPECS, "all"])
    ap.add_argument(
        "--force", action="store_true", help="download again even if the manifest is intact"
    )
    ap.add_argument(
        "--plan", action="store_true", help="only print the plan (revision, files, bytes)"
    )
    ap.add_argument(
        "--allow-large", action="store_true", help="allow a plan over 2 GB (the lead must OK it)"
    )
    args = ap.parse_args(argv)
    names = list(SPECS) if args.dataset == "all" else [args.dataset]
    for name in names:
        if args.plan:
            p = plan(name)
            print(json.dumps({k: v for k, v in p.items() if k != "files"}, indent=1)[:2000])
        else:
            download(name, force=args.force, allow_large=args.allow_large)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
