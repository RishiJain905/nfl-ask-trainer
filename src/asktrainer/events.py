"""The run's event log (`<run>/events.jsonl`) and info card (`<run>/run_info.json`).

Both are for the live dashboard, which reads the run folder while it trains. Append-only, one
JSON object per line: `{"ts": <UTC ISO>, "step": <int or null>, "kind": ..., ...fields}`.
Kinds: start, data_ready, model_loaded, resumed, eval_start, eval_end, oom_split,
verifier_failed, checkpoint_saved, artifact_uploaded, artifact_failed, epoch_end, done,
crashed. Fields are numbers, names and short codes only: never an exception's text (it could
quote a path, a query or worse), never anything secret.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

KINDS = {
    "start", "data_ready", "model_loaded", "resumed", "eval_start", "eval_end", "oom_split",
    "verifier_failed", "checkpoint_saved", "artifact_uploaded", "artifact_failed",
    "epoch_end", "done", "crashed", "val_loss", "best",
}  # fmt: skip


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


class Events:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.step: int | None = None  # the loop's current step (stamped on every event)

    def emit(self, kind: str, step: int | None = None, **fields: Any) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        row = {"ts": utc_now(), "step": self.step if step is None else step, "kind": kind,
               **fields}  # fmt: skip
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row


NULL = Events(None)


def write_run_info(path: Path, info: dict[str, Any]) -> None:
    """Rewrite `run_info.json` atomically (a reader never sees half a file)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
