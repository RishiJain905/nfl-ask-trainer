"""The training mix: the NFL export plus seeded samples of the public datasets.

Inputs: `<nfl dir>/train.jsonl` and `val.jsonl` (the NFL repo's `nfl ask train-data`) and
`<training>/public/<name>/examples_{train,val}.jsonl` (`asktrainer.data.public`). Sampling is
seeded by the run's seed and independent of file order (ids are sorted first), so the same
config and seed always give the same mix; the run folder records every id used.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asktrainer.config import TrainConfig
from asktrainer.data.schema import Example, read_jsonl
from asktrainer.settings import Paths


@dataclass
class Mix:
    train: list[Example]
    val: list[Example]
    nfl_dir: Path | None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def nfl_val_path(self) -> Path | None:
        return self.nfl_dir / "val.jsonl" if self.nfl_dir else None


def _sample(examples: list[Example], n: int, rng: random.Random) -> list[Example]:
    ordered = sorted(examples, key=lambda e: e.id)
    if n < 0 or n >= len(ordered):
        return ordered
    return rng.sample(ordered, n)


def counts(examples: list[Example]) -> dict[str, int]:
    return dict(sorted(Counter(f"{e.origin}/{e.task}/{e.group}" for e in examples).items()))


def load_mix(cfg: TrainConfig, paths: Paths, length: Callable[[Example], int] | None = None) -> Mix:
    """The run's examples. `length` (tokens of an example's full chat) filters the NFL train
    pool to `nfl_max_tokens` before the per-task samples are drawn."""
    from asktrainer.evaluation import stratified_subset

    rng = random.Random(cfg.seed)
    train: list[Example] = []
    val: list[Example] = []
    info: dict[str, Any] = {"sources": {}}
    d = cfg.data
    wants_nfl = bool(d.nfl) and any(n != 0 for n in (d.nfl_route, d.nfl_generate, d.nfl_val))
    nfl_dir = d.nfl_dir(paths) if wants_nfl else None
    if nfl_dir is not None:
        tr = [e for e in read_jsonl(nfl_dir / "train.jsonl") if e.split == "train"]
        va = [e for e in read_jsonl(nfl_dir / "val.jsonl") if e.split == "val"]
        limit = cfg.data.nfl_max_tokens or cfg.optim.max_len
        too_long = []
        if length is not None:
            keep = []
            for e in tr:
                (keep if length(e) <= limit else too_long).append(e)
            tr = keep
        for task, n in (("route", cfg.data.nfl_route), ("generate", cfg.data.nfl_generate)):
            train += _sample([e for e in tr if e.task == task], n, rng)
        n_val, sub_seed = cfg.data.nfl_val, cfg.eval.subset_seed  # the run's seed doesn't move it
        val += va if n_val < 0 else stratified_subset(va, n_val, sub_seed)
        info["nfl_train_too_long"] = len(too_long)
        info["nfl_max_tokens"] = limit
        manifest = nfl_dir / "manifest.json"
        version = nfl_dir.name
        if manifest.exists():
            version = json.loads(manifest.read_text(encoding="utf-8")).get("version", version)
        info["nfl_version"] = version
        info["sources"]["nfl"] = {"dir": str(nfl_dir), "train_available": len(tr),
                                  "val_available": len(va)}  # fmt: skip
    for name, spec in sorted(cfg.data.public.items()):
        d = paths.public / name
        got: dict[str, Any] = {"dir": str(d)}
        for split, n, sink in (("train", spec.train, train), ("val", spec.val, val)):
            if n == 0:
                continue
            f = d / f"examples_{split}.jsonl"
            if not f.exists():
                raise FileNotFoundError(f"public dataset {name}: {f} is missing")
            # sample line numbers, then parse only those (the train files hold up to 30k long
            # examples on a slow disk); the same file and seed always give the same sample
            lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            pick = range(len(lines)) if n < 0 or n >= len(lines) else rng.sample(
                range(len(lines)), n)  # fmt: skip
            exs = [Example.model_validate_json(lines[i]) for i in sorted(pick)]
            bad = [e.id for e in exs if e.split != split]
            if bad:
                raise ValueError(f"{f}: examples not in split {split}: {bad[:3]}")
            sink += exs
            got[f"{split}_available"] = len(lines)
        info["sources"][name] = got
    ids = [e.id for e in train + val]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        raise ValueError(f"duplicate example ids in the mix: {dup[:5]}")
    rng.shuffle(train)
    info["counts"] = {"train": counts(train), "val": counts(val)}
    return Mix(train=train, val=val, nfl_dir=nfl_dir, info=info)
