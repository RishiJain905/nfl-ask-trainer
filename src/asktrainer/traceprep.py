"""Preparing a thinking-trace run: which questions, in which tool-call prompt, how many tries.

    ask-trainer traces prep --name tsmoke-1 --per-set 25 --plan high:4,low:2,off:1 [--seed 0]

Writes `<traces>/<run>/`:
- `questions.jsonl`: one line per training question: the tool-call prompt (`data/tool_prompt`),
  the reference query, and what the checker needs (`schema_script`: gretel's own CREATE + INSERT
  context, SQaLe's CREATE TABLEs from the prompt; nothing for Cypher yet; `real_rows`: whether
  the script holds real rows);
- `config.json`: the plan (tries per thinking level), the sampling settings, the seed.

Questions come from the converted *training* sets (`public/<name>/examples_train.jsonl`), a
seeded sample per set; validation and benchmark examples are never used for traces. The run
itself (`tracegen`) reads only these two files, so it can run on Windows with the standard
library next to llama-server.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asktrainer.data import tool_prompt
from asktrainer.publiceval import PublicData

SETS = ("sqale", "gretel", "text2cypher", "cypherbench")
DEFAULT_PLAN = {"high": 4, "low": 2, "off": 1}
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192}  # Spark's generation_config


def parse_plan(text: str) -> dict[str, int]:
    plan: dict[str, int] = {}
    for part in text.split(","):
        level, _, n = part.partition(":")
        level = level.strip()
        if level not in ("high", "low", "off") or not n.strip().isdigit():
            raise ValueError(f"bad plan entry {part!r}: use high:N,low:N,off:N")
        plan[level] = int(n)
    return plan


def _sample(path: Path, n: int, rnd: random.Random) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rnd.shuffle(rows)
    return rows[:n]


def prep(
    run_dir: Path,
    public_root: Path,
    per_set: int,
    plan: dict[str, int],
    seed: int = 0,
    sets: tuple[str, ...] = SETS,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "results.jsonl").exists():
        raise FileExistsError(f"{run_dir} already has results; use a new run name")
    data = PublicData(public_root)
    rnd = random.Random(seed)
    out, counts = [], {}
    for name in sets:
        rows = _sample(
            public_root / name / "examples_train.jsonl", per_set, random.Random(f"{seed}-{name}")
        )
        counts[name] = len(rows)
        for ex in rows:
            p = tool_prompt.from_example_dict(ex, rnd)
            gold = json.loads(ex["messages"][-1]["content"])["query"]
            script, real = None, False
            if ex["origin"] == "gretel":
                script = data.gretel_context(
                    ex["meta"].get("row_id", ""), ex["meta"].get("source_split") or "train"
                )
                real = script is not None and "INSERT" in script.upper()
            elif ex["dialect"] in ("sqlite", "duckdb"):
                script = tool_prompt.split_old_system(ex["messages"][0]["content"])[0]
            out.append(
                {
                    "qid": ex["id"],
                    "origin": ex["origin"],
                    "dialect": ex["dialect"],
                    "question": ex["jepa"]["text"],
                    "system": p["system"],
                    "user": p["user"],
                    "tools": p["tools"],
                    "tool": p["tool"],
                    "distractor": p["distractor"],
                    "gold": gold,
                    "schema_script": script,
                    "real_rows": real,
                    "meta": {k: ex["meta"].get(k) for k in ("db_id", "row_id", "source_split")},
                }
            )
    with (run_dir / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    cfg = {
        "name": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "per_set": per_set,
        "sets": counts,
        "plan": plan,
        "sampling": SAMPLING,
        "questions": len(out),
        "requests": len(out) * sum(plan.values()),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg
