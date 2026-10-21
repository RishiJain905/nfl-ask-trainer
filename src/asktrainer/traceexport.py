"""Turning a trace run's passing answers into training examples (task `call`).

    ask-trainer traces export --run tsmoke-1 [--val-share 0.1] [--seed 0]

For each (question, thinking level) the first passing answer of the run becomes one `call`
example: the question's tool-call prompt and tools, and as the target the model's own thought
(`reasoning_content`, none for thinking off) and its own tool call (the query that passed the
check, not the reference: it is the one the thought leads to). Whole questions go to the
validation split (a seeded `val_share`), so no question is in both.

Writes `<training>/public/traces-<run>/examples_{train,val}.jsonl` and `stats.json`, so a config
picks the set up like any public dataset (`data.public: {traces-<run>: {train: -1, val: -1}}`).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from asktrainer.data.schema import Example, write_jsonl


def _lines(p: Path) -> list[dict[str, Any]]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def example(q: dict[str, Any], r: dict[str, Any], split: str, run: str) -> Example:
    tool, arg = q["tool"]["name"], q["tool"]["arg"]
    turn: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": tool, "arguments": {arg: r["query"]}}}
        ],
    }
    if r["level"] != "off":
        turn["reasoning_content"] = r["reasoning"]
    meta = {
        **(q.get("meta") or {}),
        "qid": q["qid"],
        "trace_run": run,
        "try": r["try"],
        "checked_by": r.get("checked_by"),
        "think_tokens": r.get("think_tokens"),
        "gold": q["gold"],
    }
    return Example.model_validate(
        {
            "id": f"{q['qid']}~{r['level']}",
            "origin": q["origin"],
            "task": "call",
            "split": split,
            "group": "traces",
            "messages": [
                {"role": "system", "content": q["system"]},
                {"role": "user", "content": q["user"]},
                turn,
            ],
            "jepa": {"text": q["question"], "code": r["query"]},
            "dialect": q["dialect"],
            "meta": meta,
            "tools": q["tools"],
            "thinking": r["level"],
        }
    )


EVAL_THOUGHT = "(the model writes its own thought here: this example is scored, not trained on)"


def evalset(
    public_root: Path, out_dir: Path, per_set: int = 50, seed: int = 0, sets=None
) -> dict[str, Any]:
    """Held-out questions in the tool-call format, for the scored evaluation of trained runs:
    a seeded `per_set` from each public set's *validation* split (never traced or trained on),
    with the dataset's gold query as the call. Group `evalset`: the trainer generates and scores
    these but leaves them out of the validation loss (their thought is a placeholder)."""
    from asktrainer.data import tool_prompt

    sets = sets or ("sqale", "gretel", "text2cypher", "cypherbench")
    rnd = random.Random(seed)
    out, counts = [], {}
    for name in sets:
        rows = _lines(public_root / name / "examples_val.jsonl")
        random.Random(f"{seed}-{name}").shuffle(rows)
        rows = rows[:per_set]
        counts[name] = len(rows)
        for ex in rows:
            p = tool_prompt.from_example_dict(ex, rnd)
            gold = json.loads(ex["messages"][-1]["content"])["query"]
            turn = tool_prompt.assistant_turn(EVAL_THOUGHT, p["tool"], gold)
            out.append(
                Example.model_validate(
                    {
                        "id": f"eval~{ex['id']}",
                        "origin": ex["origin"],
                        "task": "call",
                        "split": "val",
                        "group": "evalset",
                        "messages": [
                            {"role": "system", "content": p["system"]},
                            {"role": "user", "content": p["user"]},
                            turn,
                        ],
                        "jepa": {"text": ex["jepa"]["text"], "code": gold},
                        "dialect": ex["dialect"],
                        "meta": {**(ex.get("meta") or {}), "gold": gold},
                        "tools": p["tools"],
                        "thinking": "high",
                    }
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "examples_val.jsonl", out)
    stats = {"per_set": counts, "examples": len(out), "seed": seed}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def results(run_dir: Path) -> list[dict[str, Any]]:
    """The run's answers, with any later re-check (`rechecks.jsonl`) applied over the verdict."""
    rows = _lines(run_dir / "results.jsonl")
    rp = run_dir / "rechecks.jsonl"
    if rp.exists():
        later = {r["rid"]: r for r in _lines(rp)}
        for r in rows:
            if r["rid"] in later:
                rc = later[r["rid"]]
                r.update({k: rc[k] for k in ("verdict", "checked_by", "reason") if k in rc})
    return rows


def cut_by_budget(r: dict[str, Any], budget: int | None) -> bool:
    """A thought the server's --reasoning-budget ended (it then forces the end of thinking)."""
    return bool(budget) and (r.get("think_tokens") or 0) >= budget * 0.98


def export(run_dir: Path, out_dir: Path, val_share: float = 0.1, seed: int = 0) -> dict[str, Any]:
    questions = {q["qid"]: q for q in _lines(run_dir / "questions.jsonl")}
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    budget = cfg.get("reasoning_budget")
    best: dict[tuple[str, str], dict[str, Any]] = {}
    cut = 0
    for r in results(run_dir):
        if r.get("verdict") != "pass" or not r.get("query"):
            continue
        if r["level"] != "off" and not (r.get("reasoning") or "").strip():
            continue  # thinking on but nothing thought: not a usable trace
        if r["level"] != "off" and cut_by_budget(r, budget):
            cut += 1
            continue
        key = (r["qid"], r["level"])
        if key not in best or r["try"] < best[key]["try"]:
            best[key] = r
    qids = sorted({k[0] for k in best})
    random.Random(seed).shuffle(qids)
    val_q = set(qids[: round(len(qids) * val_share)])
    train, val = [], []
    for (qid, _level), r in sorted(best.items()):
        split = "val" if qid in val_q else "train"
        (val if split == "val" else train).append(example(questions[qid], r, split, run_dir.name))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "examples_train.jsonl", train)
    write_jsonl(out_dir / "examples_val.jsonl", val)
    stats = {
        "run": run_dir.name,
        "train": len(train),
        "val": len(val),
        "questions": len(qids),
        "val_questions": len(val_q),
        "left_out_cut_by_budget": cut,
        "by_origin_level": dict(
            sorted(Counter(f"{e.origin}/{e.thinking}" for e in train + val).items())
        ),
        "checked_by": dict(Counter(e.meta.get("checked_by") or "" for e in train + val)),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
