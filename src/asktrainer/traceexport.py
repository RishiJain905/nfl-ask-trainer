"""Turning a trace run's passing answers into training examples (task `call`).

    ask-trainer traces export --run tsmoke-1 [--val-share 0.1] [--seed 0]

For each (question, thinking level) the first passing answer of the run becomes one `call`
example: the question's tool-call prompt and tools, and as the target the model's own thought
(`reasoning_content`, none for thinking off) and its own tool call (the query that passed the
check, not the reference: it is the one the thought leads to). Whole questions go to the
validation split (a seeded `val_share`), so no question is in both.

Writes `<training>/public/traces-<run>/examples_{train,val}.jsonl` and `stats.json`, so a config
picks the set up like any public dataset (`data.public: {traces-<run>: {train: -1, val: -1}}`).

Also here: `gold` (the datasets' own queries as thinking-off calls, for the questions without a
kept trace), `evalset` (the scored held-out questions) and its sibling the test set.
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
    public_root: Path,
    out_dir: Path,
    per_set: int = 50,
    seed: int = 0,
    sets=None,
    exclude: set[str] | None = None,
    group: str = "evalset",
) -> dict[str, Any]:
    """Held-out questions in the tool-call format, for the scored evaluation of trained runs:
    a seeded `per_set` (-1: all) from each public set's *validation* split (the dataset's own
    test split: never traced or trained on), with the dataset's gold query as the call. Group
    `evalset`: the trainer generates and scores these but leaves them out of the validation loss
    (their thought is a placeholder). The test set (`group="testset"`) is the rest of the same
    splits (`exclude`: the evalset's source ids), scored only once, on the finished run."""
    from asktrainer.data import tool_prompt

    sets = sets or ("sqale", "gretel", "text2cypher", "cypherbench")
    rnd = random.Random(seed)
    prefix = "eval" if group == "evalset" else "test"
    out, counts = [], {}
    for name in sets:
        rows = [r for r in _lines(public_root / name / "examples_val.jsonl")
                if r["id"] not in (exclude or set())]  # fmt: skip
        random.Random(f"{seed}-{name}").shuffle(rows)
        rows = rows if per_set < 0 else rows[:per_set]
        counts[name] = len(rows)
        for ex in rows:
            p = tool_prompt.from_example_dict(ex, rnd)
            gold = json.loads(ex["messages"][-1]["content"])["query"]
            turn = tool_prompt.assistant_turn(EVAL_THOUGHT, p["tool"], gold)
            out.append(
                Example.model_validate(
                    {
                        "id": f"{prefix}~{ex['id']}",
                        "origin": ex["origin"],
                        "task": "call",
                        "split": "val",
                        "group": group,
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
    stats = {"per_set": counts, "examples": len(out), "seed": seed, "group": group,
             "excluded": len(exclude or ())}  # fmt: skip
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def gold_example(ex: dict[str, Any], split: str, seed: int = 0) -> Example:
    """A public example's gold query as a thinking-off call: the tool-call prompt the traces
    use (its tool name and distractor drawn per example, so a different sample never shifts
    them), no thought, and the dataset's own query as the call."""
    from asktrainer.data import tool_prompt

    p = tool_prompt.from_example_dict(ex, random.Random(f"{seed}-{ex['id']}"))
    gold = json.loads(ex["messages"][-1]["content"])["query"]
    return Example.model_validate(
        {
            "id": f"{ex['id']}~gold",
            "origin": ex["origin"],
            "task": "call",
            "split": split,
            "group": "gold",
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
                tool_prompt.assistant_turn(None, p["tool"], gold),
            ],
            "jepa": {"text": ex["jepa"]["text"], "code": gold},
            "dialect": ex["dialect"],
            "meta": {**(ex.get("meta") or {}), "qid": ex["id"], "gold": gold},
            "tools": p["tools"],
            "thinking": "off",
        }
    )


def gold(
    public_root: Path,
    out_dir: Path,
    sizes: dict[str, int],
    traced: set[str],
    kept: set[str],
    val_per_set: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """The datasets' gold queries as thinking-off examples (group `gold`), from each set's
    *training* split, in a seeded order:

    - left out: the questions with a kept trace (`kept`: the trace export's train *and* val, so
      no question is in two examples) and those whose text repeats one of the set's test split
      (`examples_val.jsonl`, where the evalset and the test set come from);
    - the first `val_per_set` questions never traced (`traced`) are the validation split, so no
      trace, training example or test question shares one;
    - then up to `sizes[set]` training examples (-1: all the rest).

    With `kept == traced` no traced question is used at all (the P9 check: new questions only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, info = [], [], {}
    for name, n in sizes.items():
        rows = sorted(_lines(public_root / name / "examples_train.jsonl"), key=lambda r: r["id"])
        random.Random(f"{seed}-{name}").shuffle(rows)
        tests = {_norm(r["jepa"]["text"])
                 for r in _lines(public_root / name / "examples_val.jsonl")}  # fmt: skip
        repeats = {r["id"] for r in rows if _norm(r["jepa"]["text"]) in tests}
        rows = [r for r in rows if r["id"] not in kept and r["id"] not in repeats]
        v = [r for r in rows if r["id"] not in traced][:val_per_set]
        v_ids = {r["id"] for r in v}
        t = [r for r in rows if r["id"] not in v_ids]
        t = t if n < 0 else t[:n]
        val += [gold_example(r, "val", seed) for r in v]
        train += [gold_example(r, "train", seed) for r in t]
        info[name] = {"train": len(t), "val": len(v), "test_repeats_left_out": len(repeats),
                      "traced_in_train": sum(r["id"] in traced for r in t)}  # fmt: skip
    write_jsonl(out_dir / "examples_train.jsonl", train)
    write_jsonl(out_dir / "examples_val.jsonl", val)
    stats = {"train": len(train), "val": len(val), "seed": seed, "sets": info,
             "kept_traces_left_out": len(kept), "traced": len(traced)}  # fmt: skip
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
