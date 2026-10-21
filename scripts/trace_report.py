"""Summarise a finished thinking-trace run: pass rates, lenient rates, exploration, lengths, speed.

    python scripts/trace_report.py --run D:\\nfl-ml-data\\ask\\traces\\tsmoke-1 [--json out.json]

Standard library only. "Lenient" re-checks the strict failures that only differ in shape: the
answer returns the reference's columns plus others, or the same columns in another order (the
strict check, which training data and benchmarks use, rejects both). "Exploration" counts
failing calls that look at the data instead of answering (`SELECT DISTINCT col FROM t`,
`SELECT ... LIMIT n` with no filter or aggregate, `MATCH (n) RETURN ... LIMIT n`), a sign the
model wanted a second turn.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asktrainer import querycheck as qc  # noqa: E402

AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b", re.IGNORECASE)


def exploratory(q: str, dialect: str) -> bool:
    s = " ".join((q or "").split())
    if dialect == "cypher":
        return bool(re.search(r"\b(db\.labels|db\.schema|apoc\.meta)", s, re.I)) or (
            bool(re.search(r"\bLIMIT\s+\d+\s*$", s, re.I))
            and "WHERE" not in s.upper()
            and not AGG.search(s)
        )
    if re.match(r"^SELECT\s+DISTINCT\s+[\w.\"`]+\s+FROM\s+[\w.\"`]+\s*;?$", s, re.I):
        return True
    if re.match(r"^SELECT\s+\*\s+FROM\s+[\w.\"`]+(\s+LIMIT\s+\d+)?\s*;?$", s, re.I):
        return True
    return (
        bool(re.search(r"\bLIMIT\s+\d+\s*;?$", s, re.I))
        and " WHERE " not in f" {s.upper()} "
        and not AGG.search(s)
    )


def lenient(q: dict, r: dict) -> bool:
    """Does a strict SQL failure only differ in shape (extra columns, column order)?"""
    if q["dialect"] == "cypher" or r.get("reason") not in ("rows_differ", "fuzz_rows_differ"):
        return False
    try:
        creates = qc._creates(q["schema_script"])
    except Exception:
        return False
    dbs = []
    if q.get("real_rows"):
        dbs.append(qc._connect(q["schema_script"]))
    lits = qc._literals([q["gold"], r["query"]])
    dbs += [qc.fuzz_db(creates, s, lits) for s in qc.FUZZ_SEEDS]
    informative = 0
    try:
        for db in dbs:
            try:
                g, p = qc._run(db, q["gold"]), qc._run(db, r["query"].strip().rstrip(";"))
            except Exception:
                return False
            if not g and not p:
                continue
            if not g or not p or len(p[0]) < len(g[0]) or len(p[0]) > 8:
                return False
            ok = any(
                qc.same_results(g, [tuple(row[c] for c in cols) for row in p])
                for cols in itertools.permutations(range(len(p[0])), len(g[0]))
            )
            if not ok:
                return False
            informative += 1
    finally:
        for db in dbs:
            db.close()
    return informative > 0


def pctl(v, q):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    qs = {
        q["qid"]: q
        for q in (
            json.loads(x)
            for x in (run / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        )
    }
    res = [
        json.loads(x)
        for x in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    events = [
        json.loads(x)
        for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    progress = [
        json.loads(x)
        for x in (run / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    for r in res:
        q = qs[r["qid"]]
        r["lenient"] = r.get("verdict") == "pass" or (
            r.get("verdict") == "fail" and r.get("query") and lenient(q, r)
        )
        r["explore"] = (
            r.get("verdict") == "fail"
            and bool(r.get("query"))
            and exploratory(r["query"], q["dialect"])
        )
    out: dict = {
        "run": run.name,
        "answers": len(res),
        "planned": cfg["requests"],
        "plan": cfg["plan"],
    }
    # wall-clock generation time: sum of the gaps between progress lines under 60 s (drops restarts)
    ts = [datetime.fromisoformat(p["ts"]) for p in progress]
    busy = sum(
        min((b - a).total_seconds(), 60)
        for a, b in zip(ts, ts[1:], strict=False)
        if (b - a).total_seconds() < 600
    )
    comp = sum(r.get("completion_tokens") or 0 for r in res)
    prompt = sum(r.get("prompt_tokens") or 0 for r in res)
    out["speed"] = {
        "busy_minutes": round(busy / 60, 1),
        "completion_tokens": comp,
        "prompt_tokens": prompt,
        "completion_tok_s": round(comp / busy, 1) if busy else None,
        "answers_per_min": round(len(res) / busy * 60, 1) if busy else None,
        "restarts": sum(1 for e in events if e.get("kind") == "start") - 1,
    }

    def block(rows):
        v = Counter(r.get("verdict") for r in rows)
        checked = v["pass"] + v["fail"]
        th = [r.get("think_tokens") or 0 for r in rows]
        kept = [r.get("think_tokens") or 0 for r in rows if r.get("verdict") == "pass"]
        return {
            "n": len(rows),
            "pass": v["pass"],
            "fail": v["fail"],
            "unverified": v["unverified"],
            "pass_rate": round(v["pass"] / checked, 3) if checked else None,
            "lenient_rate": round(sum(1 for r in rows if r["lenient"]) / checked, 3)
            if checked
            else None,
            "explore_share_of_fails": round(sum(1 for r in rows if r["explore"]) / v["fail"], 3)
            if v["fail"]
            else None,
            "think_p50": pctl(th, 0.5),
            "think_p90": pctl(th, 0.9),
            "think_mean": round(statistics.fmean(th), 1) if th else None,
            "kept_think_mean": round(statistics.fmean(kept), 1) if kept else None,
            "completion_mean": round(
                statistics.fmean([r.get("completion_tokens") or 0 for r in rows]), 1
            )
            if rows
            else None,
        }

    by_set, by_level, by_sl = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in res:
        by_set[r["origin"]].append(r)
        by_level[r["level"]].append(r)
        by_sl[(r["origin"], r["level"])].append(r)
    out["overall"] = block(res)
    out["by_set"] = {k: block(v) for k, v in sorted(by_set.items())}
    out["by_level"] = {k: block(v) for k, v in by_level.items()}
    out["by_set_level"] = {f"{k[0]}|{k[1]}": block(v) for k, v in sorted(by_sl.items())}
    out["reasons"] = dict(
        Counter(
            (r.get("reason") or "").split(":")[0] for r in res if r.get("verdict") != "pass"
        ).most_common(15)
    )
    # questions solved within k tries, per level and overall (any level)
    solved = defaultdict(set)
    seen = defaultdict(set)
    for r in res:
        seen[r["origin"]].add(r["qid"])
        if r.get("verdict") == "pass":
            solved[r["origin"]].add(r["qid"])
    out["questions_solved"] = {k: f"{len(solved[k])} of {len(seen[k])}" for k in sorted(seen)}
    pass_at = {}
    levels = {lv: cfg.get("tries") or 1 for lv in cfg["mix"]} if cfg.get("mix") else cfg["plan"]
    for lv, n in levels.items():
        per_q = defaultdict(dict)
        for r in by_level.get(lv, []):
            per_q[r["qid"]][r["try"]] = r.get("verdict")
        ks = {}
        for k in range(1, n + 1):
            qq = [
                t
                for t in per_q.values()
                if (any(t.get(i) == "pass" for i in range(k)) or all(i in t for i in range(k)))
                and any(t.get(i) in ("pass", "fail") for i in range(k))
            ]
            if qq:
                ks[k] = round(
                    sum(1 for t in qq if any(t.get(i) == "pass" for i in range(k))) / len(qq), 3
                )
        pass_at[lv] = ks
    out["pass_at_k"] = pass_at
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
