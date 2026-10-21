"""Turn a thinking-trace run's files into dashboard ticks (same store and page as training runs).

    python scripts/trace_feed.py --run D:\\nfl-ml-data\\ask\\traces\\tsmoke-1
        [--gpu-dir F:\\nfl-ask-trainer-ckpt\\_monitor]

Reads only the run's own files: `config.json`, `results.jsonl` (one checked answer per line),
`progress.jsonl` (totals every 10 s), `events.jsonl`, `tracegen.pid`, and the GPU monitor's
samples. Writes `<run>/dashboard/tick-NNNNN.json` (new progress rows, new events, new GPU
samples, and a snapshot of the summary recomputed from every result) and, once,
`<run>/dashboard/run.json` (the run's identity, `kind: "trace"`). Prints one JSON line saying
what it wrote. Standard library only (runs on Windows next to the generator).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HEARTBEAT_S = 600
STALL_S = 900
EDGES = [0, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
LEVELS = ("high", "low", "off")


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def parse_ts(s):
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def read_lines(p: Path) -> list[dict]:
    out = []
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip().lstrip("\ufeff")
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass  # a line mid-write: picked up next tick
    except OSError:
        pass
    return out


def cap(s, n: int):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text(encoding="utf-8-sig").strip())
    except (OSError, ValueError):
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return str(pid) in out


def gpu_samples(gpu_dir: Path, since, until) -> list:
    out = []
    for f in sorted(glob.glob(str(gpu_dir / "gpu-*.jsonl"))):
        for r in read_lines(Path(f)):
            t = parse_ts(r.get("ts"))
            if t is None or (since is not None and t <= since) or t > until:
                continue
            out.append(
                [
                    r.get("ts"),
                    r.get("gpu_util"),
                    r.get("vram_gb"),
                    r.get("cpu"),
                    r.get("ram_free_gb"),
                ]
            )
    return out


def pctl(vals: list[float], q: float):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def block(rows: list[dict]) -> dict:
    v = Counter(r.get("verdict") for r in rows)
    think = [r["think_tokens"] for r in rows if r.get("think_tokens") is not None]
    checked = v["pass"] + v["fail"]
    return {
        "n": len(rows),
        "pass": v["pass"],
        "fail": v["fail"],
        "unverified": v["unverified"],
        "pass_rate": v["pass"] / checked if checked else None,
        "think_p50": pctl(think, 0.5),
        "think_p90": pctl(think, 0.9),
        "think_mean": statistics.fmean(think) if think else None,
        "secs_p50": pctl([r["secs"] for r in rows if r.get("secs") is not None], 0.5),
        "truncated": sum(1 for r in rows if r.get("finish") == "length"),
    }


def summary(results: list[dict], plan: dict) -> dict:
    by_set, by_level, by_sl = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in results:
        by_set[r["origin"]].append(r)
        by_level[r["level"]].append(r)
        by_sl[f"{r['origin']}|{r['level']}"].append(r)
    hist = {}
    for lv, rows in by_level.items():
        counts = [0] * len(EDGES)
        for r in rows:
            t = r.get("think_tokens") or 0
            i = max(j for j, e in enumerate(EDGES) if t >= e)
            counts[i] += 1
        hist[lv] = {"edges": EDGES, "counts": counts}
    # pass@k per level: questions with a passing answer among their first k tries (checkable ones)
    pass_at = {}
    for lv in plan:
        per_q = defaultdict(dict)
        for r in by_level.get(lv, []):
            per_q[r["qid"]][r["try"]] = r.get("verdict")
        ks = {}
        for k in range(1, plan[lv] + 1):
            qs = [t for t in per_q.values() if all(i in t for i in range(k))]
            qs = [t for t in qs if any(t[i] in ("pass", "fail") for i in range(k))]
            if qs:
                ks[str(k)] = sum(1 for t in qs if any(t[i] == "pass" for i in range(k))) / len(qs)
        pass_at[lv] = ks
    q_any = defaultdict(set)
    q_pass = defaultdict(set)
    for r in results:
        q_any[r["origin"]].add(r["qid"])
        if r.get("verdict") == "pass":
            q_pass[r["origin"]].add(r["qid"])
    return {
        "overall": block(results),
        "by_set": {k: block(v) for k, v in sorted(by_set.items())},
        "by_level": {k: block(v) for k, v in by_level.items()},
        "by_set_level": {k: block(v) for k, v in sorted(by_sl.items())},
        "questions": {
            k: {"seen": len(q_any[k]), "with_pass": len(q_pass[k])} for k in sorted(q_any)
        },
        "reasons": dict(
            Counter(
                r.get("reason") or "" for r in results if r.get("verdict") != "pass"
            ).most_common(12)
        ),
        "checked_by": dict(
            Counter(r.get("checked_by") or "none" for r in results if r.get("verdict") == "pass")
        ),
        "hist": hist,
        "pass_at": pass_at,
        "think_tokens": sum(r.get("think_tokens") or 0 for r in results),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in results),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu-dir", default=r"F:\nfl-ask-trainer-ckpt\_monitor")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    cfg = read_json(run / "config.json") or {}
    state = read_json(out / "state.json") or {
        "seq": 0,
        "results_sent": 0,
        "progress_sent": 0,
        "events_sent": 0,
        "gpu_last": None,
        "last_tick_at": None,
        "run_doc": False,
    }
    results = read_lines(run / "results.jsonl")
    progress = read_lines(run / "progress.jsonl")
    events = read_lines(run / "events.jsonl")
    now = dt.datetime.now(dt.UTC)
    started = next((parse_ts(e["ts"]) for e in events if e.get("kind") == "start"), None)
    total = cfg.get("requests") or 0
    done = len(results)
    kinds = {e.get("kind") for e in events}
    last_res = max((parse_ts(r.get("ts")) for r in results), default=None, key=lambda t: t or now)
    if "DONE" in kinds:
        st = "done"
    elif kinds & {"stopped", "aborted"}:
        st = "stopped"
    elif started is None:
        st = "waiting"
    elif not alive(run / "tracegen.pid"):
        st = "stopped"
    elif last_res and (now - last_res).total_seconds() > STALL_S:
        st = "stalled"
    else:
        st = "running"
    # recent speed from the progress totals (last ~2 minutes) and the ETA from the results' pace
    tok_s = req_min = eta = None
    pr = [(parse_ts(p["ts"]), p) for p in progress if parse_ts(p.get("ts"))]
    if len(pr) >= 2:
        t1, p1 = pr[-1]
        t0, p0 = next(((t, p) for t, p in reversed(pr) if (t1 - t).total_seconds() >= 120), pr[0])
        span = (t1 - t0).total_seconds()
        if span > 0:
            tok_s = (p1["completion_tokens"] - p0["completion_tokens"]) / span
            req_min = (p1["done"] - p0["done"]) / span * 60
            if req_min and st == "running":
                eta = (total - done) / (req_min / 60)
    status = {
        "state": st,
        "updated_at": now_iso(),
        "done": done,
        "total": total,
        "errors": sum(1 for e in events if e.get("kind") == "request_failed"),
        "started_at": started.isoformat() if started else None,
        "last_result_at": last_res.isoformat() if last_res else None,
        "gen_tok_s_recent": tok_s,
        "answers_per_min": req_min,
        "eta_s": eta,
    }
    new_progress = [
        [
            p["ts"],
            p.get("done"),
            p.get("completion_tokens"),
            p.get("prompt_tokens"),
            p.get("errors"),
        ]
        for p in progress[state["progress_sent"] :]
    ]
    new_events = events[state["events_sent"] :]
    since = parse_ts(state["gpu_last"]) or started
    gpu = gpu_samples(Path(a.gpu_dir), since, now) if started else []
    new_results = len(results) - state["results_sent"]
    last_tick = parse_ts(state["last_tick_at"])
    quiet = last_tick is None or (now - last_tick).total_seconds() >= HEARTBEAT_S
    if not (new_results or new_progress or new_events or gpu or quiet or a.force):
        print(json.dumps({"tick_file": None, "reason": "nothing new", "status": st}))
        return
    run_doc = None
    if not state["run_doc"]:
        run_doc = out / "run.json"
        doc = {
            "run": run.name,
            "kind": "trace",
            "created_at": now_iso(),
            "started_at": started.isoformat() if started else cfg.get("created_at"),
            "info": {
                "objective": "traces",
                "plan": cfg.get("plan"),
                "sets": cfg.get("sets"),
                "questions": cfg.get("questions"),
                "requests": total,
                "sampling": cfg.get("sampling"),
                "seed": cfg.get("seed"),
                "server": next((e.get("url") for e in events if e.get("kind") == "start"), None),
                "workers": next(
                    (e.get("workers") for e in events if e.get("kind") == "start"), None
                ),
            },
        }
        run_doc.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    samples = [
        {
            "set": r["origin"],
            "level": r["level"],
            "question": cap(r.get("question_text") or "", 300),
            "qid": r["qid"],
            "think": cap(r.get("reasoning"), 600),
            "think_tokens": r.get("think_tokens"),
            "query": cap(r.get("query") or r.get("content") or "", 700),
            "verdict": r.get("verdict"),
            "reason": r.get("reason"),
            "checked_by": r.get("checked_by"),
            "secs": r.get("secs"),
        }
        for r in results[-10:]
    ]
    questions = {q["qid"]: q for q in read_lines(run / "questions.jsonl")} if samples else {}
    for s in samples:
        q = questions.get(s["qid"]) or {}
        s["question"] = cap(q.get("user"), 300)
        s["gold"] = cap(q.get("gold"), 700)
    seq = state["seq"] + 1
    tick = {
        "seq": seq,
        "made_at": now_iso(),
        "kind": "trace",
        "status": status,
        "progress_cols": ["ts", "done", "completion_tokens", "prompt_tokens", "errors"],
        "progress": new_progress,
        "summary": summary(results, cfg.get("plan") or {}),
        "samples": samples,
        "events": new_events,
        "gpu_cols": ["ts", "gpu_util", "vram_gb", "cpu", "ram_free_gb"],
        "gpu": gpu,
    }
    tick_file = out / f"tick-{seq:05d}.json"
    tick_file.write_text(json.dumps(tick, ensure_ascii=False), encoding="utf-8")
    state.update(
        seq=seq,
        results_sent=len(results),
        progress_sent=len(progress),
        events_sent=len(events),
        gpu_last=gpu[-1][0] if gpu else state["gpu_last"],
        last_tick_at=tick["made_at"],
        run_doc=True,
    )
    (out / "state.json").write_text(json.dumps(state), encoding="utf-8")
    print(
        json.dumps(
            {
                "tick_file": str(tick_file),
                "seq": seq,
                "run_doc": str(run_doc) if run_doc else None,
                "status": st,
                "done": done,
                "total": total,
                "new_results": new_results,
                "gpu": len(gpu),
                "bytes": os.path.getsize(tick_file),
                "pass_rate": tick["summary"]["overall"]["pass_rate"],
            }
        )
    )


if __name__ == "__main__":
    main()
