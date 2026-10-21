"""Turn a training run's real files into the live dashboard's next update ("tick").

    python scripts/dashboard_feed.py --run /mnt/d/nfl-ml-data/ask/finetune/<run> [--wandb]

Reads only what the run writes (nothing is computed that the run didn't log, except an ETA
and a smoothed loss, both labelled as such on the page):
- `run_info.json`, `config.json`, `environment.json`, `mix_manifest.json`: who the run is;
- `train_log.jsonl`: every logged training step and evaluation (the trainer's own numbers,
  the same values it sends to W&B);
- `events.jsonl`: model load, checkpoints, evaluations, resumes, OOM splits, crashes;
- `samples_<step>.jsonl` (or, for older runs, `preds_epoch<N>.jsonl` + `scores_epoch<N>.json`):
  example generations with their scores;
- the GPU monitor's samples (`scripts/gpu_monitor.ps1`, Windows performance counters);
- with `--wandb`, the run's W&B history, read back to check it matches the local log.

Writes `<run>/dashboard/tick-NNNNN.json` (only what's new since the last tick, plus a status
snapshot) and `<run>/dashboard/run.json` (the run's identity, once), and prints one JSON line
saying what it wrote. State lives in `<run>/dashboard/state.json`. Never writes anything else.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import statistics
import subprocess
from pathlib import Path

TRAIN_COLS = [
    "step", "ts", "ce", "loss", "jepa", "lr", "grad_norm", "tok_s", "peak_gb", "tokens_seen",
    "target_tokens", "epoch",
]  # fmt: skip
TRAIN_KEYS = {
    "ce": "train/ce", "loss": "train/loss", "jepa": "train/jepa", "lr": "train/lr",
    "grad_norm": "train/grad_norm", "tok_s": "train/tokens_per_s", "peak_gb": "train/peak_gb",
    "tokens_seen": "train/tokens_seen", "target_tokens": "train/target_tokens",
}  # fmt: skip
HEARTBEAT_S = 900  # a status-only tick at least this often while nothing new arrives
STALL_S = 1200  # no new step for this long while the process lives: "stalled"
TEXT_CAP = 700


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def parse_ts(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_lines(p: Path) -> list[dict]:
    out = []
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip().lstrip("﻿")  # a BOM from PowerShell 5.1
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass  # a line mid-write: picked up next tick
    except OSError:
        pass
    return out


def num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int | float):
        return round(float(v), 6)
    return None


def cap(s, n: int = TEXT_CAP):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def process_alive(run_name: str) -> bool:
    """Is an `ask-trainer train` for this run still running (inside WSL)?"""
    try:
        out = subprocess.run(["pgrep", "-af", "ask-trainer train"], capture_output=True,
                             text=True, timeout=10).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return False
    return any(run_name in line for line in out.splitlines())


def train_rows(log: list[dict]) -> tuple[list[list], list[dict]]:
    """(training points as rows of TRAIN_COLS, evaluation rows)."""
    points, evals = [], []
    for r in log:
        if any(k.startswith("val/") for k in r):
            evals.append(r)
        elif "train/ce" in r or "train/loss" in r:
            row = [r.get("step"), r.get("ts")]
            row += [num(r.get(TRAIN_KEYS[c])) for c in TRAIN_COLS[2:11]]
            row.append(num(r.get("epoch")))
            points.append(row)
    return points, evals


def eval_doc(r: dict) -> dict:
    metrics = {k: num(v) for k, v in r.items() if k.startswith("val/") and num(v) is not None}
    return {"step": r.get("step"), "ts": r.get("ts"), "epoch": num(r.get("epoch")),
            "metrics": metrics}  # fmt: skip


def samples_for(run: Path, step, epoch) -> list[dict]:
    """Up to 12 generations of the evaluation at `step` (the trainer's samples file, or an
    older run's preds + scores files)."""
    f = run / f"samples_{step}.jsonl"
    if f.exists():
        rows = read_lines(f)[:12]
        return [{k: (cap(v) if isinstance(v, str) else v) for k, v in r.items()} for r in rows]
    if epoch is None:
        return []
    preds = run / f"preds_epoch{int(round(epoch))}.jsonl"
    scores = read_json(run / f"scores_epoch{int(round(epoch))}.json") or {}
    if not preds.exists():
        return []
    items = {i.get("id"): i for i in scores.get("items", [])}
    out = []
    for p in read_lines(preds)[:12]:
        it = items.get(p.get("id"), {})
        out.append({
            "id": p.get("id"), "group": it.get("group"), "task": it.get("task"),
            "output": cap(p.get("output")), "exec_ok": it.get("exec_ok"),
            "json_ok": it.get("json_ok"), "reason": cap(it.get("reason"), 80),
        })  # fmt: skip
    return out


def gpu_samples(gpu_dir: Path | None, since: dt.datetime | None, until: dt.datetime) -> list:
    if gpu_dir is None or not gpu_dir.exists():
        return []
    out = []
    for f in sorted(glob.glob(str(gpu_dir / "gpu-*.jsonl"))):
        for r in read_lines(Path(f)):
            t = parse_ts(r.get("ts"))
            if t is None or (since is not None and t <= since) or t > until:
                continue
            out.append([r.get("ts"), num(r.get("gpu_util")), num(r.get("vram_gb")),
                        num(r.get("cpu")), num(r.get("ram_free_gb"))])  # fmt: skip
    return out


def local_wandb_path(info: dict) -> str | None:
    """entity/project/id from the run's own `wandb/run-<date>-<id>` folder (runs made before
    run_info.json existed) and configs/paths.yaml."""
    run_dir = info.get("_run_dir")
    if not run_dir:
        return None
    dirs = sorted(Path(run_dir).glob("wandb/run-*-*"))
    if not dirs:
        return None
    run_id = dirs[-1].name.rsplit("-", 1)[-1]
    try:
        import yaml

        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "paths.yaml")
                             .read_text(encoding="utf-8"))  # fmt: skip
        wb = cfg.get("wandb") or {}
        return f"{wb['entity']}/{wb['project']}/{run_id}"
    except (OSError, KeyError, TypeError, ValueError):
        return None


def wandb_check(info: dict, points: list[list]) -> dict:
    """Compare W&B's stored train/ce with the local log at the same steps (read-only)."""
    res = {"checked_at": now_iso(), "ok": None, "compared": 0, "max_abs_diff": None,
           "missing_in_wandb": 0, "error": None}  # fmt: skip
    path = info.get("wandb_path") or "/".join(
        str(info.get(k) or "") for k in ("wandb_entity", "wandb_project", "wandb_id")
    )
    if path.count("/") != 2 or path.endswith("/") or path.startswith("/"):
        path = local_wandb_path(info) or path
    if not path or path.count("/") != 2 or path.endswith("/"):
        res["error"] = "no W&B run id in run_info.json"
        return res
    try:
        import wandb

        run = wandb.Api(timeout=30).run(path)
        remote = {}
        for row in run.scan_history(keys=["_step", "train/ce"], page_size=1000):
            if row.get("train/ce") is not None:
                remote[int(row["_step"])] = float(row["train/ce"])
    except Exception as e:  # noqa: BLE001 - the type only, never the text
        res["error"] = type(e).__name__
        return res
    local = {int(p[0]): p[2] for p in points if p[0] is not None and p[2] is not None}
    diffs, missing = [], 0
    for step, v in local.items():
        if step in remote:
            diffs.append(abs(remote[step] - v))
        else:
            missing += 1
    res.update(compared=len(diffs), missing_in_wandb=missing,
               max_abs_diff=max(diffs) if diffs else None,
               ok=bool(diffs) and max(diffs) < 1e-4)  # fmt: skip
    return res


def status_of(run: Path, info: dict, points: list[list], evals: list[dict], events: list[dict],
              alive: bool) -> dict:  # fmt: skip
    now = dt.datetime.now(dt.UTC)
    last = points[-1] if points else None
    last_t = parse_ts(last[1]) if last else None
    kinds = [e.get("kind") for e in events]
    if (run / "DONE").exists() or "done" in kinds:
        state = "done"
    elif "crashed" in kinds[-1:] or (not alive and not (run / "DONE").exists() and points):
        state = "stopped"
    elif alive and last_t is not None and (now - last_t).total_seconds() > STALL_S:
        state = "stalled"
    elif alive:
        state = "running"
    else:
        state = "waiting"
    planned = info.get("planned_steps")
    step = last[0] if last else 0
    # ETA from the real step times (median of the last 20 gaps) + evaluations still due
    eta = None
    times = [parse_ts(p[1]) for p in points[-21:] if parse_ts(p[1]) is not None]
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:], strict=False) if b > a]
    if planned and gaps and state in ("running", "stalled") and step:
        eta = statistics.median(gaps) * max(0, planned - step)
        eval_secs = [e.get("seconds") for e in events if e.get("kind") == "eval_end"]
        every = info.get("eval_every_steps")
        if eval_secs and every:
            eta += statistics.median([s for s in eval_secs if s]) * max(0, planned - step) / every
    ckpts = [e for e in events if e.get("kind") == "checkpoint_saved"]
    tok_recent = [p[7] for p in points[-10:] if p[7] is not None]
    return {
        "state": state, "alive": alive, "step": step, "planned_steps": planned,
        "epoch": last[11] if last else None, "tokens_seen": last[9] if last else None,
        "planned_tokens": info.get("planned_tokens"), "last_step_at": last[1] if last else None,
        "tok_s_recent": round(statistics.median(tok_recent), 1) if tok_recent else None,
        "peak_gb": max((p[8] for p in points if p[8] is not None), default=None),
        "ce_last": last[2] if last else None, "lr_last": last[5] if last else None,
        "last_checkpoint": ckpts[-1] if ckpts else None,
        "last_eval_step": evals[-1].get("step") if evals else None,
        "eta_s": round(eta) if eta is not None else None, "updated_at": now_iso(),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu-dir", default="/mnt/f/nfl-ask-trainer-ckpt/_monitor")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--force", action="store_true", help="write a tick even with nothing new")
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "dashboard"
    out.mkdir(exist_ok=True)
    state = read_json(out / "state.json") or {
        "seq": 0, "steps_sent": [], "evals_sent": [], "events_sent": 0, "gpu_last": None,
        "last_tick_at": None, "run_doc": False,
    }  # fmt: skip
    info = read_json(run / "run_info.json") or {}
    cfg = read_json(run / "config.json") or {}
    if not info:  # an older run without run_info.json: what the config says
        info = {"run": run.name, "objective": "jepa" if (cfg.get("jepa") or {}).get("enabled")
                else "plain", "seed": cfg.get("seed")}  # fmt: skip
    info.setdefault("run", run.name)
    info["_run_dir"] = str(run)
    log = read_lines(run / "train_log.jsonl")
    points, evals = train_rows(log)
    events = read_lines(run / "events.jsonl")
    alive = process_alive(run.name)

    sent_steps = set(state["steps_sent"])
    # a resume cuts train_log back to its checkpoint: re-send what changed (the page keys by step)
    new_points = [p for p in points if p[0] not in sent_steps or p[0] == points[-1][0]]
    sent_evals = set(state["evals_sent"])
    new_evals = []
    for r in evals:
        if r.get("step") in sent_evals:
            continue
        d = eval_doc(r)
        d["samples"] = samples_for(run, r.get("step"), r.get("epoch"))
        new_evals.append(d)
    new_events = events[state["events_sent"] :]
    started = parse_ts(info.get("started_at")) or (parse_ts(points[0][1]) if points else None)
    since = parse_ts(state["gpu_last"]) or started
    # only samples inside the run's own window: none without a known start; up to its end
    ended = None
    if (run / "DONE").exists():
        ended = dt.datetime.fromtimestamp((run / "DONE").stat().st_mtime, dt.UTC)
    until = ended or dt.datetime.now(dt.UTC)
    gpu = gpu_samples(Path(args.gpu_dir), since, until) if started is not None else []
    status = status_of(run, info, points, evals, events, alive)

    last_tick = parse_ts(state["last_tick_at"])
    quiet = last_tick is None or (dt.datetime.now(dt.UTC) - last_tick).total_seconds() > HEARTBEAT_S
    has_new = bool(new_points or new_evals or new_events or gpu)
    wb = wandb_check(info, points) if args.wandb and points else None
    if not (has_new or quiet or args.force or wb):
        print(json.dumps({"tick_file": None, "reason": "nothing new", "status": status["state"]}))
        return

    run_doc = None
    if not state["run_doc"]:
        run_doc = out / "run.json"
        first_ts = points[0][1] if points else None
        doc = {"run": run.name, "created_at": now_iso(),
               "info": {k: v for k, v in info.items() if not k.startswith("_")},
               "config": cfg, "environment": read_json(run / "environment.json") or {},
               "started_at": info.get("started_at") or first_ts}  # fmt: skip
        run_doc.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    seq = state["seq"] + 1
    tick = {
        "seq": seq, "made_at": now_iso(), "status": status, "cols": TRAIN_COLS,
        "points": new_points, "evals": new_evals, "events": new_events,
        "gpu_cols": ["ts", "gpu_util", "vram_gb", "cpu", "ram_free_gb"], "gpu": gpu,
        "wandb_check": wb,
    }  # fmt: skip
    tick_file = out / f"tick-{seq:05d}.json"
    tick_file.write_text(json.dumps(tick, ensure_ascii=False), encoding="utf-8")
    state.update(
        seq=seq, steps_sent=sorted(sent_steps | {p[0] for p in new_points}),
        evals_sent=sorted(sent_evals | {e["step"] for e in new_evals}),
        events_sent=len(events), gpu_last=gpu[-1][0] if gpu else state["gpu_last"],
        last_tick_at=tick["made_at"], run_doc=True,
    )  # fmt: skip
    (out / "state.json").write_text(json.dumps(state), encoding="utf-8")
    print(json.dumps({
        "tick_file": str(tick_file), "seq": seq, "run_doc": str(run_doc) if run_doc else None,
        "new_points": len(new_points), "new_evals": len(new_evals), "new_events": len(new_events),
        "gpu": len(gpu), "status": status["state"], "step": status["step"],
        "bytes": os.path.getsize(tick_file), "wandb": wb,
    }))  # fmt: skip


if __name__ == "__main__":
    main()
