"""Trace-generation throughput under different llama-server settings (Windows side, GPU):

    C:\\Python314\\python.exe scripts\\llamacpp\\bench_trace.py [--run pilot] [--minutes 8]
        [--configs default,np24,np32,ub1024,lean_samplers,upstream] [--n 2000]

Replays the same seeded sample of a trace run's questions (each at its own planned thinking
level, the run's sampling, the real request `tracegen` sends: tools, thinking budget,
`/tokenize`, the inline check) against one server configuration after another: start it
(`serve.ps1`, port 8094), keep `-Parallel` requests in flight for a minute's warm-up plus
`--minutes`, stop it. Counted are the answers that finish inside the measured window.

Why (docs/llamacpp.md, "Trace-generation speed"): the pilot run's round 1 generated 242 tokens
a second with 16 slots, against 796 in `llama-batched-bench`. Windows where all 16 slots decode
reach ~300; windows with a prompt being processed fall to ~150. The configurations try more slots
(so a stalled step carries more tokens), larger micro-batches (fewer prompt passes), a sampler
chain without the stages that do nothing at our settings (same distribution: top-k 40, top-p,
min-p 0.05, temperature), and the clean upstream build.

Results: printed as a table and saved to D:\\nfl-ml-data\\ask\\llamacpp\\bench-trace-<time>.json.
One GPU job at a time: run it only while nothing else uses the GPU (~10 minutes a configuration).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from asktrainer.tracegen import Run  # noqa: E402

DATA = Path(r"D:\nfl-ml-data\ask")
SERVE = REPO / "scripts" / "llamacpp" / "serve.ps1"
PORT = 8094
BUDGET = "--reasoning-budget 1024"
UPSTREAM = r"F:\Personal\TQ-Optimizer-Test\llama-cpp-upstream\build-rdna3-gfx1101\bin"
# name: serve.ps1 arguments (the KV pool grows with the slots: ~36 KB a token at f16)
CONFIGS: dict[str, dict[str, Any]] = {
    "default": {"Parallel": 16, "Ctx": 65536, "Extra": BUDGET},
    "np24": {"Parallel": 24, "Ctx": 98304, "Extra": BUDGET},
    "np32": {"Parallel": 32, "Ctx": 131072, "Extra": BUDGET},
    "ub1024": {"Parallel": 16, "Ctx": 65536, "Extra": BUDGET + " -b 2048 -ub 1024"},
    "lean_samplers": {
        "Parallel": 16,
        "Ctx": 65536,
        "Extra": BUDGET + " --samplers top_k;top_p;min_p;temperature",
    },
    "upstream": {"Parallel": 16, "Ctx": 65536, "Extra": BUDGET, "Bin": UPSTREAM},
}


def sample(run: Run, n: int, seed: int = 0) -> list[tuple[str, dict[str, Any], str, int]]:
    """`n` (question, level) requests, the same for every configuration: questions drawn at
    random, each at the first level of its plan (the adaptive plan gives each one level)."""
    qs = list(run.questions)
    random.Random(seed).shuffle(qs)
    out = []
    for q in qs[:n]:
        level = next(iter(run.plan_of(q)))
        out.append((f"{q['qid']}#{level}#bench", q, level, 0))
    return out


def summarize(done: list[dict[str, Any]], t0: float, t1: float, slots: int) -> dict[str, Any]:
    """Throughput over the answers that finished in [t0, t1] (`done` rows: end, secs, tokens)."""
    rows = [r for r in done if t0 <= r["end"] <= t1 and not r.get("error")]
    span = max(t1 - t0, 1e-9)
    secs = sorted(r["secs"] for r in rows)
    return {
        "slots": slots,
        "answers": len(rows),
        "answers_per_min": round(len(rows) / span * 60, 1),
        "gen_tok_s": round(sum(r["completion_tokens"] for r in rows) / span, 1),
        "prompt_tok_s": round(sum(r["prompt_tokens"] for r in rows) / span, 1),
        "secs_p50": secs[len(secs) // 2] if secs else None,
        "per_slot_gen_tok_s": round(sum(r["completion_tokens"] for r in rows) / span / slots, 1),
        "errors": sum(1 for r in done if r.get("error")),
    }


def serve(spec: dict[str, Any], stop: bool = False) -> str:
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SERVE),
            "-Port", str(PORT)]  # fmt: skip
    if stop:
        args += ["-Stop"]
    else:
        args += ["-Parallel", str(spec["Parallel"]), "-Ctx", str(spec["Ctx"]),
                 "-Extra", spec["Extra"]]  # fmt: skip
        if spec.get("Bin"):
            args += ["-Bin", spec["Bin"]]
    log = DATA / "llamacpp" / f"bench-serve-{PORT}.log"  # a file: llama-server inherits it
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return log.read_text(encoding="utf-8", errors="replace").strip()


def load(run: Run, todo: list, slots: int, warmup: float, measure: float) -> dict[str, Any]:
    """Keep `slots` requests in flight until the window ends, then summarize it."""
    it, lock, done = iter(todo), threading.Lock(), []
    start = time.time()
    t0, t1 = start + warmup, start + warmup + measure

    def worker() -> None:
        while time.time() < t1:
            with lock:
                req = next(it, None)
            if req is None:
                return
            try:
                r = run.one(*req)
                row = {"secs": r["secs"], "completion_tokens": r.get("completion_tokens") or 0,
                       "prompt_tokens": r.get("prompt_tokens") or 0}  # fmt: skip
            except Exception as e:  # the server: count it, keep going
                row = {"secs": 0, "completion_tokens": 0, "prompt_tokens": 0, "error": str(e)}
            row["end"] = time.time()
            with lock:
                done.append(row)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=max(t1 - time.time(), 0) + 600)  # the last answers may run on
    return summarize(done, t0, t1, slots)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="pilot", help="the trace run whose questions are replayed")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--warmup", type=float, default=60.0, help="seconds")
    ap.add_argument("--n", type=int, default=2000, help="requests sampled (more than needed)")
    a = ap.parse_args()
    run = Run(DATA / "traces" / a.run, f"http://127.0.0.1:{PORT}")
    todo = sample(run, a.n)
    out: dict[str, Any] = {"run": a.run, "minutes": a.minutes, "results": {}}
    path = DATA / "llamacpp" / f"bench-trace-{time.strftime('%Y%m%d-%H%M%S')}.json"
    for name in a.configs.split(","):
        spec = CONFIGS[name]
        msg = serve(spec)
        if "ready" not in msg:
            out["results"][name] = {"error": msg[:300]}
            continue
        try:
            out["results"][name] = {**load(run, todo, spec["Parallel"], a.warmup,
                                           a.minutes * 60), "spec": spec}  # fmt: skip
        finally:
            serve(spec, stop=True)
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        r = out["results"][name]
        print(f"{name:14s} {r['answers_per_min']:6.1f}/min  gen {r['gen_tok_s']:6.1f} tok/s  "
              f"prompt {r['prompt_tok_s']:6.1f} tok/s  per slot {r['per_slot_gen_tok_s']:5.1f}",
              flush=True)  # fmt: skip
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
