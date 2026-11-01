"""Score the release GGUFs on the evalset the way the run's adapters were scored (docs/release.md):

    C:\\Python314\\python.exe scripts\\release_eval.py [--only Q8_0,Q4_K_M] [--testset]

Each `<name>-<variant>.gguf` in the release folder is served by llama-server (`serve.ps1 -Model`,
port 8092, thinking budget 1,024), answers the 200 evalset questions greedily
(`llamaeval.generate`) and is checked with checker 4 (and 3 beside it) by `ask-trainer evalrun`,
logged into the run's W&B run as `release-<variant>` at the adapter's step. Answers and scores go
to `<data>/release/eval/`, never into the folder that is uploaded. Needs the trainer's Neo4j
(`asktrainer-neo4j`) and nothing else on the GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import pilot_sweep as ps  # noqa: E402  (its WSL / server helpers' settings)
import real_run as rr  # noqa: E402  (the evalset and test set paths)

from asktrainer import llamaeval  # noqa: E402

NAME = "Spark-X2.6-4B-SQL-Cypher-Agent"
RUN, STEP = "sft-main", 1500  # the adapter the release was merged from
RELEASE = ps.DATA / "release"


def serve(model: Path | None, log: Path) -> None:
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps.SERVE),
            "-Port", str(ps.EVAL_PORT)]  # fmt: skip
    if model is None:
        args += ["-Stop"]
    else:
        args += ["-Model", str(model), "-Extra", f"--reasoning-budget {ps.BUDGET}"]
    with log.open("w", encoding="utf-8") as f:  # a file, not a pipe (see Sweep.serve)
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    out = log.read_text(encoding="utf-8", errors="replace")
    if model is not None and "ready" not in out:
        raise RuntimeError(f"llama-server did not start: {out[:300]}")


def evalrun(preds: Path, tag: str, examples: Path, checker: int, wandb: bool) -> dict:
    cmd = (f"ask-trainer evalrun --run {RUN} --preds {ps._wsl(preds)} --tag {tag} --step {STEP} "
           f"--checker {checker} --examples {ps._wsl(examples)}"
           + ("" if wandb else " --no-wandb"))  # fmt: skip
    full = f"cd {ps.WSL_REPO} && source ~/venvs/nfl-ask-trainer/bin/activate && {cmd}"
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", full], capture_output=True,
                       **ps.TEXT)  # fmt: skip
    if r.returncode != 0:
        raise RuntimeError(f"scoring {tag} (checker {checker}) failed: {r.stderr[-600:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated variants (default: every GGUF)")
    ap.add_argument("--testset", action="store_true", help="the 1,000-question test set instead")
    ap.add_argument("--gguf", default="", help="score this GGUF instead (e.g. the base's Q8_0)")
    ap.add_argument("--label", default="", help="its name in the results (with --gguf)")
    args = ap.parse_args()
    gguf_dir = RELEASE / f"{NAME}-GGUF"
    out = RELEASE / "eval"
    out.mkdir(parents=True, exist_ok=True)
    examples, kind = (rr.TESTSET, "test") if args.testset else (rr.EVALSET, "eval")
    results_path = out / f"{kind}.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    only = {v for v in args.only.split(",") if v}
    if args.gguf:
        models = [(Path(args.gguf), args.label or Path(args.gguf).stem)]
    else:
        models = [(g, g.stem.removeprefix(f"{NAME}-"))
                  for g in sorted(gguf_dir.glob(f"{NAME}-*.gguf"))]  # fmt: skip
    for gguf, variant in models:
        if (only and variant not in only) or variant in results:
            continue
        tag = f"release-{kind}-{variant}"
        preds = out / f"preds_{tag}.jsonl"
        print(f"{variant}: answering ({gguf.stat().st_size / 1e9:.2f} GB)", flush=True)
        serve(gguf, out / "serve.log")
        try:
            gen = llamaeval.generate(examples, preds, f"http://127.0.0.1:{ps.EVAL_PORT}")
        finally:
            serve(None, out / "serve-stop.log")
        r4 = evalrun(preds, tag, examples, rr.CHECKER, wandb=True)
        r3 = evalrun(preds, f"{tag}-checker{rr.ALSO}", examples, rr.ALSO, wandb=False)
        results[variant] = {"generate": gen, f"checker{rr.CHECKER}": r4, f"checker{rr.ALSO}": r3}
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"{variant}: {r4.get('val/public/exec_acc')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
