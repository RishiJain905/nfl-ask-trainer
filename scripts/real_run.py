"""The real run, end to end, from Windows (docs/real-run.md):

    C:\\Python314\\python.exe scripts\\real_run.py [--config configs/general_full.yaml]

1. trains (`ask-trainer train`, WSL; `--resume` when the run folder is there unfinished) under a
   watchdog: no new line in the run's `train_log.jsonl` / `events.jsonl` for 15 minutes while
   the trainer lives (pilot P9 stalled at step 173 with the GPU idle) stops it and resumes from
   the last checkpoint, up to 6 times, and only while each resume makes progress;
2. stops on request, so the PC can be switched off: with `D:\\nfl-ml-data\\ask\\finetune\\PAUSE`
   present, the trainer itself stops right after its next checkpoint (nothing lost: the
   checkpoint holds the whole state), writes `<run>/PAUSED` and exits, and the driver exits
   with `phase: paused`. Delete the file and start the driver again to go on (it won't start
   while the file is there);
3. after `DONE`, scores every saved adapter (`adapter_stepNNNNNN`, the final `adapter`) on the
   evalset through llama.cpp (the pilots' way: LoRA to GGUF, `serve.ps1` on port 8092,
   `llamaeval generate`, `ask-trainer evalrun` with checker 4 into the run's W&B run, checker 3
   beside it), then the test set (1,000 questions, never used for a decision) once, on the final
   adapter and on the best evalset adapter when that is another one.

State: <data root>/ask/main/state.json (scores, resumes) and status.json (what runs now, for the
watcher); driver.pid. One GPU job at a time.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import pilot_sweep as ps  # noqa: E402  (its WSL / server / LoRA-conversion helpers)

from asktrainer import llamaeval  # noqa: E402

DATA = ps.DATA
MAIN = DATA / "main"
PAUSE = DATA / "finetune" / "PAUSE"
STALL_S = 15 * 60  # no progress while training
START_S = 90 * 60  # before this launch's first step: tokenizing ~95k examples, loading
MAX_RESUMES = 6
CHECKER, ALSO = 4, 3  # into W&B / beside it (the pilots' official scores were version 3)
TEST_TAG = "test"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def last_step(run_dir: Path) -> int:
    p = run_dir / "train_log.jsonl"
    if not p.exists():
        return 0
    step = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            step = max(step, json.loads(line).get("step") or 0)
        except json.JSONDecodeError:
            pass  # a line being written
    return step


def progress_mtime(run_dir: Path) -> float:
    """The latest write to the run's step log or event log (0 if neither exists yet)."""
    ts = [p.stat().st_mtime for p in (run_dir / "train_log.jsonl", run_dir / "events.jsonl")
          if p.exists()]  # fmt: skip
    return max(ts, default=0.0)


def stalled(now_s: float, launched: float, last_progress: float, armed: bool) -> bool:
    """Armed (a step logged since this launch): no progress for STALL_S. Before that: START_S
    since the launch (the trainer tokenizes and loads before its first step)."""
    if armed:
        return now_s - max(last_progress, launched) >= STALL_S
    return now_s - launched >= START_S


def adapter_tags(run_dir: Path) -> list[tuple[str, str, int]]:
    """(folder, tag, step) of the adapters to score: the step snapshots, then the final one."""
    out = []
    for d in sorted(run_dir.glob("adapter_step*")):
        m = re.fullmatch(r"adapter_step(\d{6})", d.name)
        if d.is_dir() and m:
            out.append((d.name, f"step{m.group(1)}", int(m.group(1))))
    if (run_dir / "adapter").is_dir():
        out.append(("adapter", "final", last_step(run_dir)))
    return out


class Main(ps.Sweep):
    """The pilot driver's helpers (`wsl`, `serve`, `convert`) with the real run's own state."""

    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path
        self.name = _yaml_name(cfg_path)
        self.dir = MAIN
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = DATA / "finetune" / self.name
        self.state_path = self.dir / "state.json"
        self.state = (json.loads(self.state_path.read_text(encoding="utf-8"))
                      if self.state_path.exists()
                      else {"run": self.name, "resumes": [], "scores": {}, "log": []})  # fmt: skip

    # -- training under the watchdog ------------------------------------------------------
    def kill_trainer(self) -> None:
        pat = f"ask-trainer train.*{self.name}"
        self.wsl(f"pkill -TERM -f '{pat}'; sleep 15; pkill -KILL -f '{pat}'; true")

    def train(self) -> str:
        """'done' | 'paused' | raises when the run can't go on."""
        log = self.dir / f"{self.name}.train.log"
        while not (self.run_dir / "DONE").exists():
            if PAUSE.exists():  # a stop was asked for: nothing starts until the file is gone
                note = f"{PAUSE} exists: delete it and start the driver again"
                self.status(phase="paused", step=last_step(self.run_dir), note=note)
                return "paused"
            resume = (self.run_dir / "config.json").exists()
            if resume and len(self.state["resumes"]) >= MAX_RESUMES:
                raise RuntimeError(f"{MAX_RESUMES} resumes already: see {log}")
            cmd = (f"ask-trainer train --resume {self.name} --launched-by agent" if resume else
                   f"ask-trainer train --config {ps._wsl(self.cfg_path)} --name {self.name} "
                   "--launched-by agent")  # fmt: skip
            start_step, launched = last_step(self.run_dir), time.time()
            self.status(phase="training", note="resumed" if resume else "started",
                        step=start_step)  # fmt: skip
            full = f"cd {ps.WSL_REPO} && source ~/venvs/nfl-ask-trainer/bin/activate && {cmd}"
            with log.open("a", encoding="utf-8") as f:
                proc = subprocess.Popen(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", full],
                                        stdout=f, stderr=subprocess.STDOUT)  # fmt: skip
            why = None
            while proc.poll() is None:
                time.sleep(30)
                armed = last_step(self.run_dir) > start_step
                if stalled(time.time(), launched, progress_mtime(self.run_dir), armed):
                    why = "stalled"
                    self.kill_trainer()
                    break
            proc.wait()
            step = last_step(self.run_dir)
            if (self.run_dir / "DONE").exists():
                break
            if (self.run_dir / "PAUSED").exists():  # the trainer stopped at a checkpoint
                at = (self.run_dir / "PAUSED").read_text(encoding="utf-8").strip()
                note = (f"stopped cleanly after the checkpoint at step {at}: safe to switch "
                        f"off; to go on, delete {PAUSE} and start the driver")  # fmt: skip
                self.status(phase="paused", step=step, note=note)
                return "paused"
            if step <= start_step and resume:
                raise RuntimeError(f"a resume from step {start_step} made no progress "
                                   f"({why or f'exit {proc.returncode}'}): see {log}")  # fmt: skip
            self.state["resumes"].append({"ts": now(), "step": step, "why": why or
                                          f"exit {proc.returncode}"})  # fmt: skip
            self.save()
            self.status(phase="training", note=f"{why or 'exited'} at step {step}: resuming",
                        step=step)  # fmt: skip
        return "done"

    # -- scoring -----------------------------------------------------------------------
    def score(self, folder: str, tag: str, step: int, examples: Path) -> dict:
        gguf = self.convert(self.name, folder)
        preds = self.run_dir / f"preds_llamacpp_{tag}.jsonl"
        self.serve(ps.EVAL_PORT, f"--reasoning-budget {ps.BUDGET} --lora {gguf}")
        try:
            gen = llamaeval.generate(examples, preds, f"http://127.0.0.1:{ps.EVAL_PORT}")
        finally:
            self.serve(ps.EVAL_PORT, stop=True)
        out = {"generate": gen}
        ex = f"--examples {ps._wsl(examples)}"
        runs = ((CHECKER, "", tag), (ALSO, " --no-wandb", f"{tag}-checker{ALSO}"))
        for checker, extra, t in runs:
            r = self.wsl(f"ask-trainer evalrun --run {self.name} --preds {ps._wsl(preds)} "
                         f"--tag {t} --step {step} --checker {checker} {ex}{extra}")  # fmt: skip
            if r.returncode != 0:
                raise RuntimeError(f"scoring {tag} (checker {checker}) failed: {r.stderr[-600:]}")
            out[f"checker{checker}"] = json.loads(r.stdout.strip().splitlines()[-1])
        return out

    def score_all(self) -> None:
        evalset = DATA / "training" / "public" / "evalset" / "examples_val.jsonl"
        testset = DATA / "training" / "public" / "testset" / "examples_val.jsonl"
        scores = self.state["scores"]
        # the Cypher answers are checked on the trainer's Neo4j (a no-op when it runs already)
        subprocess.run(["docker", "start", "asktrainer-neo4j"], capture_output=True, **ps.TEXT)
        for folder, tag, step in adapter_tags(self.run_dir):
            if tag not in scores:
                self.status(phase="scoring", note=f"evalset: {tag}")
                scores[tag] = self.score(folder, tag, step, evalset)
                self.save()
        acc = {t: s[f"checker{CHECKER}"].get("val/public/exec_acc") or 0.0
               for t, s in scores.items() if not t.startswith(TEST_TAG)}  # fmt: skip
        best = max(acc, key=acc.get)
        for tag in dict.fromkeys(["final", best]):
            t = f"{TEST_TAG}-{tag}"
            if t not in scores:
                folder = next(f for f, tg, _ in adapter_tags(self.run_dir) if tg == tag)
                step = next(s for _, tg, s in adapter_tags(self.run_dir) if tg == tag)
                self.status(phase="testing", note=f"testset: {tag}")
                scores[t] = self.score(folder, t, step, testset)
                self.save()
        self.state["best_on_evalset"] = best

    def run(self) -> int:
        if self.train() == "paused":
            return 0
        self.score_all()
        self.status(phase="done", best=self.state.get("best_on_evalset"),
                    summary={t: s[f"checker{CHECKER}"].get("val/public/exec_acc")
                             for t, s in self.state["scores"].items()})  # fmt: skip
        return 0


def _yaml_name(p: Path) -> str:
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"name:\s*(\S+)", line)
        if m:
            return m.group(1)
    raise ValueError(f"{p}: no `name:`")


def main() -> int:
    ps.keep_awake()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(REPO / "configs" / "general_full.yaml"))
    a = ap.parse_args()
    m = Main(Path(a.config))
    import os

    (m.dir / "driver.pid").write_text(str(os.getpid()), encoding="ascii")
    try:
        return m.run()
    except Exception as e:
        m.status(phase="failed", error=f"{type(e).__name__}: {str(e)[:600]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
