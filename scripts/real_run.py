"""The real run, end to end, from Windows (docs/real-run.md):

    C:\\Python314\\python.exe scripts\\real_run.py [--config configs/general_full.yaml]

1. trains (`ask-trainer train`, WSL; `--resume` when the run folder is there unfinished) under a
   watchdog: no new line in the run's `train_log.jsonl` / `events.jsonl` for 15 minutes while
   the trainer lives (pilot P9 stalled at step 173 with the GPU idle) stops it and resumes from
   the last checkpoint, and so does a crash (a segfault in the GPU runtime at step 809), as long
   as each resume makes progress and the crashes don't keep coming back before the next
   checkpoint;
2. stops on request, so the PC can be switched off: with `D:\\nfl-ml-data\\ask\\finetune\\PAUSE`
   present, the trainer itself stops right after its next checkpoint (nothing lost: the
   checkpoint holds the whole state), writes `<run>/PAUSED` and exits, and the driver exits
   with `phase: paused`. Delete the file and start the driver again to go on (it won't start
   while the file is there);
3. scores each step adapter (`adapter_stepNNNNNN`, every 500 steps) on the evalset while the run
   goes on (Rishi, 2026-10-26: an accuracy that falls while the validation loss still falls
   should show at that step, not at the end): once the trainer is past the checkpoint before an
   adapter step, the driver writes `PAUSE` with the text `scoring`, so the trainer stops right
   after saving that adapter (the same exact stop as above); the driver scores it through
   llama.cpp (the pilots' way: LoRA to GGUF, `serve.ps1` on port 8092, `llamaeval generate`,
   `ask-trainer evalrun` with checker 4, checker 3 beside it), removes its `PAUSE` and resumes.
   A stop asked for meanwhile (any other text in `PAUSE`) still stops it. These scores go into
   W&B after training: a log into the run's W&B run now would move its step past the trainer's
   next one, and W&B would drop that step. A scoring that fails is noted and training goes on;
4. after `DONE`, scores what is left (the final `adapter`, any that failed), logs the mid-run
   scores into the run's W&B run, then the test set (1,000 questions, never used for a decision)
   once, on the final adapter and on the best evalset adapter when that is another one.

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
import urllib.request
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
SCORING = "scoring"  # PAUSE's text when the driver itself stops the trainer to score an adapter
EVALSET = DATA / "training" / "public" / "evalset" / "examples_val.jsonl"
TESTSET = DATA / "training" / "public" / "testset" / "examples_val.jsonl"
NEO4J_HTTP = "http://127.0.0.1:7476/"  # asktrainer-neo4j, auth off (docs/traces.md)
STALL_S = 15 * 60  # no progress while training
START_S = 90 * 60  # before this launch's first step: tokenizing ~95k examples, loading
# WSL's GPU residency failures (`dxgkio_make_resident ... -12`) crash or stall the trainer now
# and then all through a run (sft-main: a segfault at step 809), and a resume goes on cleanly. So
# the limit is on crashes that come back before the run gets past the next checkpoint (a crash
# loop), with a high cap on the whole run as a backstop.
MAX_RESUMES_IN_A_ROW = 3
MAX_RESUMES = 30
# PyTorch's cache of GPU memory grew to 14.4-15.6 GB of the card's 16 (the trainer's own peak is
# ~10.7-12.4): more than Windows keeps resident for WSL, so `dxgkio_make_resident` failed and the
# HIP runtime segfaulted in the backward pass (step 810, twice), or the trainer stalled (835). So
# PyTorch gets 80% of the card (12.8 GiB; the largest step so far needed 12.36 GB), frees cached
# blocks past 90% of that, and doesn't split big blocks (fragmentation). The threshold alone did
# nothing: it only works with a cap (train.limit_cuda). Numbers unchanged.
GPU_FRACTION = 0.8
ALLOC_CONF = "garbage_collection_threshold:0.9,max_split_size_mb:256"
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


def resumes_in_a_row(resumes: list[dict], ckpt_every: int) -> int:
    """The latest resumes that all restarted from the same checkpoint: the run got no further
    than that checkpoint between them."""
    n, at = 0, None
    for r in reversed(resumes):
        c = r["step"] // ckpt_every
        if at is not None and c != at:
            break
        at, n = c, n + 1
    return n


def adapter_steps(every: int, total: int) -> list[int]:
    """The steps the trainer saves an `adapter_stepNNNNNN` at (not the last step: `adapter`)."""
    return list(range(every, total, every)) if every > 0 else []


def pause_due(step: int, adapter_step: int, ckpt_every: int) -> bool:
    """Past the last checkpoint before `adapter_step`: a PAUSE made now stops the trainer at the
    first checkpoint at or after it, which comes after that step's adapter is saved."""
    return step > (adapter_step - 1) // ckpt_every * ckpt_every


def neo4j_up() -> bool:
    try:
        with urllib.request.urlopen(NEO4J_HTTP, timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


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
        self._plan: tuple[list[int], int] | None = None

    # -- stops for a scoring -----------------------------------------------------------
    def plan(self) -> tuple[list[int], int]:
        """(the step adapters' steps, the checkpoint interval), from the run's saved config."""
        if self._plan is None:
            ck = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
            info = json.loads((self.run_dir / "run_info.json").read_text(encoding="utf-8"))
            ck = ck["checkpoint"]
            self._plan = (adapter_steps(ck["adapter_every_steps"], info["planned_steps"]),
                          ck["every_steps"])  # fmt: skip
        return self._plan

    def failed(self) -> set[str]:
        return {e["tag"] for e in self.state.get("score_errors", [])}

    def pending(self) -> list[int]:
        """Step adapters (saved or still to come) not scored yet; one whose scoring failed is
        left to the scoring after training."""
        try:
            steps = self.plan()[0]
        except (OSError, ValueError, KeyError):  # not written yet, or being written (a start)
            return []
        done = set(self.state["scores"]) | self.failed()
        return [s for s in steps if f"step{s:06d}" not in done]

    def stop_asked(self) -> bool:
        """`PAUSE` is there and isn't the driver's own (a scoring stop)."""
        try:
            return PAUSE.read_text(encoding="utf-8").strip() != SCORING
        except OSError:
            return False

    def trainer_running(self) -> bool:
        # `trai[n]`: the pattern doesn't match the shell that runs pgrep
        return self.wsl(f"pgrep -f 'ask-trainer trai[n].*{self.name}'").returncode == 0

    def ensure_neo4j(self) -> None:
        """The Cypher answers are checked on the trainer's Neo4j. Docker Desktop doesn't start by
        itself after a switch-off: start it when it's down, then the container, and wait for
        Neo4j to answer."""
        up = ["docker", "version", "--format", "{{.Server.Version}}"]

        def engine() -> bool:  # (it can hang while Docker Desktop starts: bounded)
            try:
                r = subprocess.run(up, capture_output=True, timeout=60, **ps.TEXT)
                return r.returncode == 0
            except subprocess.TimeoutExpired:
                return False

        if not engine():
            subprocess.run(["docker", "desktop", "start"], capture_output=True, timeout=600,
                           **ps.TEXT)  # fmt: skip
            deadline = time.time() + 600
            while not engine():  # (the start can return before the engine answers)
                if time.time() > deadline:
                    raise RuntimeError("Docker Desktop didn't come up within 10 minutes")
                time.sleep(10)
        subprocess.run(["docker", "start", "asktrainer-neo4j"], capture_output=True, **ps.TEXT)
        deadline = time.time() + 300
        while not neo4j_up():
            if time.time() > deadline:
                raise RuntimeError(f"Neo4j didn't answer on {NEO4J_HTTP} within 5 minutes")
            time.sleep(5)

    def score_pending(self) -> None:
        """Each saved step adapter not scored yet, on the evalset, while the trainer is stopped
        (no W&B yet: the module's docstring says why). A failure is noted and training goes on;
        the scoring after training tries it again."""
        for folder, tag, step in adapter_tags(self.run_dir):
            if tag == "final" or tag in self.state["scores"] or tag in self.failed():
                continue
            if self.trainer_running():  # never two GPU jobs: leave it to the next stop
                return
            self.status(phase="scoring", note=f"evalset: {tag} (during the run)",
                        step=last_step(self.run_dir))  # fmt: skip
            try:
                self.ensure_neo4j()
                self.state["scores"][tag] = self.score(folder, tag, step, EVALSET, wandb=False)
            except Exception as e:  # a scoring never stops the training
                self.state.setdefault("score_errors", []).append(
                    {"ts": now(), "tag": tag, "error": f"{type(e).__name__}: {str(e)[:400]}"}
                )
            self.save()

    # -- training under the watchdog ------------------------------------------------------
    def kill_trainer(self) -> None:
        pat = f"ask-trainer train.*{self.name}"
        self.wsl(f"pkill -TERM -f '{pat}'; sleep 15; pkill -KILL -f '{pat}'; true")

    def train(self) -> str:
        """'done' | 'paused' | raises when the run can't go on."""
        log = self.dir / f"{self.name}.train.log"
        while not (self.run_dir / "DONE").exists():
            if not self.stop_asked():
                self.score_pending()  # step adapters saved and not scored yet
            if self.stop_asked():  # a stop was asked for: nothing starts until the file is gone
                note = f"{PAUSE} exists: delete it and start the driver again"
                self.status(phase="paused", step=last_step(self.run_dir), note=note)
                return "paused"
            PAUSE.unlink(missing_ok=True)  # the driver's own, done with
            resume = (self.run_dir / "config.json").exists()
            if resume and len(self.state["resumes"]) >= MAX_RESUMES:
                raise RuntimeError(f"{MAX_RESUMES} resumes already: see {log}")
            if resume and self.state["resumes"] and resumes_in_a_row(
                    self.state["resumes"], self.plan()[1]) >= MAX_RESUMES_IN_A_ROW:  # fmt: skip
                raise RuntimeError(f"{MAX_RESUMES_IN_A_ROW} crashes in a row from the same "
                                   f"checkpoint: see {log}")  # fmt: skip
            cmd = (f"ask-trainer train --resume {self.name} --launched-by agent" if resume else
                   f"ask-trainer train --config {ps._wsl(self.cfg_path)} --name {self.name} "
                   "--launched-by agent")  # fmt: skip
            start_step, launched = last_step(self.run_dir), time.time()
            self.status(phase="training", note="resumed" if resume else "started",
                        step=start_step)  # fmt: skip
            full = (f"cd {ps.WSL_REPO} && source ~/venvs/nfl-ask-trainer/bin/activate && "
                    f"export PYTORCH_ALLOC_CONF={ALLOC_CONF} PYTORCH_HIP_ALLOC_CONF={ALLOC_CONF}"
                    f" ASKTRAINER_GPU_MEMORY_FRACTION={GPU_FRACTION} && {cmd}")  # fmt: skip
            with log.open("a", encoding="utf-8") as f:
                proc = subprocess.Popen(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", full],
                                        stdout=f, stderr=subprocess.STDOUT)  # fmt: skip
            why = None
            while proc.poll() is None:
                time.sleep(30)
                step = last_step(self.run_dir)
                if stalled(time.time(), launched, progress_mtime(self.run_dir), step > start_step):
                    why = "stalled"
                    self.kill_trainer()
                    break
                todo = self.pending()
                if todo and not PAUSE.exists() and pause_due(step, todo[0], self.plan()[1]):
                    PAUSE.write_text(f"{SCORING}\n", encoding="utf-8")
                    self.status(
                        phase="training",
                        step=step,
                        note=f"stopping at the next checkpoint to score step {todo[0]}",
                    )
            proc.wait()
            step = last_step(self.run_dir)
            if (self.run_dir / "DONE").exists():
                break
            if (self.run_dir / "PAUSED").exists() and not self.stop_asked():
                continue  # stopped to score an adapter: the loop's top scores it and resumes
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
    def check(self, preds: Path, tag: str, step: int, examples: Path, checker: int,
              wandb: bool) -> dict:  # fmt: skip
        extra = "" if wandb else " --no-wandb"
        r = self.wsl(f"ask-trainer evalrun --run {self.name} --preds {ps._wsl(preds)} "
                     f"--tag {tag} --step {step} --checker {checker} "
                     f"--examples {ps._wsl(examples)}{extra}")  # fmt: skip
        if r.returncode != 0:
            raise RuntimeError(f"scoring {tag} (checker {checker}) failed: {r.stderr[-600:]}")
        return json.loads(r.stdout.strip().splitlines()[-1])

    def score(self, folder: str, tag: str, step: int, examples: Path, wandb: bool = True) -> dict:
        gguf = self.convert(self.name, folder)
        preds = self.run_dir / f"preds_llamacpp_{tag}.jsonl"
        self.serve(ps.EVAL_PORT, f"--reasoning-budget {ps.BUDGET} --lora {gguf}")
        try:
            gen = llamaeval.generate(examples, preds, f"http://127.0.0.1:{ps.EVAL_PORT}")
        finally:
            self.serve(ps.EVAL_PORT, stop=True)
        return {"generate": gen, "wandb": wandb,
                f"checker{CHECKER}": self.check(preds, tag, step, examples, CHECKER, wandb),
                f"checker{ALSO}": self.check(preds, f"{tag}-checker{ALSO}", step, examples, ALSO,
                                             wandb=False)}  # fmt: skip

    def score_all(self) -> None:
        scores = self.state["scores"]
        self.ensure_neo4j()
        for folder, tag, step in adapter_tags(self.run_dir):
            if tag not in scores:
                self.status(phase="scoring", note=f"evalset: {tag}")
                scores[tag] = self.score(folder, tag, step, EVALSET)
                self.save()
            elif not scores[tag].get("wandb", True):  # scored during the run: into W&B now
                self.status(phase="scoring", note=f"evalset: {tag} into W&B")
                preds = self.run_dir / f"preds_llamacpp_{tag}.jsonl"
                scores[tag][f"checker{CHECKER}"] = self.check(preds, tag, step, EVALSET, CHECKER,
                                                              wandb=True)  # fmt: skip
                scores[tag]["wandb"] = True
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
                scores[t] = self.score(folder, t, step, TESTSET)
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
