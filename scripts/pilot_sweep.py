"""The pilot sweep, end to end, from Windows (docs/pilots.md):

    C:\\Python314\\python.exe scripts\\pilot_sweep.py [--traces pilot] [--only P1,P2]

1. waits for the trace run to finish, re-checking its Cypher answers every 30 minutes
   (`cyphercheck recheck`), then once more; exports the traces (`traces export`, in WSL);
2. scores the base model on the evalset (the trace server, no LoRA): the reference;
3. stops the trace server and runs P1-P8 one after another, each: `ask-trainer train` (WSL,
   resumed if a run folder is left unfinished), then for its step-50, step-100 and final
   adapters: convert the LoRA to GGUF (WSL), serve it on the Q8_0 base (`serve.ps1`, port
   8092, thinking budget 1024), answer the evalset (`llamaeval generate`), stop the server,
   score (`ask-trainer evalrun`, WSL, into the run's W&B run).
   P4-P8 take the "best" settings from the pilots before them, by the pre-registered rule
   (`best_of`): the higher final exec accuracy on the evalset; a tie (within half a point):
   the lower final validation loss.

State: <data root>/ask/pilots/sweep.json (results so far; a restart skips what's done) and
status.json (what runs now, for the dashboard watcher). One GPU job at a time, always.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from asktrainer import llamaeval  # noqa: E402

DATA = Path(r"D:\nfl-ml-data\ask")
WSL_DATA = "/mnt/d/nfl-ml-data/ask"
WSL_REPO = "/mnt/f/Personal/NFL/nfl-ask-trainer"
FORK = "/mnt/f/Personal/TQ-Optimizer-Test/atomic-llama-cpp-turboquant-spark"
BASE_HF = "/mnt/f/LOCAL-Models/XHToken/Spark-X2.6-4B"
SERVE = REPO / "scripts" / "llamacpp" / "serve.ps1"
EVAL_PORT = 8092
TRACE_PORT = 8090
BUDGET = 1024
TIE = 0.005  # exec accuracies this close count as a tie
# captured output is UTF-8 (WSL, progress bars, W&B's emoji), never the console code page
TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}

BASE = {
    "name": "pilot",
    "seed": 42,
    "data": {"public": {}},
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "targets": ["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "optim": {
        "lr": 1.0e-4,
        "epochs": 1,
        "batch": 1,
        "grad_accum": 32,
        "warmup": 0.03,
        "warmup_min_steps": 10,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "max_len": 4096,
        "attn_chunk": 1024,
        "attn_chunk_above": 3072,
        "hipblaslt": True,
    },
    "jepa": {"enabled": False, "lam": 1.0, "k": 0},
    "eval": {
        "every_epoch": True,
        "every_steps": 0,
        "loss_every_steps": 25,
        "public_n": 0,
        "val_loss_n": 200,
        "val_batch": 2,
    },
    "checkpoint": {"every_steps": 20, "keep": 2, "adapter_every_steps": 50},
    "log_every": 1,
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def lr_of(r: dict) -> float:
    return r["config"]["optim"]["lr"]


def score_of(r: dict) -> tuple[float, float]:
    """(final exec accuracy, -final val loss): higher is better."""
    return (r["final"].get("val/public/exec_acc") or 0.0, -(r.get("val_loss") or 9e9))


def best_of(results: list[dict]) -> dict:
    """The pre-registered rule: the higher final exec accuracy; within TIE, the lower val loss."""
    top = max(results, key=lambda r: score_of(r)[0])
    tied = [r for r in results if score_of(top)[0] - score_of(r)[0] <= TIE]
    return max(tied, key=lambda r: score_of(r)[1])


def plan(results: dict[str, dict]) -> list[tuple[str, dict]]:
    """The pilots in order, each as (name, overrides); later ones only once what they depend
    on has finished (an empty list item stops the plan there)."""
    out: list[tuple[str, dict]] = [
        ("P1", {"optim": {"lr": 1e-4}}),
        ("P2", {"optim": {"lr": 2e-4}}),
        ("P3", {"optim": {"lr": 4e-4}}),
    ]
    lr_runs = [results[p] for p in ("P1", "P2", "P3") if p in results]
    if len(lr_runs) < 3:
        return out
    lr = lr_of(best_of(lr_runs))
    out.append(("P4", {"optim": {"lr": lr}, "lora": {"r": 64, "alpha": 128}}))
    if "P4" not in results:
        return out
    plain = best_of([*lr_runs, results["P4"]])
    rank = plain["config"]["lora"]
    common = {"optim": {"lr": lr}, "lora": {"r": rank["r"], "alpha": rank["alpha"]}}
    out.append(("P5", {**common, "jepa": {"enabled": True, "lam": 1.0, "k": 0}}))
    out.append(("P6", {**common, "jepa": {"enabled": True, "lam": 0.5, "k": 1}}))
    if "P5" not in results or "P6" not in results:
        return out
    jepa = best_of([results["P5"], results["P6"]])
    out.append(("P7", {**_overrides_of(plain), "seed": 7}))
    out.append(("P8", {**_overrides_of(jepa), "seed": 7}))
    return out


def _overrides_of(r: dict) -> dict:
    c = r["config"]
    return {
        "optim": {"lr": c["optim"]["lr"]},
        "lora": {"r": c["lora"]["r"], "alpha": c["lora"]["alpha"]},
        "jepa": dict(c["jepa"]),
    }


def merged(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in over.items():
        if isinstance(v, dict):
            out[k] = {**out.get(k, {}), **v}
        else:
            out[k] = v
    return out


# ---- running things ---------------------------------------------------------------------------


class Sweep:
    def __init__(self, traces: str, after_round: int | None = None):
        self.after_round = after_round  # start the pilots once this trace round is done
        self.dir = DATA / "pilots"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.traces = traces
        self.state_path = self.dir / "sweep.json"
        self.state = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {"traces": traces, "results": {}, "base": None, "log": []}
        )

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def status(self, **kw) -> None:
        tmp = self.dir / "status.tmp"
        tmp.write_text(json.dumps({"ts": now(), **kw}), encoding="utf-8")
        tmp.replace(self.dir / "status.json")
        self.state["log"].append({"ts": now(), **kw})
        self.save()
        print(json.dumps({"ts": now(), **kw}), flush=True)

    def wsl(self, cmd: str, log: Path | None = None) -> subprocess.CompletedProcess:
        full = f"cd {WSL_REPO} && source ~/venvs/nfl-ask-trainer/bin/activate && {cmd}"
        if log is not None:
            with log.open("a", encoding="utf-8") as f:
                return subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", full],
                                      stdout=f, stderr=subprocess.STDOUT)  # fmt: skip
        return subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", full],
                              capture_output=True, **TEXT)  # fmt: skip

    def serve(self, port: int, extra: str = "", stop: bool = False) -> str:
        args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SERVE),
                "-Port", str(port)]  # fmt: skip
        args += ["-Stop"] if stop else ["-Extra", extra]
        # the output goes through a file, not a pipe: llama-server, started by the script,
        # inherits the script's handles, and a pipe it holds open would never reach its end
        log = self.dir / f"serve-{port}.log"
        with log.open("w", encoding="utf-8") as f:
            subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        out = log.read_text(encoding="utf-8", errors="replace").strip()
        if not stop and "ready" not in out:
            raise RuntimeError(f"llama-server did not start: {out[:300]}")
        return out

    # -- phase 1: the traces ------------------------------------------------------------------
    def traces_done(self) -> bool:
        ev = DATA / "traces" / self.traces / "events.jsonl"
        kinds = set()
        for line in ev.read_text(encoding="utf-8").splitlines():
            try:
                kinds.add(json.loads(line).get("kind"))
            except json.JSONDecodeError:
                pass
        if kinds & {"stopped", "aborted"}:
            raise RuntimeError(f"the trace run {self.traces} stopped early: see its events")
        if "DONE" in kinds:
            return True
        if self.after_round is not None and self._round_started(ev, self.after_round + 1):
            self.stop_tracegen()  # the rest of the traces runs after the sweep (finish_traces)
            return True
        return False

    def _round_started(self, ev: Path, n: int) -> bool:
        for line in ev.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "round" and e.get("round", -1) >= n:
                return True
        return False

    def stop_tracegen(self) -> None:
        pid_file = DATA / "traces" / self.traces / "tracegen.pid"
        pid = pid_file.read_text(encoding="ascii").strip()
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
             f"if ($p -and $p.ProcessName -eq 'python') "
             f"{{ Stop-Process -Id {pid} -Confirm:$false; 'stopped' }}"],
            capture_output=True, **TEXT,
        )  # fmt: skip
        how = r.stdout.strip() or "already gone"
        ev = DATA / "traces" / self.traces / "events.jsonl"
        why = (
            f"round {self.after_round} done; the next round runs after the pilot sweep "
            "(the GPU trains the pilots meanwhile)"
        )
        with ev.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"ts": now(), "kind": "paused", "reason": why}) + "\n")
        self.status(phase="traces", note=f"round {self.after_round} done: tracegen {pid} {how}; "
                    "the next rounds run after the sweep")  # fmt: skip

    def finish_traces(self) -> None:
        """After the sweep: the trace run's remaining rounds (it resumes where it stopped)."""
        self.serve(TRACE_PORT, f"--reasoning-budget {BUDGET}")
        log = self.dir / "trace_run.log"  # a file, not a pipe: tracegen inherits the handles
        with log.open("w", encoding="utf-8") as f:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 str(REPO / "scripts" / "llamacpp" / "trace_run.ps1"), "-Run", self.traces],
                stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )  # fmt: skip
        note = log.read_text(encoding="utf-8", errors="replace").strip()
        self.status(phase="traces-rest", note=note[:300])

    def recheck(self) -> str:
        r = subprocess.run(
            [sys.executable, "-m", "asktrainer.cyphercheck", "recheck", "--run",
             str(DATA / "traces" / self.traces)],
            capture_output=True, cwd=REPO, **TEXT,
            env={**_env(), "PYTHONPATH": str(REPO / "src")},
        )  # fmt: skip
        return (r.stdout.strip().splitlines() or [""])[-1]

    def wait_for_traces(self, every: int = 1800) -> None:
        last = 0.0
        while not self.traces_done():
            if time.time() - last >= every:
                self.status(
                    phase="traces", note="re-checking Cypher answers", result=self.recheck()
                )
                last = time.time()
            time.sleep(60)
        self.status(phase="traces", note="final re-check", result=self.recheck())

    def export(self) -> dict:
        out = DATA / "training" / "public" / f"traces-{self.traces}" / "stats.json"
        if self.state.get("export"):
            return self.state["export"]
        r = self.wsl(f"ask-trainer traces export --name {self.traces}")
        if r.returncode != 0:
            raise RuntimeError(f"traces export failed: {r.stdout[-400:]} {r.stderr[-400:]}")
        self.state["export"] = json.loads(out.read_text(encoding="utf-8"))
        self.save()
        return self.state["export"]

    # -- scoring one adapter (or the base) ----------------------------------------------------
    def score(
        self, run: str | None, tag: str, step: int, lora_gguf: Path | None, port: int
    ) -> dict:
        run_dir = DATA / "finetune" / run if run else self.dir / "base"
        run_dir.mkdir(parents=True, exist_ok=True)
        preds = run_dir / f"preds_llamacpp_{tag}.jsonl"
        evalset = DATA / "training" / "public" / "evalset" / "examples_val.jsonl"
        started = False
        if port == EVAL_PORT:
            extra = f"--reasoning-budget {BUDGET}" + (f" --lora {lora_gguf}" if lora_gguf else "")
            self.serve(port, extra)
            started = True
        try:
            gen = llamaeval.generate(evalset, preds, f"http://127.0.0.1:{port}")
        finally:
            if started:
                self.serve(port, stop=True)
        if run is None:
            r = self.wsl(f"ask-trainer evalrun --preds {_wsl(preds)} --tag {tag} --step 0")
        else:
            r = self.wsl(
                f"ask-trainer evalrun --run {run} --preds {_wsl(preds)} --tag {tag} --step {step}"
            )
        if r.returncode != 0:
            raise RuntimeError(f"scoring {run or 'base'} {tag} failed: {r.stderr[-600:]}")
        res = json.loads(r.stdout.strip().splitlines()[-1])
        res["generate"] = gen
        return res

    def convert(self, run: str, adapter: str) -> Path:
        out = DATA / "llamacpp" / "lora" / f"{run}-{adapter}.gguf"
        if out.exists():
            return out
        r = self.wsl(
            f"cd {FORK} && PYTHONPATH=gguf-py python convert_lora_to_gguf.py --base {BASE_HF} "
            f"--outtype f16 --outfile {_wsl(out)} {WSL_DATA}/finetune/{run}/{adapter}"
        )
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"LoRA conversion failed for {run}/{adapter}: {r.stderr[-600:]}")
        return out

    # -- one pilot --------------------------------------------------------------------------
    def pilot(self, p: str, over: dict) -> dict:
        name = f"pilot-{p.lower()}"
        cfg = merged(BASE, over)
        cfg["name"] = name
        cfg["data"]["public"] = {f"traces-{self.traces}": {"train": -1, "val": -1}}
        cfg_path = self.dir / "configs" / f"{name}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")  # JSON is YAML
        run_dir = DATA / "finetune" / name
        log = self.dir / f"{name}.train.log"
        if not (run_dir / "DONE").exists():
            self.status(phase="pilot", pilot=p, run=name, note="training")
            if (run_dir / "config.json").exists():
                cmd = f"ask-trainer train --resume {name} --launched-by agent"
            else:
                cmd = (
                    f"ask-trainer train --config {_wsl(cfg_path)} --name {name} --launched-by agent"
                )
            r = self.wsl(cmd, log=log)
            if r.returncode != 0 or not (run_dir / "DONE").exists():
                raise RuntimeError(f"{name} training failed (exit {r.returncode}): see {log}")
        res = self.state["results"].get(p) or {"config": cfg, "run": name, "evals": {}}
        adapters = sorted(d.name for d in run_dir.glob("adapter_step*") if d.is_dir()) + ["adapter"]
        steps = _final_step(run_dir)
        for a in adapters:
            tag = "final" if a == "adapter" else a.replace("adapter_", "")
            if tag in res["evals"]:
                continue
            step = steps if a == "adapter" else int(a.split("step")[1])
            self.status(phase="pilot", pilot=p, run=name, note=f"scoring {tag}")
            gguf = self.convert(name, a)
            res["evals"][tag] = self.score(name, tag, step, gguf, EVAL_PORT)
            self.state["results"][p] = res
            self.save()
        res["final"] = {k: v for k, v in res["evals"]["final"].items() if k.startswith("val/")}
        res["val_loss"] = _final_val_loss(run_dir)
        res["done_at"] = now()
        self.state["results"][p] = res
        self.save()
        return res

    def run(self, only: set[str] | None) -> None:
        self.status(phase="traces", note="waiting for the trace run")
        self.wait_for_traces()
        if self.state.get("base") is None:
            self.status(phase="base", note="scoring the base model on the trace server")
            self.state["base"] = self.score(None, "base", 0, None, TRACE_PORT)
            self.save()
        self.serve(TRACE_PORT, stop=True)
        self.status(phase="export", result=self.export())
        while True:
            done = {p: r for p, r in self.state["results"].items() if "final" in r}
            todo = [(p, o) for p, o in plan(done) if p not in done]
            todo = [t for t in todo if not only or t[0] in only]
            if not todo:
                break
            p, over = todo[0]
            r = self.pilot(p, over)
            self.status(
                phase="pilot", pilot=p, note="done", final=r["final"], val_loss=r["val_loss"]
            )
        self.status(phase="done", summary=summary(self.state))
        if self.after_round is not None and not self.traces_finished():
            self.finish_traces()

    def traces_finished(self) -> bool:
        ev = DATA / "traces" / self.traces / "events.jsonl"
        return any('"DONE"' in line for line in ev.read_text(encoding="utf-8").splitlines())


def summary(state: dict) -> dict:
    res = state["results"]
    out = {p: {"exec_acc": r["final"].get("val/public/exec_acc"), "val_loss": r.get("val_loss"),
               "lr": lr_of(r), "r": r["config"]["lora"]["r"], "jepa": r["config"]["jepa"]}
           for p, r in res.items() if r.get("final")}  # fmt: skip
    if "P7" in out and "P8" in out:
        noise = abs((out["P7"]["exec_acc"] or 0) - (out["P1"]["exec_acc"] or 0))
        out["rule"] = {"seed_noise": noise}
    return out


def _final_step(run_dir: Path) -> int:
    steps = [json.loads(x).get("step", 0) for x in (run_dir / "train_log.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]  # fmt: skip
    return max(steps or [0])


def _final_val_loss(run_dir: Path) -> float | None:
    rows = [json.loads(x) for x in (run_dir / "train_log.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]  # fmt: skip
    losses = [r["val/loss"] for r in rows if "val/loss" in r]
    return losses[-1] if losses else None


def _wsl(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if s[1:2] == ":" else s


def _env() -> dict:
    import os

    return dict(os.environ)


def keep_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs (released when it exits; no
    setting changes): a sleep would stop training and the servers mid-run."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> int:
    keep_awake()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default="pilot")
    ap.add_argument("--only", default=None, help="run only these pilots, e.g. P1,P2")
    ap.add_argument(
        "--after-round", type=int, default=None,
        help="start the pilots once this trace round is done (0: the first tries); the rest of "
        "the trace run resumes after the sweep",
    )  # fmt: skip
    a = ap.parse_args()
    sweep = Sweep(a.traces, a.after_round)
    import os

    (sweep.dir / "driver.pid").write_text(str(os.getpid()), encoding="ascii")  # for watchers
    try:
        sweep.run(set(a.only.split(",")) if a.only else None)
    except Exception as e:
        sweep.status(phase="failed", error=f"{type(e).__name__}: {str(e)[:600]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
