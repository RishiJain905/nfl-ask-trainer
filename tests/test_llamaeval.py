"""The llama.cpp evaluation path and the pilot sweep's decisions, on the CPU: answers from a
fake llama-server in Spark's text form, the scoring command, and the pre-registered rule."""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from asktrainer import llamaeval

REPO = Path(__file__).resolve().parents[1]


def _sweep():
    spec = importlib.util.spec_from_file_location(
        "pilot_sweep", REPO / "scripts" / "pilot_sweep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_as_text_rebuilds_spark_calls():
    msg = {
        "reasoning_content": "count rows",
        "content": "",
        "tool_calls": [{"function": {"name": "run_query", "arguments": '{"query": "SELECT 1"}'}}],
    }
    assert llamaeval.as_text(msg) == (
        "count rows</think><tool_call>run_query<arg_key>query</arg_key>"
        "<arg_value>SELECT 1</arg_value></tool_call>"
    )
    assert llamaeval.as_text({"content": "no can do"}) == "</think>no can do"


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["temperature"] == 0.0 and body["chat_template_kwargs"]["enable_thinking"]
        tool = body["tools"][0]["function"]["name"]
        out = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "reasoning_content": "think",
                        "tool_calls": [
                            {"function": {"name": tool, "arguments": json.dumps({"query": "Q"})}}
                        ],
                    },
                }
            ],
            "usage": {"completion_tokens": 9},
        }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_generate_answers_every_example_and_resumes(tmp_path):
    exs = [
        {
            "id": f"e{i}",
            "thinking": "high",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "tools": [{"type": "function", "function": {"name": "run_query"}}],
        }
        for i in range(5)
    ]
    src = tmp_path / "ex.jsonl"
    src.write_text("".join(json.dumps(e) + "\n" for e in exs))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        out = tmp_path / "preds.jsonl"
        r = llamaeval.generate(src, out, url, workers=2)
        assert r["answered"] == 5 and r["errors"] == 0
        rows = [json.loads(x) for x in out.read_text().splitlines()]
        assert {x["id"] for x in rows} == {f"e{i}" for i in range(5)}
        assert rows[0]["output"].endswith("<arg_value>Q</arg_value></tool_call>")
        assert llamaeval.generate(src, out, url)["answered"] == 0  # nothing left
    finally:
        srv.shutdown()


def _result(lr, acc, loss, r=16, jepa=None):
    return {
        "config": {
            "optim": {"lr": lr},
            "lora": {"r": r, "alpha": 2 * r},
            "jepa": jepa or {"enabled": False, "lam": 1.0, "k": 0},
        },
        "final": {"val/public/exec_acc": acc},
        "val_loss": loss,
    }


def test_the_sweep_plans_by_the_pre_registered_rule():
    sw = _sweep()
    assert [p for p, _ in sw.plan({})] == ["P1", "P2", "P3"]
    res = {"P1": _result(1e-4, 0.40, 0.5), "P2": _result(2e-4, 0.452, 0.6),
           "P3": _result(4e-4, 0.455, 0.4)}  # fmt: skip
    # P2 and P3 tie (within half a point): the lower val loss wins
    p4 = dict(sw.plan(res))["P4"]
    assert p4["optim"]["lr"] == 4e-4 and p4["lora"] == {"r": 64, "alpha": 128}
    res["P4"] = _result(4e-4, 0.43, 0.38, r=64)
    plan = dict(sw.plan(res))
    assert plan["P5"]["lora"]["r"] == 16 and plan["P5"]["jepa"] == {
        "enabled": True,
        "lam": 1.0,
        "k": 0,
    }
    assert plan["P6"]["jepa"] == {"enabled": True, "lam": 0.5, "k": 1} and "P7" not in plan
    res["P5"] = _result(4e-4, 0.47, 0.41, jepa={"enabled": True, "lam": 1.0, "k": 0})
    res["P6"] = _result(4e-4, 0.44, 0.40, jepa={"enabled": True, "lam": 0.5, "k": 1})
    plan = dict(sw.plan(res))
    assert plan["P7"]["seed"] == 7 and plan["P7"]["jepa"]["enabled"] is False
    assert plan["P8"]["seed"] == 7 and plan["P8"]["jepa"]["lam"] == 1.0
    cfg = sw.merged(sw.BASE, plan["P8"])
    assert cfg["optim"]["grad_accum"] == 32 and cfg["optim"]["lr"] == 4e-4 and cfg["seed"] == 7
    assert sw._wsl(Path(r"D:\nfl-ml-data\ask\x.jsonl")) == "/mnt/d/nfl-ml-data/ask/x.jsonl"
    assert "P9" not in plan
    res["P7"] = _result(4e-4, 0.41, 0.5)
    res["P8"] = _result(4e-4, 0.46, 0.4, jepa={"enabled": True, "lam": 1.0, "k": 0})
    # P9, the gold-answer check: P1 exactly (its lr, rank, plain) plus the gold set, final only
    p9 = sw.merged(sw.BASE, dict(sw.plan(res))["P9"])
    assert p9["optim"]["lr"] == 1e-4 and p9["seed"] == 42 and not p9["jepa"]["enabled"]
    assert p9["data"]["public"] == {"gold-p9": {"train": -1, "val": -1}}
    assert p9["checkpoint"] == {"every_steps": 20, "keep": 2, "adapter_every_steps": 0}
    assert p9["eval"]["loss_every_steps"] == 100 and p9["eval"]["val_batch"] == 2


def _bench():
    spec = importlib.util.spec_from_file_location(
        "bench_trace", REPO / "scripts" / "llamacpp" / "bench_trace.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_trace_bench_replays_one_sample_and_counts_the_window():
    bt = _bench()

    class FakeRun:
        questions = [{"qid": f"q{i}", "plan": {("high", "low")[i % 2]: 2}} for i in range(10)]

        def plan_of(self, q):
            return q["plan"]

    a, b = bt.sample(FakeRun(), 4), bt.sample(FakeRun(), 4)
    assert a == b and len(a) == 4  # the same requests for every configuration
    assert all(level == next(iter(q["plan"])) and rid.endswith("#bench") for rid, q, level, _ in a)
    done = [
        {"end": 5, "secs": 9, "completion_tokens": 999, "prompt_tokens": 9},  # warm-up: not counted
        {"end": 70, "secs": 20, "completion_tokens": 300, "prompt_tokens": 600},
        {"end": 120, "secs": 30, "completion_tokens": 300, "prompt_tokens": 600},
        {"end": 100, "secs": 0, "completion_tokens": 0, "prompt_tokens": 0, "error": "x"},
    ]
    s = bt.summarize(done, 60, 180, slots=2)
    assert s["answers"] == 2 and s["gen_tok_s"] == 5.0 and s["prompt_tok_s"] == 10.0
    assert s["answers_per_min"] == 1.0 and s["per_slot_gen_tok_s"] == 2.5 and s["errors"] == 1
    assert set(bt.CONFIGS) >= {"default", "np24", "lean_samplers", "upstream"}


def _real():
    spec = importlib.util.spec_from_file_location("real_run", REPO / "scripts" / "real_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_real_run_watchdog_and_adapter_list(tmp_path):
    rr = _real()
    t0 = 1_000_000.0
    # before this launch's first step: only the long start-up limit applies
    assert not rr.stalled(t0 + rr.STALL_S + 60, t0, t0, armed=False)
    assert rr.stalled(t0 + rr.START_S, t0, t0, armed=False)
    # once stepping: no progress for STALL_S (counted from the launch at the earliest)
    assert not rr.stalled(t0 + 600, t0, t0 + 100, armed=True)
    assert rr.stalled(t0 + 100 + rr.STALL_S, t0, t0 + 100, armed=True)
    run = tmp_path / "sft-main"
    run.mkdir()
    assert rr.last_step(run) == 0 and rr.progress_mtime(run) == 0.0
    (run / "train_log.jsonl").write_text('{"step": 3}\n{"step": 5, "val/loss": 1}\n{"st', "utf-8")
    assert rr.last_step(run) == 5  # a half-written last line is skipped
    for d in ("adapter_step001000", "adapter_step000500", "adapter", "adapter_epoch1"):
        (run / d).mkdir()
    assert rr.adapter_tags(run) == [("adapter_step000500", "step000500", 500),
                                    ("adapter_step001000", "step001000", 1000),
                                    ("adapter", "final", 5)]  # fmt: skip
    cfg = tmp_path / "c.yaml"
    cfg.write_text("# x\nname: sft-main\nseed: 42\n", encoding="utf-8")
    assert rr._yaml_name(cfg) == "sft-main"
    assert rr.PAUSE.name == "PAUSE" and rr.PAUSE.parent.name == "finetune"


def test_the_real_run_knows_when_to_stop_for_a_scoring():
    rr = _real()
    assert rr.adapter_steps(500, 2961) == [500, 1000, 1500, 2000, 2500]
    assert rr.adapter_steps(500, 2500) == [500, 1000, 1500, 2000]  # the last step is `adapter`
    assert rr.adapter_steps(0, 2961) == []
    # step 981 logged: the step-980 checkpoint is behind, so a PAUSE now stops at 1000
    assert not rr.pause_due(980, 1000, 20) and rr.pause_due(981, 1000, 20)
    assert rr.pause_due(999, 1000, 20)
    # an adapter step between checkpoints: the stop comes at the next one (520), after it
    assert not rr.pause_due(500, 510, 20) and rr.pause_due(501, 510, 20)


class _FakeTrainer:
    """`ask-trainer train --resume` as the driver sees it: a step per poll, an adapter every
    `adapter` steps (not the last), a checkpoint every `ckpt` that stops on PAUSE; it starts
    from the last checkpoint and crashes (exit 139) at each step in `crashes`, once."""

    def __init__(self, run: Path, pause: Path, step: int, crashes: list, total=70, ckpt=10,
                 adapter=30):  # fmt: skip
        self.run, self.pause, self.rc, self.crashes = run, pause, None, crashes
        self.step = step // ckpt * ckpt
        self.total, self.ckpt, self.adapter = total, ckpt, adapter
        (run / "PAUSED").unlink(missing_ok=True)  # the trainer's resume does this

    def poll(self):
        if self.rc is not None:
            return self.rc
        self.step += 1
        with (self.run / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": self.step}) + "\n")
        if self.crashes and self.step == self.crashes[0]:
            self.crashes.pop(0)
            self.rc = 139
            return self.rc
        if self.step % self.adapter == 0 and self.step < self.total:
            (self.run / f"adapter_step{self.step:06d}").mkdir(exist_ok=True)
        if self.step == self.total:
            (self.run / "adapter").mkdir()
            (self.run / "DONE").write_text("", encoding="utf-8")
            self.rc = 0
        elif self.step % self.ckpt == 0 and self.pause.exists():
            (self.run / "PAUSED").write_text(f"{self.step}\n", encoding="utf-8")
            self.rc = 0
        return self.rc

    def wait(self):
        return self.rc

    @property
    def returncode(self):
        return self.rc


def _driver(tmp_path, monkeypatch, score, crashes=()):
    rr = _real()
    crashes = list(crashes)
    pause = tmp_path / "finetune" / "PAUSE"
    run = tmp_path / "finetune" / "sft-main"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps({"checkpoint": {"every_steps": 10, "adapter_every_steps": 30}}), encoding="utf-8"
    )
    (run / "run_info.json").write_text(json.dumps({"planned_steps": 70}), encoding="utf-8")
    monkeypatch.setattr(rr, "PAUSE", pause)
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    stops = []

    def popen(argv, **k):
        stops.append(rr.last_step(run))
        m.cmds.append(argv[-1])
        return _FakeTrainer(run, pause, stops[-1], crashes)

    monkeypatch.setattr(rr.subprocess, "Popen", popen)
    m = rr.Main.__new__(rr.Main)  # no state on D:
    m.cmds = []
    m.name, m.dir, m.run_dir, m._plan = "sft-main", tmp_path, run, None
    m.state = {"run": "sft-main", "resumes": [], "scores": {}, "log": []}
    m.statuses = []
    m.status = lambda **kw: m.statuses.append(kw)
    m.save = lambda: None
    m.ensure_neo4j = lambda: None
    m.trainer_running = lambda: False
    m.score = lambda folder, tag, step, examples, wandb=True: score(m, tag, step, wandb)
    return rr, m, pause, stops


def test_the_real_run_stops_scores_each_adapter_and_resumes(tmp_path, monkeypatch):
    scored = []

    def score(m, tag, step, wandb):
        scored.append((tag, step, wandb, rr.last_step(m.run_dir)))
        return {"wandb": wandb, "checker4": {"val/public/exec_acc": 0.5}}

    rr, m, pause, stops = _driver(tmp_path, monkeypatch, score)
    assert m.train() == "done"
    # each adapter scored right at its step (the trainer stopped there), without W&B for now
    assert scored == [("step000030", 30, False, 30), ("step000060", 60, False, 60)]
    assert stops == [0, 30, 60]  # launches: the start, then a resume after each scoring
    assert not pause.exists() and m.state["resumes"] == []  # scoring stops aren't resumes
    assert rr.last_step(m.run_dir) == 70


def test_a_stop_asked_for_during_a_scoring_still_stops(tmp_path, monkeypatch):
    def score(m, tag, step, wandb):
        rr.PAUSE.write_text("stop for the night\n", encoding="utf-8")  # Rishi, meanwhile
        return {"wandb": wandb, "checker4": {}}

    rr, m, pause, stops = _driver(tmp_path, monkeypatch, score)
    assert m.train() == "paused"
    assert stops == [0] and pause.read_text(encoding="utf-8").startswith("stop")
    assert m.statuses[-1]["phase"] == "paused" and "step000030" in m.state["scores"]


def test_a_failed_scoring_never_stops_the_training(tmp_path, monkeypatch):
    def score(m, tag, step, wandb):
        if tag == "step000030":
            raise RuntimeError("Neo4j down")
        return {"wandb": wandb, "checker4": {}}

    rr, m, pause, stops = _driver(tmp_path, monkeypatch, score)
    assert m.train() == "done"
    assert [e["tag"] for e in m.state["score_errors"]] == ["step000030"]  # left to the end
    assert stops == [0, 30, 60] and list(m.state["scores"]) == ["step000060"]


def test_the_resume_limit_counts_crashes_from_the_same_checkpoint():
    rr = _real()

    def r(*steps):
        return [{"step": s} for s in steps]

    assert rr.resumes_in_a_row([], 20) == 0
    assert rr.resumes_in_a_row(r(809), 20) == 1
    assert rr.resumes_in_a_row(r(809, 815), 20) == 2  # both back to the step-800 checkpoint
    assert rr.resumes_in_a_row(r(809, 815, 1503), 20) == 1  # it got further in between
    assert rr.resumes_in_a_row(r(173, 809, 812, 819), 20) == 3


def _scored(m, tag, step, wandb):
    return {"wandb": wandb, "checker4": {}}


def test_crashes_now_and_then_are_resumed_and_the_run_finishes(tmp_path, monkeypatch):
    rr, m, pause, stops = _driver(tmp_path, monkeypatch, _scored, crashes=[15, 45])
    assert m.train() == "done"
    # every launch keeps PyTorch's GPU cache below what Windows keeps resident
    assert all(f"PYTORCH_HIP_ALLOC_CONF={rr.ALLOC_CONF}" in c for c in m.cmds) and m.cmds
    assert [r["step"] for r in m.state["resumes"]] == [15, 45]
    assert stops == [0, 15, 30, 45, 60]  # each crash resumed from its checkpoint (10, 40)
    assert list(m.state["scores"]) == ["step000030", "step000060"]


def test_a_crash_loop_stops_the_driver(tmp_path, monkeypatch):
    # three crashes before the run gets past the step-10 checkpoint, each a little further on
    rr, m, pause, stops = _driver(tmp_path, monkeypatch, _scored, crashes=[15, 17, 19])
    with pytest.raises(RuntimeError, match="3 crashes in a row"):
        m.train()
    assert [r["step"] for r in m.state["resumes"]] == [15, 17, 19]
