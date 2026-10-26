"""The llama.cpp evaluation path and the pilot sweep's decisions, on the CPU: answers from a
fake llama-server in Spark's text form, the scoring command, and the pre-registered rule."""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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
