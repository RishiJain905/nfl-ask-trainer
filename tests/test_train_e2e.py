"""`train()` end to end on the CPU: a tiny Llama written to disk as the base model, public
data with a gretel raw context, step evaluations with the public scorer, checkpoints, a
simulated crash and `train(resume=...)`, and the W&B artifact calls (W&B disabled)."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from conftest import TEMPLATE
from safetensors.torch import load_file
from test_train_smoke import LLAMA_TARGETS, tiny_llama

from asktrainer import train as train_mod
from asktrainer.config import TrainConfig
from asktrainer.data.schema import Example, write_jsonl
from asktrainer.events import KINDS, Events
from asktrainer.settings import Paths

CTX = "CREATE TABLE t (id INT, v FLOAT); INSERT INTO t (id, v) VALUES (1, 1.5), (2, 2.5);"


def _gretel(i: int, split: str) -> Example:
    q = f"SELECT id, v FROM t WHERE id = {1 + i % 2}"
    return Example(
        id=f"gretel-{split}-{i}", origin="gretel", task="generate", split=split, group="public",
        messages=[{"role": "system", "content": "Schema: t(id, v). " * (1 + i % 3)},
                  {"role": "user", "content": f"Question: row {i}?"},
                  {"role": "assistant", "content": json.dumps({"query": q})}],
        jepa={"text": f"row {i}?", "code": json.dumps({"query": q})}, dialect="sqlite",
        meta={"row_id": str(i), "source_split": "test" if split == "val" else "train"},
    )  # fmt: skip


@pytest.fixture
def env(tmp_path, tiny_tok, monkeypatch):
    base = tmp_path / "base"
    tiny_llama(tiny_tok).save_pretrained(str(base), safe_serialization=True)
    tok = tiny_tok
    tok.chat_template = TEMPLATE
    tok.save_pretrained(str(base))
    paths = Paths(data_root=tmp_path / "data", base_model=base, nfl_repo=tmp_path / "nfl",
                  uv_exe=tmp_path / "uv.exe")  # fmt: skip
    g = paths.public / "gretel"
    write_jsonl(g / "examples_train.jsonl", [_gretel(i, "train") for i in range(8)])
    write_jsonl(g / "examples_val.jsonl", [_gretel(i, "val") for i in range(4)])
    (g / "raw").mkdir(parents=True)
    pq.write_table(pa.table({"id": list(range(4)), "sql_context": [CTX] * 4}),
                   g / "raw" / "synthetic_text_to_sql_test.snappy.parquet")  # fmt: skip
    monkeypatch.setattr(train_mod, "wandb_mode", lambda: ("disabled", None))
    artifacts = []
    monkeypatch.setattr(train_mod, "log_adapter",
                        lambda run, cfg, d, aliases, meta, *rest: artifacts.append(
                            (d.name, tuple(aliases))))  # fmt: skip
    cfg = TrainConfig.model_validate({
        "name": "e2e", "seed": 5,
        "data": {"public": {"gretel": {"train": 8, "val": 4}}},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.1, "targets": LLAMA_TARGETS},
        "optim": {"lr": 1e-2, "epochs": 2, "batch": 2, "grad_accum": 1, "warmup": 1,
                  "max_len": 4096, "hipblaslt": False},
        "eval": {"every_steps": 3, "public_n": 4, "max_new_tokens": {"generate": 6}},
        "checkpoint": {"dir": str(tmp_path / "ck"), "every_steps": 3, "keep": 2},
    })  # fmt: skip
    return paths, cfg, artifacts


def _ce(run_dir) -> dict[int, float]:
    rows = [json.loads(x) for x in (run_dir / "train_log.jsonl").read_text().splitlines()]
    return {r["step"]: r["train/ce"] for r in rows if "train/ce" in r}


def test_train_crash_resume_equals_straight_run(env):
    paths, cfg, artifacts = env
    a = train_mod.train(cfg, name="a", paths=paths, device="cpu")
    assert (a / "DONE").exists() and (a / "adapter" / "adapter_model.safetensors").exists()
    assert artifacts == [("adapter_epoch1", ("latest", "epoch-1")),
                         ("adapter_epoch2", ("latest", "epoch-2")),
                         ("adapter", ("latest", "final"))]  # fmt: skip
    scores = json.loads((a / "scores_public_step000003.json").read_text())
    assert scores["summary"]["set"]["gretel"]["n"] == 4  # the step-3 evaluation ran
    log = [json.loads(x) for x in (a / "train_log.jsonl").read_text().splitlines()]
    assert any("val/gretel/json_ok" in r for r in log)  # a random model writes no JSON
    assert any("val/loss" in r and r["step"] == 4 for r in log)  # the epoch-end evaluation

    artifacts.clear()
    b = train_mod.train(cfg, name="b", paths=paths, device="cpu", stop_after=7)
    assert not (b / "DONE").exists()
    assert [p.name for p in (paths.data_root.parent / "ck" / "b").iterdir()
            if p.name.startswith("step-")] == ["step-000004", "step-000006"]  # fmt: skip
    b2 = train_mod.train(resume="b", paths=paths, device="cpu")
    assert b2 == b and (b / "DONE").exists()
    assert _ce(b) == pytest.approx(_ce(a), abs=1e-9)  # the log continues seamlessly
    assert sorted(_ce(b)) == list(range(1, 9))
    wa = load_file(str(a / "adapter" / "adapter_model.safetensors"))
    wb = load_file(str(b / "adapter" / "adapter_model.safetensors"))
    assert wa.keys() == wb.keys()
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k
    assert ("adapter", ("latest", "final")) in artifacts


def test_resume_refuses_changed_data(env):
    paths, cfg, _ = env
    train_mod.train(cfg, name="c", paths=paths, device="cpu", stop_after=3)
    g = paths.public / "gretel" / "examples_train.jsonl"
    write_jsonl(g, [_gretel(i, "train") for i in range(100, 108)])
    with pytest.raises(RuntimeError, match="changed"):
        train_mod.train(resume="c", paths=paths, device="cpu")


def test_log_adapter_builds_the_artifact(tmp_path):
    d = tmp_path / "adapter_epoch1"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    calls = []

    class FakeRun:
        def log_artifact(self, art, aliases):
            calls.append((art.name, art.type, art.metadata["epoch"], aliases))

    cfg = TrainConfig()
    ev = Events(tmp_path / "events.jsonl")
    train_mod.log_adapter(FakeRun(), cfg, d, ["latest", "epoch-1"], {"epoch": 1, "run": "r1"},
                          print, ev)  # fmt: skip
    assert calls == [("ask-finetune-adapter-r1", "model", 1, ["latest", "epoch-1"])]

    class Broken:
        def log_artifact(self, art, aliases):
            raise ConnectionError("offline")

    seen = []
    train_mod.log_adapter(Broken(), cfg, d, ["latest"], {"epoch": 1, "run": "r1"}, seen.append, ev)
    assert seen and "failed" in seen[0] and "offline" not in seen[0]  # the type only
    rows = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["artifact_uploaded", "artifact_failed"]
    assert rows[1]["error"] == "ConnectionError" and "offline" not in json.dumps(rows)


def _jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_dashboard_files(env, monkeypatch):
    paths, cfg, _ = env
    a = train_mod.train(cfg, name="d", paths=paths, device="cpu", config_path="configs/x.yaml")
    info = json.loads((a / "run_info.json").read_text())
    for k in ("run", "config_path", "objective", "seed", "started_at", "planned_steps",
              "planned_tokens", "examples_per_set", "max_len", "effective_batch", "wandb",
              "checkpoint_dir", "base_model", "git_commit"):  # fmt: skip
        assert k in info, k
    assert info["planned_steps"] == 8 and info["effective_batch"] == 2
    assert info["examples_per_set"] == {"gretel": 8} and info["config_path"] == "configs/x.yaml"
    assert set(info["wandb"]) == {"id", "url", "project", "entity", "mode"}
    ev = _jsonl(a / "events.jsonl")
    assert all({"ts", "step", "kind"} <= set(r) and r["kind"] in KINDS for r in ev)
    kinds = [r["kind"] for r in ev]
    for k in ("start", "data_ready", "model_loaded", "eval_start", "eval_end",
              "checkpoint_saved", "epoch_end", "done"):  # fmt: skip
        assert k in kinds, k
    assert kinds[0] == "start" and kinds[-1] == "done"
    saved = [r for r in ev if r["kind"] == "checkpoint_saved"]
    assert {"name", "seconds", "bytes", "deleted"} <= set(saved[0]) and saved[0]["bytes"] > 0
    ends = [r for r in ev if r["kind"] == "eval_end"]
    assert {"seconds", "loss_s", "generate_s", "public_s"} <= set(ends[0])
    rows = _jsonl(a / "train_log.jsonl")
    assert all("ts" in r for r in rows)
    samples = _jsonl(a / "samples_000003.jsonl")
    assert 0 < len(samples) <= 12
    for r in samples:
        assert {"id", "set", "group", "question", "gold", "output", "ok", "exec_ok",
                "json_ok", "reason"} <= set(r)  # fmt: skip
    assert (a / "samples_000008.jsonl").exists()  # the epoch-end evaluation's

    # a crash mid-run is recorded (the exception's type only), then the resume
    real = train_mod.forward_losses
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += args[0].training  # training forwards only (not the evaluations)
        if args[0].training and calls["n"] == 7:  # step 7
            raise MemoryError("secret-ish text that must not be logged")
        return real(*args, **kw)

    monkeypatch.setattr(train_mod, "forward_losses", flaky)
    with pytest.raises(MemoryError):
        train_mod.train(cfg, name="e", paths=paths, device="cpu")
    monkeypatch.setattr(train_mod, "forward_losses", real)
    ev = _jsonl(paths.runs / "e" / "events.jsonl")
    assert ev[-1]["kind"] == "crashed" and ev[-1]["error"] == "MemoryError"
    assert "secret" not in (paths.runs / "e" / "events.jsonl").read_text()
    train_mod.train(resume="e", paths=paths, device="cpu")
    ev = _jsonl(paths.runs / "e" / "events.jsonl")
    kinds = [r["kind"] for r in ev]
    assert "resumed" in kinds and kinds[-1] == "done"
    info = json.loads((paths.runs / "e" / "run_info.json").read_text())
    assert len(info["resumed_at"]) == 1 and info["resumed_from_step"] == 6  # ckpts 3, 4, 6
