"""Crash safety: step checkpoints, atomic writes, and a resume that equals an uninterrupted run
(a deterministic tiny model on the CPU, LoRA dropout and the JEPA gate on, so every RNG
matters)."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import example
from test_train_smoke import LLAMA_TARGETS, tiny_llama

from asktrainer import checkpoint as ck
from asktrainer.chat import ChatTokenizer
from asktrainer.config import JepaSpec, LoraSpec, OptimConfig, TrainConfig
from asktrainer.modelio import apply_lora
from asktrainer.train import Logger, run_loop


def _cfg() -> TrainConfig:
    return TrainConfig(
        name="t",
        seed=3,
        lora=LoraSpec(r=4, alpha=8, dropout=0.1, targets=LLAMA_TARGETS),
        optim=OptimConfig(lr=1e-2, epochs=2, batch=2, grad_accum=1, warmup=1, max_len=4096),
        jepa=JepaSpec(enabled=True, lam=1.0, k=1, ratio=0.5),
    )


def _setup(tiny_tok, cfg):
    model = apply_lora(tiny_llama(tiny_tok), cfg.lora, gradient_checkpointing=True)
    ct = ChatTokenizer(tiny_tok, cfg.optim.max_len, jepa=True, k=1)
    feats, _ = ct.tokenize_all([example(i, "route" if i % 2 else "generate") for i in range(8)])
    return model, ct, feats


def _ce(logger: Logger) -> dict[int, float]:
    return {r["step"]: r["train/ce"] for r in logger.history if "train/ce" in r}


def test_resume_equals_an_uninterrupted_run(tiny_tok, tmp_path):
    cfg = _cfg()
    # A: straight through (8 steps: 2 epochs x 4)
    model_a, ct, feats = _setup(tiny_tok, cfg)
    log_a = Logger(None, echo=None)
    run_loop(model_a, ct, feats, cfg, tmp_path / "a", log_a, "cpu",
             checkpointer=ck.Checkpointer(tmp_path / "ck_a", every_steps=3, keep=2))  # fmt: skip
    # B: the same run "crashes" after step 7 (checkpoints at steps 3, 4 = epoch end, 6)
    model_b, ct, feats = _setup(tiny_tok, cfg)
    log_b = Logger(None, echo=None)
    cp = ck.Checkpointer(tmp_path / "ck_b", every_steps=3, keep=2)
    res_b = run_loop(model_b, ct, feats, cfg, tmp_path / "b", log_b, "cpu", checkpointer=cp,
                     stop_after=7)  # fmt: skip
    assert res_b.steps == 7 and res_b.final_adapter is None
    kept = [p.name for p in ck.complete(tmp_path / "ck_b")]
    assert kept == ["step-000004", "step-000006"]  # keep = 2, oldest deleted
    meta = ck.read_meta(tmp_path / "ck_b" / "step-000006")
    assert (meta["epoch"], meta["next_group"], meta["step"]) == (1, 2, 6)
    assert meta["tokens_seen"] > 0
    # C: a fresh process (new model, other RNG draws) resumes from the newest checkpoint
    torch.manual_seed(12345)
    model_c, ct, feats = _setup(tiny_tok, cfg)
    log_c = Logger(None, echo=None)
    res_c = run_loop(model_c, ct, feats, cfg, tmp_path / "c", log_c, "cpu",
                     checkpointer=ck.Checkpointer(tmp_path / "ck_b", every_steps=3, keep=2),
                     resume_from=ck.latest(tmp_path / "ck_b"))  # fmt: skip
    ce_a, ce_b, ce_c = _ce(log_a), _ce(log_b), _ce(log_c)
    assert sorted(ce_c) == [7, 8] and res_c.steps == 8
    for s in range(1, 8):
        assert ce_b[s] == ce_a[s]
    for s in (7, 8):
        assert ce_c[s] == pytest.approx(ce_a[s], rel=0, abs=1e-12)
    a, c = ck.trainable_state(model_a), ck.trainable_state(model_c)
    assert a.keys() == c.keys()
    for k in a:
        assert torch.equal(a[k], c[k]), k
    tok_a = [r["train/tokens_seen"] for r in log_a.history if r["step"] == 8]
    tok_c = [r["train/tokens_seen"] for r in log_c.history if r["step"] == 8]
    assert tok_a == tok_c


def test_resume_from_an_epoch_end(tiny_tok, tmp_path):
    cfg = _cfg()
    model_a, ct, feats = _setup(tiny_tok, cfg)
    log_a = Logger(None, echo=None)
    epochs_a = []
    run_loop(model_a, ct, feats, cfg, tmp_path / "a", log_a, "cpu",
             on_epoch_end=lambda e, s: epochs_a.append(e) or {})  # fmt: skip
    model_b, ct, feats = _setup(tiny_tok, cfg)
    cp = ck.Checkpointer(tmp_path / "ck", every_steps=0, keep=2)  # epoch ends only
    run_loop(model_b, ct, feats, cfg, tmp_path / "b", Logger(None, echo=None), "cpu",
             checkpointer=cp, stop_after=5)  # fmt: skip
    assert [p.name for p in ck.complete(tmp_path / "ck")] == ["step-000004"]
    model_c, ct, feats = _setup(tiny_tok, cfg)
    epochs_c = []
    log_c = Logger(None, echo=None)
    run_loop(model_c, ct, feats, cfg, tmp_path / "c", log_c, "cpu", checkpointer=cp,
             resume_from=ck.latest(tmp_path / "ck"),
             on_epoch_end=lambda e, s: epochs_c.append(e) or {})  # fmt: skip
    assert epochs_a == [1, 2] and epochs_c == [2]  # epoch 1's end is not redone
    assert _ce(log_c)[8] == pytest.approx(_ce(log_a)[8], abs=1e-12)


def test_incomplete_checkpoints_are_never_loaded(tmp_path):
    root = tmp_path / "ck"
    (root / "step-000003").mkdir(parents=True)
    (root / "step-000003" / ck.MARKER).write_text("ok")
    (root / "step-000009").mkdir()  # a rename that never got its marker: invalid
    (root / ".tmp-step-000012").mkdir()  # an interrupted write
    assert ck.latest(root).name == "step-000003"
    assert [p.name for p in ck.complete(root)] == ["step-000003"]
    with pytest.raises(RuntimeError):
        ck.load(root / "step-000009", None, None, None)


def test_save_cleans_interrupted_writes_and_keeps_the_newest(tiny_tok, tmp_path):
    cfg = _cfg()
    model, ct, feats = _setup(tiny_tok, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    cp = ck.Checkpointer(tmp_path / "ck", every_steps=1, keep=2)
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / ".tmp-step-000001").mkdir()
    for step in (1, 2, 3):
        cp.save(model, opt, sched, ck.LoopState(epoch=0, next_group=step, step=step))
    assert [p.name for p in ck.complete(tmp_path / "ck")] == ["step-000002", "step-000003"]
    assert not list((tmp_path / "ck").glob(".tmp-*"))
    files = {p.name for p in (tmp_path / "ck" / "step-000003").iterdir()}
    assert files == {"trainable.safetensors", "optimizer.pt", "scheduler.pt", "rng.pt",
                     "state.json", ck.MARKER}  # fmt: skip
    st = json.loads((tmp_path / "ck" / "step-000003" / "state.json").read_text())
    assert {"python", "numpy"} <= set(st["rng"])


def test_rng_round_trip():
    import random

    import numpy as np

    from asktrainer.jepa import JepaGate

    gate = JepaGate(0.5, seed=1)
    js, tensors = ck.rng_state(gate)
    a = (random.random(), np.random.rand(), torch.rand(1).item(), gate.apply(), gate.rng.random())
    ck.set_rng_state(json.loads(json.dumps(js)), tensors, gate)  # through JSON, as saved
    b = (random.random(), np.random.rand(), torch.rand(1).item(), gate.apply(), gate.rng.random())
    assert a == b
