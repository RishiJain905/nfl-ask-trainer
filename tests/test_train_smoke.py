"""The real training loop on tiny random models (CPU): plain and JEPA, the packed-row JEPA
mask, and batched left-padded generation. A tiny Spark-X2.6 (its own remote code, sliding
window on) runs too when the model folder is there."""

from __future__ import annotations

import pytest
import torch
from conftest import example, spark_dir

from asktrainer.chat import ChatTokenizer, collate
from asktrainer.config import JepaSpec, LoraSpec, OptimConfig, TrainConfig
from asktrainer.evaluation import generate
from asktrainer.modelio import apply_lora, backbone_of
from asktrainer.train import Logger, forward_losses, run_loop, validation_loss

LLAMA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SPARK_TARGETS = ["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_llama(tok):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id, tie_word_embeddings=True, attn_implementation="eager",
    )  # fmt: skip
    return LlamaForCausalLM(cfg).eval()


def tiny_spark(tok):
    d = spark_dir()
    if d is None:
        pytest.skip("the Spark-X2.6 model folder (its remote code) is not available")
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(str(d), trust_remote_code=True)
    cfg.update(dict(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["sliding_attention"] * 3 + ["full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, sliding_window=8, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=True,
    ))  # fmt: skip
    m = AutoModelForCausalLM.from_config(
        cfg, trust_remote_code=True, attn_implementation="eager", dtype=torch.float32
    )
    m.tie_weights()
    return m.eval()


MODELS = {"llama": (tiny_llama, LLAMA_TARGETS), "spark": (tiny_spark, SPARK_TARGETS)}


def _cfg(targets, jepa: bool, epochs: int = 4) -> TrainConfig:
    return TrainConfig(
        name="t",
        seed=1,
        lora=LoraSpec(r=8, alpha=16, dropout=0.0, targets=targets),
        optim=OptimConfig(
            lr=1e-2, epochs=epochs, batch=2, grad_accum=2, warmup=0, max_len=4096
        ),  # fmt: skip
        jepa=JepaSpec(enabled=jepa, lam=1.0, k=1),
    )


@pytest.mark.parametrize("arch", ["llama", "spark"])
def test_packed_views_leave_the_main_sequence_alone(tiny_tok, arch):
    make, _ = MODELS[arch]
    model = make(tiny_tok)
    ct = ChatTokenizer(tiny_tok, jepa=True, k=2)
    feats = [ct.features(example(i, "route" if i % 2 else "generate")) for i in range(3)]
    with torch.no_grad():
        ce_plain, d0 = forward_losses(model, collate(feats, ct.pad_id))
        ce_packed, dist = forward_losses(model, collate(feats, ct.pad_id, with_views=True))
    assert d0 is None and dist.shape == (3,)
    assert torch.allclose(ce_plain, ce_packed, rtol=1e-5)
    # a packed view's embedding equals the view run on its own (the blocks are isolated)
    bb = backbone_of(model)
    with torch.no_grad():
        h = bb(input_ids=torch.tensor([feats[0].text_ids]), use_cache=False).last_hidden_state
        packed = bb(
            input_ids=collate(feats[:1], ct.pad_id, with_views=True).input_ids,
            attention_mask=_packed_mask(model, feats[:1], ct),
            position_ids=collate(feats[:1], ct.pad_id, with_views=True).position_ids,
            use_cache=False,
        ).last_hidden_state
    n = len(feats[0].input_ids)
    assert torch.allclose(h[0, feats[0].text_pos], packed[0, n + feats[0].text_pos], atol=1e-4)


def _packed_mask(model, feats, ct):
    from asktrainer.jepa import packed_attention_mask

    b = collate(feats, ct.pad_id, with_views=True)
    return packed_attention_mask(backbone_of(model).config, b.segments, b.input_ids.shape[1],
                                 torch.float32, "cpu")  # fmt: skip


@pytest.mark.parametrize("arch", ["llama", "spark"])
@pytest.mark.parametrize("jepa", [False, True])
def test_loop_learns_and_saves(tiny_tok, tmp_path, arch, jepa):
    make, targets = MODELS[arch]
    cfg = _cfg(targets, jepa)
    model = apply_lora(make(tiny_tok), cfg.lora, gradient_checkpointing=True)
    ct = ChatTokenizer(tiny_tok, cfg.optim.max_len, jepa=jepa, k=cfg.jepa.k)
    feats, dropped = ct.tokenize_all([example(i, "route" if i % 2 else "generate")
                                      for i in range(8)])  # fmt: skip
    assert not dropped
    before = validation_loss(model, feats, ct, cfg, "cpu")
    calls = []
    logger = Logger(tmp_path / "log.jsonl", echo=None)
    res = run_loop(model, ct, feats, cfg, tmp_path, logger, "cpu",
                   on_epoch_end=lambda e, s: calls.append(e) or {"val/x": 1.0})  # fmt: skip
    after = validation_loss(model, feats, ct, cfg, "cpu")
    assert after["val/loss"] < before["val/loss"] - (0.03 if jepa else 0.1)
    assert res.steps == 8 and calls == [1, 2, 3, 4]
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert (tmp_path / "adapter_epoch4" / "adapter_config.json").exists()
    train_rows = [r for r in logger.history if "train/ce" in r]
    assert len(train_rows) == 8 and all("train/tokens_per_s" in r for r in train_rows)
    assert train_rows[-1]["train/ce"] < train_rows[0]["train/ce"]
    if jepa:
        assert all("train/jepa" in r for r in train_rows)
        assert after["val/jepa"] < before["val/jepa"]  # the term is being minimised
        assert train_rows[-1]["train/loss"] > train_rows[-1]["train/ce"]
    else:
        assert all(r["train/loss"] == r["train/ce"] for r in train_rows)


def test_validation_loss_splits_by_thinking_mode(tiny_tok):
    cfg = _cfg(LLAMA_TARGETS, False)
    model = tiny_llama(tiny_tok)
    ct = ChatTokenizer(tiny_tok, cfg.optim.max_len)
    feats, _ = ct.tokenize_all([example(i, "generate") for i in range(6)])
    assert not any(k.startswith("val/loss/thinking") for k in validation_loss(
        model, feats, ct, cfg, "cpu"))  # fmt: skip
    for i, f in enumerate(feats):  # as call examples at these levels
        f.thinking = ("high", "off", "low")[i % 3]
    out = validation_loss(model, feats, ct, cfg, "cpu")
    n_on = sum(f.n_target for f in feats if f.thinking != "off")
    n_off = sum(f.n_target for f in feats if f.thinking == "off")
    both = (out["val/loss/thinking_on"] * n_on + out["val/loss/thinking_off"] * n_off) / (
        n_on + n_off
    )
    assert abs(both - out["val/loss"]) < 1e-5


def test_jepa_ratio_drops_views(tiny_tok, tmp_path):
    cfg = _cfg(LLAMA_TARGETS, True, epochs=2)
    cfg.jepa.ratio = 0.5
    model = apply_lora(tiny_llama(tiny_tok), cfg.lora)
    ct = ChatTokenizer(tiny_tok, jepa=True, k=1)
    feats, _ = ct.tokenize_all([example(i) for i in range(8)])
    logger = Logger(None, echo=None)
    run_loop(model, ct, feats, cfg, tmp_path, logger, "cpu")
    shares = [r["train/jepa_on"] for r in logger.history if "train/jepa_on" in r]
    assert 0.0 < sum(shares) / len(shares) < 1.0


def test_eval_failure_does_not_stop_training(tiny_tok, tmp_path):
    cfg = _cfg(LLAMA_TARGETS, False, epochs=2)
    model = apply_lora(tiny_llama(tiny_tok), cfg.lora)
    ct = ChatTokenizer(tiny_tok)
    feats, _ = ct.tokenize_all([example(i) for i in range(4)])

    def boom(epoch, step):
        raise RuntimeError("verifier down")

    logger = Logger(None, echo=None)
    res = run_loop(model, ct, feats, cfg, tmp_path, logger, "cpu", on_epoch_end=boom)
    assert res.steps == 2
    assert sum(1 for r in logger.history if r.get("val/eval_failed") == 1.0) == 2


@pytest.mark.parametrize("arch", ["llama", "spark"])
def test_batched_left_padded_generation_equals_single(tiny_tok, arch):
    make, targets = MODELS[arch]
    model = apply_lora(make(tiny_tok), LoraSpec(r=4, alpha=8, dropout=0.0, targets=targets))
    model.eval()
    ct = ChatTokenizer(tiny_tok)
    prompts = [ct.prompt_ids(example(i)) for i in (0, 1, 2, 3)]
    assert len({len(p) for p in prompts}) > 1  # different lengths: real left padding
    batched = generate(model, prompts, 6, ct.pad_id, ct.eos_id, "cpu", batch_size=4)
    single = [generate(model, [p], 6, ct.pad_id, ct.eos_id, "cpu", batch_size=1)[0]
              for p in prompts]  # fmt: skip
    assert batched == single
    assert sum(len(b) for b in batched) > 0


def test_generation_uses_the_kv_cache(tiny_tok):
    """After the prefill, each step feeds one token (training turns config.use_cache off)."""
    model = apply_lora(tiny_llama(tiny_tok), LoraSpec(r=4, alpha=8, dropout=0.0,
                                                      targets=LLAMA_TARGETS))  # fmt: skip
    model.eval()
    assert model.config.use_cache is False
    seen = []
    hook = backbone_of(model).register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append(kwargs["input_ids"].shape[1]), with_kwargs=True
    )
    ct = ChatTokenizer(tiny_tok)
    p = ct.prompt_ids(example(0))
    try:
        model.generation_config.eos_token_id = None  # never stop early
        generate(model, [p], 5, ct.pad_id, None, "cpu")
    finally:
        hook.remove()
    assert seen[0] == len(p) and seen[1:] == [1] * (len(seen) - 1) and len(seen) == 5


def test_generation_splits_a_batch_that_runs_out_of_memory(tiny_tok):
    model = tiny_llama(tiny_tok)
    real = model.generate
    sizes = []

    def flaky(*a, **k):
        sizes.append(k["input_ids"].shape[0])
        if k["input_ids"].shape[0] > 1:
            raise torch.OutOfMemoryError("fake")
        return real(*a, **k)

    model.generate = flaky
    ct = ChatTokenizer(tiny_tok)
    prompts = [ct.prompt_ids(example(i)) for i in range(3)]
    out = generate(model, prompts, 3, ct.pad_id, ct.eos_id, "cpu", batch_size=3)
    assert sizes[0] == 3 and len(out) == 3 and sizes.count(1) == 3


def test_the_gpu_memory_cap_is_off_unless_asked_and_never_on_the_cpu(monkeypatch):
    from asktrainer import train

    monkeypatch.delenv(train.GPU_FRACTION_VAR, raising=False)
    assert train.limit_cuda("cpu") is None
    monkeypatch.setenv(train.GPU_FRACTION_VAR, "0.8")
    assert train.limit_cuda("cpu") is None  # a cap on the card only
