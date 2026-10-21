"""Chunked eager attention == Spark's own eager attention (values and gradients)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from conftest import example
from test_train_smoke import SPARK_TARGETS, tiny_spark

from asktrainer.attn_chunked import _module_of, make_chunked, use_chunked
from asktrainer.chat import ChatTokenizer, collate
from asktrainer.config import LoraSpec
from asktrainer.modelio import apply_lora
from asktrainer.train import forward_losses


def _loss_and_grads(model, batch):
    model.zero_grad(set_to_none=True)
    ce, dist = forward_losses(model, batch)
    loss = ce / batch.n_target + (dist.mean() if dist is not None else 0.0)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


def test_chunked_equals_eager_on_tiny_spark(tiny_tok):
    model = apply_lora(tiny_spark(tiny_tok), LoraSpec(r=4, alpha=8, dropout=0.0,
                                                      targets=SPARK_TARGETS))  # fmt: skip
    model.train()
    with torch.no_grad():  # LoRA B starts at zero: make the adapters do something
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    ct = ChatTokenizer(tiny_tok, jepa=True, k=1)
    feats = [ct.features(example(i)) for i in range(2)]
    for with_views in (False, True):
        batch = collate(feats, ct.pad_id, with_views=with_views)
        use_chunked(model, on=False)
        ref_loss, ref_grads = _loss_and_grads(model, batch)
        use_chunked(model, chunk=7, threshold=16)  # every sequence here is longer than 16
        assert _module_of(model).eager_attention_forward.__name__.startswith("chunked")
        loss, grads = _loss_and_grads(model, batch)
        use_chunked(model, on=False)
        assert abs(loss - ref_loss) < 1e-5
        assert ref_grads.keys() == grads.keys() and ref_grads
        for n in ref_grads:
            assert torch.allclose(grads[n], ref_grads[n], atol=1e-6, rtol=1e-4), n


def test_short_sequences_take_the_original():
    calls = []

    def original(module, q, k, v, attention_mask=None, scaling=None, dropout=0.0, **kw):
        calls.append(q.shape[-2])
        return q, None

    fn = make_chunked(original, chunk=4, threshold=8)
    mod = SimpleNamespace(num_key_value_groups=1, training=False)
    q = torch.randn(1, 2, 8, 4)
    out, _ = fn(mod, q, q, q)
    assert calls == [8] and out is q  # at the threshold: the original, untouched
    out, _ = fn(
        mod, torch.randn(1, 2, 9, 4), q[:, :, :1].expand(1, 2, 9, 4), q[:, :, :1].expand(1, 2, 9, 4)
    )
    assert calls == [8] and out.shape == (1, 2, 9, 4)  # above it: the chunked path
