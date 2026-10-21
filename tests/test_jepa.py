from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from asktrainer.jepa import (
    JepaGate,
    block_causal_mask,
    jepa_distance,
    jepa_embeddings,
    packed_attention_mask,
)


def test_cosine_distance():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, 0.0], [0.0, 5.0], [-1.0, 0.0], [3.0, 4.0]])
    d = jepa_distance(a, b, "cosine")
    assert torch.allclose(d, torch.tensor([0.0, 1.0, 2.0, 0.0]), atol=1e-6)


def test_l2_and_mse():
    a = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    b = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    assert torch.allclose(jepa_distance(a, b, "l2"), torch.tensor([5.0, 0.0]))
    assert torch.allclose(jepa_distance(a, b, "mse"), torch.tensor([12.5, 0.0]))
    with pytest.raises(ValueError):
        jepa_distance(a, b, "infonce")


def test_distance_is_fp32_and_differentiable_through_both_views():
    a = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    d = jepa_distance(a, b)
    assert d.dtype == torch.float32
    d.sum().backward()
    assert a.grad is not None and b.grad is not None  # no stop-gradient (as the reference)


def test_embeddings_gather():
    h = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    pred, code = jepa_embeddings(h, torch.tensor([[0, 1], [1, 4]]), torch.tensor([[0, 3], [1, 0]]))
    assert torch.equal(pred, torch.stack([h[0, 1], h[1, 4]]))
    assert torch.equal(code, torch.stack([h[0, 3], h[1, 0]]))


def test_gate_ratio():
    g = JepaGate(0.25, seed=3)
    hits = sum(g.apply() for _ in range(4000))
    assert abs(hits / 4000 - 0.25) < 0.03
    assert all(JepaGate(1.0).apply() for _ in range(50))
    assert all(JepaGate(-1.0).apply() for _ in range(50))  # the reference's "off" value
    a = [JepaGate(0.5, seed=9).apply() for _ in range(1)]
    assert a == [JepaGate(0.5, seed=9).apply()]


def test_block_causal_mask_segments_and_padding():
    m = block_causal_mask([[3, 2], [2]], length=6, dtype=torch.float32)
    allowed = m[:, 0] == 0
    expect0 = torch.tensor([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],  # the second segment never sees the first
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],  # padding sees only itself
    ], dtype=torch.bool)  # fmt: skip
    assert torch.equal(allowed[0], expect0)
    assert torch.equal(allowed[1][:2, :2], torch.tensor([[1, 0], [1, 1]], dtype=torch.bool))
    assert allowed.any(-1).all()  # no fully masked row
    assert m.min() == torch.finfo(torch.float32).min


def test_block_causal_mask_sliding_window():
    m = block_causal_mask([[6]], length=6, dtype=torch.float32, sliding_window=3)
    allowed = (m[0, 0] == 0).int()
    # a query sees itself and the 2 keys before it (transformers: kv > q - window)
    assert allowed[5].tolist() == [0, 0, 0, 1, 1, 1]
    assert allowed[1].tolist() == [1, 1, 0, 0, 0, 0]


def test_packed_mask_kinds():
    sparkish = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"], sliding_window=4
    )
    llamaish = SimpleNamespace(layer_types=None, sliding_window=None)
    d = packed_attention_mask(sparkish, [[5, 3]], 8, torch.float32, "cpu")
    assert set(d) == {"full_attention", "sliding_attention"}
    assert not torch.equal(d["full_attention"], d["sliding_attention"])
    t = packed_attention_mask(llamaish, [[5, 3]], 8, torch.float32, "cpu")
    assert isinstance(t, torch.Tensor) and t.shape == (1, 1, 8, 8)


def test_cosine_matches_reference_formula():
    torch.manual_seed(0)
    u, a = torch.randn(4, 16), torch.randn(4, 16)
    ref = 1.0 - torch.mean(torch.nn.functional.cosine_similarity(u, a, dim=-1))
    assert math.isclose(float(jepa_distance(u, a).mean()), float(ref), rel_tol=1e-6)
