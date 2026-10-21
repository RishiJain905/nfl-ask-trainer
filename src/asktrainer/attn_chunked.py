"""Spark-X2.6's own eager attention, computed in query chunks for long sequences.

The remote code (`modeling_spark.eager_attention_forward`) materialises the whole
[batch, heads, L, L] score matrix, plus its fp32 softmax, which autograd keeps for backward.
That is fine at 2k tokens but runs out of the GPU's 16 GB around 4-5k tokens, and the NFL
generate prompts reach 5.2k. Softmax is row-wise, so computing it a block of query rows at a
time is the same math, op for op (the same scale, additive mask, max subtraction, fp32
softmax cast back to bf16, dropout, matmul with V); only the GEMMs' row count changes. In
training each chunk is checkpointed, so backward holds one chunk's probabilities at a time
(the chunk's scores are recomputed once more, and only for sequences over `threshold`).
Shorter sequences take the original function unchanged.

Exactness: on the CPU in fp32 the loss and every LoRA gradient match the original
(`tests/test_attn_chunked.py`). On the GPU in bf16 it is not bit-identical (a GEMM with
another row count rounds differently), but it is as close to a higher-precision reference as
the original is: on a real 3,000-token example, relative distance of the last hidden states
to an fp32-attention reference 2.1% (chunked) vs 2.4% (original); CE 0.4347 / 0.4332 vs the
reference's 0.4363 (`tests/test_gpu.py`). The SDPA patch in `attention.py` is a different
kernel altogether (measured slower here, logits up to 4.0 apart).
"""

from __future__ import annotations

import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_ORIGINAL: dict[str, Any] = {}
CHUNK = 1024
THRESHOLD = 3072  # query length above which the chunked path is used


def _repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def _chunk(q, key, value, mask, scaling: float, dropout: float, training: bool):
    w = torch.matmul(q, key.transpose(2, 3)) * scaling
    if mask is not None:
        w = w + mask
    w = w - w.max(dim=-1, keepdim=True).values
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    w = F.dropout(w, p=dropout, training=training)
    return torch.matmul(w, value)


def make_chunked(original, chunk: int = CHUNK, threshold: int = THRESHOLD):
    def chunked_eager_attention_forward(
        module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs
    ):
        # training: chunk only long sequences (short ones keep the original, bit for bit);
        # without grad (val loss, generation prefill over a batch) chunk anything longer than
        # a chunk: nothing is saved for backward, so it only lowers the peak
        limit = threshold if torch.is_grad_enabled() else chunk
        if query.shape[-2] <= limit:
            return original(module, query, key, value, attention_mask=attention_mask,
                            scaling=scaling, dropout=dropout, **kwargs)  # fmt: skip
        n_rep = module.num_key_value_groups
        key, value = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        lk = key.shape[-2]
        training = module.training
        ckpt = torch.is_grad_enabled() and training
        # without grad, keep batch x rows about constant (a batched prefill of 8 prompts uses
        # 256-row chunks): the transient score block stays ~ the size of one 2k-row block
        step = chunk if ckpt else max(128, (2 * chunk) // max(1, query.shape[0]))
        outs = []
        for s in range(0, query.shape[-2], step):
            q = query[:, :, s : s + step]
            m = None if attention_mask is None else attention_mask[:, :, s : s + step, :lk]
            if ckpt:
                o = checkpoint(_chunk, q, key, value, m, scaling, dropout, training,
                               use_reentrant=False)  # fmt: skip
            else:
                o = _chunk(q, key, value, m, scaling, dropout, training)
            outs.append(o)
        return torch.cat(outs, dim=2), None

    chunked_eager_attention_forward.original = original
    return chunked_eager_attention_forward


def _module_of(model) -> Any:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    mod = sys.modules.get(type(base).__module__)
    if mod is None or not hasattr(mod, "eager_attention_forward"):
        raise RuntimeError(f"{type(base).__module__} has no eager_attention_forward")
    return mod


def use_chunked(model, on: bool = True, chunk: int = CHUNK, threshold: int = THRESHOLD) -> None:
    """Switch Spark's attention to the chunked eager path (on) or back to the original."""
    mod = _module_of(model)
    key = mod.__name__
    if key not in _ORIGINAL:
        fn = mod.eager_attention_forward
        _ORIGINAL[key] = getattr(fn, "original", fn)
    original = _ORIGINAL[key]
    mod.eager_attention_forward = make_chunked(original, chunk, threshold) if on else original
