"""A faster attention for Spark-X2.x: PyTorch SDPA in place of the remote code's eager matmul.

Spark's `modeling_spark.py` (trust_remote_code) computes attention as an explicit
softmax(QK^T) V in `eager_attention_forward`: the full [batch, heads, seq, seq] matrix in
float32, quadratic memory. `use_sdpa(model)` swaps the module-level function the attention
layers call for one that runs `torch.nn.functional.scaled_dot_product_attention` with the same
additive mask (causal / sliding window, built by the model), the same scale and GQA expansion.
`scripts/probe_model.py` checks the logits agree with eager on this GPU before it's used.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
import torch.nn.functional as F

_ORIGINAL: dict[str, Any] = {}


def _modeling_module(model: torch.nn.Module):
    """The remote code's module (`transformers_modules...modeling_spark`)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    name = type(base).__module__
    mod = sys.modules.get(name)
    if mod is None or not hasattr(mod, "eager_attention_forward"):
        raise RuntimeError(f"{name} has no eager_attention_forward: not a Spark-X2.x model?")
    return mod


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:  # GQA: expand K / V to the query heads (as the eager path does)
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    mask = None
    if attention_mask is not None:
        mask = attention_mask[:, :, :, : key.shape[-2]]
        if mask.dtype != torch.bool and mask.dtype != query.dtype:
            mask = mask.to(query.dtype)
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling, is_causal=False
    )
    return out, None


def use_sdpa(model: torch.nn.Module, on: bool = True) -> None:
    """Switch the model's attention to SDPA (on) or back to the remote code's eager (off)."""
    mod = _modeling_module(model)
    key = mod.__name__
    if key not in _ORIGINAL:
        _ORIGINAL[key] = mod.eager_attention_forward
    mod.eager_attention_forward = sdpa_attention_forward if on else _ORIGINAL[key]
