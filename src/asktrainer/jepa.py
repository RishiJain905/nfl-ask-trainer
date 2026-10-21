"""The LLM-JEPA term (Huang, LeCun, Balestriero, "LLM-JEPA: Large Language Models Meet Joint
Embedding Predictive Architectures", arXiv 2509.14252; reference code
github.com/galilai-group/llm-jepa, Apache-2.0, `finetune.py` `RepresentationTrainer`).

What we follow from the reference, and how:

- **The objective** (paper Eq. 2 / 3): `loss = gamma * CE + lambda * d(Pred(Enc(Text)), Enc(Code))`
  with gamma = 1 (the reference's `--gamma` default).
- **The embedding** `Enc(x)`: the last layer's hidden state (after the final norm: the
  reference reads `outputs.hidden_states[-1]`) at one token of the view, index
  `len(view) + last_token` (`_last_token_index`). See `chat.py` for the index we use.
- **The predictor**: the LLM itself, with k predictor tokens appended to the text view
  (`--predictors`); `Pred(Enc(Text))` is the embedding at the last predictor token. k = 0
  means Pred is the identity. There are no extra weights.
- **The distance** (`--jepa_l2`, `--jepa_mse`, default cosine): cosine `1 - mean(cos)`,
  l2 `mean ||p - c||_2`, mse `mean((p - c)^2)`. Computed in fp32 here.
- **Gradients flow through both views**: the reference never detaches either embedding
  (no stop-gradient), so neither do we.
- **Random JEPA-loss dropout** (`--jepa_ratio`, paper "LD = 1 - ratio"): per micro-batch, with
  probability `1 - ratio` the views are not run at all and the step is CE only. As in the
  reference the term is not re-weighted on the steps that keep it (the paper suggests holding
  `lambda * ratio` roughly constant when you lower the ratio).
- **One forward pass** (`--additive_mask`): the reference packs the two views into one extra
  batch row with a block-diagonal causal mask, so the views cannot attend to each other. We pack
  them after the main sequence *in the same row* (see `chat.collate`), with a block-diagonal
  mask over the three segments, so a JEPA step costs only the views' few extra tokens instead
  of a second full-length row. The main sequence's attention is unchanged (a test pins that
  the CE is identical to the plain path).
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

DISTANCES = ("cosine", "l2", "mse")


def jepa_distance(
    pred: torch.Tensor, target: torch.Tensor, distance: str = "cosine"
) -> torch.Tensor:
    """Per-example distance [B] between predicted and target embeddings ([B, H]), in fp32."""
    p, t = pred.float(), target.float()
    if distance == "cosine":
        return 1.0 - F.cosine_similarity(p, t, dim=-1)
    if distance == "l2":
        return torch.linalg.norm(p - t, ord=2, dim=-1)
    if distance == "mse":
        return ((p - t) ** 2).mean(dim=-1)
    raise ValueError(f"unknown JEPA distance {distance!r} (one of {DISTANCES})")


def jepa_embeddings(
    hidden: torch.Tensor, text_index: torch.Tensor, code_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather `Pred(Enc(Text))` and `Enc(Code)` ([B, H]) from the last hidden states
    ([B, L, H]) at the (row, position) indices from `chat.collate`."""
    pred = hidden[text_index[:, 0], text_index[:, 1]]
    code = hidden[code_index[:, 0], code_index[:, 1]]
    return pred, code


class JepaGate:
    """Random JEPA-loss dropout: `apply()` is True with probability `ratio` (seeded).
    `ratio >= 1` (or <= 0, the reference's "off" value -1) always applies."""

    def __init__(self, ratio: float = 1.0, seed: int = 0):
        self.ratio = ratio
        self.rng = random.Random(seed)

    def apply(self) -> bool:
        if self.ratio >= 1.0 or self.ratio <= 0.0:
            return True
        return self.rng.random() < self.ratio


def block_causal_mask(
    segments: list[list[int]],
    length: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
    sliding_window: int | None = None,
) -> torch.Tensor:
    """The additive attention mask [B, 1, L, L] for rows of packed segments: causal inside
    each segment, nothing across segments (the reference's `_build_additive_mask`, one block per
    view), optionally a sliding window (a query sees the `sliding_window` most recent keys,
    itself included: transformers' `kv_idx > q_idx - sliding_window`). Padding positions only
    see themselves (never a fully masked row: no NaN in the softmax)."""
    B = len(segments)
    seg_id = torch.full((B, length), -1, dtype=torch.long)
    for b, seg in enumerate(segments):
        start = 0
        for j, n in enumerate(seg):
            seg_id[b, start : start + n] = j
            start += n
    seg_id = seg_id.to(device)
    q = torch.arange(length, device=device).view(1, length, 1)
    k = torch.arange(length, device=device).view(1, 1, length)
    same = (seg_id.unsqueeze(2) == seg_id.unsqueeze(1)) & (seg_id.unsqueeze(2) >= 0)
    allowed = same & (k <= q)
    if sliding_window is not None:
        allowed &= k > q - sliding_window
    allowed |= (q == k).expand(B, -1, -1)  # padding rows attend to themselves
    mask = torch.zeros((B, 1, length, length), dtype=dtype, device=device)
    mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return mask


def packed_attention_mask(model_config, segments, length, dtype, device):
    """The mask(s) the model's forward takes for packed rows: a dict
    {"full_attention", "sliding_attention"} for models with sliding-window layers (Spark-X2.6
    accepts a prepared mapping and skips its own mask creation), else one 4D tensor (which
    transformers' `create_causal_mask` returns as-is)."""
    full = block_causal_mask(segments, length, dtype, device)
    layer_types = getattr(model_config, "layer_types", None) or []
    window = getattr(model_config, "sliding_window", None)
    if "sliding_attention" in layer_types and window:
        sliding = block_causal_mask(segments, length, dtype, device, sliding_window=window)
        return {"full_attention": full, "sliding_attention": sliding}
    return full
