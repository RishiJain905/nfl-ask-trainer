"""Turn an `Example` into token ids, labels and the LLM-JEPA views; collate and batch them.

The model sees exactly what it gets at inference: the chat template rendered with
`enable_thinking=False` (Spark-X2.6's no-thinking mode: the bot turn opens `<|Bot|></think>`)
for JSON-reply examples; a `call` example (thinking + one tool call) is rendered with its tools
and its thinking level (`enable_thinking`, `reasoning_effort`), so its prompt ends
`<|Bot|><think>` and the trained turn is `reasoning</think><tool_call>...</tool_call>` plus the
end token (thinking off: the prompt ends `<|Bot|></think>`).
The loss is on the assistant turn only: every token up to and including the generation prompt
is masked (-100), the assistant content and its closing end token (`<｜end▁of▁sentence｜>`)
are trained. The prompt rendering must be a token-level prefix of the full rendering (checked
per example: a template that breaks it would silently train on the wrong tokens).

The two LLM-JEPA views follow the reference implementation (github.com/galilai-group/llm-jepa,
`finetune.py`, `load_and_prepare_dataset`): the **text** view is the question alone as a user
turn, chat-templated, with the k predictor tokens appended to the user content
(`<|predictor_k|>...<|predictor_1|>`, in that order); the **code** view is the target alone as
an assistant turn, chat-templated. The embedding of a view is the last-layer hidden state at
`len(view) + last_token` (the reference's `_last_token_index`; `last_token=-2` here is the
token before the turn's closing `<｜end▁of▁sentence｜>`: the last predictor token of the text
view, the last target token of the code view, as `-2` is for Llama-3's `<|eot_id|>`). Spark's
spare special tokens `<||unused2||>`... are the predictor tokens (no new embeddings needed).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from asktrainer.data.schema import Example

PREDICTOR_TOKEN = "<||unused{n}||>"  # predictor i (1-based) -> <||unused{i+1}||>
MAX_PREDICTORS = 10  # the reference defines <|predictor_1|> ... <|predictor_10|>


def predictor_tokens(k: int) -> list[str]:
    """The k predictor tokens in the order the reference appends them (k, k-1, ..., 1)."""
    if not 0 <= k <= MAX_PREDICTORS:
        raise ValueError(f"k must be in 0..{MAX_PREDICTORS}")
    return [PREDICTOR_TOKEN.format(n=i + 1) for i in range(k, 0, -1)]


@dataclass
class Features:
    """One tokenized example. `input_ids[:prompt_len]` is the prompt (no loss)."""

    id: str
    origin: str
    task: str
    group: str
    input_ids: list[int]
    prompt_len: int
    text_ids: list[int] = field(default_factory=list)
    text_pos: int = -1  # index of the text view's embedding token (within text_ids)
    code_ids: list[int] = field(default_factory=list)
    code_pos: int = -1

    @property
    def n_target(self) -> int:
        return len(self.input_ids) - self.prompt_len

    @property
    def length(self) -> int:
        """Tokens in the packed training row (main sequence + the JEPA views when present)."""
        return len(self.input_ids) + len(self.text_ids) + len(self.code_ids)


class ChatTokenizer:
    """Renders and tokenizes examples for one tokenizer (Spark-X2.6's, or a test one)."""

    def __init__(
        self,
        tokenizer,
        max_len: int = 4096,
        jepa: bool = False,
        k: int = 0,
        last_token: int = -2,
    ):
        self.tok = tokenizer
        self.max_len = max_len
        self.jepa = jepa
        self.k = k
        self.last_token = last_token
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id
        self.predictors = predictor_tokens(k) if jepa else []
        for t in self.predictors:
            ids = self.encode(t)
            if len(ids) != 1:
                raise ValueError(f"predictor token {t!r} is not a single token: {ids}")

    # -- rendering -----------------------------------------------------------------------
    def render(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = False,
        ex: Example | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {"enable_thinking": False}
        if ex is not None and ex.task == "call":
            kwargs["tools"] = ex.tools
            if ex.thinking in ("high", "low"):
                kwargs.update(enable_thinking=True, reasoning_effort=ex.thinking)
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def encode(self, text: str) -> list[int]:
        # the template writes BOS itself; never let the tokenizer add another
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def prompt_ids(self, ex: Example) -> list[int]:
        """The generation prompt (ends `<|Bot|></think>`), as Ask sends it."""
        return self.encode(self.render(ex.prompt_messages, add_generation_prompt=True, ex=ex))

    def full_ids(self, ex: Example) -> tuple[list[int], int]:
        full = self.encode(self.render(ex.chat_messages, ex=ex))
        prompt = self.prompt_ids(ex)
        if full[: len(prompt)] != prompt:
            raise ValueError(f"{ex.id}: the prompt rendering is not a token prefix of the chat")
        if not full or full[-1] != self.eos_id:
            raise ValueError(f"{ex.id}: the chat does not end with the end token")
        return full, len(prompt)

    def views(self, ex: Example) -> tuple[list[int], int, list[int], int]:
        text = ex.jepa.text + "".join(self.predictors)
        text_ids = self.encode(self.render([{"role": "user", "content": text}]))
        code_ids = self.encode(self.render([{"role": "assistant", "content": ex.jepa.code}]))
        tp, cp = len(text_ids) + self.last_token, len(code_ids) + self.last_token
        if not (0 <= tp < len(text_ids) and 0 <= cp < len(code_ids)):
            raise ValueError(f"{ex.id}: last_token {self.last_token} is outside a JEPA view")
        return text_ids, tp, code_ids, cp

    def features(self, ex: Example) -> Features:
        ids, plen = self.full_ids(ex)
        f = Features(ex.id, ex.origin, ex.task, ex.group, ids, plen)
        if self.jepa:
            f.text_ids, f.text_pos, f.code_ids, f.code_pos = self.views(ex)
        return f

    def tokenize_all(self, examples: Iterable[Example]) -> tuple[list[Features], list[str]]:
        """Tokenize; drop (and return the ids of) examples longer than `max_len` (the main
        sequence; the JEPA views are short and packed after it)."""
        kept: list[Features] = []
        dropped: list[str] = []
        for ex in examples:
            f = self.features(ex)
            if len(f.input_ids) > self.max_len:
                dropped.append(ex.id)
            else:
                kept.append(f)
        return kept, dropped


@dataclass
class Batch:
    """A right-padded training batch. Each row is the main sequence, then (JEPA) the text and
    code views packed after it; `segments[b]` lists the row's segment lengths (one segment =
    one independent causal sequence: the packed views never attend to the main sequence)."""

    input_ids: torch.Tensor  # [B, L]
    attention_mask: torch.Tensor  # [B, L] 1 = real token
    position_ids: torch.Tensor  # [B, L] restarting at 0 per segment
    target_index: torch.Tensor  # [N, 2] (row, position) whose hidden state predicts a target
    target_ids: torch.Tensor  # [N]
    segments: list[list[int]]
    text_index: torch.Tensor | None = None  # [B, 2] (row, position) of the text embedding
    code_index: torch.Tensor | None = None
    ids: list[str] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return int(self.attention_mask.sum())

    @property
    def n_target(self) -> int:
        return int(self.target_ids.numel())

    @property
    def packed(self) -> bool:
        return self.text_index is not None

    def to(self, device) -> Batch:
        def mv(t):
            return None if t is None else t.to(device, non_blocking=True)

        return Batch(
            mv(self.input_ids),
            mv(self.attention_mask),
            mv(self.position_ids),
            mv(self.target_index),
            mv(self.target_ids),
            self.segments,
            mv(self.text_index),
            mv(self.code_index),
            self.ids,
        )


def collate(
    feats: Sequence[Features], pad_id: int, with_views: bool = False, pad_to: int = 0
) -> Batch:
    """Right-pad a list of features into a `Batch`. With `with_views`, each row also carries
    its text and code views packed after the main sequence (the reference's additive-mask
    trick, done in-row: one forward pass, the views cost only their own few tokens). `pad_to`
    rounds the padded length up to a multiple (length buckets)."""
    rows, segs, tgt_idx, tgt_ids, text_idx, code_idx = [], [], [], [], [], []
    for b, f in enumerate(feats):
        row = list(f.input_ids)
        seg = [len(row)]
        for t in range(f.prompt_len - 1, len(f.input_ids) - 1):  # hidden t predicts token t+1
            tgt_idx.append((b, t))
            tgt_ids.append(f.input_ids[t + 1])
        if with_views:
            if not f.text_ids:
                raise ValueError(f"{f.id}: no JEPA views (tokenize with jepa=True)")
            text_idx.append((b, len(row) + f.text_pos))
            row += f.text_ids
            code_idx.append((b, len(row) + f.code_pos))
            row += f.code_ids
            seg += [len(f.text_ids), len(f.code_ids)]
        rows.append(row)
        segs.append(seg)
    L = max(len(r) for r in rows)
    if pad_to > 1:  # a few distinct shapes (TunableOp tunes each GEMM shape once)
        L = -(-L // pad_to) * pad_to
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    pos = torch.zeros((len(rows), L), dtype=torch.long)
    for b, (r, seg) in enumerate(zip(rows, segs, strict=True)):
        ids[b, : len(r)] = torch.tensor(r)
        att[b, : len(r)] = 1
        start = 0
        for n in seg:
            pos[b, start : start + n] = torch.arange(n)
            start += n
    return Batch(
        input_ids=ids,
        attention_mask=att,
        position_ids=pos,
        target_index=torch.tensor(tgt_idx, dtype=torch.long).reshape(-1, 2),
        target_ids=torch.tensor(tgt_ids, dtype=torch.long),
        segments=segs,
        text_index=torch.tensor(text_idx, dtype=torch.long) if with_views else None,
        code_index=torch.tensor(code_idx, dtype=torch.long) if with_views else None,
        ids=[f.id for f in feats],
    )


def left_pad(prompts: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad prompts for batched generation: (input_ids, attention_mask)."""
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(prompts), L), dtype=torch.long)
    for b, p in enumerate(prompts):
        if p:
            ids[b, L - len(p) :] = torch.tensor(list(p))
            att[b, L - len(p) :] = 1
    return ids, att


def length_grouped_batches(
    lengths: Sequence[int], batch_size: int, seed: int, mega: int = 64
) -> list[list[int]]:
    """Batches of similar lengths in a seeded random order (HF's `LengthGroupedSampler` idea):
    shuffle, cut into mega-batches of `mega * batch_size`, sort each by length (longest
    first), split into batches, then shuffle the batch order (the longest batch goes first,
    so an out-of-memory shows up on step one, not an hour in)."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    size = max(1, mega * batch_size)
    batches: list[list[int]] = []
    for s in range(0, len(idx), size):
        chunk = sorted(idx[s : s + size], key=lambda i: -lengths[i])
        batches += [chunk[j : j + batch_size] for j in range(0, len(chunk), batch_size)]
    if not batches:
        return []
    longest = max(range(len(batches)), key=lambda j: max(lengths[i] for i in batches[j]))
    first = batches.pop(longest)
    rng.shuffle(batches)
    return [first, *batches]
