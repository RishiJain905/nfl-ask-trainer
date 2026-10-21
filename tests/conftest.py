"""Shared fixtures: a tiny byte-level tokenizer with Spark-X2.6's special tokens and its
no-thinking chat layout, tiny models, and a few hand-made examples. CPU only, no network."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# the GPU stays hidden unless the GPU tests were asked for (ASKTRAINER_GPU_TESTS=1 pytest -m gpu)
if os.environ.get("ASKTRAINER_GPU_TESTS") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("WANDB_MODE", "disabled")

from asktrainer.data.schema import Example  # noqa: E402
from asktrainer.settings import get_paths  # noqa: E402

BOS, EOS, PAD = "<｜start▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>"
SPECIAL = ["<|System|>", "<|User|>", "<|Bot|>", "<think>", "</think>"] + [
    f"<||unused{i}||>" for i in range(2, 8)
]
# Spark-X2.6's template, reduced to what single-turn chats use (same strings, same order)
TEMPLATE = (
    "{%- set enable_thinking = enable_thinking | default(true) %}"
    "{{- '" + BOS + "<|System|>' + '\\n' + 'you are a helpful assistant.' }}"
    "{%- if messages[0].role == 'system' %}{{- '\\n\\n' + messages[0].content }}{%- endif %}"
    "{{- '" + EOS + "' }}"
    "{%- for m in messages %}"
    "{%- if m.role == 'user' %}{{- '" + BOS + "<|User|>' + m.content + '" + EOS + "' }}"
    "{%- elif m.role == 'assistant' %}"
    "{{- '" + BOS + "<|Bot|></think>' + m.content + '" + EOS + "' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '" + BOS + "<|Bot|>' }}"
    "{%- if not enable_thinking %}{{- '</think>' }}{%- else %}{{- '<think>' }}{%- endif %}"
    "{%- endif %}"
)


def make_tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tk.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token=BOS, eos_token=EOS,
                                  pad_token=PAD)  # fmt: skip
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    tok.chat_template = TEMPLATE
    return tok


@pytest.fixture(scope="session")
def tiny_tok():
    return make_tiny_tokenizer()


def spark_dir() -> Path | None:
    try:
        d = get_paths().base_model
    except Exception:
        return None
    return d if (d / "tokenizer.json").exists() and (d / "modeling_spark.py").exists() else None


def example(i: int, task: str = "route", group: str = "seen", split: str = "train",
            origin: str = "nfl") -> Example:  # fmt: skip
    q = f"How many touchdowns did player {i} score in {2015 + i % 9}?"
    if task == "route":
        target = json.dumps({"action": "template", "template": f"t{i % 3}", "params": {"n": i}})
    else:
        target = json.dumps({"query": f"SELECT count(*) FROM plays WHERE id = {i}"})
    return Example(
        id=f"{origin}-{task}-{split}-{i}",
        origin=origin,
        task=task,
        split=split,
        group=group,
        messages=[
            {
                "role": "system",
                "content": "You route questions. " + "Schema: plays(id). " * (i % 4),
            },
            {"role": "user", "content": f"Question: {q}"},
            {"role": "assistant", "content": target},
        ],
        jepa={"text": q, "code": target},
        dialect=None if task == "route" else "duckdb",
    )


@pytest.fixture
def examples():
    return [example(i, "route" if i % 2 else "generate") for i in range(8)]
