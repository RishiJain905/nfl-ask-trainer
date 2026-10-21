"""Load the base model fast, add LoRA, save adapters, merge for serving.

Loading: `from_pretrained` memory-maps the safetensors shards, and the 9P bridge WSL uses for
`/mnt/d` is very slow at mmap's random page reads (200 s for 8 GB, measured). A plain
sequential read of a shard runs at ~300 MB/s from F:, so `load_base` builds the model with
its weights on the meta device, reads each shard whole (`Path.read_bytes`), deserialises it with
`safetensors.torch.load`, moves it to the GPU and assigns it, then re-ties the output head
to the embedding (Spark-X2.6 ties them; the shards hold only `model.embedding.weight`).

Attention stays the remote code's eager softmax(QK^T)V (`attn_implementation="eager"`, which
also makes transformers build the float masks that path needs): the SDPA patch in
`attention.py` measured slower for training on this GPU and not numerically identical.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from asktrainer.config import LoraSpec

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
THINKING_DEFAULT = "{%- set enable_thinking = enable_thinking | default(true) %}"
NO_THINKING_DEFAULT = "{%- set enable_thinking = enable_thinking | default(false) %}"


def load_tokenizer(model_dir: Path | str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        return [model_dir / f for f in files]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors weights in {model_dir}")


def load_base(
    model_dir: Path | str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    log=print,
) -> tuple[Any, Any, dict[str, Any]]:
    """(model, tokenizer, info): the base model in `dtype` on `device`, eager attention,
    `use_cache` off (training); `info` has the load time and the tie check."""
    from safetensors.torch import load as st_load
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir = Path(model_dir)
    t0 = time.perf_counter()
    tok = load_tokenizer(model_dir)
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    config.use_cache = False
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):  # weights on meta; buffers (RoPE) real
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, dtype=dtype, attn_implementation="eager"
        )
    model.config._name_or_path = str(model_dir)
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    read_s = 0.0
    nbytes = 0
    for shard in _shards(model_dir):
        t = time.perf_counter()
        raw = shard.read_bytes()
        read_s += time.perf_counter() - t
        nbytes += len(raw)
        part = st_load(raw)
        del raw
        part = {k: v.to(device=device, dtype=dtype) for k, v in part.items()}
        unexpected = set(part) - expected
        if unexpected:
            raise RuntimeError(f"{shard.name}: unexpected weights {sorted(unexpected)[:5]}")
        model.load_state_dict(part, strict=False, assign=True)
        loaded |= set(part)
        del part
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if tied:
        model.tie_weights()
    missing = expected - loaded
    if tied:
        missing -= {"lm_head.weight"}
    if missing:
        raise RuntimeError(f"weights missing from the shards: {sorted(missing)[:5]}")
    model.to(device)  # the buffers (the parameters are already there)
    meta = [n for n, p in [*model.named_parameters(), *model.named_buffers()]
            if p.device.type == "meta"]  # fmt: skip
    if meta:
        raise RuntimeError(f"parameters left on the meta device: {meta[:5]}")
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    if tied and head.data_ptr() != emb.data_ptr():
        raise RuntimeError("the output head is not tied to the embedding after loading")
    model.eval()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    info = {
        "load_s": round(time.perf_counter() - t0, 1),
        "read_s": round(read_s, 1),
        "read_mb_s": round(nbytes / 1e6 / max(read_s, 1e-9)),
        "tied": tied and head.data_ptr() == emb.data_ptr(),
        "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
    }
    log(f"loaded {model_dir.name} in {info['load_s']} s (read {info['read_mb_s']} MB/s)")
    return model, tok, info


def apply_lora(model, spec: LoraSpec, gradient_checkpointing: bool = True):
    """Wrap `model` with PEFT LoRA (adapters in fp32 over the bf16 base: PEFT's
    `autocast_adapter_dtype`), with non-reentrant gradient checkpointing and no KV cache."""
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False  # training; generation passes use_cache=True itself
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        bias="none",
        target_modules=list(spec.targets),
        task_type="CAUSAL_LM",
    )
    pm = get_peft_model(model, cfg, autocast_adapter_dtype=True)
    bad = [n for n, p in pm.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"trainable parameters not in fp32: {bad[:3]}")
    return pm


def trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def backbone_of(model):
    """The decoder stack that returns `last_hidden_state` (Spark2_5Model, LlamaModel, ...)
    under a PEFT wrapper: the CE and JEPA terms are computed from its hidden states."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(base, base.base_model_prefix)


def head_weight(model) -> torch.Tensor:
    """The output projection's weight (Spark-X2.6: the tied embedding)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.get_output_embeddings().weight


def save_adapter(model, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), save_embedding_layers=False)  # embeddings are frozen
    return out


def no_thinking_template(template: str) -> str:
    """Spark's chat template with `enable_thinking` defaulting to false: any server that
    renders it without the flag serves the model the way it was trained (`</think>` opened)."""
    if template.count(THINKING_DEFAULT) != 1:
        raise ValueError("the chat template's enable_thinking default line was not found once")
    return template.replace(THINKING_DEFAULT, NO_THINKING_DEFAULT)


def merge_and_save(
    adapter_dir: Path | str,
    out_dir: Path | str,
    base_dir: Path | str,
    card: dict[str, Any] | None = None,
    device: str = "cuda",
    log=print,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Merge a LoRA adapter into the bf16 base and write a servable model folder: the merged
    weights (safetensors), the remote code files, the tokenizer, a `chat_template.jinja` whose
    `enable_thinking` defaults to false (also in `tokenizer_config.json`), a greedy
    `generation_config.json`, and `ask_model_card.json` (base, run, data version, objective)."""
    from peft import PeftModel

    base_dir, out = Path(base_dir), Path(out_dir)
    model, tok, _ = load_base(base_dir, device=device, dtype=dtype, log=log)
    pm = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = pm.merge_and_unload()
    merged.config.use_cache = True
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="2GB")
    for f in base_dir.glob("*.py"):  # the remote code (trust_remote_code)
        shutil.copy2(f, out / f.name)
    for name in ("LICENSE",):
        if (base_dir / name).exists():
            shutil.copy2(base_dir / name, out / name)
    template = no_thinking_template(tok.chat_template)
    tok.chat_template = template
    tok.save_pretrained(str(out))
    (out / "chat_template.jinja").write_text(template, encoding="utf-8")
    tc_path = out / "tokenizer_config.json"
    tc = json.loads(tc_path.read_text(encoding="utf-8"))
    tc["chat_template"] = template
    tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False), encoding="utf-8")
    gen_path = base_dir / "generation_config.json"
    gen = json.loads(gen_path.read_text(encoding="utf-8")) if gen_path.exists() else {}
    gen.update({"do_sample": False, "temperature": None, "top_p": None})
    gen = {k: v for k, v in gen.items() if v is not None}
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2), encoding="utf-8")
    card = {
        "base": str(base_dir),
        "adapter": str(adapter_dir),
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dtype": str(dtype).removeprefix("torch."),
        "chat_template": "enable_thinking defaults to false",
        **(card or {}),
    }
    (out / "ask_model_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    log(f"merged model written to {out}")
    return out
