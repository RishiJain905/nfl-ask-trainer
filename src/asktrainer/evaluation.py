"""Per-epoch evaluation: the validation loss, batched greedy generation on a fixed stratified
subset of the NFL val examples, and the NFL repo's verifier (`nfl ask verify`, run on the
Windows side through WSL interop) that scores the generations by executing them.

Generation is batched with left padding. That is exact for Spark-X2.6: its RoPE angles come
from the cache positions (pads included), so every real token keeps its relative offsets, and
the attention mask hides the pads (`tests/test_train_smoke.py` checks batched == one at a time
on a small model; `tests/test_gpu.py` on the real one).
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from asktrainer.chat import ChatTokenizer, left_pad
from asktrainer.data.schema import Example
from asktrainer.settings import Paths, to_windows


def stratified_subset(
    examples: Sequence[Example], n: int, seed: int, key=lambda e: (e.group, e.task)
) -> list[Example]:
    """A fixed subset of `n` examples with every stratum (`key`: by default (group, task))
    represented: one each first, the rest proportional to stratum size (largest remainders),
    seeded. `n < 0` = all, `n == 0` = none."""
    if n == 0:
        return []
    if n < 0:
        n = len(examples)
    strata: dict[tuple, list[Example]] = defaultdict(list)
    for e in sorted(examples, key=lambda e: e.id):
        strata[key(e)].append(e)
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    total = sum(len(v) for v in strata.values())
    if n >= total:
        return [e for k in keys for e in strata[k]]
    take = {k: min(1, len(strata[k])) for k in keys}
    left = n - sum(take.values())
    if left < 0:  # fewer slots than strata: the largest strata keep theirs
        keep = sorted(keys, key=lambda k: -len(strata[k]))[:n]
        take = {k: (1 if k in keep else 0) for k in keys}
        left = 0
    room = {k: len(strata[k]) - take[k] for k in keys}
    pool = sum(room.values())
    if left and pool:
        quota = {k: left * room[k] / pool for k in keys}
        extra = {k: min(room[k], math.floor(quota[k])) for k in keys}
        rest = left - sum(extra.values())
        for k in sorted(keys, key=lambda k: -(quota[k] - math.floor(quota[k]))):
            if rest <= 0:
                break
            if extra[k] < room[k]:
                extra[k] += 1
                rest -= 1
        take = {k: take[k] + extra[k] for k in keys}
    return [e for k in keys for e in strata[k][: take[k]]]


@torch.no_grad()
def generate(
    model,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    pad_id: int,
    eos_id: int,
    device,
    batch_size: int = 8,
    prefill_tokens: int = 8192,
    on_oom=None,
) -> list[list[int]]:
    """Greedy continuations (token ids, the end token excluded) for each prompt, in order.
    Prompts are sorted by length and batched (at most `batch_size` rows and
    `prefill_tokens` padded prompt tokens per batch), left-padded."""
    from transformers import GenerationConfig

    gc = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=True,
    )
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out: list[list[int] | None] = [None] * len(prompts)

    def run(idx: list[int]) -> None:
        ids, att = left_pad([prompts[j] for j in idx], pad_id)
        try:
            # use_cache passed explicitly: transformers replaces a GenerationConfig field left
            # at its default with the model's own value, and training sets use_cache = False
            gen = model.generate(
                input_ids=ids.to(device),
                attention_mask=att.to(device),
                generation_config=gc,
                use_cache=True,
            )
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            del ids, att
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if on_oom is not None:
                on_oom(len(idx))
            half = len(idx) // 2  # too big a batch for what's free: split it and go on
            run(idx[:half])
            run(idx[half:])
            return
        new = gen[:, ids.shape[1] :].tolist()
        for j, row in zip(idx, new, strict=True):
            if eos_id in row:
                row = row[: row.index(eos_id)]
            out[j] = [t for t in row if t != pad_id]

    i = 0
    while i < len(order):
        L = len(prompts[order[i]])
        bs = max(1, min(batch_size, prefill_tokens // max(L, 1)))
        run(order[i : i + bs])
        i += bs
    return [o or [] for o in out]


def predict(
    model, ct: ChatTokenizer, examples: Sequence[Example], max_new_tokens: dict[str, int],
    device, batch_size: int = 8, prefill_tokens: int = 8192,
    stats: dict[str, float] | None = None, on_oom=None,
) -> list[dict[str, str]]:  # fmt: skip
    """`{"id", "output"}` for each example (the assistant text only), generated per task with
    that task's `max_new_tokens`. `stats` (when given) gets per-task seconds and tokens."""
    was_training = model.training
    model.eval()
    by_task: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_task[e.task].append(e)
    texts: dict[str, str] = {}
    try:
        for task, exs in sorted(by_task.items()):
            prompts = [ct.prompt_ids(e) for e in exs]
            t0 = time.perf_counter()
            outs = generate(
                model, prompts, max_new_tokens.get(task, 256), ct.pad_id, ct.eos_id, device,
                batch_size=batch_size, prefill_tokens=prefill_tokens, on_oom=on_oom,
            )  # fmt: skip
            if stats is not None:
                stats[f"val/generate_s/{task}"] = time.perf_counter() - t0
                stats[f"val/generated_tokens/{task}"] = float(sum(len(o) for o in outs))
            for e, o in zip(exs, outs, strict=True):
                texts[e.id] = ct.tok.decode(o, skip_special_tokens=True).strip()
    finally:
        model.train(was_training)
    return [{"id": e.id, "output": texts[e.id]} for e in examples]


def write_preds(path: Path, preds: Sequence[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def verifier_command(paths: Paths, data: Path, preds: Path, out: Path) -> list[str]:
    return [
        str(paths.uv_exe), "run", "--project", to_windows(paths.nfl_repo), "nfl", "ask", "verify",
        "--data", to_windows(data), "--preds", to_windows(preds), "--out", to_windows(out),
    ]  # fmt: skip


def run_verifier(
    paths: Paths, data: Path, preds: Path, out: Path, timeout_s: int = 1800,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:  # fmt: skip
    """Run `nfl ask verify` and return its JSON (`summary`, `items`, ...). Raises on failure
    (the caller logs it and carries on). Its stdout / stderr go to `<out>.log`."""
    cmd = verifier_command(paths, data, preds, out)
    t0 = time.perf_counter()
    proc = runner(cmd, cwd=str(paths.nfl_repo), capture_output=True, text=True,
                  timeout=timeout_s)  # fmt: skip
    log = out.with_suffix(".log")
    log.write_text(
        f"$ {' '.join(cmd)}\nexit {proc.returncode} in {time.perf_counter() - t0:.0f} s\n"
        f"--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nfl ask verify exited {proc.returncode} (see {log})")
    if not out.exists():
        raise RuntimeError(f"nfl ask verify wrote no scores (see {log})")
    return json.loads(out.read_text(encoding="utf-8"))


METRICS = (
    "exec_acc",
    "exec_acc_nonempty",
    "json_ok",
    "action_acc",
    "template_acc",
    "params_exact",
)


def score_metrics(scores: dict[str, Any]) -> dict[str, float]:
    """The verifier's summary as flat W&B keys: val/<metric> (all), val/<task>/<metric>,
    val/<group>/exec_acc."""
    s = scores.get("summary", scores)
    out: dict[str, float] = {}
    for m in METRICS:
        v = (s.get("all") or {}).get(m)
        if v is not None:
            out[f"val/{m}"] = float(v)
    for task, block in (s.get("task") or {}).items():
        for m in METRICS:
            if block.get(m) is not None:
                out[f"val/task/{task}/{m}"] = float(block[m])
    for group, block in (s.get("group") or {}).items():
        for m in ("exec_acc", "exec_acc_nonempty"):
            if block.get(m) is not None:
                out[f"val/{group}/{m}"] = float(block[m])
    if s.get("all", {}).get("n") is not None:
        out["val/verified_n"] = float(s["all"]["n"])
    return out
