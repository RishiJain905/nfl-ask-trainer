"""The training loop: LoRA SFT on the Ask examples, plain or with the LLM-JEPA term.

Why our own loop rather than a `transformers.Trainer` subclass: every part the Trainer would
give us is one we'd have to override or switch off. The loss is computed from the decoder's
hidden states at the target positions only (never full-vocabulary logits over a 2k-token
prompt), the JEPA views are packed into the same row with a custom block mask, batches are
length-grouped our way, the per-epoch evaluation generates in batches and shells out to the
NFL verifier, and W&B gets our tags and a config checked for secret-named keys (the Trainer's
W&B callback would log its own `TrainingArguments` dump). What remains (AdamW, cosine with
warmup, grad accumulation, clipping) is a few lines, and the loop runs unchanged on a tiny
CPU model in the tests.

Loss normalisation over one optimizer step (all `grad_accum` micro-batches together, as if
they were one big batch): CE = sum of target-token CE / target tokens in the step; JEPA =
lambda * sum of per-example distances / examples in the step (the reference's
`gamma * lm_loss + lbd * jepa_loss`, with HF's `/ grad_accum` averaging; micro-batches that
dropped the term, `jepa.ratio < 1`, contribute zero, as in the reference).
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from asktrainer.chat import Batch, ChatTokenizer, Features, collate, length_grouped_batches
from asktrainer.checkpoint import Checkpointer, LoopState
from asktrainer.checkpoint import load as load_checkpoint
from asktrainer.config import TrainConfig, assert_no_secrets
from asktrainer.events import NULL as NO_EVENTS
from asktrainer.events import Events, utc_now, write_run_info
from asktrainer.jepa import JepaGate, jepa_distance, jepa_embeddings, packed_attention_mask
from asktrainer.modelio import backbone_of, head_weight

CE_CHUNK = 1024  # target tokens per CE chunk (logits recomputed in backward: low memory)


# -- losses ------------------------------------------------------------------------------
def _ce_chunk(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")


def target_ce_sum(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of CE over target positions: `h` [N, H] hidden states, `w` [V, H] the head, `y`
    [N] targets. fp32 logits, in chunks; under grad each chunk is checkpointed so only one
    chunk's logits live at a time."""
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], CE_CHUNK):
        hs, ys = h[s : s + CE_CHUNK], y[s : s + CE_CHUNK]
        if torch.is_grad_enabled():
            total = total + checkpoint(_ce_chunk, hs, w, ys, use_reentrant=False)
        else:
            total = total + _ce_chunk(hs, w, ys)
    return total


def forward_losses(
    model, batch: Batch, distance: str = "cosine"
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(sum of target CE, per-example JEPA distances [B] or None) for one batch, from one
    forward pass of the decoder."""
    backbone = backbone_of(model)
    w = head_weight(model)
    if batch.packed:
        mask = packed_attention_mask(
            backbone.config, batch.segments, batch.input_ids.shape[1], w.dtype,
            batch.input_ids.device,
        )  # fmt: skip
    else:
        mask = batch.attention_mask
    h = backbone(
        input_ids=batch.input_ids,
        attention_mask=mask,
        position_ids=batch.position_ids,
        use_cache=False,
    ).last_hidden_state
    sel = h[batch.target_index[:, 0], batch.target_index[:, 1]]
    ce = target_ce_sum(sel, w, batch.target_ids)
    dist = None
    if batch.packed:
        pred, code = jepa_embeddings(h, batch.text_index, batch.code_index)
        dist = jepa_distance(pred, code, distance)
    return ce, dist


# -- logging -----------------------------------------------------------------------------
class Logger:
    """Where metrics go: a JSONL file in the run folder, plus W&B when a run is given."""

    def __init__(self, path: Path | None = None, wandb_run=None, echo: Callable | None = print):
        self.path = path
        self.run = wandb_run
        self.echo = echo
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        row = {"step": step, "ts": utc_now(), **metrics}
        self.history.append(row)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=float) + "\n")
        if self.run is not None:
            self.run.log(metrics, step=step)
        if self.echo:
            keys = [k for k in metrics if not k.startswith("val/task/")][:12]
            self.echo(f"step {step}: " + ", ".join(f"{k}={_fmt(metrics[k])}" for k in keys))

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            for k, v in values.items():
                self.run.summary[k] = v


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


# -- the loop ----------------------------------------------------------------------------
@dataclass
class LoopResult:
    steps: int
    epochs: int
    adapters: list[Path] = field(default_factory=list)
    final_adapter: Path | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    dropped_too_long: list[str] = field(default_factory=list)


def _peak_gb(device) -> float | None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    return None


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def free_cuda(device) -> None:
    """Return cached blocks to the allocator (the eval's shapes differ from training's)."""
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync(device) -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def validation_loss(
    model, feats: list[Features], ct: ChatTokenizer, cfg: TrainConfig, device
) -> dict[str, float]:
    """val/loss (token-mean CE over all val examples) and per origin; val/jepa when on."""
    was = model.training
    model.eval()
    sums: dict[str, list[float]] = {}
    jsum, jn = 0.0, 0
    try:
        # batches never mix origins (so the per-origin sums stay exact), longest first
        order = sorted(range(len(feats)), key=lambda i: (feats[i].origin, -feats[i].length))
        bs = max(1, cfg.eval.val_batch)
        batches: list[list[Features]] = []
        for i in order:
            if batches and len(batches[-1]) < bs and batches[-1][0].origin == feats[i].origin:
                batches[-1].append(feats[i])
            else:
                batches.append([feats[i]])
        for group in batches:
            b = collate(group, ct.pad_id, with_views=cfg.jepa.enabled).to(device)
            ce, dist = forward_losses(model, b, cfg.jepa.distance)
            acc = sums.setdefault(group[0].origin, [0.0, 0])
            acc[0] += float(ce)
            acc[1] += b.n_target
            if dist is not None:
                jsum += float(dist.sum())
                jn += dist.numel()
    finally:
        model.train(was)
    out: dict[str, float] = {}
    tot_l = sum(v[0] for v in sums.values())
    tot_n = sum(v[1] for v in sums.values())
    if tot_n:
        out["val/loss"] = tot_l / tot_n
    for origin, (loss_sum, n) in sorted(sums.items()):
        if n:
            out[f"val/loss/{origin}"] = loss_sum / n
    if jn:
        out["val/jepa"] = jsum / jn
    return out


def _eval_safely(fn, args: tuple, label: str, logger: Logger) -> dict[str, Any]:
    try:
        return fn(*args) or {}
    except Exception as e:  # an evaluation failure never stops training
        if logger.echo:
            logger.echo(f"{label} evaluation failed: {e!r}"[:500])
        return {"val/eval_failed": 1.0}


def run_loop(
    model,
    ct: ChatTokenizer,
    train_feats: list[Features],
    cfg: TrainConfig,
    run_dir: Path,
    logger: Logger,
    device,
    on_epoch_end: Callable[[int, int], dict[str, Any]] | None = None,
    on_eval: Callable[[int, float], dict[str, Any]] | None = None,
    checkpointer: Checkpointer | None = None,
    resume_from: Path | None = None,
    stop_after: int | None = None,
    events: Events = NO_EVENTS,
) -> LoopResult:
    """Train `model` (a PEFT model) on `train_feats` for `cfg.optim.epochs` epochs.

    Saves the adapter after each epoch (`adapter_epoch<N>/`) and at the end (`adapter/`).
    `on_eval(step, epoch)` runs every `cfg.eval.every_steps` optimizer steps (not at an
    epoch's last step: `on_epoch_end(epoch, step)` covers that). With a `checkpointer`, a
    full resumable state is written every `checkpoint.every_steps` steps (after that step's
    evaluation) and after each epoch's end; `resume_from` (a checkpoint folder) restores it
    and continues from the exact position, so the run is the same as an uninterrupted one.
    `stop_after` stops dead after that many steps, as a crash would (tests)."""
    from transformers import get_cosine_schedule_with_warmup

    o, j = cfg.optim, cfg.jepa
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=o.lr, weight_decay=o.weight_decay, betas=(0.9, 0.999))
    lengths = [f.length for f in train_feats]
    n_batches = math.ceil(len(train_feats) / o.batch)
    steps_per_epoch = math.ceil(n_batches / o.grad_accum)
    total = steps_per_epoch * o.epochs
    warmup = int(o.warmup if o.warmup >= 1 else round(o.warmup * total))
    sched = get_cosine_schedule_with_warmup(opt, warmup, max(total, 1))
    gate = JepaGate(j.ratio if j.enabled else 1.0, seed=cfg.seed)
    if resume_from is not None:
        state = load_checkpoint(resume_from, model, opt, sched, gate)  # RNGs restored too
        events.emit("resumed", step=state.step, checkpoint=resume_from.name,
                    epoch=state.epoch)  # fmt: skip
        if logger.echo:
            logger.echo(f"resumed from {resume_from.name}: epoch {state.epoch}, "
                        f"step {state.step}")  # fmt: skip
    else:
        torch.manual_seed(cfg.seed)
        state = LoopState()
    result = LoopResult(steps=state.step, epochs=o.epochs)
    model.train()

    def save(st: LoopState) -> None:
        if checkpointer is not None:
            checkpointer.save(model, opt, sched, st, gate)

    every_eval = cfg.eval.every_steps
    for epoch in range(state.epoch, o.epochs):
        order = length_grouped_batches(lengths, o.batch, seed=cfg.seed + epoch)
        groups = [order[s : s + o.grad_accum] for s in range(0, len(order), o.grad_accum)]
        first = state.next_group if epoch == state.epoch else 0
        for g in range(first, len(groups)):
            group = [[train_feats[i] for i in b] for b in groups[g]]
            n_tgt = sum(f.n_target for b in group for f in b)
            n_ex = sum(len(b) for b in group)
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            t0 = time.perf_counter()
            ce_tot, j_tot, j_n, tokens = 0.0, 0.0, 0, 0
            for feats in group:
                with_views = j.enabled and gate.apply()
                batch = collate(feats, ct.pad_id, with_views=with_views, pad_to=o.pad_to).to(device)
                ce, dist = forward_losses(model, batch, j.distance)
                loss = ce / n_tgt
                if dist is not None:
                    loss = loss + j.lam * dist.sum() / n_ex
                    j_tot += float(dist.detach().sum())
                    j_n += dist.numel()
                loss.backward()
                ce_tot += float(ce.detach())
                tokens += batch.n_tokens
            gnorm = torch.nn.utils.clip_grad_norm_(params, o.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            _sync(device)
            dt = time.perf_counter() - t0
            state.step += 1
            events.step = state.step
            state.tokens_seen += tokens
            state.epoch, state.next_group = epoch, g + 1
            step = state.step
            frac = epoch + (g + 1) / len(groups)
            if step % cfg.log_every == 0 or step == total:
                ce_mean = ce_tot / n_tgt
                m: dict[str, Any] = {
                    "train/ce": ce_mean,
                    "train/loss": ce_mean + (j.lam * j_tot / n_ex if j.enabled else 0.0),
                    "train/lr": sched.get_last_lr()[0],
                    "train/grad_norm": float(gnorm),
                    "train/tokens_per_s": tokens / dt,
                    "train/target_tokens": n_tgt,
                    "train/tokens_seen": state.tokens_seen,
                    "epoch": frac,
                }
                if j.enabled:
                    m["train/jepa"] = j_tot / j_n if j_n else None
                    m["train/jepa_on"] = j_n / n_ex
                    m = {k: v for k, v in m.items() if v is not None}
                peak = _peak_gb(device)
                if peak is not None:
                    m["train/peak_gb"] = peak
                logger.log(m, step)
            last_of_epoch = g == len(groups) - 1
            if on_eval is not None and every_eval > 0 and step % every_eval == 0 \
                    and not last_of_epoch:  # fmt: skip
                metrics = _eval_safely(on_eval, (step, frac), f"step {step}", logger)
                if metrics:
                    logger.log({**metrics, "epoch": frac}, step)
            if checkpointer is not None and checkpointer.due(step) and not last_of_epoch:
                save(state)
            if stop_after is not None and step >= stop_after:
                result.steps = step
                result.history = logger.history
                return result
        adapter = run_dir / f"adapter_epoch{epoch + 1}"
        model.save_pretrained(str(adapter), save_embedding_layers=False)
        result.adapters.append(adapter)
        if on_epoch_end is not None:
            ep = epoch + 1
            metrics = _eval_safely(on_epoch_end, (ep, state.step), f"epoch {ep}", logger)
            if metrics:
                logger.log({**metrics, "epoch": float(epoch + 1)}, state.step)
        events.emit("epoch_end", step=state.step, epoch=epoch + 1,
                    tokens_seen=state.tokens_seen)  # fmt: skip
        state.epoch, state.next_group = epoch + 1, 0
        save(state)
    final = run_dir / "adapter"
    model.save_pretrained(str(final), save_embedding_layers=False)
    result.final_adapter = final
    result.steps = state.step
    result.history = logger.history
    return result


# -- the run -----------------------------------------------------------------------------
def environment() -> dict[str, Any]:
    import importlib.metadata as md

    env: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "wandb",
                "safetensors", "nfl-ask-trainer"):  # fmt: skip
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env[pkg] = None
    env["hip"] = getattr(torch.version, "hip", None)
    env["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    try:
        repo = Path(__file__).resolve().parents[2]
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "HEAD"],
                           capture_output=True, text=True, timeout=20)  # fmt: skip
        env["git_commit"] = r.stdout.strip() if r.returncode == 0 else None  # None: no commit
    except Exception:
        env["git_commit"] = None
    return env


def wandb_mode() -> tuple[str, Any]:
    """('online', settings) when a key is available (the settings module's WANDB_API_KEY,
    else wandb's own stored login), else ('offline', None) with a warning. Never prints or
    stores the key (no `wandb.login(key=...)`: that would write it into the netrc)."""
    import wandb

    from asktrainer.settings import get_secrets

    secrets = get_secrets()
    if secrets.is_set("wandb_api_key"):
        return "online", wandb.Settings(api_key=secrets.wandb_api_key.get_secret_value())
    try:
        if wandb.login(prompt=False, verify=False):
            return "online", None
    except Exception:
        pass
    warnings.warn(
        "W&B: no API key and no stored login: the run logs OFFLINE (sync it later with "
        "`wandb sync <run dir>/wandb/offline-run-*`)",
        stacklevel=2,
    )
    return "offline", None


def configure_backends(cfg: TrainConfig, runs_root: Path) -> dict[str, Any]:
    """ROCm options: TunableOp (GEMM tuning; results cached in `<runs>/tunableop/`) and
    hipBLASLt preference."""
    out: dict[str, Any] = {}
    if not torch.cuda.is_available():
        return out
    if cfg.optim.hipblaslt:
        torch.backends.cuda.preferred_blas_library("cublaslt")  # hipBLASLt on ROCm
        out["blas"] = "hipblaslt"
    if cfg.optim.tunableop:
        d = runs_root / "tunableop"
        d.mkdir(parents=True, exist_ok=True)
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_filename(str(d / "tunableop_results%d.csv"))
        out["tunableop"] = str(d)
    return out


def make_run_name(cfg: TrainConfig, seed: int, name: str | None) -> str:
    return name or f"{cfg.name}-{cfg.objective}-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}"


def resolve_run(resume: str | Path, paths) -> Path:
    """A run folder from its path or its name (under the runs dir)."""
    p = Path(resume)
    if (p / "config.json").exists():
        return p
    q = paths.runs / str(resume)
    if (q / "config.json").exists():
        return q
    raise FileNotFoundError(f"no run folder (with config.json) at {p} or {q}")


def saved_config(run_dir: Path) -> tuple[TrainConfig, dict[str, Any]]:
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    fields = set(TrainConfig.model_fields)
    return TrainConfig.model_validate({k: v for k, v in saved.items() if k in fields}), saved


def _truncate_log(path: Path, step: int) -> None:
    """Drop log rows after the checkpoint's step (they are re-run after a resume)."""
    if not path.exists():
        return
    rows = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    keep = [ln for ln in rows if json.loads(ln).get("step", 0) <= step]
    path.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")


def artifact_name(cfg: TrainConfig, run: str) -> str:
    """One artifact per run (`<artifact_name>-<run>`): runs never move each other's aliases."""
    return f"{cfg.checkpoint.artifact_name}-{run}"


def log_adapter(run, cfg: TrainConfig, adapter_dir: Path, aliases: list[str],
                meta: dict[str, Any], echo=print, events: Events = NO_EVENTS) -> None:  # fmt: skip
    """The adapter as a W&B artifact (`<artifact_name>-<run>`, type model); a failure is
    reported (event `artifact_failed`, the exception's type only), never fatal."""
    if run is None or not cfg.checkpoint.artifacts:
        return
    import wandb

    name = artifact_name(cfg, meta.get("run") or "run")
    try:
        art = wandb.Artifact(name, type="model", metadata=meta)
        art.add_dir(str(adapter_dir))
        run.log_artifact(art, aliases=aliases)
        events.emit("artifact_uploaded", name=name, folder=adapter_dir.name, aliases=aliases)
    except Exception as e:
        echo(f"W&B artifact upload failed ({adapter_dir.name}): {type(e).__name__}")
        events.emit("artifact_failed", name=name, folder=adapter_dir.name,
                    error=type(e).__name__)  # fmt: skip


SAMPLES = 12  # generations kept per evaluation for the dashboard


def write_samples(path: Path, examples, preds: dict[str, str], scored: dict[str, dict]) -> None:
    """`samples_<step>.jsonl`: the question, the gold, the output and its score fields for a
    fixed few examples of the eval subset."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for e in examples:
            it = scored.get(e.id, {})
            exec_ok = it.get("exec_ok")
            ok = exec_ok if exec_ok is not None else it.get("exact")
            row = {"id": e.id, "set": e.origin, "group": e.group, "task": e.task,
                   "question": e.jepa.text, "gold": e.target, "output": preds.get(e.id, ""),
                   "ok": ok, "exec_ok": exec_ok, "json_ok": it.get("json_ok"),
                   "parse_ok": it.get("parse_ok"), "exact": it.get("exact"),
                   "reason": str(it.get("reason") or "")[:40]}  # fmt: skip
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train(
    cfg: TrainConfig | None = None,
    seed: int | None = None,
    name: str | None = None,
    launched_by: str = "agent",
    paths=None,
    device: str = "cuda",
    resume: str | Path | None = None,
    stop_after: int | None = None,
    config_path: str | None = None,
) -> Path:
    """The whole run: data mix, model, W&B, the loop with its evaluations, checkpoints and
    artifacts. With `resume` (a run folder or name) it reloads that run's config and newest
    checkpoint and continues it, the same W&B run included. Returns the run folder.
    `stop_after` stops dead after that many optimizer steps, as a crash would (tests).

    The run folder also gets the dashboard's files: `run_info.json`, `events.jsonl`
    (`events.py`) and `samples_<step>.jsonl` per evaluation; a crash is recorded as a
    `crashed` event (the exception's type only) before it propagates."""
    from asktrainer.checkpoint import latest as latest_checkpoint
    from asktrainer.checkpoint import read_meta
    from asktrainer.settings import get_paths

    paths = paths or get_paths()
    ckpt = None
    wandb_id = None
    if resume is not None:
        run_dir = resolve_run(resume, paths)
        cfg, saved = saved_config(run_dir)
        run_name, launched_by = saved["run"], saved.get("launched_by", launched_by)
        ckpt = latest_checkpoint(cfg.checkpoint.root(run_name))
        if ckpt is None:
            raise FileNotFoundError(f"no complete checkpoint under {cfg.checkpoint.root(run_name)}")
        wandb_id = read_meta(ckpt).get("wandb_id")
    else:
        if cfg is None:
            raise ValueError("a config (or resume) is required")
        if seed is not None:
            cfg = cfg.model_copy(update={"seed": seed})
        run_name = make_run_name(cfg, cfg.seed, name)
        run_dir = paths.runs / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
    events = Events(run_dir / "events.jsonl")
    events.step = read_meta(ckpt)["step"] if ckpt else 0
    events.emit("start", step=read_meta(ckpt)["step"] if ckpt else 0, run=run_name,
                resume=ckpt is not None, objective=cfg.objective)  # fmt: skip
    try:
        return _train(cfg, paths, run_dir, run_name, launched_by, ckpt, wandb_id, device,
                      stop_after, config_path, events)  # fmt: skip
    except BaseException as e:  # a crash (or Ctrl-C): recorded, then re-raised
        events.emit("crashed", error=type(e).__name__)
        raise


def _train(cfg, paths, run_dir, run_name, launched_by, ckpt, wandb_id, device, stop_after,
           config_path, events: Events) -> Path:  # fmt: skip
    import wandb

    from asktrainer import publiceval
    from asktrainer.attn_chunked import use_chunked
    from asktrainer.checkpoint import read_meta
    from asktrainer.evaluation import (
        predict,
        run_verifier,
        score_metrics,
        stratified_subset,
        write_preds,
    )
    from asktrainer.mix import load_mix
    from asktrainer.modelio import apply_lora, load_base, load_tokenizer, trainable_params

    resume = ckpt is not None
    backends = configure_backends(cfg, paths.runs)

    # the data first (fails fast, before the slow model load)
    tok = load_tokenizer(cfg.base_model_path(paths))
    ct = ChatTokenizer(tok, cfg.optim.max_len, jepa=cfg.jepa.enabled, k=cfg.jepa.k,
                       last_token=cfg.jepa.last_token)  # fmt: skip
    mix = load_mix(cfg, paths, length=lambda e: len(ct.full_ids(e)[0]))
    train_feats, drop_train = ct.tokenize_all(mix.train)
    val_feats, drop_val = ct.tokenize_all(mix.val)
    dropped_val = set(drop_val)
    val_ok = [e for e in mix.val if e.id not in dropped_val]
    nfl_subset = stratified_subset([e for e in val_ok if e.origin == "nfl"], cfg.eval.nfl_n,
                                   cfg.seed)  # fmt: skip
    public_subset = stratified_subset([e for e in val_ok if e.origin != "nfl"],
                                      cfg.eval.public_n, cfg.seed,
                                      key=lambda e: (e.origin, e.task))  # fmt: skip
    sample_set = stratified_subset(nfl_subset + public_subset, SAMPLES, cfg.seed,
                                   key=lambda e: (e.origin, e.group))  # fmt: skip
    if cfg.eval.val_loss_n >= 0:
        keep = {
            e.id
            for e in stratified_subset(
                val_ok, cfg.eval.val_loss_n, cfg.seed, key=lambda e: (e.origin, e.group, e.task)
            )
        }
        val_feats = [f for f in val_feats if f.id in keep]  # fmt: skip
    manifest = {
        **mix.info,
        "train_ids": [f.id for f in train_feats],
        "val_ids": [f.id for f in val_feats],
        "eval_generate_ids": {"nfl": [e.id for e in nfl_subset],
                              "public": [e.id for e in public_subset]},
        "dropped_too_long": {"train": drop_train, "val": drop_val},
    }  # fmt: skip
    if resume:  # the same examples in the same order, or nothing
        before = json.loads((run_dir / "mix_manifest.json").read_text(encoding="utf-8"))
        if before["train_ids"] != manifest["train_ids"]:
            raise RuntimeError("the training data changed since the run started: can't resume")
    train_tokens = sum(f.length for f in train_feats)
    events.emit("data_ready", examples=len(train_feats), tokens=train_tokens,
                dropped=len(drop_train) + len(drop_val), val_examples=len(val_feats),
                eval_generate=len(nfl_subset) + len(public_subset))  # fmt: skip

    model, tok, load_info = load_base(cfg.base_model_path(paths), device=device)
    if cfg.optim.attn_chunk > 0:
        use_chunked(model, chunk=cfg.optim.attn_chunk, threshold=cfg.optim.attn_chunk_above)
    seed_everything(cfg.seed)  # the LoRA init is drawn here: the same seed, the same start
    model = apply_lora(model, cfg.lora, cfg.optim.gradient_checkpointing)
    events.emit("model_loaded", seconds=load_info["load_s"], peak_gb=_peak_gb(device),
                trainable_m=round(trainable_params(model) / 1e6, 2))  # fmt: skip

    env = environment()
    if not resume:
        run_config = {
            **cfg.logged(),
            "run": run_name,
            "objective": cfg.objective,
            "launched_by": launched_by,
            "config_path": config_path,
            "nfl_version": mix.info.get("nfl_version"),
            "n_train": len(train_feats),
            "n_val": len(val_feats),
            "n_eval_generate": {"nfl": len(nfl_subset), "public": len(public_subset)},
            "train_target_tokens": sum(f.n_target for f in train_feats),
            "train_tokens": train_tokens,
            "dropped_too_long": len(drop_train) + len(drop_val),
            "trainable_params_m": round(trainable_params(model) / 1e6, 2),
            "load_s": load_info["load_s"],
            "backends": backends,
            "checkpoint_root": str(cfg.checkpoint.root(run_name)),
        }
        assert_no_secrets(run_config)
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=1, default=str),
                                             "utf-8")  # fmt: skip
        (run_dir / "environment.json").write_text(json.dumps(env, indent=1), "utf-8")
        (run_dir / "mix_manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    else:
        run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        _truncate_log(run_dir / "train_log.jsonl", read_meta(ckpt)["step"])

    mode, settings = wandb_mode()
    w = paths.wandb
    tags = ["ask-finetune", f"objective:{cfg.objective}", f"seed:{cfg.seed}",
            f"launched-by:{launched_by}"]  # fmt: skip
    init: dict[str, Any] = dict(
        entity=w.entity, project=w.project, group=w.group, job_type=w.job_type,
        name=run_name, tags=tags, config=run_config, dir=str(run_dir), mode=mode,
        settings=settings,
    )  # fmt: skip
    if wandb_id:
        init.update(id=wandb_id, resume="must")
    run = wandb.init(**init)
    (run_dir / "wandb_run_id").write_text(run.id + "\n", encoding="utf-8")
    run.define_metric("epoch")
    run.define_metric("val/*", step_metric="epoch")

    o = cfg.optim
    steps_per_epoch = math.ceil(math.ceil(len(train_feats) / o.batch) / o.grad_accum)
    info_path = run_dir / "run_info.json"
    old_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    write_run_info(info_path, {
        "run": run_name,
        "config_path": run_config.get("config_path", config_path),
        "objective": cfg.objective,
        "seed": cfg.seed,
        "launched_by": launched_by,
        "started_at": old_info.get("started_at") or utc_now(),
        "resumed_at": [*old_info.get("resumed_at", []), *([utc_now()] if resume else [])],
        "resumed_from_step": read_meta(ckpt)["step"] if resume else None,
        "planned_steps": steps_per_epoch * o.epochs,
        "steps_per_epoch": steps_per_epoch,
        "epochs": o.epochs,
        "planned_tokens": train_tokens * o.epochs,
        "tokens_per_epoch": train_tokens,
        "examples": len(train_feats),
        "examples_per_set": dict(sorted(Counter(f.origin for f in train_feats).items())),
        "val_examples": len(val_feats),
        "eval_generate": {"nfl": len(nfl_subset), "public": len(public_subset)},
        "eval_every_steps": cfg.eval.every_steps,
        "max_len": o.max_len,
        "effective_batch": o.batch * o.grad_accum,
        "lr": o.lr,
        "wandb": {"id": run.id, "url": getattr(run, "url", None), "project": w.project,
                  "entity": w.entity, "mode": mode},
        "checkpoint_dir": str(cfg.checkpoint.root(run_name)),
        "checkpoint_every_steps": cfg.checkpoint.every_steps,
        "base_model": str(cfg.base_model_path(paths)),
        "git_commit": env.get("git_commit"),
    })  # fmt: skip
    logger = Logger(run_dir / "train_log.jsonl", run)
    checkpointer = Checkpointer(cfg.checkpoint.root(run_name), cfg.checkpoint.every_steps,
                                cfg.checkpoint.keep)  # fmt: skip
    checkpointer.wandb_id, checkpointer.run = run.id, run_name
    checkpointer.events = events
    tuning = cfg.optim.tunableop and torch.cuda.is_available()
    art_meta = {"run": run_name, "objective": cfg.objective, "seed": cfg.seed,
                "base": str(cfg.base_model_path(paths)), "lora": cfg.lora.model_dump(),
                "nfl_version": mix.info.get("nfl_version")}  # fmt: skip

    def evaluate(tag: str, step: int) -> dict[str, Any]:
        events.emit("eval_start", step=step, tag=tag)
        t = time.perf_counter()
        if not tuning:
            metrics = _evaluate(tag, step)
        else:
            torch.cuda.tunable.write_file()  # keep what was tuned so far
            torch.cuda.tunable.tuning_enable(False)  # eval shapes are one-offs: never tune
            try:
                metrics = _evaluate(tag, step)
            finally:
                torch.cuda.tunable.tuning_enable(True)
        parts = {
            k.split("/", 1)[1]: round(v, 2)
            for k, v in metrics.items()
            if k in ("val/loss_s", "val/generate_s", "val/public_s", "val/verify_s")
        }
        events.emit("eval_end", step=step, tag=tag, seconds=round(time.perf_counter() - t, 2),
                    **parts)  # fmt: skip
        return metrics

    def _evaluate(tag: str, step: int) -> dict[str, Any]:
        """Each part fails on its own (logged as val/<part>_failed); training goes on."""
        metrics: dict[str, Any] = {}
        free_cuda(device)
        t0 = time.perf_counter()
        try:
            metrics.update(validation_loss(model, val_feats, ct, cfg, device))
        except Exception as e:
            logger.echo(f"val loss failed ({tag}): {type(e).__name__}")
            metrics["val/loss_failed"] = 1.0
        metrics["val/loss_s"] = time.perf_counter() - t0
        subset = nfl_subset + public_subset
        if not subset:
            return metrics
        free_cuda(device)
        t1 = time.perf_counter()
        try:
            preds = predict(model, ct, subset, cfg.eval.max_new_tokens, device,
                            batch_size=cfg.eval.gen_batch,
                            prefill_tokens=cfg.eval.gen_prefill_tokens, stats=metrics,
                            on_oom=lambda bs: events.emit("oom_split", step=step,
                                                          where="generate", batch=bs))  # fmt: skip
        except Exception as e:
            logger.echo(f"generation failed ({tag}): {type(e).__name__}")
            metrics["val/generate_failed"] = 1.0
            return metrics
        finally:
            metrics["val/generate_s"] = time.perf_counter() - t1
            free_cuda(device)
        metrics["val/empty_output"] = sum(not p["output"] for p in preds) / len(preds)
        nfl_ids = {e.id for e in nfl_subset}
        scored: dict[str, dict] = {}
        if public_subset:
            pub = [p for p in preds if p["id"] not in nfl_ids]
            write_preds(run_dir / f"preds_public_{tag}.jsonl", pub)
            t2 = time.perf_counter()
            try:
                res = publiceval.verify(public_subset, pub, publiceval.PublicData(paths.public))
                (run_dir / f"scores_public_{tag}.json").write_text(
                    json.dumps(res, indent=1, default=str), encoding="utf-8")  # fmt: skip
                metrics.update(publiceval.metrics(res))
                scored.update({i["id"]: i for i in res["items"]})
            except Exception as e:
                logger.echo(f"public scoring failed ({tag}): {type(e).__name__}")
                events.emit("verifier_failed", step=step, part="public",
                            reason=type(e).__name__)  # fmt: skip
                metrics["val/public_failed"] = 1.0
            metrics["val/public_s"] = time.perf_counter() - t2
        if nfl_subset:
            pred_path = write_preds(run_dir / f"preds_{tag}.jsonl",
                                    [p for p in preds if p["id"] in nfl_ids])  # fmt: skip
            if cfg.eval.nfl_verify and mix.nfl_val_path is not None:
                t3 = time.perf_counter()
                try:
                    scores = run_verifier(paths, mix.nfl_val_path, pred_path,
                                          run_dir / f"scores_{tag}.json",
                                          cfg.eval.verify_timeout_s)  # fmt: skip
                    metrics.update(score_metrics(scores))
                    metrics["val/verify_failed"] = 0.0
                    scored.update({i["id"]: i for i in scores.get("items", []) if "id" in i})
                except Exception as e:
                    logger.echo(f"verifier failed ({tag}): {type(e).__name__}")
                    events.emit("verifier_failed", step=step, part="nfl",
                                reason=type(e).__name__)  # fmt: skip
                    metrics["val/verify_failed"] = 1.0
                metrics["val/verify_s"] = time.perf_counter() - t3
        try:
            write_samples(run_dir / f"samples_{step:06d}.jsonl", sample_set,
                          {p["id"]: p["output"] for p in preds}, scored)  # fmt: skip
        except Exception as e:
            logger.echo(f"samples not written ({tag}): {type(e).__name__}")
        return metrics

    def on_epoch_end(epoch: int, step: int) -> dict[str, Any]:
        metrics = evaluate(f"epoch{epoch}", step) if cfg.eval.every_epoch else {}
        log_adapter(run, cfg, run_dir / f"adapter_epoch{epoch}", ["latest", f"epoch-{epoch}"],
                    {**art_meta, "epoch": epoch, "step": step}, logger.echo,
                    events)  # fmt: skip
        return metrics

    def on_eval(step: int, epoch: float) -> dict[str, Any]:
        return evaluate(f"step{step:06d}", step)

    try:
        result = run_loop(model, ct, train_feats, cfg, run_dir, logger, device,
                          on_epoch_end=on_epoch_end, on_eval=on_eval,
                          checkpointer=checkpointer, resume_from=ckpt,
                          stop_after=stop_after, events=events)  # fmt: skip
        if result.final_adapter is None:  # stopped early (tests: a simulated crash)
            return run_dir
        log_adapter(run, cfg, result.final_adapter, ["latest", "final"],
                    {**art_meta, "step": result.steps, "final": True}, logger.echo,
                    events)  # fmt: skip
        logger.summary({"steps": result.steps, "final_adapter": str(result.final_adapter)})
    finally:
        run.finish()
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    events.emit("done", step=result.steps)
    return run_dir
