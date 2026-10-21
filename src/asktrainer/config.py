"""The training config (`TrainConfig`), read from YAML (`configs/*.yaml`).

The data block names the NFL export (a version folder under `<training>/nfl/`, an absolute
folder holding `train.jsonl` + `val.jsonl`, or `latest`, read from `<training>/nfl/LATEST`)
and the public datasets to mix in (`<training>/public/<name>/examples_{train,val}.jsonl`,
sampled with the run's seed). Nothing secret ever goes in a config: `assert_no_secrets`
refuses a key that looks like one (the NFL repo's `tracking.assert_no_secrets` rule), since the
whole config is logged to `wandb.config`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from asktrainer.settings import CONFIG_DIR, Paths, get_paths

# "token" but not "tokens" (a count: max_new_tokens, train_tokens are fine)
SECRET_KEY = re.compile(r"token(?!s)|secret|password|passwd|api_?key|credential", re.IGNORECASE)
# config keys that contain a secret-looking word but name a position, not a secret
_ALLOWED = {"last_token"}


def assert_no_secrets(obj: Any, path: str = "") -> None:
    """Raise if any key in a (nested) config is named like a secret."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            if key not in _ALLOWED and SECRET_KEY.search(key):
                raise ValueError(f"config key {path + key!r} looks like a secret: not logged")
            assert_no_secrets(v, f"{path}{key}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_secrets(v, f"{path}{i}.")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PublicSpec(_Strict):
    train: int = 0  # examples sampled from examples_train.jsonl (0 = none, -1 = all)
    val: int = 0  # from examples_val.jsonl (val loss only)


class DataConfig(_Strict):
    # The NFL track is parked (2026-10-20: the target is a general database agent): no NFL
    # examples unless a config asks for them. The export is read only when a count is not 0.
    nfl: str = "latest"
    # seeded samples of the NFL train set per task (-1 = all, 0 = none), drawn from the
    # examples of at most `nfl_max_tokens` tokens (None = optim.max_len): exact counts
    nfl_route: int = 0
    nfl_generate: int = 0
    nfl_max_tokens: int | None = None
    nfl_val: int = 0  # val examples kept (-1 = all; a sample is stratified by group / task)
    public: dict[str, PublicSpec] = Field(default_factory=dict)

    def nfl_dir(self, paths: Paths) -> Path:
        root = paths.training / "nfl"
        if self.nfl == "latest":
            latest = root / "LATEST"
            if not latest.exists():
                raise FileNotFoundError(f"no NFL export yet ({latest} is missing)")
            return root / latest.read_text(encoding="utf-8").strip()
        p = Path(self.nfl)
        return p if p.is_absolute() else root / p


class LoraSpec(_Strict):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: list[str] = Field(
        default_factory=lambda: ["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
    )


class OptimConfig(_Strict):
    lr: float = 1e-4
    epochs: int = 2
    batch: int = 1
    grad_accum: int = 16
    warmup: float = 0.03  # a fraction of the optimizer steps (or a step count when >= 1)
    warmup_min_steps: int = 0  # ... but at least this many (3% of a short run is a few steps)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_len: int = 5300  # longer examples are dropped (and counted)
    gradient_checkpointing: bool = True
    # Spark's eager attention in query chunks above this length (attn_chunked.py; 0 = off)
    attn_chunk: int = 1024
    attn_chunk_above: int = 3072
    # ROCm GEMM backends (measured on the RX 7800 XT with scripts/bench_train.py): hipBLASLt
    # is free (+23% at 2k tokens); TunableOp adds ~8% more but tunes every new GEMM shape
    # (~2-3 min per sequence length, cached in <runs>/tunableop/): it needs pad_to buckets
    hipblaslt: bool = True  # TORCH_BLAS_PREFER_HIPBLASLT
    tunableop: bool = False  # PYTORCH_TUNABLEOP_ENABLED
    pad_to: int = 0  # pad each batch to a multiple of this (use 256 with tunableop)


class JepaSpec(_Strict):
    enabled: bool = False
    lam: float = 2.0  # lambda
    k: int = 0  # predictor tokens
    distance: Literal["cosine", "l2", "mse"] = "cosine"
    ratio: float = 1.0  # jepa_ratio: share of micro-batches that run the JEPA term
    last_token: int = -2


class EvalConfig(_Strict):
    every_epoch: bool = True  # evaluate after each epoch
    every_steps: int = 0  # ... and every N optimizer steps (0 = off), on the same fixed subsets
    # the validation loss alone (no generation) every N steps between those (0 = off)
    loss_every_steps: int = 0
    # the scored metric that picks `<run>/best/` (higher is better; ties: the lower val/loss)
    best_metric: str = "val/public/exec_acc"
    # public val: generated and scored here (publiceval.py: gretel executed in sqlite, the
    # others exact match + a parse check), a fixed subset stratified over the datasets
    public_n: int = 200
    # NFL val: generated and scored by the NFL repo's verifier (off: the NFL track is parked)
    nfl_n: int = 0
    nfl_verify: bool = False
    # NFL generate targets reach 460-490 tokens: 384 would cut the longest ones
    max_new_tokens: dict[str, int] = Field(default_factory=lambda: {"route": 96, "generate": 512})
    gen_batch: int = 8  # rows per generation batch (at most)
    # padded prompt tokens per generation batch (at most): the prefill's activations, not the
    # KV cache, set the ceiling (~14k tokens fit next to the training state on 16 GB)
    gen_prefill_tokens: int = 14000
    val_batch: int = 2
    val_loss_n: int = -1  # val examples in the val loss (-1 = all; else stratified, fixed)
    verify_timeout_s: int = 1800


class CheckpointSpec(_Strict):
    """Crash-safe step checkpoints (checkpoint.py) and the W&B adapter artifacts."""

    dir: str = "/mnt/f/nfl-ask-trainer-ckpt"  # <dir>/<run>/step-NNNNNN/ (F: is the NVMe)
    every_steps: int = 20  # optimizer steps between checkpoints (also after each epoch)
    # also keep the adapter alone every N steps (`<run>/adapter_stepNNNNNN/`, never deleted):
    # what the scored evaluation (llamaeval) runs on after training (0 = off)
    adapter_every_steps: int = 0
    keep: int = 2  # complete checkpoints kept
    artifacts: bool = True  # log the adapter to W&B after each epoch and at the end
    artifact_name: str = "ask-finetune-adapter"

    def root(self, run: str) -> Path:
        return Path(self.dir) / run


class TrainConfig(_Strict):
    name: str = "ask"
    seed: int = 42
    base_model: Path | None = None  # None = paths.yaml base_model
    data: DataConfig = Field(default_factory=DataConfig)
    lora: LoraSpec = Field(default_factory=LoraSpec)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    jepa: JepaSpec = Field(default_factory=JepaSpec)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    checkpoint: CheckpointSpec = Field(default_factory=CheckpointSpec)
    log_every: int = 1  # optimizer steps between train/* logs

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", v):
            raise ValueError("name: letters, digits, _ . - only")
        return v

    @property
    def objective(self) -> str:
        return "jepa" if self.jepa.enabled else "plain"

    def base_model_path(self, paths: Paths | None = None) -> Path:
        return self.base_model or (paths or get_paths()).base_model

    def backend_flags(self, runs_root: Path) -> dict[str, str]:
        """Process flags the GEMM backends read when torch starts (the CLI sets them before
        importing torch; `train.configure_backends` also switches them at runtime)."""
        flags = {}
        if self.optim.hipblaslt:
            flags["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
        if self.optim.tunableop:
            flags["PYTORCH_TUNABLEOP_ENABLED"] = "1"
            flags["PYTORCH_TUNABLEOP_FILENAME"] = str(
                runs_root / "tunableop" / "tunableop_results%d.csv"
            )
        return flags

    def logged(self) -> dict[str, Any]:
        """The config as logged to W&B / the run folder (checked for secret-named keys)."""
        d = self.model_dump(mode="json")
        assert_no_secrets(d)
        return d

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> TrainConfig:
        p = Path(path)
        if not p.exists() and (CONFIG_DIR / p).exists():
            p = CONFIG_DIR / p
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        assert_no_secrets(raw)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(raw)
