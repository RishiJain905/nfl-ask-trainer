"""Paths and secrets for the trainer.

Paths come from `configs/paths.yaml` (no secrets there). Secrets come only from this module, as
Pydantic `SecretStr`: `WANDB_API_KEY` from the environment or the repo's git-ignored env file;
when it isn't set, wandb uses its own stored login (`wandb login` inside the WSL venv). Never
print a secret, log it, put it in an exception message or in `wandb.config`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"


class WandbConfig(BaseModel):
    entity: str = "models-ontario-tech-university"
    project: str = "nfl-analytics-engine"
    group: str = "ask-finetune"
    job_type: str = "ask-finetune"


class Paths(BaseModel):
    data_root: Path
    training_dir: str = "ask/training"
    runs_dir: str = "ask/finetune"
    traces_dir: str = "ask/traces"
    base_model: Path
    nfl_repo: Path
    uv_exe: Path
    wandb: WandbConfig = WandbConfig()

    @property
    def training(self) -> Path:
        return self.data_root / self.training_dir

    @property
    def runs(self) -> Path:
        return self.data_root / self.runs_dir

    @property
    def traces(self) -> Path:
        """Thinking-trace generation runs (`traces/<run>/`)."""
        return self.data_root / self.traces_dir

    @property
    def public(self) -> Path:
        """Public datasets, as downloaded and as converted (`public/<name>/`)."""
        return self.training / "public"

    @property
    def mixes(self) -> Path:
        return self.training / "mixes"


@lru_cache(maxsize=4)
def get_paths(path: str | None = None) -> Paths:
    p = Path(path) if path else CONFIG_DIR / "paths.yaml"
    return Paths.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")) or {})


class Secrets(BaseSettings):
    """Optional secrets (all SecretStr; None when unset)."""

    model_config = SettingsConfigDict(
        env_file=str(REPO / ".env"), env_file_encoding="utf-8", extra="ignore"
    )
    wandb_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    def is_set(self, name: str) -> bool:
        v = getattr(self, name.lower(), None)
        return v is not None and bool(v.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()


_MNT = re.compile(r"^/mnt/([a-z])(/.*)?$")


def to_windows(path: Path | str) -> str:
    """`/mnt/d/x/y` -> `D:\\x\\y` (for the Windows-side verifier). Other paths are returned
    unchanged (a path inside WSL's own disk has no drive letter)."""
    s = str(PurePosixPath(str(path)))
    m = _MNT.match(s)
    if not m:
        return s
    rest = (m.group(2) or "/").lstrip("/")
    return str(PureWindowsPath(f"{m.group(1).upper()}:/") / PureWindowsPath(rest))
