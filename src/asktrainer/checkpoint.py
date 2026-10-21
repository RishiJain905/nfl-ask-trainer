"""Step checkpoints: everything needed to resume a run exactly where it stopped.

The PC this trains on sometimes loses power. Every `checkpoint.every_steps` optimizer steps
(and after each epoch's evaluation) the loop writes a full resumable state:

- `trainable.safetensors`: the LoRA weights (the only trainable parameters),
- `optimizer.pt`, `scheduler.pt`: AdamW's moments and the LR schedule's position,
- `rng.pt` + the python / numpy states in `state.json`: every RNG (python, numpy, torch CPU,
  torch CUDA, the JEPA-dropout gate), so dropout masks and gate draws continue identically,
- `state.json`: the position (epoch, index of the next optimizer step's batches in that
  epoch's seeded order), the global step, tokens seen, the W&B run id, the run name.

Writes are atomic: everything goes into `.tmp-step-NNNNNN/`, a `COMPLETE` marker is written
last, then the folder is renamed to `step-NNNNNN/`, and only then are older checkpoints
beyond `keep` deleted. A power cut at any point leaves the previous checkpoints valid; a
folder without `COMPLETE` (or a `.tmp-*`) is never loaded.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

PREFIX = "step-"
TMP = ".tmp-"
MARKER = "COMPLETE"


@dataclass
class LoopState:
    """Where the loop is: `next_group` indexes the optimizer steps of `epoch` (each is
    `grad_accum` batches of that epoch's seeded order); `epoch_done` means the epoch's end
    (adapter, evaluation) has been handled, so a resume starts the next epoch."""

    epoch: int = 0
    next_group: int = 0
    step: int = 0
    tokens_seen: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def rng_state(gate=None) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """(JSON-able states, tensor states) of every RNG the loop draws from."""
    py = random.getstate()
    npy = np.random.get_state()
    js: dict[str, Any] = {
        "python": [py[0], list(py[1]), py[2]],
        "numpy": [npy[0], npy[1].tolist(), int(npy[2]), int(npy[3]), float(npy[4])],
    }
    if gate is not None:
        g = gate.rng.getstate()
        js["gate"] = [g[0], list(g[1]), g[2]]
    tensors = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        for i, s in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"torch_cuda_{i}"] = s
    return js, tensors


def set_rng_state(js: dict[str, Any], tensors: dict[str, torch.Tensor], gate=None) -> None:
    p = js["python"]
    random.setstate((p[0], tuple(p[1]), p[2]))
    n = js["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if gate is not None and "gate" in js:
        g = js["gate"]
        gate.rng.setstate((g[0], tuple(g[1]), g[2]))
    torch.set_rng_state(tensors["torch_cpu"])
    if torch.cuda.is_available():
        cuda = [tensors[k] for k in sorted(t for t in tensors if t.startswith("torch_cuda_"))]
        if cuda:
            torch.cuda.set_rng_state_all(cuda)


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()
            if p.requires_grad}  # fmt: skip


def load_trainable(model, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    if set(state) != want:
        missing, extra = sorted(want - set(state))[:3], sorted(set(state) - want)[:3]
        raise RuntimeError(f"checkpoint weights don't match the model: {missing} / {extra}")
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))


def complete(root: Path) -> list[Path]:
    """Valid checkpoints under `root`, oldest first."""
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir()
             if p.is_dir() and p.name.startswith(PREFIX) and (p / MARKER).exists()]  # fmt: skip
    return sorted(found, key=lambda p: int(p.name[len(PREFIX) :]))


def latest(root: Path) -> Path | None:
    c = complete(root)
    return c[-1] if c else None


class Checkpointer:
    def __init__(self, root: Path, every_steps: int = 20, keep: int = 2):
        self.root = Path(root)
        self.every = every_steps
        self.keep = max(1, keep)
        self.wandb_id: str | None = None
        self.run: str | None = None
        self.events = None  # an events.Events: `checkpoint_saved` after each save

    def due(self, step: int) -> bool:
        return self.every > 0 and step > 0 and step % self.every == 0

    def save(self, model, opt, sched, state: LoopState, gate=None) -> Path:
        from safetensors.torch import save_file

        t0 = time.perf_counter()
        self.root.mkdir(parents=True, exist_ok=True)
        for stale in self.root.glob(TMP + "*"):  # a save interrupted earlier
            shutil.rmtree(stale, ignore_errors=True)
        name = f"{PREFIX}{state.step:06d}"
        tmp = self.root / f"{TMP}{name}"
        tmp.mkdir()
        save_file(trainable_state(model), str(tmp / "trainable.safetensors"))
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
        torch.save(sched.state_dict(), tmp / "scheduler.pt")
        js, tensors = rng_state(gate)
        torch.save(tensors, tmp / "rng.pt")
        meta = {**asdict(state), "rng": js, "wandb_id": self.wandb_id, "run": self.run,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
        (tmp / "state.json").write_text(json.dumps(meta), encoding="utf-8")
        (tmp / MARKER).write_text("ok\n", encoding="utf-8")
        final = self.root / name
        if final.exists():  # the same step saved twice (a step save, then the epoch end)
            shutil.rmtree(final)
        tmp.rename(final)
        deleted = []
        for old in complete(self.root)[: -self.keep]:
            shutil.rmtree(old, ignore_errors=True)
            deleted.append(old.name)
        if self.events is not None:
            size = sum(f.stat().st_size for f in final.iterdir() if f.is_file())
            self.events.emit("checkpoint_saved", step=state.step, name=final.name,
                             seconds=round(time.perf_counter() - t0, 2), bytes=size,
                             deleted=deleted)  # fmt: skip
        return final


def load(path: Path, model, opt, sched, gate=None) -> LoopState:
    """Restore a checkpoint into the model / optimizer / scheduler / RNGs; returns where the
    loop stands."""
    from safetensors.torch import load_file

    path = Path(path)
    if not (path / MARKER).exists():
        raise RuntimeError(f"{path} is not a complete checkpoint")
    load_trainable(model, load_file(str(path / "trainable.safetensors")))
    opt.load_state_dict(torch.load(path / "optimizer.pt", weights_only=True))
    sched.load_state_dict(torch.load(path / "scheduler.pt", weights_only=True))
    meta = json.loads((path / "state.json").read_text(encoding="utf-8"))
    set_rng_state(meta["rng"], torch.load(path / "rng.pt", weights_only=True), gate)
    return LoopState(
        epoch=meta["epoch"],
        next_group=meta["next_group"],
        step=meta["step"],
        tokens_seen=meta["tokens_seen"],
        extra=meta.get("extra") or {},
    )


def read_meta(path: Path) -> dict[str, Any]:
    return json.loads((Path(path) / "state.json").read_text(encoding="utf-8"))
