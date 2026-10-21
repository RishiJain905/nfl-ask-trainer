from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from conftest import example

from asktrainer.config import CONFIG_DIR, TrainConfig, assert_no_secrets
from asktrainer.data.schema import write_jsonl
from asktrainer.evaluation import (
    run_verifier,
    score_metrics,
    stratified_subset,
    verifier_command,
)
from asktrainer.mix import load_mix
from asktrainer.settings import Paths

CONFIGS = ["smoke.yaml", "general_pilot.yaml", "general_full.yaml", "nfl_sft_plain.yaml",
           "nfl_sft_jepa.yaml"]  # fmt: skip


@pytest.mark.parametrize("name", CONFIGS)
def test_shipped_configs_load(name):
    cfg = TrainConfig.from_yaml(CONFIG_DIR / name)
    cfg.logged()  # no secret-named keys
    assert cfg.objective == ("jepa" if name == "nfl_sft_jepa.yaml" else "plain")
    assert cfg.lora.targets == ["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
    if not name.startswith("nfl_"):  # the NFL track is parked: no NFL data, no NFL verifier
        assert (cfg.data.nfl_route, cfg.data.nfl_generate, cfg.data.nfl_val) == (0, 0, 0)
        assert not cfg.eval.nfl_verify and cfg.eval.nfl_n == 0


def test_general_configs_match_the_brief():
    pilot = TrainConfig.from_yaml(CONFIG_DIR / "general_pilot.yaml")
    full = TrainConfig.from_yaml(CONFIG_DIR / "general_full.yaml")
    for c in (pilot, full):
        assert (c.optim.batch, c.optim.grad_accum, c.optim.max_len, c.optim.lr) == (
            1,
            32,
            3072,
            1e-4,
        )
        assert (c.checkpoint.every_steps, c.checkpoint.keep) == (20, 2)
        assert set(c.data.public) == {"sqale", "gretel", "text2cypher", "cypherbench"}
    assert pilot.eval.every_steps == 100 and full.eval.every_steps == 500
    assert {k: v.train for k, v in full.data.public.items()} == {
        "sqale": 30000,
        "text2cypher": -1,
        "gretel": 20000,
        "cypherbench": 4000,
    }


def test_plain_and_jepa_differ_only_in_jepa():
    a = TrainConfig.from_yaml(CONFIG_DIR / "nfl_sft_plain.yaml").model_dump()
    b = TrainConfig.from_yaml(CONFIG_DIR / "nfl_sft_jepa.yaml").model_dump()
    assert {k for k in a if a[k] != b[k]} == {"jepa"}


@pytest.mark.parametrize("key", ["api_key", "WANDB_API_KEY", "hf_token", "password", "secret",
                                 "credentials", "apikey"])  # fmt: skip
def test_secret_keys_refused(key):
    with pytest.raises(ValueError):
        assert_no_secrets({"a": {key: "x"}})


def test_count_keys_allowed():
    assert_no_secrets({"max_new_tokens": {"route": 1}, "last_token": -2, "train_tokens": 9,
                       "gen_prefill_tokens": 1})  # fmt: skip


def test_unknown_key_refused(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("optim: {lr: 1.0e-4, nope: 1}\n")
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(p)


def _paths(tmp_path: Path) -> Paths:
    return Paths(data_root=tmp_path, base_model=tmp_path / "m", nfl_repo=tmp_path / "nfl",
                 uv_exe=tmp_path / "uv.exe")  # fmt: skip


def _export(paths: Paths, version: str = "v1") -> Path:
    d = paths.training / "nfl" / version
    tr = [example(i, "route" if i % 2 else "generate") for i in range(10)]
    va = [example(i, "route" if i % 2 else "generate", group=g, split="val")
          for i, g in zip(range(100, 112), ["seen", "unseen_template", "refuse"] * 4,
                          strict=True)]  # fmt: skip
    write_jsonl(d / "train.jsonl", tr)
    write_jsonl(d / "val.jsonl", va)
    (d / "manifest.json").write_text(json.dumps({"version": version}))
    (paths.training / "nfl" / "LATEST").write_text(version + "\n")
    return d


def test_latest_and_mix(tmp_path):
    paths = _paths(tmp_path)
    d = _export(paths)
    pub = paths.public / "gretel"
    write_jsonl(
        pub / "examples_train.jsonl",
        [example(i, "generate", origin="gretel", group="public") for i in range(30)],
    )
    write_jsonl(pub / "examples_val.jsonl",
                [example(i, "generate", origin="gretel", group="public", split="val")
                 for i in range(5)])  # fmt: skip
    cfg = TrainConfig.model_validate(
        {
            "data": {
                "nfl": "latest",
                "nfl_route": 4,
                "nfl_generate": 2,
                "nfl_val": -1,
                "public": {"gretel": {"train": 7, "val": 2}},
            }
        }
    )
    assert cfg.data.nfl_dir(paths) == d
    m = load_mix(cfg, paths)
    assert len(m.train) == 13 and len(m.val) == 14
    assert m.info["nfl_version"] == "v1" and m.nfl_val_path == d / "val.jsonl"
    again = load_mix(cfg, paths)
    assert [e.id for e in again.train] == [e.id for e in m.train]  # seeded
    other = load_mix(cfg.model_copy(update={"seed": 99}), paths)
    assert {e.id for e in other.train} != {e.id for e in m.train}


def test_nfl_is_off_by_default(tmp_path):
    paths = _paths(tmp_path)  # no NFL export at all: the default mix must not need one
    pub = paths.public / "gretel"
    write_jsonl(
        pub / "examples_train.jsonl",
        [example(i, "generate", origin="gretel", group="public") for i in range(6)],
    )
    cfg = TrainConfig.model_validate({"data": {"public": {"gretel": {"train": 3}}}})
    m = load_mix(cfg, paths)
    assert len(m.train) == 3 and m.nfl_dir is None and not m.val  # fmt: skip


def test_missing_public_dataset_is_an_error(tmp_path):
    paths = _paths(tmp_path)
    _export(paths)
    cfg = TrainConfig.model_validate({"data": {"public": {"sqale": {"train": 5}}}})
    with pytest.raises(FileNotFoundError):
        load_mix(cfg, paths)


def test_stratified_subset():
    exs = [example(i, t, group=g, split="val") for i, (t, g) in enumerate(
        [("route", "seen")] * 50 + [("generate", "seen")] * 30 + [("route", "refuse")] * 3
        + [("route", "unseen_template")] * 1)]  # fmt: skip
    sub = stratified_subset(exs, 20, seed=1)
    assert len(sub) == 20 and len({e.id for e in sub}) == 20
    assert {(e.group, e.task) for e in sub} == {(e.group, e.task) for e in exs}
    assert sub == stratified_subset(list(reversed(exs)), 20, seed=1)  # order independent
    assert len(stratified_subset(exs, 1000, seed=1)) == len(exs)


SCORES = {
    "summary": {
        "all": {"n": 4, "exec_acc": 0.5, "json_ok": 1.0, "action_acc": 0.75,
                "template_acc": 0.5, "params_exact": None},
        "task": {"route": {"n": 3, "exec_acc": 0.33, "json_ok": 1.0}},
        "group": {"seen": {"n": 2, "exec_acc": 1.0}, "refuse": {"n": 2, "exec_acc": 0.0}},
    }
}  # fmt: skip


def test_score_metrics():
    m = score_metrics(SCORES)
    assert m["val/exec_acc"] == 0.5 and m["val/action_acc"] == 0.75
    assert "val/params_exact" not in m  # None is not logged
    assert m["val/seen/exec_acc"] == 1.0 and m["val/refuse/exec_acc"] == 0.0
    assert m["val/task/route/exec_acc"] == 0.33 and m["val/verified_n"] == 4.0


def test_verifier_command_and_run(tmp_path):
    paths = Paths(data_root=Path("/mnt/d/nfl-ml-data"), base_model=Path("/mnt/d/m"),
                  nfl_repo=Path("/mnt/f/Personal/NFL/NFL-ML-AI"),
                  uv_exe=Path("/mnt/c/Users/x/.local/bin/uv.exe"))  # fmt: skip
    cmd = verifier_command(paths, Path("/mnt/d/a/val.jsonl"), Path("/mnt/d/r/p.jsonl"),
                           Path("/mnt/d/r/s.json"))  # fmt: skip
    assert cmd[:4] == ["/mnt/c/Users/x/.local/bin/uv.exe", "run", "--project",
                       "F:\\Personal\\NFL\\NFL-ML-AI"]  # fmt: skip
    assert cmd[4:7] == ["nfl", "ask", "verify"] and "D:\\a\\val.jsonl" in cmd

    out = tmp_path / "scores.json"
    seen = {}

    def ok_runner(cmd, cwd, capture_output, text, timeout):
        seen["cwd"], seen["timeout"] = cwd, timeout
        out.write_text(json.dumps(SCORES))
        return subprocess.CompletedProcess(cmd, 0, "fine", "")

    p = paths.model_copy(update={"nfl_repo": tmp_path})
    assert run_verifier(p, tmp_path / "v", tmp_path / "p", out, 77, runner=ok_runner) == SCORES
    assert seen == {"cwd": str(tmp_path), "timeout": 77}
    assert out.with_suffix(".log").exists()

    def bad_runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 3, "", "boom")

    with pytest.raises(RuntimeError):
        run_verifier(p, tmp_path / "v", tmp_path / "p", tmp_path / "s2.json", runner=bad_runner)
