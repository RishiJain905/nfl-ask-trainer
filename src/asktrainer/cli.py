"""`ask-trainer`: the trainer's command line.

    ask-trainer doctor                      # paths, GPU, base model, NFL repo + uv, W&B, data
    ask-trainer train --config configs/sft_plain.yaml [--seed N] [--name X] [--launched-by rishi]
    ask-trainer train --resume <run folder or name>
    ask-trainer predict --run DIR --data FILE [--n 160] [--verify]
    ask-trainer merge --run DIR [--out DIR]
    ask-trainer probe [probe_model.py args]
    ask-trainer public ...                  # = python -m asktrainer.data.public ...

Never prints a secret: `doctor` reports W&B as set / not set only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from asktrainer.settings import REPO, get_paths


def _count_lines(p: Path) -> int:
    with p.open("rb") as f:
        return sum(1 for line in f if line.strip())


def doctor(args) -> int:
    paths = get_paths()
    ok = True

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else 'XX'}] {label}" + (f": {detail}" if detail else ""))

    check("data root", paths.data_root.is_dir(), str(paths.data_root))
    check("training dir", paths.training.is_dir(), str(paths.training))
    check("runs dir", paths.runs.is_dir() or paths.runs.parent.is_dir(), str(paths.runs))
    bm = paths.base_model
    need = ["config.json", "modeling_spark.py", "configuration_spark.py", "tokenizer.json",
            "chat_template.jinja", "model.safetensors.index.json"]  # fmt: skip
    missing = [f for f in need if not (bm / f).exists()]
    shards = sorted(bm.glob("model-*.safetensors"))
    check("base model", not missing and bool(shards),
          f"{bm} ({len(shards)} shards)" + (f", missing {missing}" if missing else ""))  # fmt: skip
    try:
        import torch

        gpu = torch.cuda.is_available()
        detail = "no GPU visible"
        if gpu:
            p = torch.cuda.get_device_properties(0)
            detail = f"{p.name}, {p.total_memory / 1e9:.1f} GB, torch {torch.__version__}"
        check("GPU", gpu, detail)
    except Exception as e:
        check("GPU", False, f"torch failed: {type(e).__name__}")
    check("NFL repo", (paths.nfl_repo / "pyproject.toml").exists(), str(paths.nfl_repo))
    try:
        r = subprocess.run([str(paths.uv_exe), "--version"], capture_output=True, text=True,
                           timeout=60)  # fmt: skip
        check("uv.exe", r.returncode == 0, r.stdout.strip() or f"exit {r.returncode}")
    except Exception as e:
        check("uv.exe", False, f"{paths.uv_exe}: {type(e).__name__}")
    from asktrainer.settings import get_secrets

    key_set = get_secrets().is_set("wandb_api_key")
    stored = False
    if not key_set:
        try:
            import wandb

            stored = bool(wandb.login(prompt=False, verify=False))
        except Exception:
            stored = False
    print(f"[--] W&B: WANDB_API_KEY {'set' if key_set else 'not set'}; stored login "
          f"{'found' if stored else ('not checked' if key_set else 'not found')}; runs log "
          f"{'online' if key_set or stored else 'OFFLINE (sync later)'}")  # fmt: skip
    latest = paths.training / "nfl" / "LATEST"
    if latest.exists():
        d = paths.training / "nfl" / latest.read_text(encoding="utf-8").strip()
        files = {f: (d / f).exists() for f in ("train.jsonl", "val.jsonl", "manifest.json")}
        detail = f"{d.name}: " + ", ".join(
            f"{f} {_count_lines(d / f) if f.endswith('.jsonl') else ''}".strip()
            for f, there in files.items() if there
        )  # fmt: skip
        check("NFL export (latest)", all(files.values()), detail)
    else:
        print(f"[--] NFL export: none yet ({latest} missing)")
    pub = paths.public
    if pub.is_dir():
        for d in sorted(p for p in pub.iterdir() if p.is_dir()):
            parts = [f"{s} {_count_lines(d / f'examples_{s}.jsonl')}" for s in ("train", "val")
                     if (d / f"examples_{s}.jsonl").exists()]  # fmt: skip
            print(f"[--] public/{d.name}: " + (", ".join(parts) or "not converted yet"))
    else:
        print(f"[--] public datasets: none ({pub} missing)")
    return 0 if ok else 1


def train_cmd(args) -> int:
    from asktrainer.config import TrainConfig

    runs = get_paths().runs
    if args.resume:  # the run's own config (read here: the backend flags precede torch)
        run = Path(args.resume)
        run = run if (run / "config.json").exists() else runs / args.resume
        saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
        cfg = TrainConfig.model_validate({k: v for k, v in saved.items()
                                          if k in TrainConfig.model_fields})  # fmt: skip
    else:
        if not args.config:
            raise SystemExit("train: --config (or --resume) is required")
        cfg = TrainConfig.from_yaml(args.config)
        if args.nfl:
            cfg.data.nfl = args.nfl
        if args.jepa:
            cfg.jepa.enabled = True
    for k, v in cfg.backend_flags(runs).items():  # before torch starts
        os.environ.setdefault(k, v)
    if cfg.optim.tunableop:
        (runs / "tunableop").mkdir(parents=True, exist_ok=True)
    from asktrainer.train import train

    if args.resume:
        run_dir = train(resume=args.resume, device=args.device, stop_after=args.stop_after)
    else:
        run_dir = train(cfg, seed=args.seed, name=args.name, launched_by=args.launched_by,
                        device=args.device, stop_after=args.stop_after,
                        config_path=args.config)  # fmt: skip
    print(f"run folder: {run_dir}")
    return 0


def _load_run(run: Path, device: str):
    from peft import PeftModel

    from asktrainer.config import TrainConfig
    from asktrainer.modelio import load_base

    cfg_json = json.loads((run / "config.json").read_text(encoding="utf-8"))
    keys = set(TrainConfig.model_fields)
    cfg = TrainConfig.model_validate({k: v for k, v in cfg_json.items() if k in keys})
    model, tok, _ = load_base(cfg.base_model_path(), device=device)
    if cfg.optim.attn_chunk > 0:
        from asktrainer.attn_chunked import use_chunked

        use_chunked(model, chunk=cfg.optim.attn_chunk, threshold=cfg.optim.attn_chunk_above)
    adapter = run / "adapter"
    if not adapter.exists():
        epochs = sorted(run.glob("adapter_epoch*"), key=lambda p: int(p.name[13:]))
        if not epochs:
            raise FileNotFoundError(f"no adapter in {run}")
        adapter = epochs[-1]
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return cfg, cfg_json, model, tok, adapter


def predict_cmd(args) -> int:
    from asktrainer.chat import ChatTokenizer
    from asktrainer.data.schema import read_jsonl
    from asktrainer.evaluation import (
        predict,
        run_verifier,
        score_metrics,
        stratified_subset,
        write_preds,
    )

    run = Path(args.run)
    cfg, _, model, tok, adapter = _load_run(run, args.device)
    examples = list(read_jsonl(Path(args.data)))
    if args.n:
        examples = stratified_subset(examples, args.n, cfg.seed)
    ct = ChatTokenizer(tok, cfg.optim.max_len)
    preds = predict(model, ct, examples, cfg.eval.max_new_tokens, args.device,
                    batch_size=cfg.eval.gen_batch,
                    prefill_tokens=cfg.eval.gen_prefill_tokens)  # fmt: skip
    out = Path(args.out) if args.out else run / f"preds_{Path(args.data).stem}.jsonl"
    write_preds(out, preds)
    print(f"{len(preds)} predictions from {adapter.name} -> {out}")
    if args.verify:
        scores = run_verifier(get_paths(), Path(args.data), out, out.with_suffix(".scores.json"),
                              cfg.eval.verify_timeout_s)  # fmt: skip
        print(json.dumps(score_metrics(scores), indent=1))
    return 0


def evalrun_cmd(args) -> int:
    """Score a run's llama.cpp answers to the evalset (llamaeval generate) and add the result
    to the run's folder and W&B run."""
    from asktrainer import publiceval
    from asktrainer.data.schema import read_jsonl
    from asktrainer.train import log_eval_run, log_scored_eval, resolve_run

    paths = get_paths()
    run = resolve_run(args.run, paths) if args.run else None
    exs = read_jsonl(
        Path(args.examples) if args.examples else paths.public / "evalset" / "examples_val.jsonl"
    )
    preds = [
        json.loads(x)
        for x in Path(args.preds).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    res = publiceval.verify(exs, preds, publiceval.PublicData(paths.public))
    res["tag"], res["step"], res["preds"] = args.tag, args.step, str(args.preds)
    out = (run or Path(args.preds).parent) / f"scores_llamacpp_{args.tag}.json"
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    metrics = publiceval.metrics(res)
    if args.no_wandb:
        wid = None
    elif run is not None:
        wid = log_scored_eval(run, args.tag, args.step, metrics)
    else:  # the base model: an evaluation run of its own
        wid = log_eval_run(f"evalset-{args.tag}-{time.strftime('%Y%m%d-%H%M%S')}", args.tag,
                           metrics, {"preds": str(args.preds), "evalset": len(exs)},
                           Path(args.preds).parent)  # fmt: skip
    print(json.dumps({"run": run.name if run else None, "tag": args.tag, "step": args.step,
                      "scores": str(out),
                      "wandb": wid, **{k: round(v, 4) for k, v in metrics.items()}}))  # fmt: skip
    return 0


def merge_cmd(args) -> int:
    from asktrainer.modelio import merge_and_save

    run = Path(args.run)
    cfg_json = json.loads((run / "config.json").read_text(encoding="utf-8"))
    adapter = run / "adapter"
    out = Path(args.out) if args.out else run / "merged"
    card = {
        "run": cfg_json.get("run", run.name),
        "data_version": cfg_json.get("nfl_version"),
        "objective": cfg_json.get("objective"),
        "seed": cfg_json.get("seed"),
        "lora": cfg_json.get("lora"),
        "jepa": cfg_json.get("jepa") if cfg_json.get("objective") == "jepa" else None,
    }
    base = cfg_json.get("base_model") or str(get_paths().base_model)
    merge_and_save(adapter, out, base, card=card, device=args.device)
    return 0


def probe_cmd(args, rest: list[str]) -> int:
    cmd = [sys.executable, str(REPO / "scripts" / "probe_model.py")]
    if "--model" not in rest:
        cmd += ["--model", str(get_paths().base_model)]
    return subprocess.call(cmd + rest)


def public_cmd(rest: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", "asktrainer.data.public", *rest])


def traces_cmd(args) -> int:
    """`traces prep` writes a run's questions and plan (generation runs on Windows);
    `traces export` turns its passing answers into `call` training examples."""
    from asktrainer import traceprep

    paths = get_paths()
    run_dir = paths.traces / args.name
    if args.action == "export":
        from asktrainer import traceexport

        out = paths.public / f"traces-{args.name}"
        print(json.dumps(traceexport.export(run_dir, out, args.val_share, args.seed)))
        return 0
    if args.action == "evalset":
        from asktrainer import traceexport

        n = int(args.per_set) if args.per_set.isdigit() else 50
        print(json.dumps(traceexport.evalset(paths.public, paths.public / args.name, n, args.seed)))
        return 0
    plan = traceprep.parse_plan(args.plan)
    cfg = traceprep.prep(
        run_dir,
        paths.public,
        traceprep.parse_sizes(args.per_set),
        plan,
        seed=args.seed,
        mix=traceprep.parse_mix(args.mix) if args.mix else None,
        tries=args.tries,
        budget=args.budget,
    )
    keys = ("questions", "requests", "plan", "mix", "levels", "sets")
    print(json.dumps({"run": str(run_dir), **{k: cfg[k] for k in keys}}))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["public"]:
        return public_cmd(argv[1:])
    ap = argparse.ArgumentParser(prog="ask-trainer", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check paths, GPU, base model, NFL repo, W&B, data")
    t = sub.add_parser("train", help="train one run")
    t.add_argument("--config", default=None)
    t.add_argument(
        "--resume",
        default=None,
        help="a run folder or name: continue its newest checkpoint (same data order, same W&B run)",
    )
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--name", default=None)
    t.add_argument("--launched-by", default="agent")
    t.add_argument("--nfl", default=None, help="the NFL export (version, folder or latest)")
    t.add_argument("--jepa", action="store_true", help="turn the config's JEPA term on")
    t.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="stop dead after N optimizer steps, as a crash would (resume tests)",
    )
    t.add_argument("--device", default="cuda")
    p = sub.add_parser("predict", help="generate with a run's adapter")
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=0, help="a stratified subset of this size")
    p.add_argument("--out", default=None)
    p.add_argument("--verify", action="store_true", help="score with `nfl ask verify`")
    p.add_argument("--device", default="cuda")
    ev = sub.add_parser("evalrun", help="score a run's llama.cpp evalset answers (llamaeval)")
    ev.add_argument("--run", default=None, help="none: the base model (its own W&B eval run)")
    ev.add_argument("--preds", required=True)
    ev.add_argument("--tag", required=True, help="e.g. final, step000050")
    ev.add_argument("--step", type=int, required=True, help="the training step of the adapter")
    ev.add_argument("--examples", default=None, help="default: <training>/public/evalset")
    ev.add_argument("--no-wandb", action="store_true")
    m = sub.add_parser("merge", help="merge a run's adapter into a servable bf16 model")
    m.add_argument("--run", required=True)
    m.add_argument("--out", default=None)
    m.add_argument("--device", default="cuda")
    tr = sub.add_parser(
        "traces", help="thinking-trace runs: `traces prep` writes questions and plan"
    )
    tr.add_argument("action", choices=["prep", "export", "evalset"])
    tr.add_argument("--name", required=True)
    tr.add_argument("--per-set", default="25", help="N for every set, or sqale:N,gretel:N,...")
    tr.add_argument("--plan", default="high:4,low:2,off:1", help="tries per thinking level")
    tr.add_argument(
        "--mix", default=None, help="one level per question, drawn with these weights: high:W,..."
    )
    tr.add_argument("--tries", type=int, default=2, help="with --mix: tries at most, adaptive")
    tr.add_argument("--budget", type=int, default=None, help="the server's --reasoning-budget")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--val-share", type=float, default=0.1, help="export: questions held out")
    sub.add_parser("probe", help="run scripts/probe_model.py (extra args passed through)")
    sub.add_parser("public", help="public datasets (python -m asktrainer.data.public ...)")
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "probe":
        return probe_cmd(args, rest)
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.cmd == "doctor":
        return doctor(args)
    if args.cmd == "train":
        return train_cmd(args)
    if args.cmd == "predict":
        return predict_cmd(args)
    if args.cmd == "merge":
        return merge_cmd(args)
    if args.cmd == "evalrun":
        return evalrun_cmd(args)
    if args.cmd == "traces":
        return traces_cmd(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
