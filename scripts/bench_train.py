"""Measure LoRA training throughput on this GPU with the real loss code (AE04).

    python scripts/bench_train.py --seqs 2048,4096,5200 --batches 1,2 [--jepa] [--out b.jsonl]

Run once per backend setting (TunableOp / hipBLASLt are process-wide; set them in the
environment: PYTORCH_TUNABLEOP_ENABLED=1, PYTORCH_TUNABLEOP_FILENAME=...,
TORCH_BLAS_PREFER_HIPBLASLT=1). Synthetic rows: `seq` tokens with the last `--target`
tokens as the target (like an Ask prompt + reply); with `--jepa` each row also carries a
40-token text view and a `--target`-token code view, packed in-row. Attention is the chunked
eager path above `--chunk-above` tokens, as in training (`--chunk-above 0` = always the
original). Prints one JSON line per (seq, batch): tokens/s (all tokens processed, views
included), peak GB, step seconds (median after a warm-up step).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="2048")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--batches", default="1,2")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--jepa", action="store_true")
    ap.add_argument("--chunk-above", type=int, default=3072)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from asktrainer.attn_chunked import use_chunked
    from asktrainer.chat import Features, collate
    from asktrainer.config import LoraSpec
    from asktrainer.modelio import apply_lora, load_base
    from asktrainer.settings import get_paths
    from asktrainer.train import forward_losses

    model, tok, info = load_base(get_paths().base_model)
    if args.chunk_above > 0:
        use_chunked(model, chunk=1024, threshold=args.chunk_above)
    model = apply_lora(model, LoraSpec(r=16, alpha=32, dropout=0.05))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    g = torch.Generator().manual_seed(0)
    setting = {
        "tunableop": os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0"),
        "hipblaslt": os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "0"),
        "jepa": args.jepa,
        "chunk_above": args.chunk_above,
        "load_s": info["load_s"],
        "read_mb_s": info["read_mb_s"],
        "tied": info["tied"],
    }
    print(json.dumps(setting), flush=True)

    def feats(n: int, seq: int) -> list[Features]:
        out = []
        for i in range(n):
            ids = torch.randint(100, 20000, (seq,), generator=g).tolist()
            f = Features(f"x{i}", "nfl", "route", "seen", ids, seq - args.target)
            if args.jepa:
                f.text_ids = torch.randint(100, 20000, (40,), generator=g).tolist()
                f.code_ids = torch.randint(100, 20000, (args.target,), generator=g).tolist()
                f.text_pos, f.code_pos = 38, args.target - 2
            out.append(f)
        return out

    rows = []
    for seq in [int(x) for x in args.seqs.split(",")]:
        for bs in [int(x) for x in args.batches.split(",")]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times = []
            try:
                for _ in range(args.steps):
                    b = collate(feats(bs, seq), tok.pad_token_id, with_views=args.jepa).to("cuda")
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    ce, dist = forward_losses(model, b)
                    loss = ce / b.n_target + (dist.mean() if dist is not None else 0.0)
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t)
                rest = sorted(times[1:])
                s = rest[len(rest) // 2]
                row = {**setting, "seq": seq, "batch": bs, "step_s": round(s, 3),
                       "tok_per_s": round(b.n_tokens / s), "first_step_s": round(times[0], 2),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}  # fmt: skip
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                row = {**setting, "seq": seq, "batch": bs, "oom": True}
            print(json.dumps(row), flush=True)
            rows.append(row)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
