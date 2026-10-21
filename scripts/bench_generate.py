"""Measure batched greedy generation on this GPU (AE04): prefill vs decode, per batch size.

    python scripts/bench_generate.py --prompt 2400 --batches 1,4,8 --new 32

For each batch: the time of `generate` with 1 new token (the prefill) and with `--new` tokens
(forced: no early stop), so decode s/step = (t_new - t_1) / (new - 1). LoRA r16 is applied
(unmerged, as in the per-epoch eval), the attention is the chunked eager path.
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=2400)
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--new", type=int, default=32)
    args = ap.parse_args()

    from transformers import GenerationConfig

    from asktrainer.attn_chunked import use_chunked
    from asktrainer.config import LoraSpec
    from asktrainer.modelio import apply_lora, load_base
    from asktrainer.settings import get_paths

    model, tok, _ = load_base(get_paths().base_model)
    use_chunked(model)
    model = apply_lora(model, LoraSpec())
    model.eval()
    g = torch.Generator().manual_seed(0)
    for bs in [int(x) for x in args.batches.split(",")]:
        ids = torch.randint(100, 20000, (bs, args.prompt), generator=g).cuda()
        att = torch.ones_like(ids)
        times = {}
        for n in (1, args.new):
            gc = GenerationConfig(
                do_sample=False,
                max_new_tokens=n,
                min_new_tokens=n,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    input_ids=ids, attention_mask=att, generation_config=gc, use_cache=True
                )
            torch.cuda.synchronize()
            times[n] = time.perf_counter() - t
        step = (times[args.new] - times[1]) / (args.new - 1)
        print(json.dumps({"batch": bs, "prompt": args.prompt, "prefill_s": round(times[1], 2),
                          "decode_s_per_step": round(step, 3),
                          "decode_tok_per_s": round(bs / step, 1),
                          "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}),
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
