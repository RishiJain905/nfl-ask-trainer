"""Probe the base model on this GPU before designing runs around it (AE04).

    python scripts/probe_model.py --model /mnt/d/LOCAL-Models/XHToken/Spark-X2.6-4B

Prints (and writes as JSON with --out): load time, the chat template's no-thinking rendering
and special tokens, eager vs SDPA attention (max difference and speed), LoRA training step
time and peak memory per sequence length (with gradient checkpointing), and greedy generation
speed. Nothing is saved but the JSON.
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", default="512,1024,2048,3072")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from asktrainer.attention import use_sdpa

    report: dict = {"torch": torch.__version__, "hip": torch.version.hip}
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0}
    )
    torch.cuda.synchronize()
    report["load_s"] = round(time.perf_counter() - t0, 1)
    report["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
    report["mem_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    print("load", report["load_s"], "s;", report["params_b"], "B params;",
          report["mem_after_load_gb"], "GB")  # fmt: skip

    msgs = [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
        {"role": "assistant", "content": '{"action": "refuse"}'},
    ]
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    report["template_full"] = full
    report["template_prompt"] = prompt
    report["prompt_is_prefix"] = full.startswith(prompt)
    report["special"] = {
        "bos": tok.bos_token, "eos": tok.eos_token, "pad": tok.pad_token,
        "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id,
        "added": [t for t in tok.get_added_vocab()][:60],
    }  # fmt: skip
    print(json.dumps({k: report[k] for k in ("prompt_is_prefix", "special")}, indent=1))
    print(repr(full))

    # eager vs SDPA: the same logits?
    ids = torch.randint(100, 20000, (1, 1024), device="cuda")
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        t = time.perf_counter()
        a = model(input_ids=ids).logits.float()
        torch.cuda.synchronize()
        eager_s = time.perf_counter() - t
        use_sdpa(model)
        torch.cuda.synchronize()
        t = time.perf_counter()
        b = model(input_ids=ids).logits.float()
        torch.cuda.synchronize()
        sdpa_s = time.perf_counter() - t
        use_sdpa(model, on=False)
    report["sdpa_max_abs_diff"] = float((a - b).abs().max())
    report["sdpa_argmax_agree"] = float((a.argmax(-1) == b.argmax(-1)).float().mean())
    report["fwd_1024_eager_s"], report["fwd_1024_sdpa_s"] = round(eager_s, 3), round(sdpa_s, 3)
    print("sdpa diff", report["sdpa_max_abs_diff"], "argmax agree", report["sdpa_argmax_agree"],
          "eager", eager_s, "sdpa", sdpa_s)  # fmt: skip
    del a, b

    # LoRA training steps
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        target_modules=["q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )  # fmt: skip
    pm = get_peft_model(model, lora)
    report["trainable_m"] = round(
        sum(p.numel() for p in pm.parameters() if p.requires_grad) / 1e6, 2
    )
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-4)
    pm.train()
    inner = pm.get_base_model().model
    emb = pm.get_base_model().model.embedding.weight
    steps = {}
    for impl in ("eager", "sdpa"):
        use_sdpa(pm.get_base_model(), on=impl == "sdpa")
        for n in [int(x) for x in args.lens.split(",")]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randint(100, 20000, (1, n), device="cuda")
            times = []
            try:
                for _ in range(3):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    h = inner(input_ids=x).last_hidden_state
                    tgt = h[:, -200:-1, :]  # loss on the last 200 tokens only (the target)
                    logits = torch.nn.functional.linear(tgt, emb).float()
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), x[:, -199:].reshape(-1)
                    )
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t)
                s = sorted(times)[1]
                steps[f"{impl}_{n}"] = {
                    "step_s": round(s, 3), "tok_per_s": round(n / s),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }  # fmt: skip
            except torch.OutOfMemoryError:
                steps[f"{impl}_{n}"] = {"oom": True}
            print(impl, n, steps[f"{impl}_{n}"])
    report["train_steps"] = steps

    # generation (greedy, no thinking), SDPA on
    use_sdpa(pm.get_base_model(), on=True)
    pm.eval()
    p_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    pad = torch.full((1, 1500), 500, device="cuda")
    p_ids = torch.cat([p_ids[:, :1], pad, p_ids[:, 1:]], dim=1)  # a ~1.5k-token prompt
    with torch.no_grad():
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = pm.generate(input_ids=p_ids, max_new_tokens=128, do_sample=False, min_new_tokens=128)
        torch.cuda.synchronize()
    gen_s = time.perf_counter() - t
    report["gen_128_after_1500_s"] = round(gen_s, 2)
    report["gen_tok_per_s"] = round((out.shape[1] - p_ids.shape[1]) / gen_s, 1)
    print("generate", report["gen_128_after_1500_s"], "s", report["gen_tok_per_s"], "tok/s")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
