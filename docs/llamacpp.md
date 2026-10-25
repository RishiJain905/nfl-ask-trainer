# llama.cpp: serving Spark-X2.6-4B for trace generation (and RL rollouts later)

The trainer trains in bf16 with transformers. Generating *many* answers (thinking traces now, RL
rollouts later) runs on llama.cpp instead: a Q8_0 GGUF of the base model behind `llama-server`,
which runs many requests at once with continuous batching. This file is what was built, how it
was checked, what it measured and how to run it. All numbers are from 2026-10-20 on this
machine (RX 7800 XT 16 GB, Ryzen 7 7700X, Windows 11, ROCm/HIP 7.1).

## What is where

| Thing | Where |
|---|---|
| llama.cpp build used (default in `serve.ps1`) | **Rishi's TurboQuant fork + the Spark port**: worktree `F:\Personal\TQ-Optimizer-Test\atomic-llama-cpp-turboquant-spark` (branch `merge/upstream-spark`: AtomicBot release b10298-2.0.0 + upstream's Spark commit `3ad1ba733` + Rishi's `reasoning_effort` server edit, uncommitted), built in its `build-rdna3-gfx1101\`. Verified against upstream (same perplexity and speed; TurboQuant unchanged on a 27B): its `docs\perf\spark-port\REPORT.md` |
| Clean upstream build (reference) | `ggml-org/llama.cpp` tag **b11779** in `F:\Personal\TQ-Optimizer-Test\llama-cpp-upstream`, built in `build-rdna3-gfx1101\` (`scripts/llamacpp/build_upstream.cmd`, ~25 min); the quantisation and the measurements below were made with it |
| Rishi's fork, main checkout | `F:\Personal\TQ-Optimizer-Test\atomic-llama-cpp-turboquant`: untouched (its own release predates Spark support, which needs upstream b10828+). A full upstream merge is paused in a third worktree, `atomic-llama-cpp-turboquant-upstream` (`MERGE_STATUS.md` there) |
| Q8_0 model (served) | `F:\LOCAL-Models\XHToken\Spark-X2.6-4B-GGUF\Spark-X2.6-4B-Q8_0.gguf` (4.38 GB; NVMe, loads in ~2 s; sha256 `3e6b74bc...dff517a1`, the D: copy identical) |
| Archive | `D:\LOCAL-Models\XHToken\Spark-X2.6-4B-GGUF\`: the BF16 GGUF (8.23 GB, sha256 `874a8e72...97ebbc71a`), a copy of the Q8_0, and the F16-embedding variant (4.69 GB) |
| Measurement logs | `D:\nfl-ml-data\ask\quant\spark-x2.6-4b\` (`quantize-*`, `kld-*`, `bench-*`, `batched-*`, the KLD text) |
| Server logs, pid files | `D:\nfl-ml-data\ask\llamacpp\` |

The Q8 is **not a LoRA**: it is the base model, quantised for fast generation. Training stays bf16.

## How it was made

1. **BF16 GGUF** (inside WSL, trainer venv; ~2 min):
   `python convert_hf_to_gguf.py /mnt/f/LOCAL-Models/XHToken/Spark-X2.6-4B --outtype bf16 --outfile .../Spark-X2.6-4B-BF16.gguf`
   (upstream's converter has `conversion/spark2_5.py`; 290 tensors, the chat template embedded).
2. **Quantise** (`quant_eval.ps1 -Step quantize`, ~10 s each): all 217 weight matrices to Q8_0,
   the 73 norm / small tensors stay F32. The variant `Q8_0-embF16` keeps the token embedding at
   F16; Spark ties it to the output head, so that one matrix (335M parameters) is both.
3. **Fidelity vs BF16** (`-Step kldbase`, then `-Step kld`): llama.cpp's KL-divergence test.
   BF16 scores a text and saves its full next-token distributions; each quant then scores the same
   tokens and is compared position by position. The text is 240 held-out validation examples (60
   per public set: SQaLe, gretel, text2cypher, CypherBench) rendered with Spark's chat template,
   147,968 tokens (`make_kld_text.py`); the test uses 24 chunks of 2,048 tokens (24,576 scored
   positions). The saved BF16 distributions take 6.4 GB and are deleted afterwards.
4. **Speed** (`-Step bench`, `-Step batched -Variant <v>`).

Why only Q8_0 and one variant: at 8 bits the whole model is near-lossless, so the only tensor
worth protecting is the shared embedding / output head (Rishi's QuantizationResearch protects
sensitive tensors at Q8_0 when the bulk goes to 2-3 bits; at Q8 there is nothing below it to
trade). imatrix calibration does not change Q8_0, so it was not used.

## Results

Fidelity vs BF16 (BF16 perplexity on this text: 2.168 +/- 0.030):

| Variant | Size | PPL ratio | Mean KLD | 99.9% KLD | Max KLD | Same top-1 token |
|---|---|---|---|---|---|---|
| Q8_0 | 4.38 GB | 1.0040 +/- 0.0009 | 0.00080 +/- 0.00007 | 0.050 | 1.53 | 99.42 +/- 0.05 % |
| Q8_0-embF16 | 4.69 GB | 1.0038 +/- 0.0009 | 0.00073 +/- 0.00004 | 0.048 | 0.67 | 99.42 +/- 0.05 % |

Speed, one request (`llama-bench`, flash attention, 3 repeats):

| Model | Prompt 512 | Prompt 2,048 | Generation |
|---|---|---|---|
| BF16 | 3,916 tok/s | 3,771 tok/s | 57.8 tok/s |
| Q8_0 | 3,860 | 3,669 | **89.4** |
| Q8_0-embF16 | 3,842 | 3,672 | 86.5 |

Many requests at once (`llama-batched-bench`, 1,024-token prompts, 512 generated tokens each),
total generation tokens/s across all requests:

| Requests at once | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Q8_0 | 87 | 313 | 480 | **796** | 791 |
| Q8_0-embF16 | 84 | 298 | 469 | 752 | 775 |
| BF16 | 57 | 178 | 324 | 419 | 689 |

For comparison, transformers `generate` in this repo decodes at 0.05 s/step at batch 1 (~20
tok/s) and 0.095 s/step at batch 4 (~42 tok/s), and caps a batch at 8 rows.

**Choice: plain Q8_0.** Both variants pick the same top token as BF16 at 99.42% of positions;
the F16 embedding lowers the mean KL by an amount inside its error bars and costs 3-6% speed.
Every trace is checked by executing its query, so a rare near-tie flip shows up as a rejected
sample, never as wrong training data. Q8_0 is 1.5x BF16 for one request and 1.9x at 16; the
gain flattens after 16 requests, so the server uses 16 slots.

## Running it

```
powershell -File scripts\llamacpp\serve.ps1          # start: 127.0.0.1:8090, 16 slots, one shared 65,536-token KV pool
python scripts\llamacpp\server_smoke.py              # thinking high / low / off + a tool call
powershell -File scripts\llamacpp\serve.ps1 -Stop    # stop the server serve.ps1 started
```

**One GPU job at a time**: the server holds 7.6 GB of VRAM (measured: 0.96 GB before, 8.60 GB
with the server loaded); stop it before a training run.

**The slots share one KV pool** (`-kvu`). Without it llama.cpp splits the context evenly, 4,096
tokens per slot, and a long SQaLe prompt (up to ~3,000 tokens) plus a thinking trace would not
fit. With it any one request can use up to 65,536 tokens while 16 run at once; Spark's KV cache
is small (9 of 36 layers use full attention, 4 KV heads, ~36 KB per token at f16; the other 27
layers keep only a 512-token window).

**The host-RAM prompt cache is off** (`--cache-ram 0 -ctxcp 0`). By default llama-server saves
each finished request's state to a RAM cache (8 GB): for Spark that's ~250 MB per request and
1.5-3 s on the server's single main loop, during which all 16 slots wait. In the first trace run
each slot generated only 4-8 tokens/s that way (about 40 tokens/s in all); with the cache off,
16-24 tokens/s per slot while new prompts keep arriving. The cache only pays off when later
requests repeat a long prefix, which bulk generation doesn't rely on.

Two environment settings matter and `serve.ps1` sets both for its own process:
`HIP_VISIBLE_DEVICES=1` (device 0 is the CPU's integrated GPU) and the ROCm 7.1 `bin` on `PATH`
(hipBLAS / rocBLAS DLLs).

Requests go to the OpenAI-style `/v1/chat/completions` with `tools`:

- **Thinking** is on by default (the template's own default, `thinking_effort=high`). Low:
  `"chat_template_kwargs": {"reasoning_effort": "low"}`. Off: `{"enable_thinking": false}`. Spark's
  template accepts only `high` and `low` and raises an error otherwise, so never pass the
  server-wide `--reasoning-effort medium`.
- The server splits the reply: the thought goes to `message.reasoning_content`, the call to
  `message.tool_calls` (arguments as a JSON string). Upstream llama.cpp parses Spark's format
  out of the box. The raw text the model writes is
  `...</think><tool_call>run_query<arg_key>query</arg_key><arg_value>SELECT ...</arg_value></tool_call>`
  (`/completion` with the prompt from `/apply-template` shows it).
- The server warns that the template "supports preserving reasoning" and keeps earlier turns'
  thoughts in multi-turn histories; irrelevant for single-turn traces.

What the smoke test showed on the base model (one SQL question, temperature 0.6): thoughts of
~380 characters (~100 tokens) in a terse note style, 2.1-2.2 s per answer alone, valid tool
calls at all three levels, and errors a fine-tune should remove (one call selected a column from
the wrong table; one thinking-off call dropped the "won" condition).

## Trace-generation speed (measured on the pilot run, 2026-10-25)

The pilot trace run's round 1 (11,071 answers, 5.65 h, 16 slots, `--reasoning-budget 1024`) ran
at 32.6 answers a minute: 242 generated + 405 prompt tokens a second, 745 prompt and 444
generated tokens per answer, 29.4 s per answer. `llama-batched-bench` makes 796 tokens a second
at 16; three measurements say where the difference goes:

- **Not the client.** The 16 `tracegen` workers spent 100% of their time waiting on the server
  (the sum of the answers' seconds over 16 x the wall time), and the inline check costs 3 ms an
  SQL answer (6 ms with checker version 4) against ~29 s of generation.
- **Prompts stall decoding.** From the server log's per-slot readings (every ~3 s), in 6-second
  windows: with all 16 slots decoding, 301 tokens a second (18.8 a slot); with 15, 245; with
  12-13 (the others processing a new prompt), 142-166. A new prompt arrives about every 2 s,
  and a step that carries one is slow for every slot.
- **The decoding steps themselves are ~2.6x slower than the bench's** (301 vs 796 at 16): the
  bench decodes fixed tokens with no sampling and no arriving prompts; the server also samples
  each slot's token on the CPU, through the default sampler chain, every step. A single slot
  near the end of the run made 43 tokens a second (66 in a 3-s burst), against `llama-bench`'s
  89.

The settings to try are in `scripts/llamacpp/bench_trace.py` (GPU, ~10 minutes each; it replays
one seeded sample of the run's questions, the real requests, with a minute's warm-up):
`default` (as above), `np24` / `np32` (more slots, a bigger KV pool), `ub1024` (larger
micro-batches, fewer prompt passes), `lean_samplers` (`--samplers top_k;top_p;min_p;temperature`:
the default chain without the stages that do nothing at our settings, so the same distribution)
and `upstream` (the clean b11779 build, without the fork's edits). Not run yet: the GPU was in use.

```
C:\Python314\python.exe scripts\llamacpp\bench_trace.py --minutes 8
```

## Later

- **The fine-tuned model in llama.cpp.** Either merge the LoRA into the bf16 weights
  (`ask-trainer merge`), convert to GGUF and quantise to Q8_0 again (two commands above), or
  convert the adapter alone (`convert_lora_to_gguf.py`) and load it on the base Q8_0 with
  `--lora`. The second avoids re-quantising after every RL update.
- **Speculative decoding** (`--spec-type ngram-*`: drafts by copying from the prompt, and table
  and column names come from the schema) is not measured yet. It mostly speeds up single
  requests; with 16 busy slots the GPU is already full.
- Rishi's fork could gain Spark support through an upstream merge that keeps TurboQuant; Spark's
  KV cache is small (9 of 36 layers use full attention, 4 KV heads, ~36 KB per token at f16), so
  TurboQuant KV would save little here.
