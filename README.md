# nfl-ask-trainer

The SFT environment for a small **general database agent** (text-to-SQL + text-to-Cypher) on
Spark-X2.6-4B: LoRA fine-tuning, plain SFT and SFT + LLM-JEPA, crash-safe checkpoints and
resume, executable evaluation, W&B tracking. It grew out of the NFL Analytics Engine's AE04
(the local Ask router); that NFL track is parked but its code paths stay (`configs/nfl_*`).
Read `CLAUDE.md` first (security).

## Run things

Everything runs inside WSL Ubuntu with the venv `~/venvs/nfl-ask-trainer`:

    wsl -d Ubuntu -- bash -lc "cd /mnt/f/Personal/NFL/nfl-ask-trainer && ~/venvs/nfl-ask-trainer/bin/ask-trainer doctor"

| Command | What it does |
|---|---|
| `ask-trainer doctor` | paths, GPU, base model files, NFL repo + `uv.exe`, W&B login state (set / not set only), the latest NFL export, the public datasets |
| `ask-trainer train --config configs/general_pilot.yaml [--jepa] [--seed N] [--name X] [--launched-by rishi]` | one training run |
| `ask-trainer train --resume <run folder or name>` | continue a run from its newest checkpoint (same data order, same W&B run) |
| `ask-trainer predict --run <run> --data <file.jsonl> [--n 160] [--verify]` | greedy generations with a run's adapter (`--verify`: the NFL verifier) |
| `ask-trainer merge --run <run> [--out <dir>]` | a servable merged bf16 model |
| `ask-trainer probe` | `scripts/probe_model.py` on the base model |
| `ask-trainer public ...` | the public-dataset tools (`python -m asktrainer.data.public ...`) |
| `python scripts/bench_train.py --seqs 700,2048 --batches 1,2` | training throughput on this GPU |
| `python scripts/bench_generate.py --prompt 2400 --batches 1,4` | generation speed (prefill / decode) |

Tests run on the CPU: `HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python -m pytest -q`;
`ASKTRAINER_GPU_TESTS=1 python -m pytest -m gpu` runs `tests/test_gpu.py` on the real model.

## Configs

- `general_pilot.yaml` (draft): ~4M tokens, a seeded sample of the four public sets in the
  full run's proportions; 186 optimizer steps (batch 1 x grad_accum 32); eval every 100 steps.
- `general_full.yaml` (draft): ~52M tokens (sqale 30k, text2cypher all 27k, gretel 20k,
  cypherbench 4k); 2,534 steps; eval every 500 steps.
- `smoke.yaml`: 32 examples, 8 steps, a step eval, checkpoints every 3 steps (minutes).
- `nfl_sft_plain.yaml` / `nfl_sft_jepa.yaml`: the parked NFL router runs.

## A training run

1. **Data** (`mix.py`): seeded samples of `<training>/public/<name>/examples_{train,val}.jsonl`
   (and, only when a config asks, the NFL export). Same config + seed = same mix.
2. **Tokens** (`chat.py`): the chat template with `enable_thinking=False`; the loss only on
   the reply and its end token; examples over `optim.max_len` are dropped and counted. With
   JEPA on, the text view (question + k predictor tokens `<||unused2||>`...) and the code
   view (the target) are packed after the main sequence in the same row (one forward pass).
3. **Model** (`modelio.py`): sequential shard reads (35-45 s from F:), bf16, the remote
   code's eager attention, chunked above 3,072 tokens (`attn_chunked.py`), LoRA r16 in fp32
   (31M parameters), non-reentrant gradient checkpointing, hipBLASLt GEMMs.
4. **The loop** (`train.py`): CE from the hidden states at the target positions only (fp32,
   chunked) + lambda x the JEPA distance (`jepa.py`); AdamW, cosine with warmup, gradient
   accumulation, clipping; length-grouped batches.
5. **Evaluations** every `eval.every_steps` steps and at each epoch end: the val loss, greedy
   batched generation on a fixed stratified subset of public val, scored by
   `publiceval.py` (gretel executed in in-memory SQLite from its own `sql_context`; sqale
   `EXPLAIN` against its DDL + exact match; the Cypher sets exact match + a light syntax
   check). The NFL verifier (`nfl ask verify`) is an option (`eval.nfl_verify`).
6. **Checkpoints** (`checkpoint.py`) every `checkpoint.every_steps` steps and at each epoch
   end, under `checkpoint.dir/<run>/step-NNNNNN/`: LoRA weights, optimizer, scheduler, every
   RNG, the data position, tokens seen, the W&B run id. Atomic (tmp folder, marker, rename);
   the newest `keep` are kept. `--resume` continues exactly (a CPU test pins that a crashed +
   resumed run equals an uninterrupted one, step for step and tensor for tensor).
7. **Artifacts**: the adapter as W&B artifact `ask-finetune-adapter-<run>` (aliases latest /
   epoch-N / final).

### The run folder (`<data root>/ask/finetune/<run>/`)

`config.json`, `environment.json`, `mix_manifest.json` (every example id), `run_info.json`
(the plan: steps, tokens, sets, W&B link, checkpoint dir), `events.jsonl` (start,
data_ready, model_loaded, resumed, eval_start / eval_end, oom_split, verifier_failed,
checkpoint_saved, artifact_uploaded / artifact_failed, epoch_end, done, crashed: codes only,
never error text), `train_log.jsonl` (every metric, with `ts`), `preds_*` / `scores_*` per
evaluation, `samples_<step>.jsonl` (12 fixed examples: question, gold, output, score),
the adapters, `wandb/`, `DONE`. A dashboard may write only into its own `dashboard/`.

**W&B** (project `nfl-analytics-engine`, group / job type `ask-finetune`, tags `ask-finetune`,
`objective:plain|jepa`, `seed:N`, `launched-by:<who>`): `train/*` (loss, ce, jepa, lr,
grad_norm, tokens_per_s, tokens_seen, peak_gb), `epoch`, `val/loss` (per origin),
`val/public/*` and `val/<set>/{json_ok,parse_ok,exact,exec_acc}`, the eval timings. Login:
`WANDB_API_KEY` through the settings module, else wandb's stored login, else offline (sync
later with `wandb sync`). After a resume, steps logged between the last checkpoint and the
crash are not re-logged by W&B (it drops steps it has already passed); the local log is
complete.

## Merged model

`ask-trainer merge --run <run>` writes `<run>/merged/`: merged bf16 safetensors, the remote
code, the tokenizer, a `chat_template.jinja` (and `tokenizer_config.json`) whose
`enable_thinking` defaults to false, a greedy `generation_config.json`, `ask_model_card.json`.

## Measured on this machine (RX 7800 XT, 16 GB; LoRA r16, hipBLASLt, batch 1 unless noted)

| tokens | tok/s | peak GB |
|---|---|---|
| 700 | 903 (batch 2: 955, 4: 935, 8: 916) | 9.3 (b8: 13.0) |
| 1,024 | 891 | 9.6 |
| 2,048 | 761 (no hipBLASLt: 598; + TunableOp: 821) | 10.7 |
| 3,072 | 652 | 12.5 |
| 5,200 | 464 | 12.6 |
| 8,192 | 353 | 14.9 |
| 10,240+ | out of memory | |

TunableOp tunes each new GEMM shape (~2-3 min per sequence length, cached in
`<runs>/tunableop/`), so it is off (`optim.tunableop`, with `optim.pad_to: 256`). Generation:
decode 0.05 s/step at batch 1, 0.095 at batch 4 (2.4k-token prompts); batches are capped at
8 rows / 14k prompt tokens and split themselves on an out-of-memory. The base model lives on
F: (NVMe): from D: (a USB HDD) a load took 341 s.
