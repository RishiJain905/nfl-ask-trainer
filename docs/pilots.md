# The pilot sweep

Eight short training runs on the same traced data, each changing one setting, to choose the
settings of the real run (plan: `docs/sft-architecture.html`, section 08). This page is how the
sweep runs and how a pilot is scored; the results go at the end as they come in.

## The data

- **Training**: the pilot trace run's passing answers (`docs/traces.md`, "The pilot trace run"),
  exported as `call` examples to `public/traces-pilot/`: the model's own thought and its own
  tool call, Spark's native format, thinking on. 10% of the traced questions are held out as
  `examples_val.jsonl`: the **validation loss**.
- **Scored evaluation**: `public/evalset/` (`ask-trainer traces evalset --name evalset --per-set 50`),
  200 questions never traced or trained on: 50 from each public set's **validation split**, in
  the same tool-call prompt as training, thinking high, the dataset's gold query as the reference.
  CypherBench's 50 are on its seven *test* graphs (company 8, fictional_character 4,
  flight_accident 7, geography 11, movie 7, nba 7, politics 6). The traced validation questions
  are only those the model already solved once, so they would flatter it; the evalset doesn't.

## How a pilot is scored: through llama.cpp, not during training

The plan scored every 100 steps with the trainer's own generation. Measured, that is too slow:
transformers generates ~24 tokens a second on this card even 8 answers at a time
(tsmoke-train-1: 3,104 tokens in 131 s), so 200 thinking answers would take ~50 minutes, more
than a pilot's training. Instead:

1. During training: the **validation loss only**, every 25 steps (200 stratified examples,
   ~1-2 minutes), and the adapter kept at steps 50 and 100 (`checkpoint.adapter_every_steps`).
2. After training, for each kept adapter and the final one: the LoRA converted to GGUF (the
   fork's `convert_lora_to_gguf.py`: Spark's fused qkv is a plain rename, and the LoRA's alpha is
   stored, so llama.cpp applies the same alpha / r scale as PEFT), served on the Q8_0 base
   (`serve.ps1 -Port 8092 -Extra "--reasoning-budget 1024 --lora <gguf>"`), the 200 questions
   answered greedily 16 at a time (`python -m asktrainer.llamaeval generate`, ~6-10 minutes),
   then scored in WSL (`ask-trainer evalrun --run <run> --preds <file> --tag <tag> --step N`):
   SQL executed (gretel on its real rows, SQaLe on filled copies), Cypher executed on
   CypherBench's graphs or schema-built ones (`cyphercheck`). The scores go to the run's folder
   (`scores_llamacpp_<tag>.json`) and to its W&B run as `llamacpp/<metric>` against
   `llamacpp/step`.

This also scores the setup the model is served and RL-trained with (Q8_0 + llama.cpp). Checked
on the smoke run's adapter: llama.cpp with the LoRA gives the same final call as the
transformers model on all five probe questions (in one, the base alone gets it wrong and the LoRA
fixes it); the thoughts agree for their first 110-627 characters, where a near-tie token flips
(Q8_0 vs bf16).

The **base model** (no LoRA) is scored the same way once, as the reference (its own W&B
evaluation run, `evalset-base-*`).

## The pilots

| Pilot | Objective | Learning rate | LoRA rank | Seed | Question |
|---|---|---|---|---|---|
| P1 | plain | 1e-4 | 16 | 42 | the baseline |
| P2 | plain | 2e-4 | 16 | 42 | which learning rate |
| P3 | plain | 4e-4 | 16 | 42 | |
| P4 | plain | best of P1-P3 | 64 (alpha 128) | 42 | does more adapter capacity help |
| P5 | JEPA λ 1, k 0 | best | best of P1-P4 | 42 | does LLM-JEPA help, and which setting |
| P6 | JEPA λ 0.5, k 1 | best | best | 42 | |
| P7 | best plain | best | best | 7 | how much of a difference is seed noise |
| P8 | best JEPA | best | best | 7 | |

Everything else is fixed: AdamW (β 0.9 / 0.999, weight decay 0.01), warmup 3% with at least 10
steps, cosine to 0, effective batch 32 (batch 1 x 32 accumulation), one epoch, `max_len` 4,096
(the longest traced example is ~2,630 tokens), LoRA dropout 0.05 on q_k_v_proj, out_proj,
gate_proj, up_proj, down_proj, checkpoints every 20 steps (last 2 kept).

**The rule** (pre-registered, `scripts/pilot_sweep.py: best_of`): the higher final exec accuracy
on the evalset (`val/public/exec_acc`, over the answers the checkers could decide); within half
a point, the lower final validation loss. JEPA is kept only if it beats plain by more than the
seed-to-seed difference (P7 vs P8). The real run's size (~52M vs ~100M tokens) comes from the
curves: if the step-50 / step-100 / final scores are still climbing at the end, more data helps.

## Running it

```
C:\Python314\python.exe scripts\pilot_sweep.py [--traces pilot] [--only P1,P2]
```

One process does it all, one GPU job at a time: it waits for the trace run to finish
(re-checking its Cypher answers every 30 minutes, then once more), scores the base model on the
trace server, stops that server, exports the traces, then runs the pilots in order, deciding P4-P8
from the results before them. A restart skips what's done (`sweep.json`); an unfinished training
run is resumed from its last checkpoint.

Files (`D:\nfl-ml-data\ask\pilots\`): `sweep.json` (each pilot's config and scores, the base
model's, the decisions), `status.json` (what runs now), `configs/pilot-pN.yaml` (each pilot's full
config), `pilot-pN.train.log`, `base/` (the base model's answers and scores). Each pilot is an
ordinary training run in `D:\nfl-ml-data\ask\finetune\pilot-pN\` (W&B, dashboard), with its
`adapter_step000050/`, `adapter_step000100/`, `adapter/`, and its `preds_llamacpp_*.jsonl` /
`scores_llamacpp_*.json`.

**Started early (Rishi, 2026-10-21 13:45):** the pilots train on the trace run's **first round**
only (each question's first try), with `--after-round 0`: the driver stops `tracegen` once round 1
begins, and resumes it after the sweep, so the second tries (for the unsolved questions) feed the
real run instead. The pilots compare settings with each other, so ~3,400 vs ~4,700 examples
doesn't change the ranking, and the pilot results come ~5 hours sooner.

## Results

The pilot data: 3,415 training + 379 validation examples (the first round of the trace run; SQaLe
1,617, gretel 1,073, text2cypher 872, CypherBench 232; 121 more passed but were cut by the thinking
budget). The base model on the evalset: 30.6% (W&B `evalset-base` run `j61nm1fp`).

**The learning rate (P1-P3, LoRA rank 16, 2026-10-21):** exec accuracy on the evalset, at steps 50 /
100 / final (107):

| | P1 (1e-4) | P2 (2e-4) | P3 (4e-4) |
|---|---|---|---|
| all, step 50 / 100 / final | 33.2 / 34.8 / **35.0%** | 35.1 / 33.7 / 34.4% | 33.5 / 34.4 / 34.1% |
| gretel (final) | 60.0% | 63.3% | 58.0% |
| SQaLe | 34.9% | 29.5% | 37.2% |
| CypherBench | 30.0% | 34.0% | 26.5% |
| text2cypher | 10.0% | 7.0% | 11.6% |
| valid calls | 96.5% | 95.0% | 96.5% |
| final validation loss | 0.347 | 0.348 | 0.351 |

By the rule P1 (1e-4) wins: 35.0% vs 34.4% is just beyond the half-point tie band, and it also has
the lowest validation loss. The three are within the noise of ~180 checked questions (about +/-3.5
points), so the honest reading is "1e-4 is at least as good"; P7 (P1 again with seed 7) will show
how large the seed-to-seed difference is. Every pilot beats the base model by ~4 points, from gretel
and CypherBench; text2cypher, the set with the fewest checked traces, didn't move. The validation
loss barely moves on this data (0.349 -> 0.347): the targets are the model's own samples. A higher
learning rate raises it (P3: 0.364 at step 25).

**The LoRA rank (P4, rank 64 / alpha 128, lr 1e-4, 2026-10-21):** 34.6 / 33.2 / **33.5%** at steps 50 /
100 / final (gretel 58.0%, SQaLe 30.2%, CypherBench 33.3%, text2cypher 7.3%; valid calls 94.0%), final
validation loss 0.350 (P1: 0.347). More adapter capacity doesn't help at this data size: it peaks at
step 50 and drifts down, as a larger adapter overfitting ~3,400 examples would. By the rule P1's rank 16
stays for P5-P8. (Rank 64 trains ~25% slower here: ~11 steps a minute vs ~14.)

**Paused before P5** at Rishi's request (2026-10-21 22:02): `D:\nfl-ml-data\ask\finetune\HOLD` makes
`ask-trainer train` refuse new runs (exit 3), so the driver stopped when it tried to start P5 (no P5 run
folder was made). To go on: delete the file, then `C:\Python314\python.exe scripts\pilot_sweep.py
--traces pilot --after-round 0`; the driver skips P1-P4 and runs P5 (JEPA lambda 1, k 0) and P6 (JEPA
lambda 0.5, k 1) at lr 1e-4, rank 16, then P7 / P8 (seed 7), then the trace run's second round.
