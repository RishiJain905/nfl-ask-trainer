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
`scores_llamacpp_*.json`. **P1-P6's adapters were deleted on 2026-10-25** at Rishi's request (12 GB:
the four adapter folders, the three LoRA GGUFs in `llamacpp\lora\`, the two step checkpoints in
`F:\nfl-ask-trainer-ckpt\`); their answers, scores, logs and W&B runs (with the adapter artifacts
logged after each epoch) stay. P7 and P8 keep everything.

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
stays for P5-P8. (Rank 64 trains ~25% slower here: ~49 s a step vs ~40.)

**LLM-JEPA (P5, P6, lr 1e-4, rank 16, 2026-10-22):**

| | P1 (plain) | P5 (JEPA λ 1, k 0) | P6 (JEPA λ 0.5, k 1) |
|---|---|---|---|
| all, step 50 / 100 / final | 33.2 / 34.8 / 35.0% | 36.2 / 35.5 / **36.3%** | 31.5 / 35.0 / 35.3% |
| gretel (final) | 60.0% | 61.2% | 58.0% |
| SQaLe | 34.9% | 40.0% | 34.9% |
| CypherBench | 30.0% | 31.3% | 34.0% |
| text2cypher | 10.0% | 9.5% | 9.8% |
| valid calls | 96.5% | 94.5% | 96.0% |
| validation loss, step 25 / final | 0.349 / 0.347 | 0.370 / 0.349 | 0.355 / 0.348 |

By the rule P5 is the better JEPA setting (36.3% vs 35.3%), so P8 repeats it at seed 7. P5 is also
the best pilot so far, 1.3 points over P1: beyond the half-point band, but inside the noise, which is
what P7 / P8 measure. The JEPA term costs validation loss early (P5: 0.370 at step 25 against P1's
0.349, the embedding loss pulling against the next-token loss at full weight) and is almost gone by the
end; its own value falls to ~0.002. A JEPA step takes ~44 s against ~40 s plain.

**Seed noise, plain (P7 = P1 at seed 7, 2026-10-23):** 33.3 / 33.5 / **35.2%** at steps 50 / 100 / final
(gretel 60.0%, SQaLe 33.3%, CypherBench 31.2%, text2cypher 10.3%; valid calls 96.0%) against P1's 33.1 /
34.8 / 35.0%. The seed moves the final score by 0.2 points (1.3 at step 100), far less than the ~3.5-point
sampling error of a 200-question evalset suggested: the evalset is the same for both, so what changes is
only the training order and the adapter's start. One pair isn't a noise estimate on its own; P8 (P5 at
seed 7) is the second.

**Seed noise, JEPA (P8 = P5 at seed 7, 2026-10-25):** 37.3 / 34.4 / **35.5%** at steps 50 / 100 / final
(gretel 58.0%, SQaLe 34.9%, CypherBench 33.3%, text2cypher 13.3%; valid calls 98.0%) against P5's 36.2 /
35.5 / 36.3%. Its step-50 adapter is the best checkpoint of the sweep (37.3%). Validation loss 0.353 at
step 25 and 0.329 at the end, against P7's 0.329 / 0.328 on the same subset: the JEPA term's early cost
again, gone by the end. ~46 s a step (83 minutes with the evaluations). Scored with checker version 3,
like P1-P7.

**The two seeds side by side** (final exec accuracy; steps 50 / 100 in brackets):

| | seed 42 | seed 7 | mean | seed-to-seed |
|---|---|---|---|---|
| plain (P1, P7) | 35.0% (33.1 / 34.8) | 35.2% (33.3 / 33.5) | 35.1% | 0.2 points |
| JEPA λ 1, k 0 (P5, P8) | 36.3% (36.2 / 35.5) | 35.5% (37.3 / 34.4) | 35.9% | 0.8 points |
| JEPA's lead | +1.3 | +0.3 | +0.8 | |

By the rule JEPA stays: it beats plain at both seeds, by more than plain's seed-to-seed difference (0.2,
the driver's `rule.seed_noise`). The lead is small, though: +0.8 points on average is one or two of the
~186 decided questions, about the size of JEPA's own seed-to-seed difference, and it costs ~10% more time
a step. What the curves say about the data size: the plain runs still climb at the end (P1 34.8 -> 35.0,
P7 33.5 -> 35.2), the JEPA runs don't (P5 flat, P8 best at step 50), so neither setting shows clearly that
the pilots' ~3,400 examples are too few; the real run's mix is far larger either way.

**Validation loss isn't comparable across seeds.** `train.py` picks the 200 validation-loss examples
(of 379) with `stratified_subset(..., cfg.seed)`, so P7 / P8 (seed 7) measure their loss on a different
200 than P1-P6 (seed 42): P7's step-25 loss is 0.329 against P1's 0.349, from the subset, not the model.
The exec accuracy is unaffected (the evalset is a fixed 200, the same for every run), and P7 vs P8 compare
fairly with each other. Fixed after the sweep (2026-10-25), so P7 and P8 ran the same code: every
evaluation subset (the validation loss, the scored and sampled subsets, `data.nfl_val`) is drawn
with `eval.subset_seed` (default 42, the subsets P1-P6 used), whatever the run's `seed`.

**P8 couldn't train while the PC was in use (2026-10-24; it trained on 2026-10-25, below).** Started at
16:32 with Rishi working on the PC, it
froze twice at the same point, during step 8: `HSA exception: MemoryRegion::BlockAllocator::alloc
failed.` in `pilot-p8.train.log`, the process spinning with no further steps. Both attempts are kept
(`finetune\pilot-p8-failed-20261024`, `pilot-p8-failed2-20261024`). What we measured:

- Windows' commit charge was at 47.5 of 49.1 GB (31 GB RAM + page file) when the first attempt froze, and
  in the second, free RAM fell from 13.8 to 2.9 GB at step 8. Killing the frozen trainer gave back ~16 GB
  of commit: on WSL the GPU memory PyTorch reserves (~14.8 GB with its cache) is backed by Windows commit,
  so an allocation past what the card can hold spills into system RAM.
- The step itself isn't unusual: P7 (same seed) peaked at 11.4 GB at step 8 and P5 (same JEPA) at 12.6 GB
  overall. The difference is the room left around it: with the desktop in use (the compositor, Chrome and
  the AMD software hold ~1.1 GB of VRAM) and an RLM job, a MySQL server and several Claude sessions also
  running, there was ~1.6 GB of commit to spare. P1-P7 trained overnight with the PC idle, and even then
  free RAM dipped to 0.5 GB.
- `garbage_collection_threshold:0.6` (the second attempt) didn't help, and `expandable_segments` isn't
  supported on WSL ("not supported on this platform"); neither is in the driver.

So P8 runs when the PC is idle, and the trace run's second round (llama.cpp: a fixed ~7.6 GB of VRAM, little
RAM) ran first, from 17:12. Started by hand with the driver's own `finish_traces()`, so the sweep's
`status.json` shows `phase: traces-rest`. Run P8 only after the trace run has finished (a `DONE` event):
started with `--after-round 0` while round 1 is still running, the driver would stop tracegen. Round 1
finished 2026-10-25 02:51 UTC (`DONE`, 26,085 answers; its llama-server on port 8090 stopped after), so
the driver now goes straight to P8. Rishi set a larger page file (33,792 MB, 2026-10-24: the commit
limit goes from ~48 to ~64 GB at the next restart), but asked for one more try before restarting, with
less running on the PC: on 2026-10-25 05:00 UTC the commit limit was still 48.1 GB, with 18.6 GB free
(against ~1.6 GB at the failures). P8 trained through (12.6 GB of VRAM in use, ~4.6 GB of commit to spare
at step 10; the start-up `HSA exception` line appeared once and was harmless, as in P4-P6) and was scored
by 06:43. **For the real run** (many hours): train with the PC idle, or free memory first; the larger
page file, once in effect, gives the spill room instead of a crash.

**Scored with checker version 3 throughout.** The trace checker gained near-miss and rescue copies
on 2026-10-24 (version 4, `docs/traces.md`). The evalset scorer uses the same checkers, so the
sweep pins the old rules (`SCORE_CHECKER = 3`, `ask-trainer evalrun --checker 3`) and P8 is scored
exactly like P1-P7. The pilots' training data (the round-0 export) is unchanged too. Re-scoring
every pilot under version 4 afterwards needs no GPU: `evalrun --checker 4 --no-wandb` on the saved
`preds_llamacpp_*.jsonl`.

**Re-scored under checker version 4 (2026-10-25, CPU and the trainer's Neo4j only).** The same saved
answers, scored with the near-miss and rescue copies, into `scores_llamacpp_<tag>-checker4.json` next to
the official version-3 files (no W&B). Re-scoring P1's final answers with `--checker 3` gave 34.97%
again, so the differences are the checker's alone. Exec accuracy, version 3 -> 4:

| | step 50 | step 100 | final |
|---|---|---|---|
| base | | | 30.6 -> 30.2 |
| P1 (plain, lr 1e-4) | 33.2 -> 32.8 | 34.8 -> 33.9 | 35.0 -> 34.1 |
| P2 (lr 2e-4) | 35.1 -> 34.2 | 33.7 -> 33.3 | 34.4 -> 34.1 |
| P3 (lr 4e-4) | 33.5 -> 32.4 | 34.4 -> 33.9 | 34.1 -> 33.2 |
| P4 (rank 64) | 34.6 -> 34.2 | 33.2 -> 32.6 | 33.5 -> 32.4 |
| P5 (JEPA λ 1, k 0) | 36.2 -> 35.3 | 35.5 -> 34.4 | **36.3 -> 35.7** |
| P6 (JEPA λ 0.5, k 1) | 31.5 -> 30.6 | 35.0 -> 34.2 | 35.3 -> 34.2 |
| P7 (P1 at seed 7) | 33.3 -> 32.4 | 33.5 -> 33.2 | 35.2 -> 35.0 |
| P8 (P5 at seed 7) | 37.3 -> 36.3 | 34.4 -> 33.5 | 35.5 -> 35.1 |

Every score falls 0.2-1.1 points (answers version 3 passed that the near-miss copies show are wrong),
and every decision of the sweep stands: P1 and P2 now tie (34.1%) and P1 wins on validation loss;
rank 64 is still worse; P5 still beats P6 (35.7% vs 34.2%). Over the two seeds JEPA averages 35.4%
against plain's 34.6% (+0.9). Plain's seed-to-seed difference grows to 0.9 points (34.1 vs 35.0) and
JEPA's is 0.6, so at seed 7 JEPA's lead (+0.1) is inside the noise; at seed 42 it is +1.6. The reading
is the same as under version 3: JEPA is ahead at both seeds, by about one question on average.

**Paused twice at Rishi's request** (he needed the PC): before P5 (2026-10-21 22:02 to 2026-10-22 19:56)
and after P7 (2026-10-23 00:55 to 2026-10-24 16:32; P7 had finished training and all three scores). `D:\nfl-ml-data\ask\finetune\HOLD` makes `ask-trainer train`
refuse new runs (exit 3), so the driver stops when it tries to start the next pilot (no run folder is
made) and writes `phase: paused` to `status.json`. To go on: delete the file, start the trainer's Neo4j
(`docker start asktrainer-neo4j`), then `C:\Python314\python.exe scripts\pilot_sweep.py --traces pilot
--after-round 0`; the driver skips the finished pilots, runs the rest, then the trace run's second round.

## P9: does adding the datasets' gold answers help? (planned 2026-10-25)

**Why.** The ~52M / ~100M sizes in the plan were counted on the public sets in their old format (a
JSON reply, no thinking, no tools). In the trained format (Spark's tool call, with or without its own
thought) the only data is what the trace run passed: a full export of both rounds under checker 4 is
4,662 examples, 4.3M tokens, about 1.3x a pilot. The real run therefore trains on two kinds of
example (Rishi, 2026-10-25: thinking and non-thinking, no caps on the gold share):

- **traces** (thinking on, plus the few `off` passes): Spark's own checked thoughts and calls;
- **gold** (thinking off): the dataset's own query as the tool call, no thought, for every training
  question without a kept trace (`ask-trainer traces gold`, `docs/traces.md`, "Gold answers").

No pilot mixed in gold answers, so P9 checks it first, as cheaply as a fair comparison allows.

**The run.** P1 exactly (plain, lr 1e-4, rank 16, seed 42, the same 3,415 traces) plus `gold-p9`:
4,235 gold examples, 3.06M tokens (P1's traces are 3.04M), on questions never traced, in the real
run's proportions (SQaLe 1,565, text2cypher 1,430, gretel 1,035, CypherBench 205). 7,650 examples,
240 steps (the last holds 2 examples). Validation: the traces' 379 plus 100 gold (25 a set, never traced); the loss on a fixed
300 of them every 100 steps, with `val/loss/thinking_on` and `val/loss/thinking_off` (new). Only the
final adapter is scored (checker 3 into W&B like every pilot, and checker 4 next to it).

**Why not shorter.** The comparison with P1 needs P1's 107 trace steps again, plus the gold's ~130.
Cut from the first estimate (~3 h): the step-50 / step-100 adapters and their scoring, JEPA (~15%
slower a step; the objective is a separate question), and most validation passes. What is left: ~2.4 h
of training and ~15 minutes of scoring. Half as much gold would save ~35 minutes but test a weaker dose
(the real run is ~15 gold tokens to 1 trace token).

**The rule (pre-registered).** Under checker 4, P1 scored 34.05% (63 of 185 decided). The real run
adds the gold answers only if **P9 scores at least 35.1%** (P1 + 1 point; plain's seed-to-seed
difference is 0.9). A health check comes with it: P9 must still think on the evalset (P1, P5 and P7
wrote a thought for 100% of its 200 questions, median ~830 characters); a collapse in thought length
would count against it even at a higher score. If P9 falls short, the real run trains on the traces
alone, and more traces (another trace run, where the trace-speed work matters) come next.

Run (GPU, one job at a time): `docker start asktrainer-neo4j` if needed, then
`C:\Python314\python.exe scripts\pilot_sweep.py --traces pilot --only P9`.

**Result (2026-10-26 01:03 UTC): P9 passes by a wide margin.** Final exec accuracy on the evalset:

| | checker 4 | checker 3 | decided / correct (v4) | CypherBench | gretel | SQaLe | text2cypher |
|---|---|---|---|---|---|---|---|
| base | 30.2% | 30.6% | | | | | |
| P1 (traces only) | 34.05% | 34.97% | 185 / 63 | 30.0 | 60.0 | 31.1 | 10.0 |
| P5 (best of the sweep, JEPA) | 35.7% | 36.3% | 182 / 65 | | | | |
| **P9 (P1 + gold)** | **41.76%** | **41.9%** | 182 / 76 | 44.0 | 64.0 | 38.6 | 13.2 |

+7.7 points over P1 (13 more correct answers of ~182), against a bar of +1 and a seed-to-seed
difference of 0.9; every dataset gains, CypherBench and SQaLe the most. The health check holds: a
thought on all 200 questions, median 802 characters (P1 826), median 276 generated tokens (P1 282);
4 answers ended without a call (P1 2). Fewer answers fail to parse (15 syntax errors vs 27). The
validation loss: thinking on 0.356 -> 0.354 (steps 100 -> 200), thinking off 0.187 -> 0.169, overall
0.336 at the end. So the real run trains on the traces plus the gold answers. W&B run: `pilot-p9`.

**A stall at step 173, resumed.** After step 173 (its highest peak so far, 12.0 GB) the trainer
stopped making steps while using ~320% CPU, the GPU idle; WSL's kernel log showed `dxgkio_make_resident:
Ioctl failed: -12` (the GPU paravirtualisation out of memory), not a commit shortage (22 GB free with
the larger page file). Stopped after 10 minutes and resumed from the step-160 checkpoint
(`pilot-p9.train.stall1-step173.log` kept): the same batches passed at 10-11 GB with the fresh
allocator, so the cause was the cached allocator's growth over a long run rather than one batch. The
real run is ~10x longer, so it gets a guard rather than luck: the allocator's cache released at
every checkpoint, and a watchdog that resumes a run stuck for 15 minutes from its last checkpoint
(planned with the real run's config).
