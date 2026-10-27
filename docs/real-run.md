# The real run (`sft-main`)

The general SQL + Cypher agent, trained once at full size with the settings and data the pilots
chose (`docs/pilots.md`; Rishi approved the layout on 2026-10-26, after P9). Config:
`configs/general_full.yaml`. The plan page explains it with diagrams (`docs/sft-architecture.html`,
section 15).

## What it trains on

| Set | Train | Val | What a target is |
|---|---|---|---|
| `traces-pilot-full` | 4,196 | 466 | Spark's own checked thought + tool call (thinking high 2,829 / low 1,129 / off 238) |
| `gold-full` | 90,549 | 400 | the dataset's reference query as a thinking-off tool call (SQaLe 27,887, gretel 28,695, text2cypher 25,871, CypherBench 8,096) |
| **All** | **94,745** | **866** | 2,961 optimizer steps at 32 examples a step, ~66M tokens |

Never trained on: the evalset (200) and the test set (1,000), drawn from each dataset's own test
split. A CPU dry run through the trainer's loaders found none of their questions, and none of the
validation questions, in the training data. How each set is built and kept apart:
`docs/traces.md`, "Gold answers, validation and the held-out tests". **Don't rebuild
`traces-pilot-full` or `gold-full` while the run is unfinished**: a resume checks that the
training examples are exactly the ones it started with and refuses otherwise.

Settings: P9's (plain objective, lr 1e-4 with 3% warmup and cosine decay, LoRA rank 16 / alpha 32
on every linear layer, AdamW, weight decay 0.01, batch 32, max length 4,096, one epoch, seed 42).
At P9's ~815 training tokens a second: ~23 hours, plus ~1 hour of validation passes.

## Running it

```
C:\Python314\python.exe scripts\real_run.py
```

One process does everything (`scripts/real_run.py`): it trains (`ask-trainer train` in WSL,
`--resume` when the run folder is already there), stops at every adapter snapshot to score it on
the evalset and goes on, then scores the final adapter, and the final one (plus the best, if
another) on the test set. Docker Desktop and the trainer's Neo4j (`asktrainer-neo4j`, for checking
the Cypher answers) are started when a scoring needs them. Started again after any stop, it
carries on where things stand (an adapter saved but not scored yet is scored first). State: `D:\nfl-ml-data\ask\main\` (`status.json`, `state.json`
with the scores and every resume, `sft-main.train.log`, `driver.pid`). The run itself:
`D:\nfl-ml-data\ask\finetune\sft-main\`; its checkpoints `F:\nfl-ask-trainer-ckpt\sft-main\`.

## Stopping it to switch the PC off (and going on later)

1. Create the file `D:\nfl-ml-data\ask\finetune\PAUSE` (empty is fine; any text but `scoring`,
   which is the driver's own stop for a scoring. If the driver has just written that one,
   overwrite it with anything else: the stop then holds once the scoring is done).
2. The trainer finishes the step it is on, writes its next checkpoint (every 20 steps, so within
   ~9 minutes), writes `sft-main\PAUSED` (the step) and exits on its own; the driver exits with
   `phase: paused` in `status.json`. **Nothing is lost**: the checkpoint holds the LoRA weights,
   the optimizer's moments, the schedule, every random-number state and the position in the data,
   and nothing runs past it. A CPU test stops a run this way and resumes it: the losses and the
   final weights are exactly those of an uninterrupted run
   (`tests/test_train_e2e.py::test_a_pause_stops_at_a_checkpoint_and_the_resume_loses_nothing`).
3. Once `status.json` says `paused` (no `ask-trainer` process in WSL), switch the PC off.
4. To go on: delete `PAUSE`, then start `scripts\real_run.py` again. It refuses to start while
   `PAUSE` exists (it would only stop again at the next checkpoint). The W&B run continues on the
   same line.

**Without a pause** (a power cut, a crash, a hard switch-off): the resume starts from the last
checkpoint, so at most 20 steps (~9 minutes) are redone, with the same result. Pausing first is
still better: the run folder is on D: (a USB drive, exFAT), and a switch-off in the middle of a
write can damage the file system (it happened once, `docs/d-drive-repair.md`); the checkpoints
themselves are on F: and written to a temporary folder first.

## The watchdog

Pilot P9 stopped making steps once (step 173: the GPU idle, the trainer spinning, WSL's
`dxgkio_make_resident ... -12`), and a resume went straight through. So:

- the trainer releases PyTorch's cached GPU memory after every checkpoint (the cause of the stall
  was the cache growing over a long run);
- the driver watches `train_log.jsonl` and `events.jsonl`: no new line for **15 minutes** (once
  the first step of a launch is logged; 90 minutes before that, for loading and tokenizing
  ~95k examples) stops the trainer and resumes it from its last checkpoint. A crash is resumed the
  same way. A resume that makes no step at all, or 3 crashes in a row that all go back to the
  same checkpoint (a crash loop), stop the driver (`phase: failed`) instead of looping; 30
  resumes in the whole run is the backstop. Every resume is listed in `state.json`.
- **The crashes seen**: at step 809 (2026-10-27 01:39 UTC) the trainer died with a segfault in the
  GPU runtime (`libamdhip64`, exit 139) right after two of WSL's `dxgkio_make_resident ... -12`
  failures (Windows couldn't keep the GPU memory resident); the driver resumed it from step 800
  and the run went on (the 9 steps redone give the same numbers, so W&B's warning about steps
  801-809 being logged again drops copies it already has). The kernel log had nine such failures
  in the 2.5 hours before, most of them harmless. The first limit (6 resumes in the whole run)
  could have stopped a healthy run over ~20 hours at that rate, hence the rule above (Rishi was
  told, 2026-10-27).

## What gets saved

| What | When | Kept | Size |
|---|---|---|---|
| Full checkpoint (resume point) | every 20 steps | the newest 2 | ~355 MB each, F: |
| Adapter snapshot (`adapter_stepNNNNNN`) | every 500 steps | all | 119 MB each, D: |
| Final adapter (`adapter`, `adapter_epoch1`) | at the end | all, and on W&B | 119 MB |

## Scoring (during the run, on the GPU)

Each adapter (steps 500, 1000, 1500, 2000, 2500 and the final one) is converted to a GGUF LoRA,
attached to the Q8_0 base in llama-server (port 8092, thinking budget 1,024) and answers the
evalset greedily; `ask-trainer evalrun` scores it with checker 4 (`<tag>`) and checker 3 beside it
(`<tag>-checker3`, the pilots' official version, for the comparison).

**The step adapters are scored as the run goes** (Rishi's call, 2026-10-26): the evalset accuracy
can fall while the validation loss still falls (the loss grades the reference tokens fed in; the
accuracy grades what the model writes on its own, and ~94% of the training tokens are thinking-off
gold answers that could wear the thinking-on answers down), and that should show at that step, not
after ~23 hours. One GPU job at a time, so the training stops for it:

1. once the trainer is past the checkpoint before an adapter step (step 981 logged, for the
   adapter at 1000), the driver writes `PAUSE` with the text `scoring`;
2. the trainer saves that step's adapter, then its checkpoint, and stops there (the same exact stop
   as a switch-off; nothing is lost or redone);
3. the driver scores the adapter, removes its `PAUSE` and resumes the trainer. These stops are not
   counted as resumes.

Measured at step 500 (2026-10-27): **~15 minutes a stop**, the trainer stopped at 01:13 and going
again at 01:28: ~3 minutes to start Docker Desktop and Neo4j (only after a switch-off), ~6 to
answer the 200 questions, ~5 to check them against a Neo4j just started (cold), then the ~2-minute
resume. Five stops: ~1.3 hours on top of the training.

A stop asked for meanwhile (`PAUSE` with any other text) still holds. A scoring that fails (Neo4j
down, say) is noted in `state.json` (`score_errors`) and the training goes on; the scoring at the
end tries it again. The mid-run scores go into W&B after training: logging into the training's W&B
run while it is stopped moves that run's step past the trainer's next one, and W&B would then drop
that step's training metrics. Until then they are in the run folder (`scores_llamacpp_<tag>.json`,
which the dashboard reads) and in `state.json`.

### Evalset accuracy so far

| Adapter | Checker 4 | Checker 3 | CypherBench | gretel | SQaLe | text2cypher | Thought |
|---|---|---|---|---|---|---|---|
| step 500 (learning rate still ~9.5e-5) | **38.5%** (70/182) | 39.7% | 34.7% | 61.2% | 40.0% | 12.8% | 200/200, median 1,044 chars |
| *P9 final, for reference (240 steps, annealed to 0)* | *41.8%* (76/182) | | *44.0%* | *64.0%* | *38.6%* | *13.2%* | *200/200, 810 chars* |
| *base model* | | *30.6%* | | | | | |

How to read it: a snapshot in the middle of the run is taken at a high learning rate (the cosine
has hardly started), which usually scores a little below an annealed adapter; 6 questions of 182
is within the noise of a 200-question set (one standard error is ~3.6 points). The trend over the
snapshots is the signal: a fall from one snapshot to the next while the validation loss still
falls is what the stops are there to catch. The thought is intact (every answer has one, longer
than P9's).

After training: the final adapter, the mid-run scores into the run's W&B run (`llamacpp/step`
axis, a re-check of the saved answers, no new generation), then the test set (1,000 questions,
~25 minutes) once on the final adapter, and on the best evalset adapter if that is another one. The
test set is never used to choose anything.

## The dashboard

The real run gets its own page, "Spark SFT Main Run" (`docs/dashboard/README.md`: a copy of the
trials page under a new name, so its data store starts empty), updated every ~10 minutes.
