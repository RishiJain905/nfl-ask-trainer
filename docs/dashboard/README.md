# The live training dashboard

A claude.ai Artifact that shows a training run as it happens, built only from the run's own
files, never from typed-in numbers. Two pages, one per stage:

| Page | Holds | URL |
|---|---|---|
| **Spark SFT Trials** (`spark-sft-trials.html`) | smoke runs and the pilot sweep | https://claude.ai/artifact/FWzig1Kco8arSVFh7yzDZJ |
| **Spark SFT Main Run** (a copy of the same HTML under a new file name → a new artifact with its own data store) | the real run only | made when the real run starts |

The trials page keeps every smoke and pilot run; the main-run page starts empty, so nothing is
ever overwritten or mixed.

## Where the numbers come from

| Shown | Source | Also in W&B? |
|---|---|---|
| Training loss, learning rate, gradient norm, tokens/s, peak memory per step | `<run>/train_log.jsonl` (the trainer's own values, the same ones it sends to W&B) | yes (cross-checked) |
| Validation loss and accuracy per dataset, evaluation timings | the `val/*` rows of `train_log.jsonl` | partly |
| Sample generations with their scores | `<run>/samples_<step>.jsonl` (older runs: `preds_epoch<N>.jsonl` + `scores_epoch<N>.json`) | no |
| Model load, checkpoints saved, resumes, OOM splits, verifier failures, crashes | `<run>/events.jsonl` | no |
| GPU memory in use (whole card), CPU, free RAM over wall-clock time | `scripts/gpu_monitor.ps1` (Windows performance counters, every 15 s) | no |
| ETA, smoothed loss | computed on the fly from the logged values, and labelled as such | no |
| W&B cross-check | the run's W&B history read back (`train/ce` at every step) and compared with the local log | |

GPU **utilisation** isn't shown: Windows doesn't attribute WSL's ROCm work to any GPU engine (all
engines read ~0% while the card trains), so there's no honest number for it.

## How an update flows

```
train_log.jsonl ─┐
events.jsonl ────┤
samples_*.jsonl ─┼─► scripts/dashboard_feed.py ─► <run>/dashboard/tick-NNNNN.json ─► ArtifactData set
run_info.json ───┤        (inside WSL)                (only what's new)                runs/<run>/ticks/tNNNNN
gpu-*.jsonl ─────┘
W&B history ─────┘ (--wandb: read-only check)
```

The page's data store has two collections: `runs/<run>` (the run's identity: `run.json`, written
once) and `runs/<run>/ticks/t<seq>` (append-only: every update creates a new document, so no
update ever overwrites another; the page merges them, keyed by step). A run of ~2,500 steps
updated every 5 minutes is ~230 documents (the store holds 25,000). Only the owner and editors
can write (rules: read `view`, write `admin`); the page itself never writes.

## The loop (what Claude does while a run is live)

1. Start the monitor with the run:
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\gpu_monitor.ps1 -Hours 30`
   (background; samples to `F:\nfl-ask-trainer-ckpt\_monitor\gpu-<date>.jsonl`).
2. Every ~5 minutes (pilots) or ~10 minutes (the real run):
   `wsl -d Ubuntu -- bash scripts/dashboard_feed.sh <run> -- [--wandb]`
   (the W&B check every third tick or so). It prints one JSON line; when `tick_file` isn't null,
   push it: `ArtifactData set` on `runs/<run>/ticks`, doc `t<seq:05d>`, `file_path` = the tick
   file (as a Windows path, `D:\nfl-ml-data\ask\finetune\<run>\dashboard\tick-NNNNN.json`); on a
   run's first tick also `runs` / `<run>` from `run.json`.
3. Read the printed status every tick: `stalled` (no step for 20 minutes while the process
   lives), `stopped` (the process is gone without `DONE`), a W&B mismatch, or a falling speed is
   reported to Rishi at once, with what the files say.
4. After `done`: one last tick, then stop the monitor.

Each tick file stays in `<run>/dashboard/`, so a lost update can be pushed again (a document that
already exists is refused, so a double push is harmless). `state.json` there remembers what has
been sent; deleting the folder re-sends everything as new ticks to a fresh store.

## Thinking-trace runs

Trace runs (`docs/traces.md`) use the same page and store: their run document says
`kind: "trace"` and the page switches to the trace view (answers over time, generation speed,
thinking length by level, pass rate by dataset and level, solved within k tries, why answers
failed, the latest answers with their thoughts, GPU memory). The feed is `scripts/trace_feed.py`
(standard library, run on Windows, where the generator runs):

    C:\Python314\python.exe -I scripts\trace_feed.py --run D:\nfl-ml-data\ask\traces\<run>

It reads `results.jsonl`, `progress.jsonl`, `events.jsonl`, `tracegen.pid` and the GPU
monitor, and writes `<run>/dashboard/tick-NNNNN.json` (new progress rows, events and GPU
samples, plus a summary recomputed from every result) and `run.json` once; push them exactly as
for training runs (`runs/<run>` and `runs/<run>/ticks/t<seq:05d>`). Status: `running`,
`stalled` (no answer for 15 minutes), `stopped` (the generator's process is gone without
`DONE`), `done`.

## Changing the page

The HTML is `spark-sft-trials.html` (one file: SVG charts drawn by its own script, no
libraries, the dataviz reference palette, light and dark). Republish it to the same URL from
this file. For the main run: copy it to `spark-sft-main.html`, change the `<title>` and heading
to "Spark SFT Main Run", publish it as a new artifact with the same `capabilities`, and add its
URL to the table above.
