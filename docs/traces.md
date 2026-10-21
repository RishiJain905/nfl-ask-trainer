# Thinking traces: how the training replies get their reasoning

Spark-X2.6-4B is trained with thinking **on**: every training reply is the model's own reasoning
followed by one query tool call. The four public datasets only have question and query, so a
stage before the pilots makes the reasoning. The model answers each training question itself
(served by llama.cpp), every answer is checked by running its query, and only reasoning that
reached a correct query is kept. This is rejection sampling, as in STaR (Zelikman et al. 2022).
Questions it never gets right get STaR's "rationalisation": the reference query is shown as a
hint, and the reasoning that reaches it is trained without the hint.

Why the model's own traces and not a bigger model's: they keep its own style and length, so
training sharpens its reasoning instead of overwriting it, and the server and checker are the
ones the RL stage will use.

## The pieces

| Piece | What it does | Where it runs |
|---|---|---|
| `ask-trainer traces prep` (`src/asktrainer/traceprep.py`) | samples training questions per dataset, renders each as a tool-call prompt, attaches what the checker needs, writes the plan | WSL (trainer venv) |
| `data/tool_prompt.py` | the tool-call prompt: the query tool (its name and argument varied), sometimes an unrelated tool, the schema, up to two tested examples, the rules; the question as the user turn | both |
| `scripts/llamacpp/serve.ps1` | starts / stops llama-server with the Q8_0 GGUF (`docs/llamacpp.md`) | Windows |
| `scripts/llamacpp/trace_run.ps1` → `asktrainer.tracegen` | sends every planned request (question x thinking level x try) to the server, 16 at once, checks each answer as it arrives, appends to `results.jsonl` | Windows, standard-library Python |
| `querycheck.py` | the checker (below) | both |
| `scripts/trace_feed.py` | turns the run's files into dashboard ticks | Windows |

WSL can't reach a server on Windows' localhost (WSL runs in NAT mode), so everything that talks
to llama-server runs on Windows with nothing but the standard library.

## How an answer is checked (`querycheck.py`)

A reply must hold exactly one call to the question's query tool with a non-empty query argument
(otherwise: no call, wrong tool, several calls, bad argument, or cut off at the token limit).
Then:

- **SQL**: read only, one statement, `EXPLAIN` against the schema (unknown tables and columns
  fail). Then both queries run and their results are compared as multisets of rows (numbers at
  the reference's precision):
  - on the **real rows** where the dataset has them (gretel's own INSERTs);
  - on **four randomly filled copies of the schema** (Spider's test-suite idea, Zhong, Yu and
    Klein 2020): small integer ranges so joins and groups line up, the queries' own literals
    mixed into the value pools, a few NULLs, foreign keys picked from the referenced rows. A copy
    where the reference returns nothing proves nothing and isn't counted.
  - When the reference sorts its final result, one more copy with distinct values (where ties
    are rare) must return the rows **in the same order**.
  - **Ties at a LIMIT** (`ORDER BY wins DESC LIMIT 1` with two teams level) count neither way:
    the reference is re-run without its LIMIT and, when the sort key is a returned column, the
    answer may hold any of the rows tied at the cut. An answer sorted the wrong way still fails.
  - Pass = agreement on at least one copy that returned rows and no disagreement anywhere;
    if no copy returned rows, only an exact match passes, else **unverified**.
- **Cypher**: a syntax check, then an exact match (whitespace-normalised) passes; anything else
  is **unverified** until the graphs are downloaded (CypherBench's own graphs; text2cypher's
  demo databases).

Tested on the smoke run's own 100 questions (`tests/test_traces.py` for the unit cases): every
reference query passes against itself (SQaLe: 19 by filled copies, 6 only by exact match since
the reference returns no rows there; gretel: 16 by real rows + copies, 9 by copies), and 15 of
16 deliberately broken references fail. The one that slips through changes the sort inside a
"latest event per group" window function, which only shows when a group has two matching
events after a four-table join: random data checks are strong, not perfect.

## Files of a run (`D:\nfl-ml-data\ask\traces\<run>\`)

| File | Holds |
|---|---|
| `config.json` | the plan (tries per thinking level), sampling (temperature 1.0, top-p 0.95, Spark's own defaults; up to 8,192 new tokens), the seed, the counts |
| `questions.jsonl` | one line per question: the prompt, tools, the reference query, the checker's schema script |
| `results.jsonl` | one line per answer: the request, the thought and its token count (the server's own tokenizer), the call, the verdict and why, tokens and seconds |
| `progress.jsonl` | totals every 10 s (answers, tokens) |
| `events.jsonl` | start, failed requests, the end |
| `tracegen.pid`, `tracegen.*.txt` | the generator's process id and output |
| `dashboard/` | the feed's ticks and state |

## Running one

```
wsl -d Ubuntu -- bash -lc "cd /mnt/f/Personal/NFL/nfl-ask-trainer && ~/venvs/nfl-ask-trainer/bin/ask-trainer traces prep --name tsmoke-1 --per-set 25 --plan high:4,low:2,off:1"
powershell -File scripts\llamacpp\serve.ps1 -Bin <the llama.cpp build's bin>
powershell -File scripts\llamacpp\trace_run.ps1 -Run tsmoke-1
python scripts\trace_feed.py --run D:\nfl-ml-data\ask\traces\tsmoke-1     # every few minutes; push the tick
powershell -File scripts\llamacpp\serve.ps1 -Stop
```

The live view is the **Spark SFT Trials** dashboard (`docs/dashboard/`), which shows trace runs
with their own charts: answers over time, generation speed, thinking length by level, pass rate
by dataset and level, solved within k tries, why answers failed, the latest answers with their
thoughts.

## The smoke run (tsmoke-1, 2026-10-21)

100 training questions (25 per dataset, seed 0), each answered 4 times at thinking effort high,
2 times at low and once with thinking off: 700 answers, on Rishi's llama.cpp fork (TurboQuant
build + the Spark port, `docs/llamacpp.md`), Q8_0, 16 at once, temperature 1.0, top-p 0.95.
No request failed. Full numbers: `D:\nfl-ml-data\ask\traces\tsmoke-1\report.json`
(`scripts/trace_report.py`); live view: the dashboard's `tsmoke-1`.

| | SQaLe | gretel | text2cypher | CypherBench | all |
|---|---|---|---|---|---|
| answers passed / checkable | 41 / 165 | 43 / 175 | 7 / 28 | 0 / 17 | 91 / 385 (23.6%) |
| lenient (extra columns or column order allowed) | 33.9% | 33.1% | 25% | 0% | 31.4% |
| unverified (no graph; SQL with no rows) | 10 | 0 | 147 | 158 | 315 |
| questions with a passing answer | 11 / 25 | 11 / 25 | 2 / 25 | 0 / 25 | 24 / 100 |
| thought tokens, median / 90th pct | 109 / 1,037 | 90 / 291 | 254 / 1,801 | 215 / 1,540 | 161 / 1,079 |

By thinking level: **off 14.1%** passed, **high 25.1%**, **low 26.5%**: thinking nearly doubles
the rate, and low is as good as high with shorter thoughts (mean 471 vs 670 tokens; median
about 205 for both). Within k tries (high): 30.2% at 1, 33.3% at 2, 32.8% at 4: extra tries
add little; a question is mostly solvable or not. The traces that passed are short: mean 191
thought tokens against 517 for all answers; the long thoughts (6 hit the 8,192-token cap) are
rarely right.

Why answers failed (of 294 checkable failures): different rows on the filled copies 146, on
gretel's real rows 58, no tool call at all 32, several calls in one reply 19, invalid SQL 17,
invalid Cypher 12, cut off 6. **About 8% of the failures are exploration** (11% on gretel): the
model spends its one call looking at the data (`SELECT DISTINCT contract_status FROM ...`,
`SELECT ... LIMIT 20`) as an agent with several turns would, often a sensible move (gretel's
prompts show no rows, so it can't know the spelling of a status). A further ~8 points are
answers whose rows are right but with extra columns (the reference's columns plus an id or a
name) or another column order: right for a person, wrong for the strict match the benchmarks
use, so the training data keeps the strict rule.

Speed: 19.6 answers a minute, 190 generated tokens/s plus ~230 prompt tokens/s (prompts
average 708 tokens), 35.8 busy minutes. That is well below the 786 tokens/s of
`llama-batched-bench`: in the server, new prompts keep arriving between decoding steps, the
long thoughts hold slots for minutes, and the first 75 answers ran before the prompt-cache fix
(`docs/llamacpp.md`).

What it means for the full stage (decisions, with the numbers, on the architecture page):
- **Time.** At this pace every 1,000 answers is ~51 minutes; tracing all ~81k training
  questions twice would be ~140 hours. Options: trace a subset (e.g. 15-20k questions), cap
  thinking with `--reasoning-budget` (correct traces are short), use effort low only, or mix
  traced (thinking on) and direct (thinking off) examples.
- **Yield.** Only about half the SQL questions get a passing trace even at 7 tries, so the
  STaR hint step for the rest matters, or the traced data leans towards easy questions.
- **Cypher needs its graphs.** Without them 90% of Cypher answers can't be checked; the
  CypherBench graphs (approved download) come first.
- **Multi-turn is natural for Spark.** Exploration calls suggest letting the generator run
  the call and answer with the rows (gretel has real rows; SQaLe's populated databases are in
  the approved downloads), which yields multi-step trajectories.

The passing traces export to 47 `call` examples (38 train, 9 val; 24 questions), the input of
the training smoke run on the new format.

## Effort low with a thinking budget (tsmoke-2, 2026-10-21)

The same 100 questions and prompts, effort low × 2, server flag `--reasoning-budget 1024`
(llama.cpp ends the thought after 1,024 tokens and the model goes on to its call):

| low × 2, same questions | tsmoke-1 (no budget) | tsmoke-2 (budget 1,024) |
|---|---|---|
| passed / checkable | 27 / 102 (26.5%) | 37 / 106 (**34.9%**) |
| lenient | 35.3% | 42.5% |
| solved within 1 / 2 tries | 30.2% / 30.9% | 36.5% / **41.8%** |
| thought tokens, median / 90th pct / mean | 203 / 1,022 / 471 | 197 / 862 / **321** |
| tokens per answer (mean) | 534 | 383 |
| answers a minute | (19.6 for the whole mixed run) | **36.3** (231 generated tokens/s) |

Capping thinking costs nothing (the passing traces average 138 thought tokens; the pass-rate gain
is partly sampling noise at ~100 checkable answers) and nearly doubles the pace: ~27 minutes
per 1,000 answers, so 15k questions at 2 tries is ~14 hours. Traces cut off by the budget should
be left out of the training data (`think_tokens` at the budget).

## The training smoke run on traced data (tsmoke-train-1, 2026-10-21)

`configs/traces_smoke.yaml`: the 38 traced examples (35,703 tokens, none over the 6,144-token
limit), LoRA r16, 5 optimizer steps of 8 examples. It checks the new format end to end, not
quality:

- **Rendering and masking**: the trainer renders a `call` example with its tools and thinking
  level exactly as llama-server does (byte-identical prompts on 18 of 18 checked renders, high,
  low and off, with and without an unrelated tool); the loss covers the thought, `</think>`, the
  tool call and the end token (`tests/test_traces.py` pins it with Spark's own tokenizer).
- **Training**: answer loss 0.48 → 0.14 over the 5 steps, ~750 tokens/s, peak 10-11 GB, a
  checkpoint at step 4, W&B matching the local log at every step (largest difference 5e-7).
- **Generation and scoring** at the end: the 9 validation examples generated with thinking on
  (3,104 tokens, 131 s with transformers); every answer was a valid call to the right tool;
  8 of 9 queries valid; all 5 gretel answers returned the right rows. SQaLe validation answers
  are now executed too, on randomly filled copies of their schema (`publiceval` uses
  `querycheck.check_sql`): 3 of the 3 valid ones pass.

Run folder `D:\nfl-ml-data\ask\finetune\tsmoke-train-1`; W&B run `tsmoke-train-1`
(`3wdj2bkf`); dashboard `tsmoke-train-1`.
