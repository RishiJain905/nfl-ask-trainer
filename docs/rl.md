# The RL run (planned)

Status (2026-10-31): planning; the steps are set, no method chosen yet. Rishi's direction: after
SFT, the main gains come from RL in environments like real use: the agent plugged into a database
as the one who searches, in real harnesses, over several stages in turn. All of it on the 16 GB
card: long runs are fine, and proving it can be done there is a goal in itself. A rented H100
(Rishi has credit on Modal) is the fallback for a stage that turns out too slow locally.

**The steps, in order:**

1. **The sampling baseline:** how much room RL has, measured on the released model.
2. **The method pilots:** short training runs of each candidate method, same budget, scored with
   step 1's measurement. Picks the method for the real run.
3. **The real RL run, and onwards:** stage A single-turn, stage B multi-turn in real databases,
   optionally stage C on the NFL harness. Each stage passes the gate before the next starts.

RL trains on training-split questions only: the evalset and the test set stay out of it, as for
SFT.

## The starting point (every step)

The released model: step 1,500 of `sft-main`, merged (`docs/release.md`). RL trains a **new, small
LoRA on top of the merged SFT model**, rather than training the SFT adapter further:

- **The KL reference is free.** The penalty that keeps RL close to SFT needs the SFT model; with a
  separate adapter, that's the same model with the adapter switched off, so no second copy sits
  in the 16 GB.
- **Rollouts** run on llama-server with the release Q8_0 plus the RL adapter as a GGUF LoRA,
  swapped in each step.
- **SFT stays intact.** A stage that goes wrong is undone by dropping its adapter.

## Step 1: the sampling baseline (pass@1 against pass@8)

Today every score is greedy: one answer a question, always the model's most likely token. Here
each of the 200 evalset questions is answered **8 times, sampled** (some randomness, so the 8
answers differ), and checker 4 scores all 1,600. No training happens; it measures the model as it
is.

Each question gets a count, 0 to 8 right:

| Count | Meaning | For RL |
|---|---|---|
| 8/8 | solid already | nothing to learn |
| 1-7/8 | right sometimes | GRPO's learning pool: it rewards the right answers over their wrong siblings |
| 0/8 | never right | GRPO learns nothing; SDPO / SRPO might, from the error feedback |

What it gives:

- **pass@1** (the greedy score we have, and the mean over the 8 samples beside it) and **pass@8**
  (a question counts when any of its 8 answers is right). The gap is roughly what RL can gain.
- **Per dataset**, which environment has the most room (text2cypher scores 27.7% on the test set:
  a big gap there would put Cypher first).
- **The share of informative questions** (1-7/8), which sets how much each RL step learns and so
  the real step time.
- **The 0/8 bucket per dataset**: SDPO's possible extra territory beyond pass@8.
- **Whether temperature 1.0 is too wild:** if the sampled mean falls far below greedy, RL samples
  cooler (~0.7).
- **The yardstick** every later step is scored with.

In `sft-main`'s snapshots, 59 of the 169 evalset questions scored at every snapshot changed between
right and wrong (`docs/real-run.md`): a first hint that the 1-7/8 pool is large.

**Settings:** temperature 1.0 (the usual RL setting), the evalset's thinking budget (1,024),
checker 4 with the same questions left out (those whose reference query returns no rows). **To
build:** a sampled mode for `llamaeval.generate` (greedy only today), 8 answers a question with
fixed seeds. **Cost:** ~50 minutes of generation on the 7800 XT (~1.8 s an answer), plus the
checking.

## Step 2: the method pilots

The baseline measures the model; comparing methods means training. Each candidate gets a short
stage-A pilot (single-turn, execution reward): the same training questions, the same number of
steps, the same rollout settings. Each pilot model is then scored with step 1's measurement.

The candidates:

- **The GRPO family:** GRPO, with the 2025 fixes: Dr. GRPO (no length bias), DAPO (dynamic
  sampling, which skips groups all right or all wrong; clip-higher; token-level loss), GSPO
  (sequence-level ratios). The safe default.
- **SDPO** (Self-Distillation Policy Optimization, Hübotter et al., arXiv 2601.20802, ICML 2026):
  the model, shown the feedback on its failed attempt (the SQL / Neo4j error, the wrong rows), acts
  as its own teacher, and its corrected predictions are distilled into the policy. A dense signal
  instead of one score an answer, and it can learn before the first right answer (the 0/8
  questions). One extra forward pass of the same model, no second model, so it fits.
- **SRPO** (arXiv 2604.02288): SDPO is reported to learn faster early, then degrade and collapse;
  SRPO sends right answers to a GRPO branch and failed ones to an SDPO branch.
- **Not RLHF or DPO:** the checker verifies answers, which beats a learned preference model.

How a pilot is judged: pass@1 up, pass@8 not down, and the canaries (below) steady. The pilots also
time a real step on the 7800 XT, which decides whether a stage runs locally or on Modal. They also
settle the remaining settings with the method: the sampling temperature, the group size, the KL
weight, the learning rate.

## Step 3 and onwards: the real RL run

A curriculum; each stage starts from the one before and passes the gate before the next one starts.

- **Stage A: single turn, execution reward.** Today's task: one query a question, reward 1 when it
  passes checker 4, 0 otherwise, a small penalty for a malformed call. No partial credit for
  "close" queries (it invites reward hacking). Questions focused on the ones the model sometimes
  gets right.
- **Stage B: multi-turn agent in real databases.** The model sees the rows or the error its query
  returned and may look at sample values or try again, up to 3-4 turns, on gretel's real rows, the
  SQaLe fills and CypherBench's Neo4j graphs. The reward comes from the final query. It teaches
  what the release card lists as a limitation (reading a result and retrying). A first-try right
  answer is still the best outcome, so it doesn't unlearn stage A. If step 1 shows little room for
  single-turn RL, stage A is skipped.
- **Stage C (optional): the NFL harness**, the real DuckDB and Neo4j behind Ask the Engine's guard.

**How a step runs on 16 GB.** llama-server (~6 GB with 16 slots) and the trainer (~10-12 GB)
don't fit at once, so each step goes in phases: the server generates and the checker scores; the
server stops and the trainer moves its weights from RAM to the GPU and trains on the groups that
carry a gradient; the new adapter is converted to GGUF (seconds) for the next step. A rough guess
until the pilots time it: 15-20 minutes for 32 questions x 8, so a 300-step stage takes 3-4 days;
stage B costs 2-3x more per step. The rollout loop talks only to an OpenAI-compatible server, so
the same code runs on Modal with vLLM in place of llama.cpp (Neo4j baked into the image as a
process; Modal's Docker-in-Sandbox is alpha).

**Not frying the model:** LoRA, a low learning rate, the KL penalty toward SFT, and these canaries
logged every N steps: entropy and thought length (the thinking budget stays capped), the
well-formed-call rate, the "can't answer, say so" cases, thinking-off calls.

## After every step: the same measurement

Step 1's measurement, the same questions and settings:

1. on the released SFT model, before any RL (the baseline, step 1);
2. on each method pilot (step 2);
3. after each RL stage;
4. on the final RL model.

How to read it:

- **pass@1 should climb toward pass@8**: RL turning "sometimes right" into "right first time".
- **pass@8 should not fall.** A drop means RL is narrowing the model's answers rather than
  teaching it, an early sign of a stage going wrong. A stage that lowers it by more than noise
  (the question-by-question sign test of `docs/real-run.md`) doesn't move on.
