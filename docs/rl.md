# The RL run (planned)

Status (2026-10-28): planning, no method chosen yet. Rishi's direction: after SFT, the main gains
come from RL in environments like real use: the agent plugged into a database as the one who
searches, in real harnesses, over several stages in turn. All of it on the 16 GB card: long runs
are fine, and proving it can be done there is a goal in itself.

## Before RL: how much room it has (pass@1 against pass@8)

RL of the GRPO kind answers a question several times and rewards the answers that check out, so it
learns only from questions the model *sometimes* gets right. A question it always gets right, or
never does, teaches it nothing. In `sft-main`'s snapshots, 59 of the 169 evalset questions scored
at every snapshot changed between right and wrong (`docs/real-run.md`): that is the pool RL works
on.

**The measurement.** The SFT adapter the test set picks (the final one or step 1500,
`docs/real-run.md`) answers each of the 200 evalset questions 8 times, sampled, and checker 4
scores every answer:

- **pass@1**: the greedy score we have (and the mean over the 8 samples beside it);
- **pass@8**: a question counts when any of its 8 answers is right;
- **the gap** between them is the questions RL can turn into first-try answers. Per dataset, it
  says which environment to start with (text2cypher scores only 13-24% greedy: a big gap there
  would make Cypher the first stage).

A big gap means RL has room; a small one means better data or more thinking first.

Settings: the sampling RL itself will use (temperature and the rest, decided with the method), the
thinking budget of the evalset scoring (1,024), checker 4 with the same questions left out (those
whose reference query returns no rows). To build: a sampled mode for `llamaeval.generate` (greedy
only today, temperature 0), 8 answers a question with fixed seeds. Cost: ~50 minutes of generation
(~1.8 s an answer), plus the checking.

## After every major step

The same measurement, the same questions and settings:

1. on the chosen SFT adapter, before any RL (the baseline);
2. after each RL stage;
3. on the final RL model.

How to read it:

- **pass@1 should climb toward pass@8**: RL turning "sometimes right" into "right first time".
- **pass@8 should not fall.** A drop means RL is narrowing the model's answers rather than
  teaching it, an early sign of a stage going wrong. A stage that lowers it by more than noise
  (the question-by-question sign test of `docs/real-run.md`) doesn't move on.

RL trains on training-split questions only: the evalset and the test set stay out of it, as for
SFT.
