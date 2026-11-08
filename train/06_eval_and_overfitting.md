# 06 · Evaluation and overfitting

| Risk | Guard |
|---|---|
| RL fits the training bots (an earlier run lost 894 bb/100 to an unseen rule bot) | Selection only on held-out tables (Pluribus clones + new rule bots), paired per hand on the same deals; the final report uses fresh deals |
| RL fits its own past copies (cycling) | Past copies capped at 25% of seats; a falling held-out score while in-distribution rises is flagged (08) |
| Base models memorise players | Online data split by player; a separate later shard range (34–36) for test; early stopping on validation; test read once |
| Hand model showdown bias | Quote the Pluribus-only figures as the honest ones (+0.31 nats) |
| Personal model fits 50 decisions | Dirichlet shrinkage towards the next-move prior; scored before each action; shown next to the usual-move baseline with its count |

## How the RL numbers are read

- A difference is real only if its **paired** 95% interval excludes 0.
- Every claim is against bots. The held-out Pluribus clones are copies of strong human play, not strong humans.
- The in-distribution tables include very weak bots (calling stations), so numbers like +3,687 bb/100 only say the
  policy found their weaknesses.
- "How much the policy relies on the hand inputs" comes from zeroing them. "Whether predicting hands helps" comes
  from the matched v3-ranges / v3-noranges comparison at equal updates.

## Sealed final test (registered at 03:20 on 2026-11-08, before any result)

**Why:** each run picks its model on the held-out tables, so its held-out score is biased upwards. Fresh deals don't
remove that. The sealed test uses opponents that no run trained against and no run was picked on. It runs once,
after training ends, with `scripts/sealed_eval.py`.

**Tables:**

| Table | Opponents | Why they are new |
|---|---|---|
| `sealed_styles` | Next-move bots at tilt +1.2 towards calling, raising or folding, and at temperatures 0.6 and 1.5 | Training used tilts within ±0.5 and temperatures 0.8–1.2; selection used Pluribus clones at 0.9–1.2 |
| `sealed_rules` | Three rule bots written for this test: `overbettor`, `limper`, `checkraiser` | Never used anywhere before |
| `sealed_mix` | Both kinds | — |

**Candidates:** `export_heldout/` of v3-ranges, v3-ranges-s2, v3-noranges and v2-eddie as they stand when training
ends (~07:20). Each policy plays greedily.

**Set-up:**
- 30,000 hands per table and hero; seeds start at 700,000 (in-training evaluations use 10,000 and final ones 50,000).
- Every hero plays the same deals, so all comparisons are paired per hand, with 95% intervals over hands.

**Questions, decided now:**
1. Does each candidate beat the equity bot on the sealed tables pooled?
2. Are the predicted hands worth anything?
   - Within each policy that uses them: the policy against itself with range inputs blanked.
   - Across runs: v3-ranges and v3-ranges-s2 head to head against v3-noranges. The gap between the two seeds with
     ranges is reported as the noise level.
3. Is watching the table worth anything? Each policy against itself with the opponent-style inputs blanked.

**Rules:**
- The shipped model is still picked on the held-out tables, as before. The sealed test picks nothing.
- Every result is reported, whatever it shows.

**Before this was written:** one pipeline check with 200 hands per table on the 03:1x exports. At that size the
intervals are ±150–450 bb/100, so those numbers carry no information and are not used.

**Added at 03:49, before any sealed result:** v3-noranges-s2, a second seed without range inputs (5090, started 03:48),
so the "are predicted hands worth anything?" comparison is two seeds against two. It runs to ~07:20 like the others
and joins the candidates with its `export_heldout/`.

**Added at 03:51, before any sealed result:** v3-bc (Eddie, range inputs on), PPO started from a network trained to
imitate the Base model (`scripts/bc_init.py`) instead of from scratch. Same trainer, opponents and selection; it joins
the candidates with its `export_heldout/`. It starts about two hours later than the others, so it gets fewer updates.

**Added at 04:35, before any sealed result:** v3-bc-s2, a second imitation-start seed (Eddie), to tell whether v3-bc's early
lead is the method or luck. It joins the candidates with its `export_heldout/`.
