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

**Added at 04:47, before any sealed result:**
- **Stopped early:** v3-ranges (seed 1) and v2-eddie stopped at 04:46, and v3-noranges (seed 1) at 04:47, to free Eddie
  slots. Their best held-out exports stay candidates.
- **v3-bc-kl10 and v3-bc-kl30:** PPO from v3-bc's imitation start with a penalty on KL(policy ‖ imitation policy) of
  weight 0.1 and 0.3 (`--kl-bc`), entropy coefficient 0.005. The question: does keeping the coach close to the
  Pluribus imitation help, and how strongly?
- **imitation-only:** the imitation network itself (v3-bc's `bc_init.pt`, no RL), exported as a candidate. It
  measures what RL adds on top of copying Pluribus.
- **Reading:** together with v3-bc and v3-bc-s2 (no pull) this gives a ladder: imitation only → strong pull → light
  pull → free. The v3-bc vs v3-bc-s2 gap is the noise level.

**Added at 05:25, before any sealed result:** `scripts/sealed_eval.py` gained a `--seed0` option, used only for a
pipeline smoke test on other deals (seed 900,000, 500 hands per table, v3-bc and imitation-only); its numbers were not
read. The registered test keeps the default seeds (700,000). Speed: about 220 hands/s on 6 busy cores.

**Added at 05:40, before any sealed result:** v3-bc (held-out +150 → +138 → +68 → +38: it drifts away from the
imitation start) and v3-ranges-s2 (flat at about +90 since update 100) stopped early to free Eddie. Their best held-out
exports stay candidates.

**Added at 05:43, before any sealed result:** training runs to its planned end (~07:15), and the full test (10
candidates, each also with range inputs and style inputs blanked, 30,000 hands per table) takes about 80 minutes on
this machine, so it would end after judging (08:00). It therefore runs at **10,000 hands per table** (the script's
default), all candidates and questions unchanged. The intervals are about √3 wider (roughly ±25 bb/100 pooled for each
policy against the equity bot).

**Added at 05:57, before any sealed result (supersedes the 05:43 timing):** the sealed test runs now, once, on an Eddie
CPU node (a separate checkout, so the training jobs are untouched), at 10,000 hands per table. The candidates are each
run's best held-out export **as of 05:57**, copied to a frozen folder first; training continues, but later checkpoints
are not candidates. Frozen candidates (update, held-out score):

| Candidate | Update | Held-out | Range inputs |
|---|---|---|---|
| v3-noranges-s2 | 150 | +185.9 | off |
| v3-bc | 50 | +150.4 | on |
| v3-bc-kl10 | 100 | +123.0 | on |
| v3-bc-s2 | 100 | +120.6 | on |
| v3-ranges-s2 | 100 | +110.2 | on |
| v3-noranges | 300 | +95.7 | off |
| v3-ranges | 150 | +93.9 | on |
| v3-bc-kl30 | 100 | +86.6 | on |
| v2-eddie | 150 | +73.5 | on |
| imitation-only | 0 | — | on |

## Sealed test results (06:01, run once; `data/rl/sealed_eval.json`)

10,000 hands per table on the three sealed tables, every hero on the same deals; bb/100 against the equity bot, pooled,
with 95% intervals over hands. Every result is reported.

| Candidate | vs equity bot | − imitation-only (paired) | Range inputs worth | Style inputs worth |
|---|---|---|---|---|
| v3-bc-s2 (imitation start, free) | **+207 ± 30** | +210 ± 29 | +6 ± 31 | +33 ± 30 |
| v3-bc (imitation start, free; live coach) | **+149 ± 26** | +152 ± 24 | **+75 ± 24** | −5 ± 25 |
| v3-noranges-s2 (scratch) | +120 ± 27 | +123 ± 25 | — | −71 ± 34 |
| v3-ranges-s2 (scratch) | +73 ± 24 | | −5 ± 25 | −10 ± 24 |
| v3-bc-kl10 (imitation start, KL 0.1) | +69 ± 20 | +72 ± 15 | +23 ± 17 | −23 ± 17 |
| v3-noranges (scratch) | +59 ± 27 | | — | +23 ± 30 |
| v3-bc-kl30 (imitation start, KL 0.3) | +32 ± 19 | +35 ± 11 | **+28 ± 12** | −11 ± 11 |
| v3-ranges (scratch) | +21 ± 28 | | −44 ± 27 | −69 ± 34 |
| v2-eddie | +15 ± 26 | | −14 ± 23 | +1 ± 32 |
| imitation-only (no RL) | −3 ± 18 | — | +11 ± 7 | −4 ± 6 |

**Answers to the registered questions:**
1. **Every candidate except imitation-only beats the equity bot** (v2-eddie and v3-ranges only within noise of it).
   **RL adds a lot over copying Pluribus:** the same network after PPO is +152 ± 24 (v3-bc) and +210 ± 29 (v3-bc-s2)
   better than the imitation alone, on the same deals.
2. **Predicted hands:** within a policy, blanking them costs v3-bc 75 ± 24, kl30 28 ± 12 and kl10 23 ± 17; for the
   runs trained from scratch they are worth nothing or less (v3-ranges −44 ± 27, v3-ranges-s2 −5 ± 25). Across runs from
   scratch, the two seeds with ranges vs the two without: v3-ranges-s2 − v3-noranges +14 ± 25, v3-ranges − v3-noranges
   −38 ± 29, v3-noranges-s2 − v3-ranges-s2 +47 ± 26; the gap between the two seeds with ranges (+52) is as large, so
   from scratch the ranges show no reliable effect. They help the coaches that start from the Pluribus imitation.
3. **Watching the table (style inputs):** no reliable gain; for two scratch runs blanking them helps (−71, −69).
4. **Imitation start vs scratch:** the imitation-start runs without a penalty (+149, +207) beat every run from scratch
   (+21 to +120). **The KL penalty** keeps the policy close to the imitation and costs win rate on these tables
   (kl10 +69, kl30 +32; v3-bc − kl10 +80 ± 24): here, exploiting the opponents paid more than staying close to Pluribus.

**Shipped model:** v3-bc update 50 stays the coach. It was live since 04:42, and at 05:13 the team decided that the
sealed test would settle v3-bc against v3-noranges-s2 (then the held-out leader): v3-noranges-s2 − v3-bc = −29 ± 27.
v3-bc-s2 scores highest here but is not shipped: choosing the best of ten on this test would bias its number upwards.

**Added at 06:53, before the 30,000-hand result:** at the team's request the same frozen candidates (05:57) run again
at the registered **30,000 hands per table**, same seeds, on Eddie (`data/rl/sealed_eval_30k.json`). Each batch starts
with the same deals as the 10,000-hand run, so this extends it rather than repeating it. Both results are reported; the
30,000-hand one is the registered size. It picks nothing: the shipped coach stays as decided at 06:01.

**06:56:** v3-bc-s2 (held-out down to −11 at update 250) and v3-noranges-s2 stopped. Training after the 05:57 freeze
cannot change the candidates; kl10 and kl30 run to their planned end (~07:05).
