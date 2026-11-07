# Poker Mirror — build prompt and ordered checklist

Companion to [PLAN.md](PLAN.md). PLAN.md says what to build. This file says in what order, how to prove each step, and what to paste into a coding agent.

**How to use it.**

1. Everyone does Phase 0 together.
2. Each person pastes the master prompt, then their workstream prompt, into their coding agent.
3. Tick the boxes here. At every gate the integrator (the Demo owner) collects the reports and the team decides to continue or cut.
4. Hours count from the start of the build. If time is already spent, shift every gate by the same amount and apply the cut order at once.

Commands use `python3`; use `python` on Windows.

## Facts checked while writing this file

- The code and samples are committed (`a9aca46`) and 30 unit tests pass on Python 3.12.9.
- PokerKit 0.7.7 exists on PyPI and needs Python 3.11 or later. `pokerkit` and `pyarrow` are not installed by default.
- Takara shards download from `https://huggingface.co/datasets/takara-ai/poker_hands/resolve/6acb5afb6f43082e6a468fde578890d9188be393/data/train-{i:05d}-of-00037.parquet` for `i` from 0 to 36, about 40 MB each. Shards 35 and 36 match the SHA-256 values in PLAN.md §2.
- Replay runs at about 2,700 hands per second per core. The evaluator does about 70,000 seven-card evaluations per second.

## Traps already found

- Takara rows: five hand-annotated rows in shard 0 have a broken `actions` field, date fields are floats, Pluribus and hand-annotated rows have a null venue, and the stake appears only in `source_file`.
- Hole cards revealed at showdown never include a fold. Never calibrate anything on them.
- Enumerating a whole range by nesting costs about fifteen seconds per board. Rank all 1,081 holdings once instead.
- The existing server holds one global game. Mirror needs its own session registry.
- Style statistics must stop at each venue's training cutoff, or the type test leaks.
- Differences between bettors need around a thousand matched deals before they mean anything.

## Who owns what

| Owner | Files |
|---|---|
| A · Data | `pokersim/mirror_data.py`, artifacts under `data/mirror/`, `tests/test_mirror_data.py` |
| B · Model | `pokersim/mirror_model.py`, `pokersim/mirror_advisor.py`, `pokersim/mirror_styles.py`, `tests/test_mirror_model.py` |
| C · Environment | `pokersim/river_env.py`, `pokersim/mirror_rl.py`, `tests/test_river_env.py` |
| D · Demo and integrator | `pokersim/mirror_session.py`, `pokersim/mirror_types.py` (shared decision record), Mirror routes in `pokersim/server.py`, `web/*`, CLI wiring in `pokersim/__main__.py`, the adapter in `pokersim/agents.py` |

Others expose plain Python functions and put their CLI subcommand in their own module. D wires the routes and the commands. A change to `mirror_types.py` needs a one-line note to the team. With three people, give C to the owner of A, because A's scans run in the background while the environment gets built.

## Master prompt

Paste this into every coding agent, then append the workstream prompt.

```text
You are an engineer on a 24-hour hackathon team building "Poker Mirror" in the HackTheBurgh repository (Python standard library plus PokerKit, browser UI in web/).

SPEC. PLAN.md is the spec and CHECKLIST.md is the order of work. Read PLAN.md sections 0 and 1 first, then the sections for your workstream (A: section 2, B: section 3, C: section 4, D: sections 5 and 6). README.md describes the existing pokersim package: engine, PHH parser, replay, agents, server and UI. Reuse it. Do not redesign or rewrite it.

GOAL. An offline, playable demo that answers: after accounting for the cards and the public betting history, how much does knowing this opponent help a bettor decide whether to bluff on a heads-up river where the defender checks and faces a half-pot bet? It must show (1) prediction and decision-making and (2) the invisible made visible: hidden folds in filtered data, hidden opponent type, hidden hand strength.

RULES
1. Follow CHECKLIST.md in order. Do not start a step before its prerequisites pass. Stop at every GATE and send the report.
2. At demo time there is no network, API key or GPU. Runtime dependencies are the standard library and PokerKit 0.7.7. pyarrow, numpy and scikit-learn are for offline preparation only.
3. Every number on screen comes from a generated artifact and shows its source and denominator. Never hard-code a result. Label synthetic opponents as synthetic. Show null or negative results plainly.
4. Verify before you claim. Run the check listed for the step and quote the output. If it fails, say so with the output. Never weaken a test to make it pass.
5. Stay inside your files. Shared files (pokersim/server.py, pokersim/__main__.py, pokersim/agents.py, web/app.js) change only through the Demo owner or in a small rebased PR.
6. Commit small on your own branch. Rebase on main before merging at a gate. Never force-push. Never commit raw datasets, caches, keys or files over 5 MB.
7. If a step is blocked or over budget, apply the cut rule from PLAN.md section 0 instead of pushing on. Cut extras first, then the type layer. The only Core fallback is the EV advisor replacing Q-learning at H12. If you are more than an hour late at a gate, cut immediately.
8. Out of scope: six-handed play, CFR or other equilibrium solvers, deep RL, other bet sizes, anything that implies live help at real-money tables.
9. If PLAN.md and reality disagree, say so in the report and propose the fix. Do not deviate silently.
10. Ask the human only when blocked by a decision the spec does not settle. Otherwise state your assumption in one line and continue.

REPORT at each gate: Done / Evidence (command and result) / Failing or cut / Decision needed / Next.
```

## Phase 0 · Setup (H0, everyone, about an hour)

- [ ] **0.1 Sync.** Commit and push PLAN.md and this file. Everyone pulls `main`. Create one branch per workstream: `ws-a-data`, `ws-b-model`, `ws-c-env`, `ws-d-demo`.
- [ ] **0.2 Owners.** A ______  B ______  C ______  D ______. D also integrates and calls the gates.
- [x] **0.3 Install.** `python3 -m pip install pokerkit==0.7.7 pyarrow numpy scikit-learn`. Put `pokerkit==0.7.7` in `requirements.txt` and the other three in `requirements-prep.txt`. Verify: `python3 -c "import pokerkit, pyarrow, numpy, sklearn"` runs clean.
- [x] **0.4 Baseline.** `python3 -m unittest discover tests` reports 30 tests OK. `python3 scripts/validate_dataset.py data/samples` runs. `python3 -m pokersim serve` opens the UI.
- [x] **0.5 Start the long job now (A).** Download all 37 shards into `data/cache/takara/`, which is gitignored. Verify: shards 35 and 36 match the SHA-256 values in PLAN.md §2.
- [x] **0.6 Read.** Each owner reads PLAN.md §0, §1 and their own section before writing code.

## Phase 1 · Shared contract and first decision (H0 to H3)

- [x] **1.1 [All] Decision record.** One shared module defines the record from PLAN.md §1: session id, decision id, context, offer and pot, legal mask, board, own cards, committed prediction, actual action, and optional type posterior, range readout and EV table. Add one JSON example. Verify: a unit test round-trips the example.
- [x] **1.2 [C] PokerKit adapter and parity fixtures.** Reproduce the PLAN.md §4 fixture: pot 100, stacks 150, and bettor stack changes of 0, +100, −50, −50 and −100 for check-back, bet-fold, bet-call, bet-raise-fold and bet-raise-call. Every path conserves 400 chips. Add one split-pot and one winning-hand fixture. Verify: the new tests pass and PokerKit payouts equal PokerSim payouts.
- [x] **1.3 [D] Session registry.** A token per browser, one lock per session, and a reset endpoint. Verify: two browsers keep separate profiles, a refresh recovers the open decision, and reset clears it.
- [x] **1.4 [B and D] First end-to-end decision.** A seeded river with pot 100, where the defender checks and the bettor offers 50. The judge clicks fold, call or raise. The prediction is stored before the click and shown afterwards beside the actual action and the new observation count. Verify: a scripted API test and one manual run.
- [x] **1.5 [A] Schema gate.** On a 1,000-row sample from shards 35 and 36, confirm the columns map onto `record_from_dict` after the integer casts and the one-action rejection in PLAN.md §2. Verify: the sample replays without a crash and rejected rows are counted.
- [x] **1.6 [A] Reproduce the reference counts exactly.** Folds over opportunities, shard 36 then shard 35.
  - All eligible offers: 592/1,057 and 688/1,220.
  - Exact half pot after a checked-through turn: 266/394 and 299/455.
  - Exact half pot after the river bettor bet or raised the turn: 52/100 and 71/141.
  - Both defender cards known: 0/205 and 0/225.
  - Warning-free hands: 592/815 and 688/925.

  Verify: one test asserts all ten numbers.

**GATE H3.** Pass means 1.1 to 1.6 are done. If 1.6 does not reproduce, fix the filters before any modelling, because no downstream number is trustworthy until it does.

## Phase 2 · Core build in parallel (H3 to H9)

**A · Data**

- [x] **2.A1** First scan over all 37 shards, parallel by shard, with per-shard input, eligible, duplicate and rejection counts, cached by source checksum. Verify: the counts table is written and a rerun is a cache hit.
- [x] **2.A2** Global deduplication on venue and hand ID plus a content fingerprint, then the chronological 80/20 manifests with largest-remainder allocation by venue. Verify: no hand or fingerprint appears on both sides, and the training count equals floor(0.8 × N).
- [x] **2.A3** Export the full-corpus counts, the two-shard exhibit and the two opening-replay hands: `3039372300` at step 15 and `3039405280` at step 14. Verify: both replays stop just before the response with unknown cards face down.
- [x] **2.A4** Commit only small artifacts, plus the CC BY 4.0 attribution. Verify: `git status` shows no data files.

**B · Model**

- [x] **2.B1** Population model with `population.json`, and the personal Dirichlet model with pre-click predictions, fold interval and legal mask. Verify: updates apply once, predictions sum to one, and the interval matches Beta(alpha_fold, alpha_call + alpha_raise).
- [x] **2.B2** Strength buckets by exact equity with cut points 0.33 and 0.67, cached by board and cards, plus a rank-once whole-range routine per board. Verify: it matches brute force on several boards and takes well under a second per board.
- [x] **2.B3** Playable twin: public table, strength table with calibration backoff, and an immutable frozen snapshot. Verify: the same seed gives the same responses, and swapping the defender's hidden cards leaves the bettor's fold prediction unchanged.
- [x] **2.B4** Bucket-level range readout and EV advisor from PLAN.md §3. Verify: EV(bet) equals 1.5 × p_fold − 0.5 on the pure-bluff fixture, break-even is one third, and readout weights sum to one.

**C · Environment**

- [x] **2.C1** `RiverResponseEnv` with seeded reset, masked actions and reward relative to the opening stack. Verify: payoffs match the PLAN.md §4 table, and a fixed seed with fixed actions reproduces the outcome.
- [x] **2.C2** Oracle fixtures: the pure-bluff subgame, where fold rates 0.2 and 0.6 give bet values −0.20 and +0.40, and a deterministic raise fixture. Verify: learned values land within 0.10 opening pots and the greedy actions are check and bet respectively.
- [x] **2.C3** Masked tabular Q-learning with checkpoints at 1k, 5k, 10k and 20k episodes, visit counts, and the `mirror-train` command. Verify: validation reward and greedy actions are stable across three seeds.
- [x] **2.C4** A seeded scenario pool with both buckets precomputed and reused by every arm. Verify: a 20,000-episode run spends its time on play, not on equity.

**D · Demo**

- [x] **2.D1** A Mirror entry in the UI, the reveal flow for the two opening hands, the three filters and the practice flow. Verify: the filters show 1,280/2,277, 0/430 and 1,280/1,740.
- [x] **2.D2** Reset between judges. Verify: nothing leaks between sessions.
- [x] **2.D3** An Evidence page skeleton that reads generated artifacts only. Verify: with no artifact present it says so instead of showing a number.

**GATE H9.** Pass means full-corpus counts exist, or the two-shard cohort is declared as the fallback with every number labelled so while the full scan keeps running. The pure-bluff oracle must pass, and a person must be able to play a river against a twin from start to finish.

## Phase 3 · Evaluation and the type layer (H9 to H15)

- [x] **3.1 [C] `Q*` for every cell.** Average the twin-exact EV over the scenario pool, as in PLAN.md §4. Learned Q must be within 0.10 opening pots on demo cells with at least 100 visits. List every disagreement with the EV advisor. **H12 gate:** if this fails, display the EV advisor as the bettor and call Q-learning exploratory.
- [x] **3.2 [A] Style scan.** The second scan, run after the manifests exist, using each venue's cutoff. Five features, no river decisions, per-shard partial tables merged with a group-by. Verify: a test shows no river action and no hand after the cutoff in the counts.
- [x] **3.3 [A] Handle gate.** Report, per venue, how many handles reach several hand-count thresholds. If handles do not repeat across shards, cluster within shards.
- [x] **3.4 [B] Types.** Choose the minimum hand count by split-half reliability of at least 0.7. Fit a Gaussian mixture for K from 2 to 8 by bootstrap stability, capped at 6. Name the types, build the type response tables and export map points without handles.
- [x] **3.5 [B] Mixture model and type-prior gate.** Run the lift test on the frozen 20%: four forecasters, bootstrap over players, keep the type prior only if its log-loss interval beats the population model. Repeat on hands outside shards 35 and 36. Record the result whichever way it falls.
- [x] **3.6 [C] Paired benchmark.** Four synthetic opponents (frequent caller, frequent folder, context-matched population control, strength-dependent raiser), probe budgets 0, 4, 16 and 64, 1,000 matched opportunities each, and a paired 95% bootstrap interval. Verify: it regenerates from the documented seeds.
- [ ] **3.7 [D] Visuals.** The style map with the judge's dot, the 13 by 13 range grid, the learning-curve chart and the policy chart.

**GATE H15.** Pass means the H12 outcome is recorded, the type-prior gate is decided and recorded, and the benchmark regenerates from seeds.

## Phase 4 · Evidence and integration (H15 to H20)

- [ ] **4.1 [All] Volunteer sessions.** Run development sessions first to calibrate the strength tables. Then run new held-out sessions with people who have not seen the model. Keep labels local to each session. Verify: the Evidence page shows the number of independent sessions.
- [ ] **4.2 [B] Held-out evaluation.** The forecast ladder on the frozen 20% and on volunteer sessions, and the range readout on held-out sessions.
- [x] **4.3 [A] Pluribus check (extras).** Only if everything above is done. Use spots from rows whose `source_file` starts with `pluribus/`, report the defender's mean true equity by response, and call it directional.
- [ ] **4.4 [D] Evidence page complete.** Five panels, each with source, denominator and date. Failed or null results are shown plainly.
- [ ] **4.5 [All] Full 90-second run offline** with the real artifacts. Fix what breaks.

**GATE H20.** Feature freeze. Tag `freeze`. Defects only from here.

## Phase 5 · Rehearse and ship (H20 to H24)

- [ ] **5.1** Fresh-checkout test on a second machine with Wi-Fi off: clone, install, launch.
- [ ] **5.2** Record a backup screen capture. Make sure the pretrained practice profile loads instantly.
- [ ] **5.3** README: launch command, what each screen shows, reproduction commands, attribution to the Takara dataset and the Zenodo original under CC BY 4.0, and the limits.
- [ ] **5.4** Rehearse three times with a timer. Decide who talks and who clicks.
- [ ] **5.5** Tick every item in the PLAN.md §7 definition of done.

## Workstream prompts

Append one of these to the master prompt.

```text
WORKSTREAM A · Historical data.
Read PLAN.md section 2 in full, and section 3 only for what B needs.
You own pokersim/mirror_data.py, the prepare command, the artifacts under data/mirror/ and tests/test_mirror_data.py.
Do first: start the shard download (CHECKLIST 0.5), run the schema gate on a 1,000-row sample, and reproduce the ten reference counts exactly before scaling to all 37 shards.
Deliver to B: population counts by context, the 80/20 manifests, and later the per-player style table with per-venue cutoffs. Deliver to D: the filter-exhibit JSON, the two opening-replay hands and the full-corpus counts.
Watch for: rows with one action, float date fields, null venue, duplicate hands across shards, memory (write per-shard partial tables), and timestamp ties.
Done when the counts, manifests and artifacts regenerate from the documented command and the reference-count test passes.
```

```text
WORKSTREAM B · Player model.
Read PLAN.md section 3 in full and section 6 for how it is scored.
You own pokersim/mirror_model.py, pokersim/mirror_advisor.py, pokersim/mirror_styles.py and tests/test_mirror_model.py.
Do first: the population and personal models with pre-click predictions, then strength buckets with the rank-once routine, then the twin, then the range readout and EV advisor. Start the type layer only after the H9 gate.
Deliver to C and D: one policy interface with the explanation fields (forecast, interval, EV table, readout) and a frozen snapshot function.
Watch for: legal masks, updates applied twice, any use of the defender's hidden cards in the bettor's view, and calibrating on showdown reveals.
Done when every check listed under Phase 2 B passes and the type-prior gate result is recorded.
```

```text
WORKSTREAM C · Game and evaluation.
Read PLAN.md section 4 in full and the decision-value part of section 6.
You own pokersim/river_env.py, pokersim/mirror_rl.py and tests/test_river_env.py.
Do first: the PokerKit adapter with the parity fixtures, then RiverResponseEnv, then the oracle fixtures, then masked Q-learning, then the scenario pool.
Deliver to D: playable scenarios, checkpoints and the paired benchmark results. Deliver to B: held-out prediction scores once they exist.
Watch for: equity recomputed inside the training loop, shared random streams, reward measured from the wrong stack, and treating a small sample as a result.
Done when the oracle and Q* checks pass or the H12 fallback to the EV advisor is recorded, and the benchmark regenerates from documented seeds.
```

```text
WORKSTREAM D · Demo and integration.
Read PLAN.md sections 5 and 6, and the reuse table in section 1.
You own pokersim/mirror_session.py, pokersim/mirror_types.py, the Mirror routes in pokersim/server.py, web/*, the CLI wiring in pokersim/__main__.py and the adapter in pokersim/agents.py.
Do first: the decision record, the session registry, then the first end-to-end decision, then the reveal flow and filters, then the Evidence page skeleton.
You also call the gates, merge the branches and run the fresh-checkout test.
Watch for: hard-coded numbers, state shared between judges, the UI depending on the network, and a demo that only works with the type layer.
Done when the full 90-second script runs offline from a fresh checkout and every number on screen traces to an artifact.
```

**One agent building everything.** Work through Phases 0 and 1, then B and C core, then D core, then Evidence, then the type layer, then extras. Stop at every gate and send the report.

## Gate report

```text
GATE H__ report
Done:
Evidence: <command> -> <result>
Failing or cut:
Decision needed:
Next:
```

## Demo day

- [ ] Laptop on power, Wi-Fi off, server started from a fresh clone.
- [ ] Pretrained practice profile loads. Backup recording is open.
- [ ] Evidence page is open in a second tab.
- [ ] Reset between judges tested twice.
- [ ] Roles agreed: one person talks, one clicks.

Likely questions and honest answers:

- **Is this cheating at poker?** It is a training and analysis tool. Sites ban real-time help, and we do not claim live use.
- **Are the types real?** The type prior ships only if it beats the population model on held-out data. Otherwise the map is labelled as a description.
- **Why only the river?** One decision with an exact break-even lets us check prediction, decision and learning against the same ground truth.
- **Is the range a real observation?** No, it is a model estimate at three-bucket resolution, checked on held-out sessions.
- **What if knowing the opponent does not help?** Then we say so. Identical actions are a valid result.
