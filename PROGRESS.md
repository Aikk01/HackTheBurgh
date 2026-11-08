# Poker Mirror — build progress log

Companion to [CHECKLIST.md](CHECKLIST.md) (order of work) and [PLAN.md](PLAN.md) (spec).
This file is the single place to look to find out **what is done, what is running, what was
decided, and how to pick up**. Update it whenever a step completes or a decision is made.

Build started: 2026-11-07 18:00 (local). Hours in the checklist count from here.
Builder so far: Claude as the single build agent. Commits go out under the team member's own git identity.
Last update: 2026-11-08 02:25 (Phase 7 below). Earlier: RL runs finished on the second PC and were committed at 19:35
(`policy_population.json`, `training_sufficiency.json`, `benchmark.json`). The post-push work is committed on top:
volunteer-session tooling, style map, stricter type gate, range-check panel and UI fixes.

## At a glance

**What we are building.** Poker Mirror, a training and analysis tool inside PokerSim, built on 21.6 million real
hands. It answers one question: on a heads-up river where the defender checks and faces a half-pot bet, how much
does knowing *this* opponent help the bettor decide whether to bluff?

**What works today.** Run `python -m pokersim serve` and open the Mirror tab:
1. Replay two real river decisions by the same player and guess which one folded.
2. See what data filters hide: folds never reveal cards.
3. Defend: the app commits a forecast of your answer before each click and learns your profile.
4. Bet against your frozen twin, with an EV advisor and a range X-ray of the twin's likely hand.
5. Compare two Q-learning bots, one trained against an average player and one against your twin.

The Evidence tab shows every number with its source and denominator.

**Results so far.** Knowing the individual helps a little, and the effect is real. Knowing their playing type does
not help. Folds never reveal cards: 0 of 4,156 offers where the defender's cards were shown. Pluribus hands confirm
that weaker hands fold more. In the RL benchmark, a bot tuned to one opponent wins more chips when that
opponent plays very differently from the crowd, such as a frequent caller or a frequent folder. Against an
average opponent it loses a little, because a few dozen hands give a noisy profile.

**Phase 6, every table size (157,926 decisions).** The situation, meaning bet size, board and earlier betting,
predicts a defender's answer about ten times better than their personal history does. The person still adds a
real, smaller amount on top. A bluffer using both wins slightly more against real players' recorded answers,
because almost everyone folds often enough that a half-pot bluff is nearly always right.

**Waiting on.** The GPU run (see "Handoff: GPU run"), the rest of Phase 6, merging `poker-mirror` into `main`,
and the team rehearsal. Volunteer sessions are now optional.

## Phase 7 · 6-max Table Lab (branch `train-v3`, 2026-11-08)

**What it is.** One clean screen (`/lab.html`): you against five bots. At every decision it shows three views of
the same moment:
- what people in general do (the Base model, learned from recorded hands);
- what wins against these opponents (the RL model);
- what *this* player does (a personal model per player, for you and for each opponent, learning live).

A 13×13 grid shows each opponent's likely starting hands in their own colour. Any missing model shows
"No model loaded". The training plan and its review are in [train/](train/README.md). Live training state is in
[train/TRAIN_PROGRESS.md](train/TRAIN_PROGRESS.md).

**Done:**
- **GPU runs from Phase 6:**
  - The neural forecaster on the RTX 5090 (`data/mirror/neural_forecaster.json`): test log loss 0.7902 against
    0.7899 for situation + personal.
  - The heads-up river Q-learning runs reproduced exactly (`benchmark.json`, `training_sufficiency.json`).
- **Hand model** (`pokersim/hand_range.py`, `data/rl/hand_range.npz`): P(each opponent holds each of 169 starting
  hands) from public information. Trained on Pluribus (all cards known) plus online showdowns. The honest test, on
  Pluribus: +0.31 nats over card counting.
- **Next-move model** (`pokersim/next_move.py`, `data/rl/next_move.npz`): P(next action) over 7 classes.
  - Data: 3.24M decisions from 5–6-player hands, with no showdown filter. Cards come only from the Pluribus logs.
  - Test: +0.12 nats over frequency baselines (public view), +0.40 with cards (top-1 81%).
  - Its uses: Base advice (plays like Pluribus), opponents' next move, the prior of every personal model, and the
    "Pro"/"Machine" bots.
- **RL (PPO, 193 inputs)** with the predicted ranges as input. The opponents are:
  - scripted bots;
  - Pro bots cloned from the 13 Pluribus-log professionals;
  - past copies of itself.

  Models are chosen only on held-out tables: Pluribus-cloned bots plus four new rule bots, with paired per-hand
  evaluation. A matched run without the predicted-hand inputs measures whether they help. Every run has a
  supervisor: checkpoints, wandb, automatic reset or restart, and a health check.
- **Personal models** (`pokersim/you_model.py`): Dirichlet shrinkage of the next-move prior per street and
  facing-a-bet bucket.
  - Every prediction is made before the action and scored after it, against the "usual move" baseline.
  - Your counts and a full decision log persist in `data/players/<name>/`.
- **Table Lab frontend** (built on the `table-lab` branch design):
  - Each opponent box shows only the next move (People / This player).
  - The grid is shaded per card combination, and real hands are outlined at showdown.
  - Your panel shows Base and RL top actions; your model's guess is revealed only after you act, with a running
    score.
- **Bots renamed for the demo:** Shark, Hustler, Calculator, Caller, Pro, Machine, Rock, Trapper, Nibbler,
  Accountant, Wildcard, Maniac.
- **Reserved demo player:** one real PartyPoker player (666 hands) that no model has trained or been evaluated on
  (`data/demo/`). Extraction scripts skip their hands.
- **Tooling:** uv with `pyproject.toml` and `uv.lock` (Python 3.12, CUDA 12.8 torch on Linux), identical on every
  machine.

**Running now:**
- three RL runs until about 07:25: v3-ranges (5090), v3-noranges (2080), v2-eddie (Eddie);
- see [train/TRAIN_PROGRESS.md](train/TRAIN_PROGRESS.md).

**Next:**
- Ship the best held-out policy to `data/rl/policy.npz`.
- A replay mode for the reserved player (watch "This player" learn them hand by hand).
- Rehearsal.

## Status board

Legend: `[x]` done and proven, `[~]` in progress / handed off, `[ ]` not started, `[-]` cut or left to a human.

### Phase 0 · Setup
- [~] 0.1 Sync — pushed to branch `poker-mirror`; merging into `main` and the other owners pulling is up to the team.
- [-] 0.2 Owners — human.
- [x] 0.3 Install — `pokerkit==0.7.7`, `pyarrow`; `requirements.txt` + `requirements-prep.txt`.
- [x] 0.4 Baseline — 30 original tests OK (Python 3.12.9).
- [x] 0.5 Shards — all 37 in `data/cache/takara/` (1.2 GB), shards 35/36 SHA-256 verified (`scripts/fetch_takara.py`).
- [x] 0.6 Read.

### Phase 1 · Shared contract and first decision — GATE H3 passed
- [x] 1.1 Decision record — `pokersim/mirror_types.py`, JSON round trip tested.
- [x] 1.2 PokerKit adapter + parity — `pokersim/river_env.py`; five paths give 0/+100/−50/−50/−100 on both engines, 400 chips conserved, split-pot and winning-hand fixtures.
- [x] 1.3 Session registry — `pokersim/mirror_session.py` (token per browser, lock per session, reset).
- [x] 1.4 First end-to-end decision — forecast committed before the click, refresh recovers it.
- [x] 1.5 Schema gate — `mirror_data.row_to_record` (integer casts, one-action rejection).
- [x] 1.6 Reference counts — all ten numbers reproduce exactly on shards 36 and 35.

### Phase 2 · Core build — GATE H9 passed
- [x] 2.A1 First scan, 37 shards, 21,616,175 rows, 78 s on 15 workers (`python -m pokersim mirror-prepare`).
- [x] 2.A2 Dedupe + chronological 80/20 manifests — train 9,925 / test 2,482 exact half-pot offers.
- [x] 2.A3 Exports in `data/mirror/`: `exhibit.json`, `scan_counts.json`, `population.json`, `demo_hands.phhs`.
- [x] 2.A4 Commit — only small artifacts are committed (`data/cache/` is ignored); the README credits the Takara corpus (CC BY 4.0).
- [x] 2.B1 Population + personal Dirichlet model, legal mask, Beta interval (own incomplete beta, no SciPy).
- [x] 2.B2 Exact river equity, rank-once per board (6–25 ms), brute-force verified (`mirror-check-ranges`).
- [x] 2.B3 Twin + frozen snapshot (hash); hidden cards do not move the public forecast.
- [x] 2.B4 Range readout + EV advisor; pure-bluff identity 1.5·p − 0.5 and one-third break-even verified.
- [x] 2.C1 `RiverResponseEnv` (seeded reset, masks, reward from the opening stack, separate RNG streams).
- [x] 2.C2 Oracle fixtures: pure bluff at fold rates 0.2/0.6 → −0.20/+0.40 within 0.10; raise fixture.
- [x] 2.C3 Q-learning + `mirror-train`: population bot, 20k episodes, seed 7, 53 s; oracle passes in all 9 stage-1 cells.
- [x] 2.C4 Cached scenario pools (`data/cache/mirror/scenarios_<seed>_<n>.json`, atomic writes).
- [x] 2.D1 Mirror tab: replay reveal, filters (1,280/2,277 · 0/430 · 1,280/1,740), defender drills (`web/mirror.js`).
- [x] 2.D2 Reset between judges; session isolation tested.
- [x] 2.D3 Evidence tab reads artifacts only and says "missing" otherwise.

### Phase 3 · Evaluation and type layer — GATE H12 passed, GATE H15 decided
- [x] 3.1 Q* for every cell — learned Q within 0.10 of Q* in every supported cell → Q-learning stays the displayed bettor.
- [x] 3.2 Style scan — 217,689 handles (`data/cache/mirror/style_table.csv`, `data/mirror/style_counts.json`).
- [x] 3.3 Handle gate — handles repeat across shards (≥100 hands: PokerStars 50,781 · PartyPoker 10,669 · Full Tilt 8,416 · Absolute 7,850).
- [x] 3.4 Types — N_min = 200 hands (reliability ≥ 0.7 on all five features); 48,059 players; K = 4, bootstrap-stable: tight-aggressive 15%, tight-passive 29%, loose-aggressive 35%, loose-passive 21%.
- [x] 3.5 Type-prior gate — **FAILED** (recorded). Re-run with the selected prior weight 24 and a stricter rule (decision 12): type prior vs population for players with no river history +0.0020, 95% CI [−0.0052, +0.0084]; types+history vs history alone +0.0005, CI [−0.0011, +0.0020]. The live model runs K = 1; the type map is shown only as a labelled description.
- [x] 3.6 Paired benchmark — full run on the second PC (9 min 38 s, 14 workers): `zero_probe_identical: true`, control 0.0 [0.0, 0.0]; personalised bot gains vs caller (+4.55 at 64 probes) and folder (+8.00), loses vs population (−2.75 [−4.4, −1.1]). Sufficiency: `all_oracle_pass: true`, seed disagreements only on near ties. Population policy reproduced exactly (only timing fields differ). One row failed its oracle check (folder, 16 probes): it made the baseline's choice in every spot, so its 0.0 is a training miss, not a finding.
- [~] 3.7 Visuals — range grid, policy chart, EV panel and the style map (Evidence tab, one small multiple per type) done. No judge dot, because the gate failed. The learning curve is shown as the forecast-ladder table, not yet as a chart.

### Phase 4 · Evidence and integration
- [~] 4.1 Volunteer sessions — tooling done, sessions need humans. In step 3, tag the judge as calibration or held-out; every answer is logged to `data/cache/mirror/sessions/` (ignored by git, never leaves the laptop). Evidence → "Recompute from sessions" (or `python -m pokersim mirror-eval --sessions`) fits the strength calibration from calibration sessions and scores held-out ones. Until then the strength table is the labelled synthetic default.
- [~] 4.2 Held-out evaluation — historical part done (`data/mirror/eval.json`); volunteer part needs 4.1.
- [x] 4.3 Pluribus check — 701 heads-up river spots, all cards known: mean defender equity fold 0.51 < call 0.74 < raise 0.84 (monotone; 211 spots in the half-pot band).
- [~] 4.4 Evidence page — every panel renders from the committed artifacts, checked in a headless browser after the RL results landed. The benchmark table marks the one row whose bot failed its oracle check. Only the human-session panel still waits for 4.1.
- [~] 4.5 Browser pass done on a fresh server: every Mirror step and the Evidence tab render with real artifacts. The timed 90-second rehearsal is for the team.

### Phase 5 · Rehearse and ship
- [ ] 5.1–5.5 — human.

### Phase 6 · Use all the data and real opponents (CHECKLIST.md Phase 6) — paused mid-way
Phase 6 last update: 2026-11-07 ~22:00, stopped on request and pushed; 6.1-6.4 done, 6.5-6.11 partly done.
- [x] 6.1 Wider extractor — any table size where exactly two players reach the river (`mirror_data.extract_spot(wide=True)`, `prefilter_wide`). Two-player hands extract exactly as before (tested on 30,000 rows of shard 36); six-handed fixture and split rule tested.
- [x] 6.2 Full scan and frozen split — `python -m pokersim mirror-prepare --wide --workers 8`, 805 s: 539,052 unique spots at any bet size; main cohort (40-60% bets, all three answers legal) 157,926 decisions by 60,918 players, train 126,340 / test 31,586. Two-player rows reproduce the original exactly (22,320 band, 12,407 exact half). `data/mirror/scan_counts_wide.json`, `population_wide.json`.
- [x] 6.3 Forecast models — `python -m pokersim mirror-wide` (`pokersim/mirror_wide.py`): situation model = gradient-boosted trees on public features (bet size, board texture, seats, earlier streets, stakes). Results in Key results below and `data/mirror/eval_wide.json`. Prior weight re-selected: 24.
- [x] 6.4 Bluff test on real decisions — `data/mirror/bluff_test.json`. Small but real gain for situation + player (see Key results).
- [~] 6.5 Evidence panels — API (`mirror_api.wide_panels`) and renderers (`phase6Panels` in `web/mirror.js`) written; headless Edge renders the new section (text check). A visual pass is still to do. Panels without an artifact say "missing".
- [~] 6.6 RL benchmark on real players — code written (`pokersim/mirror_real.py`, `python -m pokersim mirror-real --qlearning-sample 8`); the vectorised exact best response equals `mirror_rl.q_star` (difference 0.0, ~200x faster). NOT run yet. Only 90 players have 40+ decisions, so use `--probes 0,4,8,16` (needs 16 + 24 decisions per player).
- [~] 6.7 Strength table from Pluribus — code written (`python -m pokersim mirror-eval --pluribus-strength`, writes `strength_pluribus.json`, used only if the held-out half improves with a 95% interval above zero). NOT run yet.
- [ ] 6.8 Type gate on the new cohort — not started. Run `python -m pokersim mirror-prepare --styles-wide` (new cutoffs, ~10 min), then add a wide mode to `mirror_styles.build` / `compute_gate` and write `data/mirror/styles_wide.json`.
- [~] 6.9 Real-player twins in the app — code written: `MirrorSession.load_real_player`, `GET /api/mirror/real_players`, `POST /api/mirror/real/load`, a picker in step 4, deep link `#mirror=play&real=<id>`; `data/mirror/real_players.json` (90 players, anonymous IDs). Headless Edge loads a real player's twin and shows the picker (text check); a visual pass is still to do.
- [~] 6.10 GPU track — `scripts/train_neural_forecaster.py` and `data/mirror/neural/neural_dataset.npz` (539,052 rows, 43 public features + player history, 15 MB, committed). Quick CPU check passed on a small sample. Waiting for the 5090 run (see "Handoff: GPU run").
- [~] 6.11 Docs, tests, push — pushed with all 71 unit tests green (4 new for 6.1); PLAN.md scope note, README commands and unit tests for `mirror_wide` / `mirror_real` are still to do.

## Key results so far

| Result | Value | Source |
|---|---|---|
| Unique eligible river offers, all 37 shards | 22,320 (12,570 folds) | `data/mirror/scan_counts.json` |
| Modelling cohort, exact half pot, all three responses legal | 12,407 | same |
| Held-out log loss per decision: pooled / population / personal | 0.7901 / 0.7831 / 0.7803 | `data/mirror/eval.json` |
| Personal history vs population | +0.0028 nats, 95% CI [+0.0002, +0.0060] — small but real | same |
| Type prior vs population, players with no river history | +0.0020 nats, 95% CI [−0.0052, +0.0084] — no effect | `data/mirror/styles.json` → `gate` |
| Types + history vs history alone | +0.0005 nats, 95% CI [−0.0011, +0.0020] — no effect, gate failed | same |
| Prior weight selected on forward windows (historical model) | 24 (from 2, 8, 24) | `data/mirror/eval.json` |
| Pluribus check: mean defender equity by response, 701 heads-up river spots | fold 0.51 < call 0.74 < raise 0.84 | `data/mirror/eval.json` → `pluribus` |
| RL benchmark at 64 probes: bot tuned to the twin minus average-trained bot, chips per spot (pot 100) | caller +4.55 [+2.00, +7.15] · folder +8.00 [+6.10, +10.05] · population −2.75 [−4.40, −1.10] · strength +0.70 [−0.70, +2.05] | `data/mirror/benchmark.json` |
| RL training sufficiency, three seeds | every oracle check passes; 7 of 9 first-decision cells agree, the other 2 are near ties | `data/mirror/training_sufficiency.json` |
| Phase 6 main cohort (all tables, 40-60% bets) | 157,926 decisions by 60,918 players; test 31,586 by 15,138 players | `data/mirror/scan_counts_wide.json` |
| Phase 6 log loss, history up to cutoff: pooled / population / situation / personal / situation + personal | 0.8274 / 0.8182 / 0.7929 / 0.8161 / 0.7912 | `data/mirror/eval_wide.json` |
| Situation model vs average player | +0.0253 nats, 95% CI [+0.0229, +0.0280]; the same at every table size | same |
| Player history vs average player (all tables) | +0.0021, CI [+0.0013, +0.0029]; two-player tables +0.0022 — replicates the original +0.0028 | same |
| Player history on top of the situation | +0.0017, CI [+0.0010, +0.0025]; learning as you go +0.0031, CI [+0.0021, +0.0040] | same |
| Bluff test, chips per decision (pot 100): never / always = average-player forecast / situation + player / perfect foresight | 0 / 36.14 / 36.26 / 57.25 | `data/mirror/bluff_test.json` |
| Bluff test, situation + player vs average-player forecast | +0.12 chips per decision, CI [+0.02, +0.22], 1.4% of decisions change — almost everyone folds more than a third of the time, so bluffing half pot is nearly always right | same |

### Data left out (sampled 2026-11-07 evening)

Two samples of 400,000 rows each, the first rows of shards 35 and 36, then of shards 5, 14, 23 and 31, replayed with a
throwaway extractor generalised to any table where exactly two players reach the river (6.1 makes it permanent):

| Cohort | Sample 1 | Sample 2 | vs today | Full-corpus projection |
|---|---|---|---|---|
| Today: two-player tables, exact half pot, all responses legal | 310 | 543 | 1× | 12,407 (measured) |
| All tables, exact half pot, all responses legal | 1,067 | 837 | about 2× | about 28,000 |
| All tables, 40–60% of the pot | 4,506 | 3,833 | about 10× | about 120,000 |
| All tables, any bet size | 16,420 | 12,721 | about 34× | about 420,000 |

Shards differ a lot, so treat the projections as ±50%. The band rows do not yet require all three responses legal.
The style scan already reads every table size; only the river cohort was narrow, by design of the original plan.

## Gate reports

### GATE H3 (Phase 1) — passed
- Evidence: `python -m unittest tests.test_mirror_data tests.test_river_env` (reference counts exact; engine parity); `python scripts/smoke_mirror.py`.

### GATE H9 (Phase 2) — passed
- Evidence: full-corpus scan done; oracle fixtures pass (`tests/test_mirror_rl.py`); smoke test passes.
- First population training (α = n^-0.6) failed the oracle in 5/9 cells; fixed with α = 1/n (decision 2).

### GATE H12 — passed
- Learned Q within 0.10 of Q* in all supported cells (`data/mirror/policy_population.json` → `meta.oracle_check`).

### GATE H15 — decided
- Type-prior gate failed; recorded in `data/mirror/styles.json` → `gate`. Live model stays K = 1.
- Re-run after the push (`python -m pokersim mirror-cluster --gate-only`): the first run used prior weight 8 instead of the
  selected 24, and its rule (beat the population model) passes on personal history alone. With 24 that old rule passes by a
  hair (+0.0033, CI [+0.00001, +0.0068]), but both checks that isolate the types fail (see decision 12).
- Benchmark: first full run (18:47) compared bots trained with *different* seeds; its control row showed seed noise
  of the same size as the effects, so it was retired to `data/cache/mirror/benchmark_unmatched_seed.json`.
  The fixed design trains the baseline with the candidates' seed; at 0 probes the two bots are now identical
  (smoke run: `zero_probe_identical: true`).
- Full benchmark run on the second PC (19:35): `zero_probe_identical: true`; the control changed 0% of decisions,
  0.0 chips [0.0, 0.0]. See 3.6.

## Handoff: RL runs on a second PC (done at 19:35, kept for reproduction)

Everything the RL needs is in git: the code plus `data/mirror/population.json`. No dataset download, no GPU.
All runs are seeded and deterministic, so the second PC reproduces the numbers this PC would get.

```bash
git fetch origin && git checkout poker-mirror   # first time; afterwards just: git pull
python -m pip install pokerkit==0.7.7          # Python 3.11+
python -m unittest tests.test_river_env tests.test_mirror_rl tests.test_mirror_model   # ~1.5 min
python -m pokersim mirror-train --profile population --episodes 20000 --seed 7 --out data/mirror/policy_population.json
python -m pokersim mirror-sufficiency          # three seeds -> data/mirror/training_sufficiency.json
python -m pokersim mirror-benchmark --workers 8   # 16 personalised bots + baseline + control -> data/mirror/benchmark.json
```

Timing on this PC (16 threads): population bot 53 s; each benchmark job about a minute; the first run on a fresh
machine also spends a few minutes building the cached scenario pools. Commit back
`data/mirror/policy_population.json`, `data/mirror/training_sufficiency.json` and `data/mirror/benchmark.json`.
Do not commit `data/cache/`.

What to check in the output: `benchmark.json` → `zero_probe_identical: true`; each row's `ci95_chips`; the
`control` row shows how much two trainings of the same bot differ. `training_sufficiency.json` →
`all_oracle_pass` and whether disagreements between seeds sit on near-tie cells.

## Handoff: GPU run (Phase 6, item 6.10)

Everything is in git, including the dataset (`data/mirror/neural/neural_dataset.npz`, 15 MB). No shards needed.

```bash
git pull                                              # branch poker-mirror
python -m pip install numpy torch                     # a CUDA build of torch for the 5090 (CUDA 12.8 or newer)
python scripts/train_neural_forecaster.py --quick     # 1-2 minutes, checks the setup (CPU is fine)
python scripts/train_neural_forecaster.py             # full sweep: 4 configurations x 3 seeds, GPU if present
git add data/mirror/neural_forecaster.json && git commit -m "Add neural forecaster results" && git push
```

It writes `data/mirror/neural_forecaster.json`: test log loss of the network next to the CPU forecasters on the
same 31,586 test decisions, lifts with 95% intervals over players, and the bluff test. The Evidence tab shows it
as panel F. To beat: the situation + player model, log loss 0.7912 (history up to the cutoff) on these decisions.

## Handoff: finishing Phase 6 (agent or human)

```bash
python -m pokersim mirror-eval --pluribus-strength                  # 6.7, about 2 minutes (shard 0)
python -m pokersim mirror-real --probes 0,4,8,16 --qlearning-sample 8   # 6.6, a few minutes
python -m pokersim mirror-prepare --styles-wide                     # 6.8 step 1, about 10 minutes
python -m pokersim serve                                            # 6.5 and 6.9: check the Evidence tab and step 4
```

Still to write: a wide mode for the type gate (6.8), unit tests for `mirror_wide` and `mirror_real`, the PLAN.md
scope note and README commands (6.11). Memory note: this laptop had under 2 GB free RAM, so the wide scan used
8 workers and streams spots to disk; `mirror-wide` needs about 1.5 GB.

## Next steps

Phase 6 in [CHECKLIST.md](CHECKLIST.md) is half done: first run the GPU handoff above, then finish 6.5-6.11 with the
commands in "Handoff: finishing Phase 6". Volunteers are hard to get, so volunteer sessions (item 1 below) are optional.

For the team:
1. Volunteer sessions: two calibration sessions first, then at least three held-out sessions with people who have not
   seen the model, about twenty drills each in step 3. Then Evidence → "Recompute from sessions".
2. Merge `poker-mirror` into `main` when the team agrees.
3. Phase 5: roles, timed 90-second rehearsal, backup recording, fresh-clone test on the demo laptop with Wi-Fi off.

Optional, for the agent:
1. Look into the benchmark row that failed its oracle check (folder, 16 probes), for example by training it longer.
2. Calibrate the strength prior's shape on half of the 701 Pluribus spots, keep the other half for the check.
3. Draw the forecast ladder as a learning-curve chart.

Done: RL runs on the second PC (population policy, training sufficiency, benchmark) and their Evidence panels,
volunteer-session tooling, stricter type gate, style map, range-check panel, policy-chart fix, replay deep link,
browser pass of every screen, 67 tests and both smoke tests green.

## Decisions and deviations from the plan

1. **Commits.** Made only when the user asks, under the user's own git identity, on branch `poker-mirror` rather than directly on `main`.
2. **Learning rate.** α = 1/n by default (plan: n^-0.6, which missed the oracle tolerance at 20k episodes). `--alpha-exponent` keeps the old behaviour.
3. **Oracle pool.** Q* is compared on the training deals; Q* on validation deals is stored alongside.
4. **Ongame and iPoker drop out of the eligible set.** iPoker stacks are recorded as `inf`; Ongame records carry negative blind entries that cause replay warnings. Counts are in `scan_counts.json`.
5. **Strength prior.** No volunteer sessions yet: the strength table uses a synthetic default prior reconciled with the population marginal per context, labelled as such. Candidate improvement: calibrate its shape on half of the 701 Pluribus spots and keep the other half for the check.
6. **Reliability for N_min.** Binomial reliability `var_true / (var_true + E[p(1−p)/n]) ≥ 0.7` instead of an odd/even split, because the style scan stores aggregates.
7. **Pluribus check** uses heads-up river spots inside six-handed hands.
8. **Benchmark baseline.** Trained with the candidates' seed (plan: "identical deal schedule"); the earlier seed-7 baseline confounded training noise with personalisation. `--baseline <file>` still loads a fixed policy.
9. **Per-player responsibilities** (5.4 MB, keyed by handle) live in `data/cache/mirror/style_responsibilities.json`, not in git; `styles.json` is 228 KB.
10. **Prior weight.** The historical model uses the selected 24; live practice keeps the plan's starting value 8 until practice sessions exist to tune it.
11. **Windows port reuse.** `serve` no longer sets `SO_REUSEADDR` on Windows: a stale server had silently kept port 8000 and answered 404 to every Mirror route.
12. **Type gate rule.** The plan's wording (type model beats the population model) passes on personal history alone, because the type model also learns from history. Types now ship only if (a) for style-known players with no river history the type prior beats the population model and (b) types+history beat history alone, each with a 95% interval over players above zero. The gate uses the frozen prior weight (24). Stricter, not looser: the old rule's result is still reported.
13. **Volunteer data.** Raw session logs stay in `data/cache/mirror/sessions/` (ignored). Only aggregates are committed: `strength_calibration.json`, `sessions_eval.json`, `range_checks.json`.

## Where things are

- Shards: `data/cache/takara/` (ignored). Re-run `python scripts/fetch_takara.py` any time.
- Small artifacts (commit): `data/mirror/`. Large ones (ignored): `data/cache/mirror/`.
- Policies trained by judges in the UI: `data/cache/mirror/policies/<profile hash>_<seed>_<episodes>.json`.

## How to resume

1. `python -m unittest discover tests` must pass.
2. `python -m pokersim serve` → Mirror tab → steps 1–5; Evidence tab shows which panels exist.
3. Read the status board, the last gate report and "Next steps", then continue.
4. When you finish a step: tick it in CHECKLIST.md and note the evidence here.

## Commit guidance

Work lives on branch `poker-mirror`. To merge it when the team is ready:
`git checkout main && git merge poker-mirror && git push`.
`git add -A` is safe: `data/cache/` is ignored and `data/mirror/` holds only small JSON/PHH files (317 KB).
