# Independent review of the training plan (train/, 2026-11-08 ~02:00)

Scope: SUMMARY.md (the requirement) against README and 01–07, checked against the code on `poker-mirror` (8868169) and the
v2 run files. Decided by the user and taken as given here: the Base next-move model learns from **every** decision in the
corpus.

Checked locally, using quick reads only and no training:
- `encode()` takes **3.0 ms** on average (p90 4.3 ms; 401 decisions, equity-bot table).
- A non-acting player's observation has `legal_actions=None`, so its action mask is `[0,1,0,0,0,0,0]`.
- The Pluribus logs have **13 named human pros** with every hole card dealt:
  - pros: 76,187 decisions (24,786 postflop);
  - Pluribus: 15,169 decisions (4,870 postflop).
- The online shard sampled (shard 1) mixes table sizes: 2 players 19%, 7–9 players 28%.
- The top-class probability of the hand model has median **0.020** (p10 0.013, p90 0.043).
- The repo `Aikk01/HackTheBurgh` is **PUBLIC**.
- `data/players/` and `logs/` are **not** git-ignored (checked with `git check-ignore`).

---

## Critical

**C1. RL "opponent next move" (03.5) returns check/call 100% as written, and it is not a prediction.**
- *Problem:* `hand.observation(j)` for a non-actor sets `legal_actions=None` (engine.py:887). `action_targets` then masks
  everything except call (rl_features.py:218–224). Every opponent box would read "check/call 100%".
- *Problem:* at the human's decision point, the opponents' next decision depends on the human's not-yet-taken action. "Next
  move" is undefined for players who have already acted on this street.
- *Problem:* π_RL in j's seat says what *the RL agent* would do with j's likely hands. It does not say what the *bot* will do.
- *Fix:*
  - Build a hypothetical decision for j "as if the action reached j now": to_call_j = max bet − bet_j, plus min/max
    raise-to by the engine's rules.
  - Overwrite j's cards in `obs["hero"]["cards"]`. That field also feeds the dead-card set inside `encode` (via
    `predict_ranges`).
  - Use a deterministic rng keyed on (hand_id, len(history), j), as `recommend` already does (rl_policy.py:86).
  - Average softmax probabilities, not argmaxes, over n=64 sampled hands. At the measured 3 ms per encode that is
    ≈0.2 s per opponent, so the plan's "0.5–1 s for 20" is about 10× too pessimistic and no background thread is needed.
  - Show the box only for opponents still to act on this street.
  - Unit test: the output must not change when j's *real* hole cards are swapped.
  - Either label the row "RL in their seat", or measure its log-loss against the bots' real actions in a simulated match
    before calling it a prediction.

**C2. The next-move extraction inherits the showdown filter.**
- *Problem:* 02 §2.2 says "extend the same loop", but that loop keeps:
  - only hands where some hole cards are known (train_hand_range.py:61);
  - at most 30k such hands per shard (:66);
  - only decisions with a known *other* player, sub-sampled to 4 per hand (:86–88).
- For online hands, "known cards" means a showdown happened. The action rows would therefore come only from showdown hands:
  too few folds and too much calling. That contradicts the decided "every decision".
- *Fix:* write a separate sampler.
  - Take N random NT hands per shard regardless of showdown. 25k per shard ≈ 0.9M hands, about 3–5 min at the ≈6k hands/s
    the hand-model extraction achieved.
  - Keep *all* decisions of each sampled hand.
  - Keep only hands with 5–6 players.
  - Attach cards only for the Pluribus-log seats (see H5).

**C3. Every new run overwrites the only working policy.**
- *Problem:* non-smoke runs export to `data/rl/policy.npz` (rl_train.py:319). A fresh run exports at its first eval
  because `best_score=-1e9` (:523–527).
  - v3-s1 on the 5090 will replace the unpushed v2 update-50 export after about 75 min.
  - Eddie and 2080 copies do the same in their trees, and an rsync back clobbers again.
  - Promotion is by in-distribution mean; the planned held-out selection does not exist yet.
- *Fix, now:*
  - `cp data/rl/policy.{npz,json} data/rl/coaches/v2-upd50/`, then commit and push these 2 MB so the Mac integrates the real model
    tonight.
  - Add `--export-dir` (default: the run dir) and promote by hand.
  - Give every remote run a unique `--run-dir runs/<machine>-<name>`.

**C4. The 01:45–03:15 slot on K's path is about 4–5 h of work.**
- *Problem:* in 1.5 h K must write:
  - `train_next_move.py`: the new sampler, label mapping, player split, A/B halves, 3 seeds, baselines, calibration;
  - the `human:` agent, two held-out tables, `--init-from`, `--no-range-inputs`, held-out export selection, unit tests;
  - an Eddie job file that does not exist (there is no `eddie/` in the repo; the template is in another project), plus a
    smoke job.
- So v3 realistically starts at 05:00 or later, with ≤2 h left ≈ 60–80 updates at the measured ~100 s per update.
  - With `--eval-every 50`, the 06:00 go/no-go sees one evaluation per run.
  - That evaluation is compared against a v2-resume that will have about 200 updates.
  - The stall rule ("no new best in 4 evals" = 200 updates) can never fire before 07:00.
- *Problem:* the Mac list (log 1 h, you-model 1–2 h, score 0.5 h, screen 2–3 h, bot sims, `opponent_next_move`, 50–100
  rehearsal hands) adds up to 6–8.5 h, against about 5 h available.
- *Fix:* follow the minimum viable path at the end.

## High

**H1. The "held-out" tables are not held out.**
- *`scripted_mix`:*
  - `ScriptedAgent` is a test helper that replays a fixed action list (agents.py:247–263). It has no personas.
  - Equity bots at (tightness, aggression) corners are inside the training distribution: 62% of training seats draw both
    uniformly on [0,1] (rl_train.py:60).
  - `iterations=200` versus 40 only changes postflop Monte-Carlo noise. The preflop path ignores it (agents.py:174).
- *`humans_B`:* the same architecture trained on the other half of the same population, with tilt and temperature from the
  same family. Its card-conditioned part comes from the same Pluribus rows (H5). It is near-identical to `human:A`.
- Independent RL overfits to the opponents it trained with (joint-policy correlation, Lanctot et al. 2017,
  https://arxiv.org/abs/1711.00832). A held-out set needs a *different family*.
- *Fix (≈30–40 min):* write 3–4 rule personas not derived from `EquityAgent`:
  - a limp-caller who gives up postflop without a pair;
  - a preflop raiser who c-bets every flop and quits on the turn;
  - a check-raiser with sets and draws;
  - a min-raise spammer;
  - a top-pair overbet jammer.
- Use one set of personas for selection and a different set (or different seeds) for the reported number.

**H2. The selection score and confidence intervals are statistically weak.**
- *Problem:* the score is a raw mean of bb/100 over lineups (rl_train.py:515). loose_passive alone contributes 3,687 of the
  5×1,181.5 total (62%).
- *Problem:* the CIs are unpaired (rl_train.py:286–292), although every hero plays the same deals: the deck is shuffled
  from the seeded table rng (cards.py:87).
  - Per-hand pairing is lost because chunks are concatenated in arrival order (rl_train.py:502).
  - The same eval seeds are reused for every checkpoint (rl_train.py:491), so picking the best one on them carries a
    winner's curse.
  - The rule "the difference counts if the 95% CIs don't overlap" (06:48) is not the right test for paired data.
- *Measured per-hand SD from eval.json:*
  - 15 bb (tight_passive) to 112 bb (loose_passive), i.e. 150–1,120 bb/100.
  - Human 6-max is roughly 70–110 bb/100 (https://www.deucescracked.com/blog/poker-variance-explained-sample-size-win-rate).
  - 3,000 hands give ±54 bb/100 even on the calmest table. Detecting a ±25 bb/100 edge needs ≈14k hands.
- *Reference points:* Pluribus needed AIVAT to show 48 ± 25 mbb/game over 10k hands
  (https://www.greaterwrong.com/posts/6qtq6KDvj86DXqfp6/let-s-read-superhuman-ai-for-multiplayer-poker). AIVAT cuts SD by
  about 85% (Burch et al., https://arxiv.org/abs/1612.06915).
- *Fix (eval is cheap: ~40–50 s for 60k hands, from log.csv updates 50→51):*
  - Carry the chunk id in each eval message.
  - Report the paired per-hand difference (RL − equity bot) with its own CI.
  - Use 20k hands per lineup.
  - Selection score = mean over lineups of paired difference / SE, or the minimum over lineups.
  - Use separate seeds for selection and for the final report.
  - Optional: all-in EV adjustment (replace the realised all-in runout with equity × pot), an unbiased and cheap partial
    variance reduction.

**H3. The "+681 bb/100 from hand inputs" claim is overstated (README:24, 03:17–18).**
- *Problem:* zeroing inputs the network was trained with is an out-of-distribution perturbation. It measures *reliance*,
  not value.
- *Problem:* 2,609 of the 3,406 summed gain comes from loose_passive. tight_passive gives +16.8 with CIs of ±42–54, which
  is noise.
- *The proper ablation (v3-norange) is confounded:*
  - it starts from scratch with seed 3, against a warm-started v3-s1;
  - skipping the range simulation makes encode faster, so it gets more updates in the same wall-clock;
  - `use_ranges=False` must reach the hero rollout, the snapshot opponents (rl_train.py:132) and the eval heroes
    (:222–226) through the worker message. Missing any one silently trains with ranges.
- *Fix:* tonight, say "the policy relies on the hand inputs: zeroing them costs X on 4 of 5 tables".

**H4. Next-move inputs (02 §2.4): the style block is untrainable and the price to call is missing.**
- *Scope:* this applies to the next-move model. RL's style inputs do vary, because the tracker updates every hand
  (rl_train.py:146).
- *Problem:* corpus rows get population-mean style rates (02:67–68), a constant. The network cannot learn any dependence on
  them, and at the table the live estimates move inputs it never saw vary.
  - *Fix:* compute per-player style counts chronologically during extraction: a `StyleTracker` per `player_key` over that
    player's *earlier* hands, with the same shrinkage. Or drop the input tonight (recommended, for time).
- *Problem:* `ctx_public` (hand_range.py:104–156) has no to_call/pot, to_call/stack, stack/pot, number of players behind, or
  facing-a-raise feature. Fold versus call is barely identifiable without them.
  - *Fix:* add those 5–6 actor-specific numbers. At inference for opponents, compute them from the hypothetical state of C1.
- *Label mapping:* use nearest size in log space among the classes left after the mask, map an all-in call to "call", and
  check to "call".

**H5. The with-cards model has almost no data; the A/B halves collapse; the you-prior is Pluribus.**
- The with-cards rows are only Pluribus's own seat (02:73–76): 15,169 decisions, 4,870 postflop, about 12k after the test
  split.
- Splitting by `hash(player_key)%2` (02:32) sends *all* Pluribus rows to one half.
  - The other half's model never sees `has_cards=1`.
  - So `human:B` "acting with its own cards" is card-blind extrapolation, and `human:A` is noisy Pluribus, not a human.
- The you-prior is "next_move with your cards" (04:29), i.e. Pluribus. That contradicts SUMMARY's "starts from
  typical-player behaviour".
- The 76,187 selection-free *human* decisions with cards (13 pros in the same logs) go unused.
- *Fix:*
  - Train the with-cards head on all 6 seats of the Pluribus logs, with an `is_pluribus` flag:
    - flag=1 gives Base "what you should do";
    - flag=0 gives the you-prior ("strong human").
  - Split the test by hand number (and by pro, if feasible).
  - Never use online showdown rows *with* cards: every one of them reached showdown, so it has zero folds.
  - Cut `human:A/B`, models A/B and `humans_B` tonight.

**H6. The you-model prior σ=0.5 (04:33–35) barely moves.**
- *Problem:* about 50 decisions over 8 buckets is ≈6 per bucket.
  - The information per θ is ≈ n·p(1−p) ≈ 1.2, against a prior precision of 1/σ² = 4.
  - So the MAP moves only ≈23% of the way towards the data.
  - The model stays near the prior and can visibly lose to the "usual move" baseline on screen.
- *Fix:* use Dirichlet–multinomial shrinkage per bucket, applied to the card-aware prior:
  - p_you(a) ∝ p_prior(a | cards) · [(n_g + α q_g)/(N + α)] / q_g, where q_g is the prior's mean share of group g in that
    bucket.
  - This tends to the empirical frequency as N grows, and the prior dominates early (cf. prior-plus-deviation opponent
    models: Ganzfried & Sandholm DBBR, https://www.cs.huji.ac.il/~jeff/aamas11/papers/B4_R56.pdf).
  - Tune α (2–10) on the rehearsal log-loss.
  - Score on the 3 groups. At n=48 the SE of accuracy is ≈±7 points, so "31 vs 26" is within noise; do not claim it.

**H7. Leaking personal and network details into a public repo.**
- `Aikk01/HackTheBurgh` is PUBLIC.
  - 01_resources.md:8 and 07_action_plan.md:52–54 give the owner's IP, ssh user and home-folder names on a university network.
  - 01:9 gives the Eddie UUN.
- 04:13 says `data/players/` is "ignored by git". It is not. Neither is `logs/`, which 07:41–46 says not to commit.
- *Fix:*
  - Redact those lines before any `git add train/`.
  - Add `data/players/` and `logs/` to `.gitignore` now.
  - Never run `git add -A`.

## Medium

**M1. "Likely hand" is uninformative and the grid is visually biased.**
- *Problem:* a median top-class probability of 2% means the screen shows "KQo 2%". With 4% top-1 accuracy the shown hand
  will also flicker between decisions.
- *Problem:* `HR.grid` returns class mass (hand_range.py:280). Offsuit cells (12 combos) look 3× hotter than suited ones
  (4 combos).
- *Fix:*
  - Colour by per-combo density (probs / live_combos).
  - Keep the top-2 list in class mass.
  - Add a coarse summary next to the top hand, e.g. "pairs 18% · broadway 31%".

**M2. The hand model's headline number is on a biased population.**
- *Problem:* online showdown rows are 83% of test rows. The model learns "hands of players who later show down", which is
  too strong preflop. The +0.63 nats is measured on that same biased population.
  - The unbiased figure is Pluribus: +0.31 overall, +0.21 preflop.
- *Problem:* validation is a random 5% of *rows* (train_hand_range.py:175). Rows of the same hand land on both sides, so
  early stopping and config choice are leaky.
- *Problem:* no table-size filter.
- *Fix:*
  - Do not retrain tonight (RL depends on it); report the Pluribus rows as the honest numbers.
  - Add calibration on the Pluribus test set.
  - Any future display-only model should take a "showdown-conditioned" input flag, set to 0 at inference (learning from
    incomplete information: Ponsen et al. 2008, https://mlanthology.org/aaai/2008/ponsen2008aaai-bayes).

**M3. The bot-hand fine-tune (03.4) is weak, and it is extra scope.**
- *Problem:* `context_features` has no style input, so it learns the *average* of a mixed bot population. A calling station
  and a nit with identical actions get the same grid.
- *Problem:* 200k rows ≈ 7k hands of highly correlated rows (≈30 rows per hand). 2–3 epochs on a 3×512 network will
  forget or overfit.
- *Problem:* it adds a Mac → 5090 copy loop, and it departs from SUMMARY's own recommendation ("share the base grid").
- *Fix:* cut it. The RL side of the toggle shows **"No model loaded"** (spec-compliant), or the base grid labelled "RL
  uses the Base grid".

**M4. The warm-start versus from-scratch comparison is uninterpretable.**
- *Problem:* v3-s1 warm (56 + ~110 updates, GPU) against v3-s2 from scratch (~90 CPU updates) mixes initialisation,
  compute, machine and seed, with one seed per arm.
- *Problem:* resume or init restarts the LR and entropy schedule (rl_train.py:385–387).
- *Fix:* if kept, compare at equal steps with ≥2 seeds per arm. Better, cut it tonight.

**M5. Persistent style counts are keyed by name (04:21).**
- *Problem:* bot default names ("Equity bot 3") are reused across tables with different specs, so stats go stale.
- *Fix:* persist only the human's counts. For the bots, warm up the tracker with the existing `GameSession.observe()`
  (server.py:375) before the demo.

**M6. RL's "expected result of the hand" (SUMMARY's model table) has no place on the screen spec (04/05).**
- *Problem:* the critic was trained on 50–150bb random stacks, while the live table carries stacks over between hands.
- *Fix:* show it with a caveat or drop it from SUMMARY. Reset stacks to 100bb at the demo table.

**M7. Eval measures a different policy from the one deployed.**
- *Problem:* eval heroes play greedy (rl_train.py:222), while `RLAgent` samples (agents.py, `greedy` default False) and the
  advice shows the stochastic probabilities.
- *Fix:* report both, or set `greedy=true` for the demo RL bot.

**M8. The 2080 PC command (07:52–54) has several traps.**
- *Problem:* `a && b && setsid nohup … &` backgrounds the whole AND-list, so ssh stays attached until training ends.
- *Problem:* it installs multi-GB CUDA torch into a teammate's home.
- *Problem:* `--resume` writes to a run dir with the same name as on the 5090, so an rsync back clobbers it.
- *Fix (if used at all):*
  - install in a separate step;
  - start with `nohup setsid … </dev/null >rl.log 2>&1 &` inside tmux;
  - pass `--run-dir data/rl/runs/v2resume-2080`;
  - agree a stop time and a cleanup (`rm -rf ~/hackburgh`) with the owner.
- Better: run v2-resume on the idle 5090 and leave the owner's machine alone.

## Low

- **L1.** README:16 links REVIEW.md, which does not exist. README:24 calls the in-distribution lineups "test tables";
  03:15 admits they measure fit.
- **L2.** 03.5 contradicts itself: "hero's cards never used" versus "avoid the cards the viewer can see". The hero's cards
  must be **dead cards** (legitimate card removal) and never features.
- **L3.** Go/no-go 1 decides on the *test* players (07:23), which turns test into validation (versus 02:98). Gate on the
  validation players; read test once, for the report.
- **L4.** The contract cannot reveal "would" after the action through `/api/game/advice`: it returns `available:false`
  when it is not the human's turn (server.py:353–366).
  - Put `you.last` (prediction, actual, ✓/✗, score) in the `/api/game/act` response and in the state.
  - Compute `you_pred` server-side inside `human_act` from the pre-action observation, even if the client never fetched
    advice.
- **L5.** If Eddie is used: there is no job file in the repo, compute nodes may have no outbound internet (wandb), and
  `h_vmem` is enforced on virtual memory. Moot if Eddie is cut tonight.

---

## Answers to the six questions

1. **Does the plan cover SUMMARY's screen?**
   - Covered as planned:
     - table;
     - Base next move (once C2 and H4 are fixed);
     - Base likely hand;
     - grid with colours, click and top-2;
     - "what you should do" for Base (thin, H5) and RL (exists);
     - "what you would do" with score;
     - "No model loaded";
     - per-decision log.
   - Broken or mismatched:
     - RL next move (C1);
     - RL likely hand and the RL side of the toggle (rest on the cuttable M3);
     - the you-prior is Pluribus, not "typical player" (H5);
     - the base next-move model does not learn from every decision (C2);
     - the value "expected result" is not displayed (M6);
     - persistent styles collide on names (M5).
2. **Is the technique sound?**
   - PPO as an approximate best response to a fixed bot pool plus 25% snapshots is a reasonable choice for "beat these
     bots". It is not a CFR-style equilibrium (Pluribus: https://ui.adsabs.harvard.edu/abs/2019Sci...365..885B), and its
     advice holds only against bots like these.
   - What to change:
     - selection score and statistics (H2);
     - held-out families (H1);
     - next-move sampler and features (C2, H4);
     - with-cards data source (H5);
     - you-model shrinkage (H6);
     - opponent next-move definition (C1);
     - cut the bot-hand fine-tune (M3);
     - reward: keep the terminal reward; all-in-EV-adjusted reward is the cheap variance fix if anything.
3. **Overfitting, evaluation and leakage.**
   - None of the planned tables is truly held out (H1).
   - 3,000 hands is inadequate, and the CIs are unpaired (H2).
   - Selection and reporting reuse the same seeds (H2).
   - Leakage risks:
     - j's real cards in the dead-card set when building j's observation (C1, add the swap test);
     - A/B sharing the Pluribus rows (H5);
     - hand model validation split by row (M2);
     - go/no-go on test (L3).
   - Hero cards in the human's own advice are legitimate.
4. **Bugs and wrong claims, by file:line:**
   - engine.py:887 with rl_features.py:218 (C1);
   - train_hand_range.py:61, 66, 86 (C2);
   - rl_train.py:319, 523 (C3);
   - rl_train.py:515, 286, 491, 502 (H2);
   - agents.py:247, 174 and rl_train.py:60 (H1);
   - README:24 (H3);
   - 04:13 / .gitignore (H7);
   - README:16 (L1).
5. **Schedule:**
   - What will slip: v3 (starts ≥05:00), the Eddie runs, the bot-hand model, A/B models, the screen, rehearsal.
   - Cut: v3-s1, v3-s2, v3-norange, Eddie, the 2080 PC, `human:` bots, A/B, the bot-hand model, 3 seeds, persistent bot
     styles.
   - The minimum viable path is below.
6. **Danger:**
   - the owner's IP and user in a public repo (H7).
   - Long full-load jobs and a torch install on a teammate's personal PC (M8).
   - Clobbering the only working policy (C3).
   - Accidental commits of `data/players/` (personal decision logs) and `logs/` (H7).

## Minimum viable path to 08:00

| Time | K (5090) | Mac team |
|---|---|---|
| 01:50–02:05 | Back up and push `coaches/v2-upd50` policy (C3). Fix `.gitignore` and redact train/ (H7). Resume v2 on the **5090**: `--resume …/latest.pt --run-dir data/rl/runs/v2resume --hours 4.5 --eval-every 15`. It only exports above 1181.5. | Pull. Decision log (4.1) with `you_pred` computed in `human_act` (L4). |
| 02:05–04:00 | `train_next_move.py`, one seed, no A/B (details below). | You-model with the Dirichlet scheme (H6) on the population prior. Score on the 3 groups. |
| 04:00–05:00 | `opponent_next_move` per C1, the same hypothetical state for Base, leak test, push. | Screen: opponent boxes, grid (per-combo colour, M1), your panel. |
| 05:00–05:45 | `rl-eval` script (H2): paired, 20k hands, old lineups plus 3 new rule personas, fresh seeds. Choose between coaches/v2-upd50 and the v2resume best on it. | Screen continued. RL grid = "No model loaded" (M3). |
| 06:00 | **Freeze models.** Commit `policy.npz`, `next_move.npz` and the eval JSONs. Then integration help. | Switch the you-prior to the pros head. Warm up bot styles with `observe()`. |
| 06:30–07:30 | Bug-fix only. | Rehearsal hands under the **demo player's** `player_id`. Wi-Fi-off run. |

`train_next_move.py` (02:05–04:00):
- Sampler per C2.
- Features per H4: `ctx_public` plus the facing features; no style input.
- Heads:
  - no-cards (all decisions);
  - with-cards (Pluribus-log seats + `is_pluribus` flag, H5).
- Frequency baselines and validation by player. Read test once. Push `next_move.npz` and `.json`.

Report honestly:
- RL is measured against bots only.
- Give the paired difference against the equity bot with its CI on in-distribution and new-persona tables.
- Describe the hand-input result as reliance (H3).
- The hand model gains +0.31 nats on unbiased Pluribus rows.

## What the plan gets right
- It says plainly that the old eval lineups measure fit, not generalisation (03:15, 06:39–44).
- It refuses to retrain the hand model, which keeps RL inputs stable.
- It keeps v2-resume as a fallback.
- It splits the next-move data by player, adds a later-in-time test, uses frequency baselines, and checks calibration and
  action shares.
- `with_cards=False` for opponents, scoring before the action, reveal after it.
- A low-dimensional personal model with a prior: right in concept, only σ is wrong.
- A missing model gives `null` and "No model loaded".
- NumPy-only serving, small committed model files, `.pt` ignored, `OMP_NUM_THREADS=1`.
- Spike and KL stop rules with `keep/` rollbacks.
