# Poker Mirror — implementation plan

Build an offline, playable Poker Mirror demo inside the existing [HackTheBurgh repository](https://github.com/Aikk01/HackTheBurgh). Work from the repository root. Launch the finished demo with `python3 -m pokersim serve` (`python` on Windows) and expose it from the existing browser UI.

The demo answers one question: **after accounting for the cards on the table and the public betting history, how much does knowing this opponent help a bettor decide whether to bluff?** Scope the learned behaviour to a heads-up no-limit Hold’em river where the defender checks and faces a half-pot bet. Report the answer on three rungs: knowing nothing about the opponent (population), knowing their playing type, and knowing the individual. Deliver a historical replay, a three-level Bayesian opponent model, a playable behavioural twin, a range readout that shows what the twin’s response says about its hand, a closed-form expected-value advisor, and a counter-bot trained with tabular Q-learning against that twin. Compare it with an equally trained bot that has only a population model.

## 0. Pitch, theme mapping and scope

**Pitch.** Poker is a game of hidden information. Mirror uses the full Takara corpus to show three hidden things: which observations a filter silently drops, how this opponent tends to act, and what their action says about their hand. It then turns those three things into a bluffing decision with a measured payoff.

| Theme | What judges see | Where the proof is |
|---|---|---|
| Best use of data: prediction | Forecasts committed before each click, with uncertainty bands, and a population, type and person learning curve | Forward-in-time Brier score and log loss against the population baseline; the type-prior gate in §6 |
| Best use of data: decision making | An expected-value panel for check versus bet with the one-third break-even, and counter-bots compared on a frozen deal | Paired chips per opportunity with bootstrap intervals; Q-learning checked against the EV advisor in §4 |
| Make the invisible visible | The filter exhibit that shows which folds vanish, a style map of player types, and a range X-ray after the twin responds | Counts with denominators; held-out calibration of the readout; the Pluribus check |

**Scope boundaries.**

- The learned behaviour is the heads-up river slice only. Five-opponent or six-handed play, equilibrium or CFR solving, search and deep RL are out of scope. Profiles, types and readouts are computed per opponent, so a multiway table could reuse them later; the demo does not claim that.
- Every claim is conditional: 2009 online cash games, one bet size, one street. Label synthetic opponents as synthetic. Make human claims only from held-out volunteer sessions.
- Cluster names and range readouts are model estimates, not observations. Label them that way on screen.
- Present Mirror as a training and analysis tool. Poker sites ban real-time assistance at real-money tables, so do not describe it as live help.

**Tiers and cut order.** Gate hours (H12, H15, H20) are defined in §7.

| Tier | Contents | Rule |
|---|---|---|
| Core | Historical evidence and the filter exhibit; population and personal models with pre-click forecasts; playable twin; bucket-level range readout; EV advisor; PokerKit environment, Q-learning bots and paired evaluation; Evidence page | Never cut. One fallback: if Q-learning misses its oracle at the H12 gate, display the EV advisor as the bettor and report Q-learning as exploratory. |
| Type layer | Per-player style table, types, style map, type prior with type posterior bars, 13 by 13 range grid | The type prior ships only if it passes the §6 gate at H15. The map and grid may ship as labelled descriptions either way. |
| Extras | Pluribus external check, type-aware bettor arm | Only after the Core and the type layer are verified. |

Cut order when time runs short: extras first, then the type layer.

## 1. Start with one end-to-end decision

Run the repository’s baseline checks from the repository root.

```sh
python3 -m unittest discover tests
python3 scripts/validate_dataset.py data/samples
python3 -m pokersim serve
```

Use Python for extraction, modelling and evaluation. Use PokerKit 0.7.7 for Mirror practice and RL transitions, and the existing HTML/CSS/JavaScript table for the interface. Pin the tested engine version and install it with the app dependencies before the offline demo. Use PyArrow, NumPy and scikit-learn only in offline data preparation; the demo runtime needs only the standard library and PokerKit. Bundle the small derived data and evaluation results required for the demo under `data/mirror/` so a fresh launch works without network access, model keys or a GPU.

**First integration milestone.** Load a seeded river with a pot of 100. The defender checks, the bettor offers 50, and the judge chooses a legal response. Record a prediction before the click. After the response, show the previous prediction, actual action and updated number of observations. A refresh must recover the current decision within the session. Build the historical and simulation flows around this same decision format.

**Decision record.** Fix one record shape before any workstream starts: session id, decision id, public context, offer size and pot, legal mask, board, the viewer’s own cards when the viewer is the bettor, the committed prediction (three probabilities, fold interval, model arm and version), the actual action, and optional fields for the type posterior, range readout and EV table. Historical replays, practice drills and simulations all emit and consume this record.

**Session registry.** The existing server holds a single global game. Give Mirror its own registry keyed by a token the browser stores, with one lock per session and a reset endpoint. Refresh recovery, resetting between judges and the no-leakage test all depend on it.

| Reuse | Implementation point |
|---|---|
| `pokersim/phh.py`, `replay.py` | Convert Takara rows using `record_from_dict` after the schema gate in §2. Extract pre-action observations from `replay(record, snapshots=False).decisions`. Use the same replay events for the style counts. |
| `pokersim/engine.py`, `agents.py` | Keep the existing PHH replay and agent interface. Adapt the Mirror agents to PokerKit legal actions and compare payouts with the existing engine on fixtures. |
| `pokersim/server.py` | Expose the Mirror drill session through the existing server. Feed the existing table renderer a public view from the same PokerKit adapter used in training. Capture the decision before applying the action. |
| `pokersim/simulate.py`, `evaluate.py` | Reuse chip accounting and action normalisation. Build the public-action benchmark from replay decisions directly. The existing shadow evaluator can skip players with undisclosed cards. |
| `pokersim/evaluator.py`, `equity.py` | Use `evaluate` for exact river equity and strength buckets. Use `estimate_equity` only to sanity-check them. |
| `web/index.html`, `web/app.js`, `web/style.css` | Add a Poker Mirror entry and reuse the existing table renderer, replay controls and API helper. |

## 2. Prepare the historical evidence

Scan **all 37 Parquet shards** at Takara revision `6acb5afb6f43082e6a468fde578890d9188be393` (21,616,175 total rows, about 1.285 GB compressed). Enumerate the pinned file inventory and stream row groups one shard at a time. Prefilter on variant, player count, usable stacks and presence of a river response before running expensive replay. Record per-shard input, eligible, duplicate and rejection counts. Cache extracts by source checksum. Files 35 and 36 below are verified demo fixtures. The model-training input spans the complete inventory. Commit the preparation command, split manifest metadata, aggregate counts, two demo hands, the small type-layer artifacts (centroids, type tables and a sampled set of map points with no handles) and attribution. [Dataset revision](https://huggingface.co/datasets/takara-ai/poker_hands/tree/6acb5afb6f43082e6a468fde578890d9188be393) · [Original hand histories](https://zenodo.org/records/13997158) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| File under `data/` in the dataset | SHA-256 |
|---|---|
| `train-00036-of-00037.parquet` | `4f880233eb1a03d6a255986475d2be47739bf753b66dbaf0eb9516adca05401b` |
| `train-00035-of-00037.parquet` | `0b5a566bc75c10a132480fe14d1a0804ba9c9224f0655339a8752b9aed7b9516` |

**Schema gate (first hour, before the full scan).** These facts were checked on 100-row samples taken at seven offsets across the pinned revision, not on every row. The first rows of shard 0 are five hand-annotated historical files followed by Pluribus rows. Later shards hold HandHQ rows by venue. HandHQ rows carry `day`, `month`, `year`, `seat_count` and `hand_number_in_file` as floats, so cast them to integers. `time` is an `HH:MM:SS` string, `hand` is a string, and `finishing_stacks` is empty for some venues. The stake appears only in the `source_file` path, for example `…_200NLH_…`. Pluribus and hand-annotated rows have a null venue, and Pluribus rows have a `source_file` such as `pluribus/98/57.phh`. In the five hand-annotated rows the `actions` field is a single `[` element, which is a conversion fault. Reject every row whose `actions` has one element or fewer, and take the historical demo hands from `data/samples`. Before launching the scan, confirm on a 1,000-row sample that the columns map onto `record_from_dict`.

**Two scans.** The first scan produces labelled river responses, the exhibits and the manifests. The second scan runs once the manifests give each venue a training cutoff and produces the per-player style table for the type layer (step 6). Parallelise both by shard.

1. Select variant `NT`, exactly two players dealt in, two usable starting stacks, and three community-card deals. Require a venue and recorded hand ID. Deduplicate on that pair.
2. Replay hands whose river starts with a check, an opening bet and an observed response. Require a positive pot before the bet, a legal response and enough defender stack to cover the offer. At the defender’s turn, compute `pot_before_bet = total_pot − opening_bet_amount`.
3. Keep `bet / pot_before_bet` between 0.40 and 0.60 inclusive. Flag exact half pot with an absolute tolerance of 1e-8. Classify the turn as checked through, river bettor last bet or raised, or river defender last bet or raised.
4. Count fold, call and raise from the recorded action. Keep a valid response if the only replay warning concerns unknown cards at a subsequent showdown. Record hand-level warning status and whether both defender cards eventually became known. Use those flags for the historical filters only.
5. Export the fixed two-shard counts below for the judge’s evidence screen. Export full-corpus counts separately. Keep descriptive exports independent of the model-fitting pipeline.
6. **Style scan (type layer).** For every NLHE hand with reliable chronology whose timestamp is at or before its venue’s training cutoff (defined below), replay it with `snapshots=False, observations=False` and accumulate, per venue and handle, numerators and denominators for five public-action features: voluntarily put money in preflop, raised preflop, postflop aggression frequency (bets and raises over all postflop actions), folded to a bet on the flop or turn, and went to showdown after seeing the flop. Count from the replay’s action events so that a big-blind check is not voluntary. Include all table sizes. Exclude every river decision so that river responses remain the outcome being predicted. Write per-shard partial tables and merge them with a PyArrow group-by so memory stays bounded. Replay ran at about 2,700 hands per second per core on the development laptop, so this scan costs about two core-hours, or about twenty minutes on eight cores.
7. **Pluribus extract (extras).** From rows whose `source_file` starts with `pluribus/`, extract heads-up river spots where the defender checks, the bettor bets and the defender responds, with the defender’s cards known. A scan of 484 local Pluribus hands found 26 such spots, all with known cards, and 7 with a bet of 40–60% of the pot. Expect roughly 540, and roughly 145 in the band, across all 10,000 hands. These rows never enter training or the 80/20 split.

**Handle gate.** Group style counts by venue and handle. In four consecutive 1,000-hand Absolute Poker 1000NL files from the gabbler/PokerData copy of the same Zenodo release, between 24% and 56% of one file’s handles reappeared in another file, none appeared in a PokerStars sample, 38 of 115 handles had at least 100 hands, and the busiest had 754. Confirm the same on the full inventory by reporting, for every venue, how many handles reach several hand-count thresholds. If a venue’s handles do not repeat across shards, cluster within shards and say so.

### Historical training and test split

- **Unit.** One unique hand contributes at most one labelled response. Scan every shard, merge candidate rows, deduplicate globally on venue and recorded hand ID, and check a canonical fingerprint of the deal and action record for copied hands. Resolve duplicate keys and quarantine conflicting records before assignment. The source Hub labels its entire corpus `train`; create the project’s own train and test manifests.
- **Eligibility.** For the primary predictor use exactly half-pot offers from the verified heads-up river sequence, with all three exercise responses legal under the recorded stacks. Other sizes in the 40–60% band remain useful for descriptive analysis. Keep records with missing or unreliable chronology in the audit counts and outside the chronological modelling cohort. Log the final eligible total `N` and its fraction of source hands. The full-dataset counts are a required output of preparation.
- **80/20 assignment.** Order eligible hands within each venue by recorded date/time, breaking ties by hand ID. Target exactly `floor(0.80 × N)` training hands. Allocate training slots proportionally by venue using largest-remainder rounding, then take the earliest allocated hands per venue. The remaining, later hands are the final test set. Persist revision, filters, key, fingerprint, venue, timestamp and split. Verify the partition has full coverage and no hand or content fingerprint appears on both sides.
- **Cutoff.** For each venue, the training cutoff is the timestamp of the last hand in that venue’s training partition. The style table, type fit and type tables use only hands at or before it. A venue with no training partition does not enter the type layer.
- **Fit and score.** Use forward-in-time validation windows *inside the training 80%* to choose prior strength, the type count K and features. Refit on the entire 80%, freeze model settings and score the remaining 20% once. For the primary frozen historical test, fit personal counts only from earlier training hands. Report known-player and new-player scores and counts. A separate online-adaptation experiment may predict, record the error, then update after each revealed test response; report it separately.
- **Independence.** Shards 35/36 informed feature selection during research. In addition to the requested all-corpus holdout result, report the same frozen scores on test hands outside those two shards. Keep the live judge’s labels local to that session. Fit any volunteer strength calibration on separate development sessions and reserve new sessions for evaluating the twin.

### Reference counts for the demo

| Historical view · folds / opportunities | Shard 36 | Shard 35 |
|---|---:|---:|
| All eligible offers | 592/1,057 | 688/1,220 |
| Exact half pot after a checked-through turn | 266/394 | 299/455 |
| Exact half pot after the river bettor last bet or raised the turn | 52/100 | 71/141 |
| Eligible offers with both defender cards eventually known | 0/205 | 0/225 |
| Eligible offers in warning-free hands | 592/815 | 688/925 |

Across both shards, 430 of 2,277 eligible offers (18.9%) reveal both defender cards, and none of the 1,280 folds does. That gap is the headline of the filter exhibit.

**Opening replay.** Bundle these Absolute Poker 100NL hands from shard 36. They have the same defender handle. Start each replay just before the response. Keep unknown cards face down. In the type layer, show the defender’s style card beside each replay, computed from hands strictly before that replayed hand.

| Hand ID and replay step | Board | Pot and offer | Turn history | Recorded action |
|---|---|---|---|---|
| `3039372300`, step 15 | 6♦ A♣ 2♥ 2♣ 4♣ | 10 into 20 | Both checked | Fold |
| `3039405280`, step 14 | 8♥ 2♥ 4♠ A♣ 3♦ | 6 into 12 | River bettor last bet or raised | Call |

## 3. ML model — learn the defender from labelled actions

**Training example.** One row is one offer the defender actually faced. The target is the observed action in `[fold, call, raise]`. Use the actor’s response even when their cards are hidden. Historical hand records supply supervised examples of behaviour. The PokerKit environment supplies counterfactual actions and chip rewards for RL in §4.

**Population model.** Fit a categorical distribution over the three responses for each public turn context from §2, using only exact half-pot offers. For counts `n[a]`, set `q[a] = (n[a] + 1) / (sum(n) + 3)`. Fall back to the pooled exact-half distribution for an unseen context. Fit priors on the declared training partition and save counts, revision and split in `population.json`. Use the train and test manifests from §2. Fit all population counts, fallbacks and personal training histories from the 80% manifest; reserve the 20% manifest for scoring. Report a practice-session validation score before interpreting historical rates as representative of judges.

**Personal model.** Implement an online Dirichlet–categorical classifier. For a new player, initialise each context with three parameters `alpha[a] = 8 × q[a]`. On an offer, predict `p[a] = alpha[a] / sum(alpha)`, record the prediction, then increment the component for the observed legal response. Fold probability has marginal distribution `Beta(alpha[fold], alpha[call] + alpha[raise])`; display its mean, 5th and 95th percentiles and the actual player observation count. The prior weight of eight is a starting hyperparameter. For the historical model compare 2, 8 and 24 on forward validation windows inside the training 80%, select by pre-action log loss, then freeze the choice. Use a legal mask and renormalise probabilities whenever a response is unavailable. Design the primary drills so all three responses are available.

**Playable twin.** A public fold rate is enough for a bluff forecast. For realistic showdowns, the twin also needs a tendency to continue with stronger hands. In practice drills, capture the defender’s own river strength as weak, medium or strong. Compute equity against uniformly enumerated legal opponent hole cards, counting ties as half a win, and use cut points 0.33 and 0.67. Cache the calculation by board and own cards. This reference equity is a feature, separate from the RL reward. Fit a second three-action table indexed by turn context and defender strength on practice decisions where the simulator knows the actor’s own cards. For a player and cell, use their observed cell counts plus eight pseudo-observations from the matching **separate calibration-session** cell. Back off to the population context prior when calibration is absent; for an entirely unseen personal strength cell, use the player’s public-context posterior. Keep those sources visible. Evaluate this private-hand twin against the simpler public-only twin on fresh original-player decisions. Use the simpler twin if the strength model worsens development log loss. Never fit the strength model using only historical showdowns.

### Type prior: population, type, person (type layer)

**Features and types.** From the style table keep handles with at least `N_min` pre-cutoff hands, where `N_min` is the smallest count at which odd-numbered and even-numbered hands give a split-half correlation of at least 0.7 on every feature. Drop and report any feature that never reaches 0.7. Shrink each rate toward its venue mean in proportion to its denominator, apply a logit transform and standardise. Fit a Gaussian mixture for K from 2 to 8, choose K by bootstrap stability (adjusted Rand index of at least 0.7 across resamples, ties broken by BIC) and cap K at 6 so the map stays legible. Name each type from its centroid, for example loose-aggressive or tight-passive, and print the centroid statistics next to the name. Compute map coordinates by PCA on the same standardised features. Store each player’s responsibilities, not a hard label.

**Type response tables.** For type `k` and public context `c`, form soft counts `n_k[c][a]` by weighting each training offer by the defender’s responsibility for type `k`. Then set `q_k[c][a] = (n_k[c][a] + s × q[c][a]) / (sum_a n_k[c][a] + s)`, where `q` is the population distribution and `s` is the prior strength chosen for the personal model.

**Mixture personal model.** Give each player the prior `sum_k π_k × Dirichlet(s × q_k[c])` in every context `c`, with weights `π_k` shared across contexts. After observing counts `n[c][a]`, update `π_k' ∝ π_k × prod_c DirMult(n[c]; s × q_k[c])`, where `DirMult(n; α) = Γ(Σα) / Γ(Σα + Σn) × prod_a Γ(α_a + n_a) / Γ(α_a)`. Predict `p[c][a] = sum_k π_k' × (s × q_k[c][a] + n[c][a]) / (s + sum_a n[c][a])`. Compute in log space with `lgamma`. With K = 1 this is exactly the personal model above. Initialise `π` from the player’s style responsibilities when they have at least `N_min` pre-cutoff hands, and from the type shares otherwise. Display the `π` bars, the mixture’s mean fold probability and its 5th and 95th percentiles from 4,000 draws (`random.betavariate`, because the runtime has no SciPy). Place a live judge on the map at `sum_k π_k × centroid_k`, and label that the estimate comes from river responses only.

**Gate.** Keep the type prior in the live model only if it passes the lift test in §6. Otherwise run K = 1 and keep the map as a labelled description.

### Range readout: what the response says about the hand

**Definition.** For a river board `B`, the bettor’s cards `H` and a defender response `a` in public context `c`, enumerate the legal defender holdings `D`: 1,081 before and 990 after removing `H`. Assign each holding its strength bucket by the equity rule above. Weight each holding by the twin’s strength table, `w(D) ∝ R[c][bucket(D)][a]`, where `R` is the player’s strength table with its calibration backoff. Show bucket bars with the posterior mass of weak, medium and strong. In the type layer also show a 13 by 13 grid of the same weights aggregated by starting-hand class, mark the bettor’s own cells, and reveal the defender’s actual holding at showdown. Label both as bucket-level estimates: three buckets give three intensity levels. The readout narrows once, from the uniform prior to the response. It does not model earlier streets.

**Speed.** Rank all 1,081 holdings once per board and correct for card removal instead of nesting the enumeration. Nested enumeration costs about a million seven-card evaluations per board, which is about fifteen seconds at the evaluator’s speed of roughly 70,000 evaluations per second, measured on one laptop. Ranking once costs about a thousand evaluations.

**Validity.** Checking the readout against the twin’s own responses only tests the Bayes code, so treat that as a unit test. Validate it on held-out human practice sessions and on the Pluribus spots in §6.

### EV advisor

Value everything in opening-pot units, matching the RL reward. Let `p_f`, `p_c` and `p_r` be the personal posterior predictive probabilities of fold, call and raise for the offered context. Let `eq0` be the bettor’s equity against the defender range under the environment’s deal distribution, `eq_c` the equity against the range after a call and `eq_r` after a raise, all counting ties as half.

- `EV(check) = eq0`.
- `EV(bet) = p_f + p_c × (2 × eq_c − 0.5) + p_r × max(−0.5, 3 × eq_r − 1)`.

These follow from the payoffs in §4. A called bet wins 1.5 or loses 0.5. After a raise the bettor either folds for −0.5 or calls and wins 2.0 or loses 1.0. A pure bluff with `eq_c = 0` reduces to `1.5 × p_f − 0.5`, so it breaks even at a fold probability of one third. Show both values, the recommended action, an uncertainty band propagated from the fold interval, and a one-sentence reason that quotes the break-even and the current fold forecast. The advisor takes response probabilities from the public posterior and ranges from `R`. Report, per board, the gap between the public marginal and the marginal implied by `R` as a diagnostic.

**Session interface.** Provide `predict(public_context)`, `type_posterior()`, `observe(decision_id, public_context, own_strength, action)`, `readout(board, own_cards, context, action)`, `advise(own_observation)`, `freeze()` and `twin.act(own_observation, rng)`. Store a pre-action prediction and model version before accepting a click. Apply updates once under the session lock. Freeze an immutable snapshot for twin play and RL training. A bettor can observe public actions, board, its own cards and the stored public profile. The defender’s current strength and hole cards stay inside the defender policy. Balance the judge’s probe deals across turn contexts and strong/weak hands, and always offer the same legal half-pot bet during observation.

## 4. RL environment — train the bettor against the frozen twin

**Engine choice.** Use PokerKit 0.7.7 as the rules engine for both the live Mirror exercise and simulated training. Use its explicit hole/board deals and `complete_bet_or_raise_to(amount)` to reconstruct a valid prefix and permit the exact wager. Keep PokerSim’s historical player, server and browser renderer. Implement a thin Gymnasium-style single-agent adapter. The learner runs the bettor, while the environment calls the frozen defender internally. [PokerKit simulation API](https://pokerkit.readthedocs.io/en/stable/simulation.html) · [Gymnasium environment API](https://gymnasium.farama.org/api/env/)

| Alternative | Fit for this project |
|---|---|
| RLCard through PettingZoo | Provides a ready-made multi-agent no-limit game, action masks and a renderer. The documented action menu has fold, check/call, half-pot, pot and all-in. A general exact minimum raise to 100 after a wager of 50 would need an adapter or rules change. [PettingZoo action specification](https://pettingzoo.farama.org/environments/classic/texas_holdem_no_limit/) |
| OpenSpiel Universal Poker | Supports configurable betting abstractions and an unabstracted full-game mode. Useful for a future CFR or equilibrium comparison; requires mapping game parameters, chance states, scenario history and information sets to this exercise. [OpenSpiel implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/universal_poker/universal_poker.cc) |
| Existing PokerSim | Already serves the browser UI and PHH replays. Use its independent settlement on regression fixtures to cross-check the PokerKit adapter. |

**Verified integration fixture.** Seat stacks start at 200 each, blinds 1/2. The button raises to 10, big blind calls. On the flop the big blind checks, the button bets 40 and the big blind calls. Both check the turn. The big blind checks the river. With defender K♦ 8♦, bettor 6♥ 2♠ and board K♣ 9♦ 7♥ 4♠ 3♣, both PokerKit 0.7.7 and the checked-out PokerSim produced a pot of 100 with stacks of 150. From there the bettor’s stack changes were 0 on check-back, +100 on bet/fold, −50 on bet/call, −50 on bet/raise/fold and −100 on bet/raise/call. Each tested path conserved 400 chips. Add this fixture and an independent winning-hand and split-pot fixture to the integration tests. This check establishes transition parity on the tested deal; measure simulation throughput separately.

**Environment.** Implement `RiverResponseEnv` over a headless PokerKit state. One episode begins just after the out-of-position defender checks the river. The learner controls the in-position bettor for up to two decisions. An injected defender policy controls the other seat. Use a headless engine instance per trainer, leaving the judge’s live game session independent.

| Component | Specification |
|---|---|
| Reset | Inject the defender policy at construction. `reset(seed=..., options={"scenario": ...})` returns `(observation, info)` with the legal-action mask. Use a seeded generator of legal heads-up prefixes, fixed opening pot of 100 chips, at least 100 chips behind each seat, chip unit 1, and no rake. Use disjoint boards/deal seeds for train, validation and test. Include both value hands and bluffs. |
| Bettor observation | Return stage, public turn category, board, own cards, pot, amount to call and legal mask. For the Q-table use `(stage, turn_category, bettor_strength_bucket)`. Use the same river-equity buckets as §3. Fixed prices and stack requirements make the available actions consistent within each stage. Each frozen opponent gets its own trained policy table. |
| First decision | `CHECK_BACK` settles the hand. `BET_HALF` bets 50 into 100 and samples the defender response from the injected policy. A fold or call settles. A raise advances to the learner’s second decision. |
| Second decision | The defender’s raise is the legal minimum raise to 100 total. The learner can `FOLD` or `CALL` the extra 50. Either action settles. Mask re-raises in this exercise. Use the same restricted action menu during collection, training, live play and evaluation. |
| Step and reward | `step(action)` applies the legal engine action and runs the defender until terminal or the next bettor decision. Return `(observation, reward, terminated, truncated, info)`; settled deals terminate. Reward is zero on nonterminal steps. Terminal reward is `(bettor_final_stack − bettor_stack_at_episode_start) / opening_pot`. This includes chips already in the pot at the river. Checking to a loss pays 0, winning a fold after betting pays +1, betting then folding to a raise pays −0.5. Both Q stages use this opening-stack reference. |
| Sampling | Draw boards and hole cards from a documented deal distribution before either policy acts. The practice pack must contain complete legal deals. Keep separate random streams for deal generation, opponent sampling and exploration. A fixed seed and action sequence must reproduce the outcome. |

**Algorithm.** Implement masked ε-greedy tabular Q-learning. There are at most 18 stage/context/strength combinations and two legal actions per stage. Use `γ = 1` for the finite episode. Initialise legal Q-values to zero. For a visited pair update `Q[s,a] += α × (target − Q[s,a])`, where `target = reward` at terminal and `reward + max(Q[next, legal])` otherwise. Start with `α = visits(s,a)^−0.6`. Anneal ε linearly from 1.0 to 0.05 over the first 80% of episodes and keep it at 0.05 for the remainder. Use ε = 0 during evaluation, with deterministic ties favouring check at the opening and fold facing a raise. The private hand abstraction makes Q-values averages over deals sharing a bucket. Label them as simulated estimates. [Watkins and Dayan, Q-learning](https://doi.org/10.1023%2Fa%3A1022676722315)

**Training runs.**

1. **Population bot.** Train from scratch against the frozen population twin for an initial budget of 20,000 simulated river episodes. Checkpoint at 1k, 5k, 10k and 20k. Record state/action visit counts, mean reward on fixed validation seeds and greedy action frequencies per context. Cache prepared scenarios and equity features so the training loop spends its time on play.
2. **Personalised bot.** Freeze the judge’s or volunteer’s twin. Train a separate Q-table against it with the same initial values, episode budget, exploration schedule and deal distribution. A simulated episode consumes no new human label. Use a background training job keyed by profile version and seed; serve only completed matching checkpoints. Display progress and retain a labelled pretrained practice profile for the 90-second demo.
3. **Training sufficiency.** Check validation reward and greedy choices across checkpoints and three training seeds. Report support for each action in demo states. If the final checkpoint is unstable or a demo action has fewer than 100 visits, increase the simulation budget or restrict the displayed state coverage. Benchmark actual CPU training time on the demo laptop. Keep final test seeds sealed until the training settings are fixed.
4. **Frozen comparison.** Evaluate the two greedy Q policies against the same independent defender on matched fresh deals. Use an additional population-bot versus population-bot control trained with different seeds to estimate training variation. The trained counter-bot supplies the action shown in the demo.

**Numerical oracle.** Test the environment on a pure-bluff subgame with a board of K♣ 9♦ 7♥ 4♠ 3♣, bettor 6♥ 2♠ and legal one-pair defender hands. Verify each hand using the evaluator. Use a scripted defender that folds with fixed probability and otherwise calls. Checking has value 0. Betting 50 into 100 has normalised value `1.5 × p_fold − 0.5`. At fold rates 0.2 and 0.6 the correct bet values are −0.20 and +0.40. Train and evaluate each fixture in isolation, check that learned bet values are within 0.10 opening pots of those targets and that the greedy actions are check and bet respectively. Add a separate deterministic raise fixture to test the second decision and payout. The oracle remains an evaluation baseline; Q-learning selects the shipped bot’s actions.

**Oracle for every cell.** The pure-bluff fixture checks one hand. Extend it to the full abstraction. For each cell of stage, turn context and bettor strength bucket, compute `Q*` by averaging the twin-exact expected value of check and bet over the evaluation scenario pool, which follows the documented deal distribution. Use the twin’s strength table for response probabilities and exact showdown results, with no sampling of defender responses. The learned Q-values must lie within 0.10 opening pots of `Q*` in every demo cell with at least 100 visits. Compare the learned greedy action with the `Q*` greedy action, which should agree outside near-ties, and with the EV advisor’s average recommendation for the cell. List every disagreement. A learned-versus-`Q*` disagreement is a training defect. An advisor-versus-`Q*` disagreement means the public forecast and the strength table imply different response rates, or the bucket hides hand-specific information such as blockers.

**Policy chart.** Render the greedy policy as a three by three chart of turn context by bettor strength bucket. Each cell shows the chosen action, both Q-values in opening pots, the visit count and the `Q*` value, so the learned bot is not a black box.

**Compute budget.** Budget the equity work, not the Q updates. A bucket costs about 990 evaluations per hand, so about 28 ms for the two hands of a deal at the measured evaluator speed. Recomputing that for 20,000 episodes in each of about twenty trainings (the population arm plus four opponents at four probe budgets, and more if the type-aware arm is added) would cost hours. Precompute a seeded pool of boards and deals with both buckets and reuse it across arms. Run trainings in parallel processes.

**Cut line.** At H12, if either bot misses the oracle tolerance in the demo cells after the sufficiency checks above, display the EV advisor as the bettor and report Q-learning as exploratory, with its measured gaps to `Q*`.

**Code ownership.** Put the profile estimator, type prior and snapshots in `pokersim/mirror_model.py`, the range readout and EV advisor in `pokersim/mirror_advisor.py`, the Parquet scan, filters, split and style counts in `pokersim/mirror_data.py`, type fitting and map export in `pokersim/mirror_styles.py`, the headless environment in `pokersim/river_env.py`, and training, checkpoint loading and matched scoring in `pokersim/mirror_rl.py`. Wire CLI commands in `pokersim/__main__.py` and the inference adapter through `pokersim/agents.py`. Route all Mirror practice and training transitions through the same PokerKit adapter; translate its state to the existing renderer and agent formats at the boundary.

**Commands and outputs to implement.** Extend the existing CLI with `python3 -m pokersim mirror-prepare` to run both scans over all 37 pinned shards and emit the 80/20 manifests and the style table, `mirror-cluster --styles <style table> --out data/mirror/styles.json` to fit types, type tables and map points and run the lift test, `mirror-train --profile <snapshot> --episodes 20000 --seed 7 --out <policy.json>` for either arm, `mirror-evaluate --baseline <policy> --candidate <policy> --seed 17 --pairs 1000` for a matched report, and `mirror-check-ranges` for the readout checks in §6. Save the profile hash, type-fit hash, action encoding, bucket boundaries, environment version, seed, episode budget, visits and learned Q-values with every policy. Load the policy through the existing agent interface. Reject incompatible environment, profile or type-fit versions.

## 5. Build the 90-second demo

| Time | Judge interaction | Required output |
|---|---|---|
| 0–20 s | Inspect the two recorded river decisions, guess which bet got the fold, then reveal the turn histories. | Show both responses and the exact-half comparison across both shards. Fold rates were about 66–68% after a checked-through turn and 50–52% after the river bettor had also bet or raised the turn. In the type layer, show the defender’s style card. |
| 20–35 s | Switch between all observed actions, known defender cards and warning-free hands. | Display **1,280/2,277**, **0/430** and **1,280/1,740** folds respectively. Point out that only 18.9% of eligible offers ever reveal both defender cards and none of the folds do. Explain that a later missing showdown card can hide a valid earlier decision from a filtered dataset. |
| 35–60 s | Play two or three river decisions as the defender and open the personal profile. | Reveal committed predictions, actual actions, observation counts and uncertainty. In the type layer also show the type bars and the dot on the style map, and say that three responses barely move them. Start personal-policy training in the background. |
| 60–90 s | Take the bettor seat against the frozen twin on a validated bluff deal, then compare the context-only and personalised bettors on that deal. | Show the range X-ray after the twin responds, the EV panel with the one-third break-even, the two bettors’ chosen actions, the policy chart cell, learned check/bet values in chips and training visits. On the pure-bluff fixture also show the analytic break-even and EV. Display measured benchmark results with opponent identity, sample count and interval. Identical actions are a valid result. |

Use distinct labels for historical hands, the current judge, recorded practice sessions and simulated opponents. The G-Research demonstration centres on which observations disappear when data is filtered. The HRT demonstration centres on whether a better forecast changes a betting decision and its payoff. The style map and range X-ray extend the first; the EV panel and paired benchmark are the second. Keep the Evidence page open in a second tab for questions. If the type layer is cut, remove the style card, bars and map and change nothing else. If Q-learning is cut, show the EV advisor in the last segment.

## 6. Evaluate the delivered model

Report every forecast and decision result on the same three rungs: population, type and person. The type rung appears only if its gate passes.

- **Prediction.** Score committed forecasts against fresh responses using mean squared error of the fold probability (Brier score). Compare the context-only and personal models on identical opportunities. Report decision count, player count and per-session difference. Use fresh decisions by the original player to assess imitation. Report fold Brier score and three-action log loss separately, with historical, live-human and synthetic results labelled.
- **Type-prior gate.** On the frozen 20% test, compare four forecasters on identical offers: the pooled fold rate, the population context model, the type prior using the player’s style responsibilities, and the personal model fitted on that player’s earlier training offers. Start with style-known players at zero in-test observations, then report by number of earlier observations (0, 1 to 3, 4 to 15, 16 to 63, 64 or more), and separately for players with and without river history. Use three-action log loss and fold Brier score. Bootstrap over players, not offers. Keep the type prior in the live model only if its log loss beats the population context model with a 95% interval that excludes zero. If it fails, report the result, run K = 1 and keep the map as a labelled description. Repeat the test on hands outside shards 35 and 36. Style is measured on all table sizes while the outcome is a heads-up river response, so transfer across formats is part of what the gate tests.
- **Decision value.** Define independent scripted opponents before running the benchmark. For an initial reproducibility fixture, use fold/call/raise probabilities of (0.15, 0.75, 0.10) for a frequent caller and (0.65, 0.30, 0.05) for a frequent folder. Add a context-matched population control and a strength-dependent opponent that raises strong hands. Label these policies synthetic. Use probe budgets of 0, 4, 16 and 64 original-opponent responses to plot how quickly personalisation helps. Give the personal learner a fixed observation block from each independent defender. Run the population arm on the identical deal schedule with personal updates disabled. Freeze both bettors, then evaluate on 1,000 matched river opportunities per opponent with the same deals and defender random draws. Report the fraction of decisions that changed, mean chip difference per opportunity and a paired 95% bootstrap interval. Include checks in the denominator. Label the result as conditional on those river scenarios. Report the EV advisor as a second decision arm on the same deals, and add the type-aware bettor arm only if time allows.
- **Range readout.** Test it on three sources. First, held-out volunteer sessions, where the simulator knows the defender’s strength bucket: report the log score of the true bucket under the posterior against a response-blind baseline, and the mean posterior strength by response. Second, the Pluribus spots (extras), where every card is known and nothing is selected by showdown: report the defender’s mean true river equity by response with bootstrap intervals, and state whether fold is weaker than call and call weaker than raise. With roughly 540 spots, and fewer in the half-pot band, intervals will be wide, so describe the check as directional. Third, the twin’s own responses, as a unit test only. Do not calibrate on HandHQ showdown reveals: they come from players who did not fold, so no fold appears among them.
- **Human evidence.** Use held-out volunteer sessions for any claim about human behaviour. Show the number of independent sessions. Keep synthetic benchmark results available offline for the judge. Display a null or negative result when that is what the experiment produces.
- **Regression checks.** Verify legal actions and chip conservation, deterministic seeded replays, correct pre-bet pot, the one-third break-even point, no duplicate updates, clean resets and no cross-session profile leakage. Swapping hidden defender cards under identical public inputs must leave the bettor’s fold prediction unchanged. Also verify that readout weights sum to one, that the readout and the EV advisor use only public information and the viewer’s own cards, that the type posterior sums to one and K = 1 reproduces the personal model exactly, that `EV(bet)` equals `1.5 × p_f − 0.5` on the pure-bluff fixture, and that style counts contain no river decision and no hand after the venue cutoff.
- **Evidence page.** Build one offline screen with five panels, each showing its source, denominator and generation date: the filter exhibit, the forecast ladder by observation count, the type-prior gate result, the decision benchmark and the range checks. Show a failed or null result as plainly as a pass.

## 7. Work order, schedule and handoffs

| Workstream | Start after the shared decision contract | Handoff |
|---|---|---|
| A · Historical data | Run the schema gate. Process all 37 shards, deduplicate, create the 80/20 manifests, regenerate the two-shard demo evidence and export both opening hands. Then run the style scan (type layer) and the Pluribus extract (extras). | B receives context priors and the style table. D receives attributed aggregates and offline replays. |
| B · Player model | Implement the two supervised action tables, public prediction, session updates, freezing and twin sampling. Add the bucket-level readout and EV advisor. Then fit types and the type prior and run the lift test (type layer). | C and D receive the same policy interface and explanation fields. |
| C · Game and evaluation | Build the PokerKit-backed RiverResponseEnv, matched engine fixtures, masked Q-learning, policy checkpoints, the `Q*` oracle, independent defenders, the precomputed scenario pool and the paired benchmark. | D receives playable scenarios and measured results. B receives held-out prediction scores. |
| D · Demo | Implement the session registry, replay reveal, filters, practice flow, fixed-deal comparison and Evidence page using the shared fixture. Then add the style map, range grid and learning-curve chart (type layer). | Integrate A’s exports and B/C’s live results. Exercise the complete flow offline. |

Hours are measured from the start of the build. If time has already passed, shift every gate by the same amount and apply the cut order from §0.

| Window | Goal | Gate at the end |
|---|---|---|
| H0–H3 | Shared decision record and fixture; PokerKit parity; schema gate; two-shard exports; one end-to-end decision with a committed prediction | A judge can click through one decision and a refresh recovers it |
| H3–H9 | A: first scan and manifests. B: population and personal models, twin, bucket readout, EV advisor. C: environment, oracle, Q-learning. D: replay, practice flow, session registry, Evidence page skeleton | Full-corpus counts exist, or the two-shard cohort is declared as the fallback with every number labelled so while the full scan keeps running; the pure-bluff oracle passes |
| H9–H15 | A: style scan. B: types, type prior, lift test. C: scenario pool, paired benchmark, `Q*` check. D: style map, range grid, learning-curve chart | **H12:** Q-learning matches `Q*` on the demo cells, or the EV advisor becomes the displayed bettor. **H15:** the type-prior gate is decided |
| H15–H20 | Integration; benchmark runs from documented seeds; Pluribus check if time is spare; demo script | **H20:** feature freeze |
| H20–H24 | Rehearse offline; record a backup capture; fix defects only; README and attribution | A fresh-checkout launch passes the definition of done |

For a single build agent, complete the shared fixture, historical export, profile, playable twin, EV advisor and bucket readout, RL environment, Q trainer, paired evaluation, Evidence page and final UI integration in that order, then the type layer, then extras. For a team, run A–D concurrently once the shared fixture works. Integrate the first real-data replay and live update before polishing the UI.

**Definition of done.** From a fresh checkout, the documented launch command opens the demo offline. Preparation accounts for all 37 shards and reproduces the deterministic 80/20 split, two replays and all five demo cohort counts. Record train/test sizes, exclusion counts and final held-out scores. A judge can complete a practice block, play against their frozen twin, compare two bettors on the same deal and reset for the next judge. Existing tests and the new regression checks pass. The PokerKit adapter passes the settlement parity checks. Both Q policies train from the documented CLI, the oracle fixtures pass, the benchmark can be regenerated from documented seeds, and every displayed result identifies its source and denominator. The Evidence page shows every number with its source and denominator, the type-prior gate result is recorded whichever way it falls, the `Q*` comparison passes or the displayed bettor has fallen back to the EV advisor, and each extra is either reported or listed as skipped.

## Revision notes

This revision fuses the Mirror plan with the X-ray framing: make the hidden visible, predict, then decide.

- **Added §0.** Pitch, theme mapping, scope boundaries, and the Core, Type layer and Extras tiers with cut order.
- **Added a schema gate and a handle gate in §2.** They record verified Takara quirks: Pluribus rows sit in shard 0 with a null venue, five hand-annotated rows have a broken `actions` field, and date fields are floats. They also record evidence that anonymised handles repeat across files. The style scan, cutoff rule and Pluribus extract are new.
- **Added three model pieces in §3.** A mixture-of-Dirichlet type prior with a gate, a range readout and a closed-form EV advisor. The readout includes a speed note, because nested equity enumeration takes about fifteen seconds per board.
- **Extended §4.** A `Q*` oracle for every cell, a policy chart, a compute budget and the H12 fallback to the EV advisor. New modules and CLI commands are listed.
- **Updated §5 to §7.** The demo script, a type-prior gate, range-readout checks, the Evidence page, a session registry, a decision record and an hour-by-hour schedule with gates.
- **Replaced** the machine-specific checkout path with the repository root.
- **Not re-verified in this revision.** Shard counts and hashes, the fixture stacks, the reference counts and the PokerKit 0.7.7 pin are carried over unchanged from the previous version.
