# Poker Mirror · "gg ez poker"

**HackTheBurgh XIII, 7–8 November 2026.** A poker training simulator that holds up a mirror: you sit at a
six-handed no-limit hold'em table against the four classic types of player, models trained on 21.6 million real
hands show what everyone (you included) is likely to do and to hold, a second real-time Bayesian model prices your
hand like a market, a bluff check prices your bets, and a reinforcement-learning coach suggests a better move.
Nothing on screen ever sees a hidden card, and every number is reported next to its baseline on data no model was
trained on.

```bash
uv venv --python 3.12 .venv                                   # or: python3.12 -m venv .venv
uv pip install --python .venv -r requirements.txt -r requirements-market.txt
.venv/bin/python -m pokersim serve --port 8765                # then open http://localhost:8765/
```

All model weights are committed (`data/rl/*.npz`, `market/model/likelihood.pkl`), so a fresh clone runs offline.
Python 3.11+; the Mac's system Python 3.9 is too old. Press **N** for the next hand, click any range grid to see it
large, and toggle the **hole-card cam** in the header to show the audience every card (display only; the models never
see them).

## What is on the screen

| Where | What it shows | Which model |
|---|---|---|
| Each seat | The player's type (Tight/Loose × Aggressive/Passive, and the Shark) and a fold / call / raise bar for their next move; a second "This player" bar when their own habits disagree with people in general | Next move + personal model |
| Beside each opponent | Two 13×13 likely-hand grids: **Base** (the hand model, seat colour) and **Live ML** (the market model, cyan) with its top hand types and "you win X%" | Hand model; Live ML |
| Middle of the table | One line per move: did we read it (✓/✗) and how much it moved our guess of their hand | |
| Bottom left | Two coaches: **Base** copies Pluribus, **RL** was trained against players like these; a note when they disagree | Next move; RL coach |
| Bottom middle | The model of **you**: its guess of your move (revealed after you act), how often it has been right, your habit | Personal model |
| Bottom right | Bet-size buttons and the **bluff check**: if you bet X with nothing, everyone folds Y%, break-even is Z% | Each player's own model |
| After the hand | **If you had followed the coach**: what each move would have won from your first disagreement, 150 replays each, with ± | RL coach (`pokersim/what_if.py`) |
| **Details** button | The older right-hand panel: selected player's range, habit in plain words, scorecard, replays ("The Shark learns you", "A player like you") | |
| **Autopilot** | The RL coach plays your seat | RL coach |

The five opponents are named after their type. Tight Aggressive plays like the professionals in the Pluribus games;
Loose Aggressive, Loose Passive and Tight Passive are mined from 10,213 real online regulars; the Shark is the mined
tight-aggressive type that also reads the table and bluffs more when the players still in fold more than people in
general. Details and the measured numbers are in [DEMO.md](DEMO.md).

## The models

| Model | Question it answers | Trained on | Held-out result |
|---|---|---|---|
| **Next move** | What will this player do? | Every decision in 21.6M hands (5/6-max NLHE) | Calibrated: says 75% → happens 76% (670,615 decisions by unseen players) |
| **Hand model** | What are they holding? (169 hand types) | Hands where cards were shown + Pluribus | Real hand in top 10: 16% vs 8% for card counting (29% vs 5% after the flop; 44,036 Pluribus guesses) |
| **Personal model** | How does *this* person differ? | Nothing beforehand; Bayesian tilt from their own moves ("Bayes' Bluff", Southey et al. 2005) | +1.3 ± 1.2 points over people in general; sharper for 28 of 30 unseen players |
| **RL coach** | What should you do? | PPO from an imitation of Pluribus, against human-like and rule bots | **+156 ± 15 bb/100** vs a maths bot on sealed tables registered before any result |
| **Live ML (market)** | Given every bet so far, which of the 1,326 two-card hands does this opponent hold, and what is your hand worth? | Showdown hands from PokerStars 200NL (2009), heads-up postflop pots | Brier 0.183 vs 0.194 for a random-hand range, 95% CI excludes zero; median update 0.3 ms |

How the first four were trained and tested: [train/TRAINING.md](train/TRAINING.md) and the pre-registered sealed
test in [train/06_eval_and_overfitting.md](train/06_eval_and_overfitting.md). The market model: next section and
[NOTES-market.md](NOTES-market.md).

## Live ML: poker as a market

Every bet is an order. The `market/` package treats it that way: it keeps a probability over **all 1,326 hole-card
combos** an opponent could hold and, after each of their actions, does a Bayesian update
`posterior ∝ prior × P(action | combo, context)`. The prior is a position- and role-based preflop range; the
likelihood is two multinomial logistic regressions (check / small bet / big bet, and call / raise) on combo features
(board-strength percentile, flush and straight draws) crossed with context (street, position, bet size faced, pot
odds, stack-to-pot ratio, preflop role, aggression this street). The posterior times a precomputed equity vector
gives a **price**: your win probability right now. Price above the pot-odds price says call; below says fold.

- **Fast.** `market/fasteval.py` is a vectorised exact 7-card evaluator (identical to eval7's values) that scores
  all 1,326 combos on hundreds of runouts in one table lookup. A full update (Bayes + likelihood + equity + price)
  takes 0.32 ms median, 7.1 ms p99, always under the 50 ms budget.
- **Honest.** Evaluated on 46 held-out tables (551 hands where both players showed, 9,792 price events) with a
  hand-level cluster bootstrap. Brier 0.183 (model) vs 0.187 (prior only) vs 0.194 (uniform); calibration error
  0.049 vs 0.085. The gain is on the flop and turn; on the river the model is no better than uniform.
- **Two biases we found and fixed.** Hands that fold never show cards, so folds cannot be learned from showdown data
  (and an opponent's call barely narrows the range). When only one player shows, they won 2,833 of 2,833 times
  because losers muck; our first evaluation included those hands and made the baseline look better than the model.
  All evaluation now uses both-shown hands only.
- **Not claimed.** The call/fold rule's profit is not measurable on showdown-only data (model − uniform =
  −28 bb/100, CI spans zero). At the six-handed table each opponent is treated as heads-up against the field, an
  approximation that is called out in the code and the pitch.

On the table it runs through `market/live.py`, which rebuilds one posterior per live opponent from the public action
history on every `GET /api/lab/predict` call (about 5 ms per opponent) and adds a `"market"` field to each player
row. Nothing else in `pokersim/` changed. The standalone tools:

```bash
.venv/bin/python -m market.parse       # gabbler/PokerData HandHQ files in data/cache -> data/processed/hu_flop_hands.pkl
.venv/bin/python -m market.evaluate    # fit on train tables, score test tables -> figures/, results.json
.venv/bin/python -m market.server      # FastAPI websocket replaying held-out hands one event at a time, :8001
.venv/bin/python -m pytest tests/test_market.py
```

The websocket streams `meta`, then one `event` per action or board deal (posterior, 13×13 grid, price, pot-odds
price, decision, latency), then `end`. The opponent's cards appear only in `end`, so nothing leaks mid-hand. The
schema is at `GET /api/schema` and in [NOTES-market.md](NOTES-market.md).

## Documentation map

- **[DEMO.md](DEMO.md):** what is on the screen, the demo flow, every number to quote and what it was tested on,
  what we do not claim.
- **[docs/PITCH_4MIN.md](docs/PITCH_4MIN.md):** the four-speaker pitch script. Slides: [docs/slides/](docs/slides/).
- **[train/TRAINING.md](train/TRAINING.md):** how each model was trained and tested; the full plan is in
  [train/](train/).
- **[NOTES-market.md](NOTES-market.md):** the market model's data, assumptions, results and API.
- **[judges/](judges/):** mock-judge sessions and the questions they asked.
- **[PROGRESS.md](PROGRESS.md), [PLAN.md](PLAN.md), [CHECKLIST.md](CHECKLIST.md):** build log and the spec of the
  earlier heads-up river prototype (the **Mirror** and **Evidence** tabs on the PokerSim page, below).

## Repository layout

```
pokersim/                  the engine and the Poker Mirror server (standard library + PokerKit + NumPy)
  engine.py, evaluator.py, phh.py, replay.py, agents.py, equity.py, evaluate.py, simulate.py, dataset.py
  server.py                HTTP + SSE API; serves web/; / is the Poker Mirror page, /index.html the PokerSim tools
  lab_api.py               GET /api/lab/predict: every value the Poker Mirror page shows (response documented inside)
  next_move.py, hand_range.py, you_model.py, opponent_model.py     the three data models
  rl_policy.py, rl_train.py, rl_features.py, what_if.py           the PPO coach and the what-if replays
  mirror_*.py, river_env.py                                        the earlier heads-up river prototype
market/                    Live ML: cards, prior, features, likelihood, fasteval, equity, tracker, live, server
  model/likelihood.pkl     the shipped fit (train tables only)
web/lab.html, web/lab/     the Poker Mirror page ("gg ez poker")
web/index.html, app.js, mirror.js   PokerSim tools: Play, Replay, Evaluate, Simulate, Mirror, Evidence
train/                     training plan, progress and evaluation protocol
data/rl/                   next_move, hand_range and RL policy weights, sealed-test results, coach variants
data/demo/, data/mirror/   small derived artifacts the page reads (calibration, replays, player types)
data/samples/              hand sets from the dataset for offline use; data/cache/ is the ignored download cache
scripts/                   training, mining, calibration, replay-building and smoke tests
tests/                     python -m unittest discover tests   (plus tests/test_market.py under pytest)
docs/, judges/, eddie/     pitch and slides, mock-judge notes, cluster job scripts
```

Dependency files: `requirements.txt` (serve), `requirements-market.txt` (Live ML, standalone market server and
tests), `requirements-prep.txt` (offline data preparation), `requirements-train.txt` (PyTorch RL training);
`pyproject.toml` lists the same groups for uv.

## What we do not claim

- The RL coach is tested against bots built from real players' decisions, not against people.
- The demo table is not a test; the sealed tables and the held-out tables are.
- The what-if box and the market price are estimates with error bars, not verdicts.
- Nothing here is for use while playing online. Real-time assistance is banned on poker sites; this is for practice
  and review.

## Team

Built at HackTheBurgh XIII (judges: HRT and G-Research). Contributors, as in the git history: **kazut / kazutaka**
(data pipeline, hand and next-move models, PPO coach, Poker Mirror page), **silvermew** (PokerSim engine and the
river prototype), **Aikk** (Live ML market model and the gg ez poker table integration), **Amish Parekh** (frontend
redesign and Table Lab), **Aayush Deshmukh** (repository setup).

---

# PokerSim - interactive hold'em simulator + hand-history replayer

Poker Mirror is built on PokerSim, a realistic, playable No-Limit Texas Hold'em table with a browser UI, a replayer
for real recorded hands from the [gabbler/PokerData](https://huggingface.co/datasets/gabbler/PokerData)
Hugging Face dataset, and a harness for testing poker models against those hands. Its page is served at
`/index.html` next to the Poker Mirror page.

* **Play** - sit at a table against bots or your own model; full NLHE rules: blinds, antes,
  straddles, min-raise rules, all-ins, side pots, split pots, uncalled bets, showdown.
* **Replay** - load any `.phh` / `.phhs` file from the dataset (browsed and fetched on demand),
  scrub through every action with the table animating, and overlay a model's decision at each
  spot ("model lens").
* **Fork** - stop a recorded hand at any action and play it out live from that spot, with humans,
  bots or models in the seats (unknown cards are dealt at random, the recorded run-out is kept).
* **Evaluate** - shadow-evaluate a model over thousands of recorded decisions: agreement rate,
  per-street / per-action breakdown, confusion matrix, bet-sizing error, CSV export.
* **Simulate** - self-play between models for N hands, bb/100 with standard errors, and export of
  every hand in PHH format (the same format as the dataset).

The engine is standard-library Python 3.11+ (3.12 tested) plus vanilla HTML/CSS/JS - no build step.
The optional `anthropic` package enables the Claude agent.

## Quick start

```bash
python -m pokersim serve --open        # http://127.0.0.1:8000/  (Poker Mirror; the tools are at /index.html)
```

Use `--host 0.0.0.0` to share the table on a LAN. Deep links on `/index.html`: `#demo` deals a bot-only table,
`#replay=pluribus-session-30.phhs&i=7&step=12&lens=tag` opens a sample hand at a step with a model
lens, `#replay=<file>&eval=tag` runs an evaluation, `#sim=equity,tag,lag,random&hands=200` a match.

Command line equivalents:

```bash
python -m pokersim agents                                       # list built-in bots / adapters
python -m pokersim replay data/samples/dwan-ivey-2009.phh       # step through a hand in the terminal
python -m pokersim evaluate data/samples/pluribus-session-30.phhs --agent tag
python -m pokersim simulate --agents equity,tag,lag,random --hands 500 --out selfplay.phhs
python -m pokersim fetch --list pluribus/100                    # browse the Hugging Face dataset
python -m pokersim fetch "pluribus/100/0.phh"                   # download into data/cache
python scripts/validate_dataset.py data/samples                 # engine vs. recorded stacks
python -m unittest discover tests                               # test-suite
```

## The dataset

`gabbler/PokerData` is ~16 GB of hands in the **Poker Hand History (PHH)** format
(TOML; spec at <https://phh.readthedocs.io>): 21.6M anonymised online NLHE hands (HandHQ,
2009: PokerStars, Full Tilt, PartyPoker, iPoker, Ongame, Absolute), the 10,000 Pluribus hands
(Brown & Sandholm 2019), the 83 televised hands of the 2023 WSOP $50k PPC final table (9 mixed
variants) and a few historic hands. Example:

```toml
variant = "NT"
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ["d dh p1 TcQc", "d dh p2 8s4c", ..., "p3 f", "p4 cbr 210", "p5 f", "p6 f", "p1 cc", "p2 f",
           "d db 7d5h9d", "p1 cc", "p4 cc", "d db 7c", ...]
players = ["MrBlue", "MrBlonde", "MrWhite", "MrPink", "MrBrown", "Pluribus"]
finishing_stacks = [10310, 9900, 10000, 9790, 10000, 10000]
```

Actions: `d dh pN cards` deal hole cards, `d db cards` deal board, `pN f` fold, `pN cc`
check/call, `pN cbr X` bet/raise **to** X, `pN sm [cards]` show/muck. `??` is an unknown card;
players are listed from the first seat left of the button, so the last player is the button
(heads-up: blinds reversed, button posts the small blind). Online cash-game logs hide every hole
card that was not shown at showdown, so model evaluation on those hands is limited to players
whose cards became known. The Pluribus, WSOP and iPoker "hero" hands have known cards.

The engine supports the hold'em family (`NT` no-limit, `NS` short deck, `PO` pot-limit Omaha,
`FT` fixed-limit). Stud/draw/badugi hands from the WSOP set parse but cannot be replayed.

### Validation against the recorded results

`scripts/validate_dataset.py` replays hands through the engine and compares with the recorded
`finishing_stacks` / `winnings`:

| Data                                         | Hands | Exact finishing-stack match                       |
|----------------------------------------------|------:|---------------------------------------------------|
| Pluribus (sessions 30-32, 100-102)           |   484 | 484 / 484                                          |
| 2023 WSOP PPC final table, hold'em variants  |    25 | 25 / 25                                            |
| Historic hands (Dwan/Ivey, Antonius/Blom, Phua/Xuan) | 3 | complete, no stack data in record            |
| HandHQ (6 sites, ~6,000 hands)               | 5,969 | winners match 3,211 / 3,231; stacks differ only by rake |

Known data quirks (reported as warnings in the UI, never silently hidden): Ongame records use
negative blind entries for a returning player's refunded blind (7 of 999 hands end a few chips
off), PartyPoker's `winnings` include returned uncalled bets, and online `winnings` are post-rake.

## Plugging in your own model

Models receive an **observation** (JSON) and return an **action** (JSON). Three ways:

1. **HTTP adapter** (any language): start a server that accepts `POST` with the observation
   and replies with an action, then use the agent spec `http:http://127.0.0.1:5001/act`.
   `examples/model_server.py` is a complete 60-line example - copy it and replace `decide()`.
2. **Python**: subclass `pokersim.agents.Agent` and implement `act(observation) -> action`
   (register it in `pokersim.agents.AGENT_TYPES` to use it from the UI/CLI).
3. **Claude**: spec `claude` or `claude:claude-opus-5-5,effort=low` (needs `pip install anthropic`
   and `ANTHROPIC_API_KEY`). Decisions come back as structured JSON with a one-line reasoning that
   is shown in the game log. Refusal fallbacks are enabled by default.

Observation (abridged):

```json
{
  "variant": "NT", "street": "flop", "board": ["Kc", "7s", "4h"],
  "pot": 150.0, "total_pot": 150.0, "to_call": 0.0, "min_bet": 10, "chip_unit": 1,
  "blinds": [5, 10, 0, 0, 0, 0], "antes": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
  "hero": {"index": 3, "name": "p4", "position": "HJ", "cards": ["Ah", "Ac"], "stack": 1920, "bet": 0},
  "players": [{"index": 0, "name": "p1", "position": "SB", "stack": 1485, "bet": 0, "folded": true,
               "all_in": false, "is_button": false, "is_hero": false}, ...],
  "legal_actions": {"fold": false, "check": true, "call": 0.0, "raise": true,
                    "min_raise_to": 10, "max_raise_to": 1920, "raise_kind": "bet"},
  "history": [{"street": "preflop", "player": 3, "action": "raise", "amount": 40, "to": 40}, ...]
}
```

Action: `{"action": "fold" | "check" | "call" | "bet" | "raise" | "all_in", "amount": <raise-to total>,
"reasoning": "optional"}`. Illegal or out-of-range answers are normalised (clamped / converted to
call) and the adjustment is recorded, so mistakes stay visible in logs and evaluation rows.

### Evaluation metrics

For each recorded decision the model sees exactly what the player saw (their hole cards when
known in the record, optionally filled in from the showdown). Reported: agreement on the exact
action class (fold / check / call / bet / raise), agreement on intent (fold / passive /
aggressive), per-street and per-action breakdown, a confusion matrix, mean `|log(model size /
human size)|` when both bet, latency, and a per-decision CSV. In the UI every disagreement is a
link that opens the hand at that exact spot with the model's suggestion overlaid on the table.

## The Mirror tab: the earlier heads-up river prototype

The **Mirror** tab on `/index.html` was the first Poker Mirror design and still runs. It asks one question: after
accounting for the board and the public betting, how much does knowing *this* opponent help a bettor decide whether
to bluff? Scope: heads-up no-limit river, the defender checks and faces a half-pot bet. The spec is
[PLAN.md](PLAN.md); the order of work and the build log are [CHECKLIST.md](CHECKLIST.md) and
[PROGRESS.md](PROGRESS.md).

Five steps in the UI: (1) two recorded river decisions by the same player, guess which one folded;
(2) what a filter hides (folds never reach a showdown); (3) you defend, with a forecast committed
before every click; (4) you bet against your frozen twin with an EV advisor and a range X-ray;
(5) two Q-learning counter-bots, one trained against the population twin and one against yours.
The **Evidence** tab shows every generated artifact with its source and denominator.

Offline preparation (needs `pip install -r requirements-prep.txt`, 1.3 GB download):

```bash
python scripts/fetch_takara.py                      # 37 Parquet shards -> data/cache/takara/
python -m pokersim mirror-prepare --styles          # river spots, exhibit, manifests, population, style table
python -m pokersim mirror-cluster                   # player types + the type-prior gate -> data/mirror/styles.json
python -m pokersim mirror-train --profile population --episodes 20000 --seed 7 --out data/mirror/policy_population.json
python -m pokersim mirror-benchmark                 # personalised vs population bettor -> data/mirror/benchmark.json
python -m pokersim mirror-eval --pluribus-shard data/cache/takara/train-00000-of-00037.parquet
python -m pokersim mirror-check-ranges              # equity / readout / advisor sanity checks
```

The prototype itself needs only the standard library plus `pokerkit`; everything it shows is read from
`data/mirror/`.

## Credits

Hand histories in `data/samples/` are excerpts of the `gabbler/PokerData` dataset (CC-BY-4.0),
itself built from the PHH dataset by Juho Kim (University of Toronto Computer Poker Research
Group), the Pluribus supplementary material (Brown & Sandholm, *Science* 2019), HandHQ logs and
PokerGO/WSOP broadcasts. PHH format: Kim, "Recording and Describing Poker Hands", IEEE CoG 2024.

Poker Mirror scans the [takara-ai/poker_hands](https://huggingface.co/datasets/takara-ai/poker_hands) Parquet
release of the same PHH dataset (CC BY 4.0), pinned at revision `6acb5afb`. The Live ML market model is trained on
the PokerStars 200NL HandHQ files of `gabbler/PokerData`. The files committed in `data/mirror/` and `data/demo/` are
small derived artifacts: counts, model parameters, anonymous style-map points and a few demo hands.
