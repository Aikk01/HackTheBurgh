# PokerSim - interactive hold'em simulator + hand-history replayer

A realistic, playable No-Limit Texas Hold'em table with a browser UI, a replayer for
real recorded hands from the [gabbler/PokerData](https://huggingface.co/datasets/gabbler/PokerData)
Hugging Face dataset, and a harness for testing poker models against those hands.

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

Everything is standard-library Python 3.11+ (3.12 tested) plus vanilla HTML/CSS/JS - no build step.
The optional `anthropic` package enables the Claude agent.

## Quick start

```bash
python -m pokersim serve --open        # http://127.0.0.1:8000/
```

Use `--host 0.0.0.0` to share the table on a LAN. Deep links: `#demo` deals a bot-only table,
`#replay=pluribus-session-30.phhs&i=7&step=12&lens=tag` opens a sample hand at a step with a model
lens, `#replay=<file>&eval=tag` runs an evaluation, `#sim=equity,tag,lag,random&hands=200` a match.

**Table Lab** (`/lab.html`) is a separate page: you and five bots at one table. Above each opponent it shows
the base model's and the RL model's next-move and likely-hand guesses. On the right is a 13×13 likely-hand grid per
opponent. Your panel shows what you should do (base, RL) and what you would do (your own model). Values come from
`GET /api/lab/predict`; the response is documented in `pokersim/lab_api.py`. A model whose files are missing shows
"No model loaded". Filling its field in `lab_api.py` is the only change needed when a model lands.

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

## Poker Mirror (hackathon demo)

The **Mirror** tab asks one question: after accounting for the board and the public betting, how
much does knowing *this* opponent help a bettor decide whether to bluff? Scope: heads-up no-limit
river, the defender checks and faces a half-pot bet. The spec is [PLAN.md](PLAN.md); the order of
work and the build log are [CHECKLIST.md](CHECKLIST.md) and [PROGRESS.md](PROGRESS.md).

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

The demo itself needs only the standard library plus `pokerkit`; everything it shows is read from
`data/mirror/`.

## Layout

```
pokersim/engine.py     game state machine (Hand, Table, legal actions, side pots, showdown)
pokersim/evaluator.py  7-card hand evaluator (+ Omaha, short-deck ordering)
pokersim/phh.py        PHH parser / writer
pokersim/replay.py     replays records -> per-step snapshots + decision points
pokersim/agents.py     Agent interface, bots, HTTP + Claude adapters, spec registry
pokersim/equity.py     Monte-Carlo equity, Chen pre-flop score
pokersim/evaluate.py   shadow evaluation report
pokersim/simulate.py   self-play, PHH export
pokersim/dataset.py    Hugging Face browsing/downloading (cache in data/cache)
pokersim/server.py     HTTP + SSE API, serves web/
web/                   index.html, style.css, app.js (table renderer, panes)
data/samples/          small hand sets from the dataset for offline use
scripts/               validation and smoke tests
tests/                 unit tests (python -m unittest discover tests)
```

## Credits

Hand histories in `data/samples/` are excerpts of the `gabbler/PokerData` dataset (CC-BY-4.0),
itself built from the PHH dataset by Juho Kim (University of Toronto Computer Poker Research
Group), the Pluribus supplementary material (Brown & Sandholm, *Science* 2019), HandHQ logs and
PokerGO/WSOP broadcasts. PHH format: Kim, "Recording and Describing Poker Hands", IEEE CoG 2024.

Poker Mirror scans the [takara-ai/poker_hands](https://huggingface.co/datasets/takara-ai/poker_hands) Parquet
release of the same PHH dataset (CC BY 4.0), pinned at revision `6acb5afb`. The files committed in `data/mirror/`
are small derived artifacts: counts, model parameters, anonymous style-map points and a few demo hands.
