# Poker as a market — working notes

Live Bayesian posterior over the opponent's 1,326 hole-card combos in heads-up postflop
pots. The posterior becomes a win-probability **price** for the tracked player, and the
rule is to **call** when price > pot-odds price. Code is in `market/` and does not change `pokersim/`.

```
python -m market.parse          # parse data/cache -> data/processed/hu_flop_hands.pkl
python -m market.evaluate       # train on train tables, evaluate on test tables -> figures/, results.json
python -m pytest tests/test_market.py
```
Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements-market.txt`

## Data
- `gabbler/PokerData` → `handhq/PS-…200NLH_OBFU/2/`: the first 120 of 1,000 files, about 120k
  PokerStars 200NL hands from July 2009 (86 MB, in `data/cache/`, gitignored).
- Kept: 6-seat tables with ≥3 players dealt in, exactly 2 players seeing the flop, and a
  clean engine replay (no negative-blind quirks). That gives **19,930 hands**, of which
  **5,606 reach showdown**. Both hands are known in **2,773**. There are 230 tables and 2,409 obfuscated player IDs.
- Parsing uses `pokersim.replay` (pot, to_call, stacks and positions come from the engine). Each bet
  size is stored as a fraction of the pot before the bet.

## Decisions & assumptions
- **Split by table** (≈ one session): 80/20, seed 0, fixed before any fitting. The test table
  list is in `data/processed/split.json`. Caveat: 63% of test-set players also appear at
  train tables, because players sit at several tables. So this is a session-level split, not a
  player-level one. The model learns population behaviour, not individual players.
- **Evaluator:** eval7 (0.23 µs per 7-card hand, vs 1.0 µs for phevaluator and 1.9 µs for the repo's pure-Python
  evaluator). Bug found: eval7's built-in *range* equity functions ignore weights and
  give wrong exact turn equity. We use only `eval7.evaluate` and do the range maths ourselves.
- **Equity:** for each street we compute hero equity against every combo: exact on the river,
  exact over all river cards on the turn, and 150 seeded (turn, river) samples on the flop.
  Price = posterior · equity vector, so each action costs only a 1,326-length dot product.
- **Prior:** combos are ranked by preflop all-in equity vs a random hand. Each range is the top X%
  with soft edges (τ = 0.03). Flat-callers have the top few % down-weighted to 0.3. The percentages are
  *assumed*, not fitted (`market/prior.py`): opens UTG 14 / HJ 18 / CO 25 / BTN 40 / SB 35%,
  BB call 40%, 3-bettor 6%, limped pots 50%.
- **Likelihood:** two multinomial logistic regressions on combo × context interactions.
  - "Open" (no bet faced): check / bet ≤ 55% pot / bet > 55% or all-in.
  - "Facing": call / raise.
  - Combo features: board-strength percentile (s, s², s³), flush draw, OESD, gutshot. The draw
    flags only count if the hole cards add to the draw.
  - Context features: street, IP, size of the bet faced, pot odds, log SPR, preflop role, 3-bet pot,
    opponent aggression this street.
  - Folds are not modelled. Showdown players never folded, and a fold ends the hand, so no update is needed.
  - The likelihood is floored at L = 0.05 + 0.95·p for robustness. This value is not tuned yet.
- **Training data:** actions of every player whose cards were shown at showdown (train tables only).

## Selection bias (read before quoting numbers)
1. **All evaluation uses showdown hands.** Those are not representative: hands
   that ended in a fold are missing, and so are the bluffs and weak hands that got folded. M3 (EM over
   non-showdown hands) is the attempted fix for training. Evaluation stays showdown-only.
2. **Muck bias, which we found and fixed.** When only one player's cards are known, that player won
   **100%** of the time (2,833 / 2,833), because the loser mucked. The first evaluation included
   those hands and made the uniform baseline look *better* than the model. Evaluation now uses
   **only hands where both players' cards are known**, where the mean outcome is exactly 0.5.
3. The likelihood model's training data still includes one-shown (winning) players. This is a mild
   bias toward strong-hand behaviour. Train on both-shown hands only if we want to test this.

## M1 results (held-out: 46 tables, 551 both-shown hands, 9,792 price events)
95% CIs come from a hand-level cluster bootstrap (1,000 resamples). Lower is better.

| range used            | Brier                   | log-loss              | ECE   |
|-----------------------|-------------------------|-----------------------|-------|
| uniform (baseline)    | 0.1947 [0.1865, 0.2031] | 0.631 [0.612, 0.650]  | 0.085 |
| prior only            | 0.1873 [0.1789, 0.1963] | 0.616 [0.595, 0.639]  | 0.059 |
| **prior + likelihood**| **0.1833 [0.1747, 0.1927]** | **0.606 [0.584, 0.630]** | **0.048** |

- Paired differences: model − uniform Brier = **−0.0114 [−0.0177, −0.0046]**; log-loss −0.025
  [−0.042, −0.006]; model − prior Brier −0.0040 [−0.0075, −0.0005]. All three CIs exclude 0.
- By street (Brier, uniform / prior / model): flop .215/.200/.196, turn .195/.186/.181, river
  .167/.170/.169. **The gain is on the flop and turn. On the river the model is no better than uniform.**
- Action model on held-out rows: log-loss 0.723 vs 0.841 for the class-frequency baseline (open), and 0.479 vs
  0.531 (facing).
- **Decision metric is not informative yet.** On both-shown hands, always calling makes
  +1,506 bb/100 decisions [1,327, 1,699]: every hand reached showdown, and 0.5 equity beats the usual
  pot odds. Model − uniform = −28 bb/100 [−73, +12], not significant. Showdown-only data can't
  measure the cost of a wrong call against a hand that would have made us fold.
- Latency at M1: median 0.7 ms per event, but the flop deal spiked to ~55 ms → the latency test failed. Fixed in M2.
- Figures: `figures/reliability.png`, `figures/decisions.png`. All numbers are in `results.json`.

## M2: speed + websocket
- **`market/fasteval.py`** is a vectorised exact evaluator that returns the same values as `eval7.evaluate` (tested on
  60k random hands and on every combo across 63 boards, including flush-heavy ones). It has no per-combo Python loops.
  - A non-flush value depends only on the rank multiset. All 73,775 multisets of 5, 6 or 7 cards are precomputed, keyed by
    Σ 5^rank, and looked up with `np.searchsorted`.
  - A flush value comes from an 8,192-entry table indexed by the suit's rank bitmask. The result is max(non-flush, flush).
  - The tables build in 0.3 s and are cached in `market/eval_tables.npz`.
  - All runouts × 1,326 combos are evaluated in one call: 150 flop runouts take 3.5 ms (they took ~50 ms before).
- With that headroom, flop equity now uses **300** sampled runouts. Equity per street: flop 6.6 ms,
  turn 0.75 ms, river 0.06 ms.
- **Latency** (all 9,792 held-out events, cold equity cache, `results.json → latency_benchmark_m2`):
  - Compute (Bayes update + likelihood on 1,326 rows + equity + price): median **0.32 ms**, p99 **7.1 ms**, max 22.6 ms.
  - Including JSON serialisation of the full event (~12 KB): median 0.50 ms, p99 7.4 ms, max 23.2 ms.
  - Every event is under the **50 ms** budget, and `test_latency_per_action_under_50ms` passes.
- The full evaluation now runs in 14 s (it took 133 s). Results barely moved with 300 runouts: model − uniform Brier
  −0.0113 [−0.0176, −0.0045]; model 0.1831 / prior 0.1871 / uniform 0.1944; ECE .049 / .060 / .085.
- **Known weakness, seen in live replays:** an opponent's *call* barely narrows the range. The facing-bet model
  can only tell calls from raises, because folds never appear in showdown data. The informative "didn't fold" signal is
  missing, and this is the main motivation for M3.

## Websocket API (`python -m market.server` → http://127.0.0.1:8001; pokersim stays on :8000)
- `GET /api/health`, `GET /api/schema` (this schema as JSON), `GET /api/hands?limit=50`. Hands are from held-out tables only.
- `WS /ws/hand/{hand_id}?tracked=0|1&delay_ms=700&mode=model|prior|uniform` streams one hand.
- `WS /ws/stream?delay_ms=700&mode=model&seed=0` streams random held-out hands without end.
- Messages: one `meta`, then one `event` per action or board deal, then `end`. The opponent's cards appear
  **only** in `end`, so nothing leaks mid-hand (tested).

| message | fields |
|---|---|
| `meta` | `hand_id`, `big_blind`, `tracked {position, cards}`, `opponent {position, preflop_role}`, `pre_raises`, `combo_labels[1326]` (index order of every posterior, e.g. `"AsKh"`), `grid_labels[13][13]` (`"AKs"`; row/col 0 = Ace; suited above the diagonal, offsuit below) |
| `event` | `seq`, `mode`, `hand_id`, `street`, `board[]`, `tracked`, `action` (null on a board deal, else `{actor: tracked\|opponent, kind, size_frac_pot, pot, to_call}`), **`price`** (tracked player's win prob, ties = ½), **`pot_odds_price`** (to_call / (pot + to_call); only when the tracked player faces a bet), **`decision`** (`call`/`fold`/null; emitted *before* the tracked player's real action), `latency_ms`, **`posterior[1326]`** (sums to 1), `grid[13][13]` (posterior mass per hand class), `top_combos[8] {combo, p}`, `effective_combos` (exp entropy: 1326 = uniform) |
| `end` | `hand_id`, `final_board[]`, `opponent_cards` (null if mucked), `tracked_share` (1 / 0.5 / 0) |

## Next
- Tune EPS / C with cross-validation **on train tables only**. Try LightGBM if logistic regression underfits the river.
- M3: EM over non-showdown hands. Folds become learnable, which should make calls informative.

## Integration into Poker Mirror (branch `gg-ez-poker`, built on `train-v3`)
- `market/live.py` runs the market model on a live 6-max table: for every opponent still in the hand it rebuilds a
  1,326-combo posterior from the public action history on each `/api/lab/predict` call, plus the viewer's win
  probability ("price") against that posterior once the flop is out. Card removal uses the viewer's cards and the board.
- **Approximation:** the model was trained on heads-up postflop pots. At a full table each opponent is treated as if
  heads-up against the field: the context of each of their actions (bet faced, pot odds, SPR, position, preflop role,
  aggression this street) is read off the shared history, and "in position" means last to act among the players
  still in. The numbers on the table are therefore demo-grade, not the evaluated heads-up setting in `results.json`.
- Wiring: `pokersim/lab_api.py` adds `"market": {"hands", "price", "effective_combos", "updates"}` to every player
  row and `"market": {"hands": bool}` to the model status; nothing else in `pokersim/` changed. The shipped model is
  `market/model/likelihood.pkl` (the M1/M2 fit, train tables only). Extra deps: `requirements-market.txt`.
- Page: `web/lab.html` is "gg ez poker" on a black page (the felt is the only green); each opponent has two 13x13
  grids beside their plate, **Base** (their hand model, seat colour) and **Live ML** (ours, cyan), with the top two
  hand types and "you win X%" under Live ML. The old right-hand panel is behind the **Details** button.
- A predict call costs ~0.6 s in total; the market part of it is ~5 ms per opponent.
