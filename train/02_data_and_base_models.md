# 02 · Data and base models

## Corpus

- **Source:** the Takara poker_hands corpus, 37 Parquet shards, 21.6M hands, at a pinned revision.
  `scripts/fetch_takara.py` downloads it in about a minute.
- **Pluribus logs:** 10,000 six-handed hands, every hole card known (13 professionals + Pluribus).
- **Online hands (HandHQ 2009):** actions are always known; hole cards only for players who reached showdown.

## Hand model (trained, `data/rl/hand_range.npz`)

- **What it predicts:** P(opponent holds each of the 169 hand classes), given what the table can see.
- **Model:** 169-class softmax made of:
  - a context network;
  - a bilinear term between the context and each class's connection with the board;
  - `log(live combos)`, so dead cards are respected exactly.
- **Data:** Pluribus logs plus online showdown rows (2.32M rows).
- **Results** (nats over card counting):

| Test set | Gain | Note |
|---|---|---|
| Pluribus logs | +0.31 | **The honest figure** |
| Pluribus logs, flop to river | +0.89 | — |
| Pluribus logs, preflop | +0.21 | — |
| All rows | +0.63 | Mostly online showdown rows, so biased towards strong hands |

- **Known weaknesses:**
  - Validation was a random 5% of rows, so rows from the same hand land on both sides.
  - The online rows are showdown-biased.
- **We do not retrain it tonight.** RL reads its outputs, and changing it would change what those inputs mean.

## Next-move model (trained, `data/rl/next_move.npz`)

- **Code:** `pokersim/next_move.py` (NumPy inference) and `scripts/train_next_move.py` (extract and train).
- **What it predicts:** P(next action) over the RL's 7 classes: fold, check/call, min raise, ½ pot, pot, 2× pot,
  all-in. Illegal actions are masked.

**Data:** every decision of 5–6-player no-limit hands.
- Online: 15,000 random hands per shard, **no showdown filter**.
- The Pluribus logs.
- 3,235,418 decisions in total. Folds are 53% of the labels, as in real play.

**Inputs:**
- The player's public situation:
  - the 51 context features shared with the hand model;
  - price to call, stack-to-pot ratio, players left, raises this street.
- Optionally the player's own cards: hand class, made hand, draws, preflop strength. Used only with Pluribus-log rows,
  because online rows with cards come only from showdowns and contain no folds.
- Flags: `has_cards`, `is_pluribus`, `from_logs`.

**Uses:**

| Mode | Used for |
|---|---|
| Public (no cards) | An opponent's next move (Base row); the prior of "this player" and of your model |
| With cards, `pluribus=1` | "What you should do", Base row (plays like Pluribus) |
| With cards, `pluribus=0` | The "pro" bots that RL trains against |

**Splits:**
- Online rows by player (80/10/10); shards 34–36 are test only.
- Pluribus logs by hand (20% test, 10% validation).
- Early stopping on validation; 3 seeds; the median seed is shipped.

**Test results** (`next_move.json`):

| Test set | Rows | Model nats | Frequency baseline | Gain | Top-1 |
|---|---|---|---|---|---|
| Online, public | 670,615 | 0.855 | 0.975 | **+0.120 ± 0.001** | 69% |
| Pluribus logs, public | 18,439 | 0.774 | 1.044 | **+0.270 ± 0.011** | 70% |
| Pluribus logs, with cards | 18,439 | 0.472 | 0.870 | **+0.398 ± 0.011** | 81% |
| Pluribus's seat, with cards | 3,073 | 0.470 | 0.873 | **+0.403 ± 0.028** | 82% |

- **Baselines:** action frequencies by street × facing a bet × position, plus a strength bucket for the with-cards
  rows.
- **NumPy port:** matches PyTorch to 3e-7.
- **Style input:** dropped. The corpus has no per-player history at decision time, so the network could not learn
  from it. Adapting to a particular player is the personal model's job (04).
