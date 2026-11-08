# 04 · Screen, decision log and personal model

Built in the repo while the RL runs train. This is code, not compute. The Mac only runs the finished demo.

## Decision log

- **Where:** in `pokersim/server.py` (`/api/game/act`), one JSON line per human decision.
- **File:** `data/players/<player_id>/decisions.jsonl` (git-ignored).
- **Fields:**
  - the observation;
  - the opponents' styles;
  - what Base and RL advised (top 2 with %);
  - the personal model's prediction, made before the action;
  - the action, in the 7 classes plus the raw engine action.
- **Persistence:** style counts are saved only for human players. Bots are warmed up with the existing `observe()`
  (bot names are reused across tables).

## Personal model (`pokersim/you_model.py`), one code path for you and for every opponent

- **Prior:** the next-move model's public prediction (no cards) for that player in the current situation.
- **Adjustment** per situation bucket (street × facing a bet) and action group (fold / call / raise), using Dirichlet
  shrinkage:
  `p(a) ∝ p_prior(a) · [(n_g + α·q_g) / (N + α)] / q_g`.
  - `q_g` is the prior's share of group g in that bucket.
  - `n_g / N` are the player's own counts.
  - With little data it stays at the prior; with more it moves to the player's real frequencies.
  - α ≈ 5, tuned on the rehearsal log.
- **Uses:**
  - **You:** "What you would do", revealed after you act, with ✓ / ✗.
  - **Each opponent:** the "this player" row. It updates from every action they make, so it also gets a live score.
- **Score:** right / total against "always your usual move in this bucket", both computed before the action.
  - At about 50 decisions the uncertainty is ±7 points.
  - The screen shows the counts and makes no claim beyond them.

## Screen (Play tab: `web/app.js`, `web/index.html`, `web/style.css`)

- **Opponent box:** two rows × two columns.
  - Rows: Base ("people like this") and This player (personal model).
  - Columns: Next move (top action with %) and Likely hand.
  - Likely hand shows the range summary, e.g. "pair+ 35% · draw 20% · nothing 45%", with the top hand next to it.
- **The grid:** one 13×13 grid with every opponent in their own colour.
  - Each cell takes the colour of the opponent most likely to hold that hand; the shade strength is that probability
    per combination.
  - Hover a cell: every opponent's %.
  - Click an opponent: isolate them, with their top 2 hands.
  - Showdown: the real hands are outlined in their owner's colour.
- **Your panel:**
  - What you should do: Base (Pluribus mode, with your cards) and RL, top 2 actions with %.
  - The RL row also shows the expected result of the hand, labelled as an estimate (bb).
  - What you would do: shown after you act.
- **"No model loaded"** wherever a field is `null`.
- **Demo table:** stacks reset to 100 bb every hand, because RL was trained on fresh stacks. Opponents come from RL's
  training mix.
