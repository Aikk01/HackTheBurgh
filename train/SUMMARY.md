# What we want to build

You play six-handed no-limit hold'em: you and five bots. At every decision the screen shows two models side by side, plus a model of *you*.

## The screen

- **The table:** you and five opponents.
- **Above each opponent:** a small box with two rows (Base model, RL model) and two columns:
  - **Next move:** the top prediction with its %.
  - **Likely hand:** the top hand with its %.
- **Right side:** a 13×13 grid of likely starting hands for one opponent at a time.
  - Each opponent has their own colour; click a player to see their grid.
  - A toggle switches between the Base and RL model.
  - The top 2 hands are listed with %.
- **Your panel:**
  - **What you should do:** the Base and RL models' top 2 actions with %.
  - **What you would do:** your personal model's prediction of your next move, and how often it has been right, e.g. "31 of 48, vs 26 for always guessing your usual move".
- **Wherever a model is missing:** the screen says **"No model loaded"**.
- **No game-theory lens for now.**

## The models

| Model | Predicts | Learns from | Status |
|---|---|---|---|
| **Base model: hands** | Each opponent's hand over the 169 starting-hand types (the 13×13 grid) | Historical hands (Takara corpus): Pluribus, where every card is known, plus online hands that reached showdown | Code exists (`pokersim/hand_range.py`, `scripts/train_hand_range.py`). The trained file `data/rl/hand_range.npz` is **not pushed**. |
| **Base model: next move** | The next action (fold / check-call / bet-raise and size) | Pluribus's own decisions, using the same extraction pipeline with action labels added | **Not built** |
| **RL model** (PPO) | The next action out of 7 (fold, check/call, min raise, ½ pot, pot, 2× pot, all-in), plus the expected result of the hand | Self-play against bots (`pokersim/rl_train.py`), trained on the RTX 5090 | Code exists. `data/rl/policy.npz` is **not pushed**. It doesn't predict hands itself; it reads the base model's hand guesses as input (193 inputs). |
| **Your model** | Your next move | Your logged decisions. It starts from typical-player behaviour and shifts towards you as data comes in. | **Not built** |
| Style model | Each player's style: how often they play hands, raise, bet, fold to a bet and go to showdown, and which of four types they are | This table's hands, adjusted towards 48k real players | Works today (`pokersim/opponent_model.py`). Feeds the RL model and can seed your model. |

## Data we collect about you

For every decision:
- the situation: street, position, your cards, the board, the pot, the price to call, the betting so far and the opponents' styles;
- what the Base and RL models advised;
- what your model predicted;
- what you actually did.

Saved per player to disk so it survives restarts. Today nothing is saved per decision, and style stats reset with every new table.

## Order of work

1. **Kazutaka (GPU):**
   1. train the hand model → `data/rl/hand_range.npz`;
   2. build and train the base next-move model;
   3. train the RL model **after** step 1, with the 193-input code. A run started before the 193-input commit (`a83e0f5`) won't load.
2. **Us (CPU):**
   - the decision log;
   - your model;
   - the accuracy score;
   - the screen.

## Open decisions

- **Should RL have its own hand guesses?** That means adding a hand output to the network and retraining. Recommended: no, share the base model's grid.
- **Opponents' next-move predictions** must use only what everyone can see. Otherwise they give away the opponent's cards, because the RL model needs a player's own cards to decide. Options: average over the opponent's likely hands, or show them only after the hand ends.
- **When to show "what you would do":** recommended after you act ("predicted call ✓"), so it doesn't influence you.
- **Demo (4–5 minutes):** play 50–100 hands beforehand so your model already has data on you.
