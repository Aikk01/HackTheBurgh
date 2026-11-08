# Training plan: Poker Mirror 6-max demo

This folder is the training plan for the 6-max demo specified in [SUMMARY.md](SUMMARY.md): you against five bots.
At every decision the screen shows what people do, what wins against these opponents, and what you would do.

| File | What it covers |
|---|---|
| [01_resources.md](01_resources.md) | The machines, and which run goes where |
| [02_data_and_base_models.md](02_data_and_base_models.md) | The corpus, the hand model, the next-move model (both trained) |
| [03_rl.md](03_rl.md) | RL: opponents, held-out tables, the two matched runs, the v2 insurance run |
| [04_screen_and_personal_model.md](04_screen_and_personal_model.md) | Screen, decision log, personal model, accuracy score |
| [05_contracts.md](05_contracts.md) | Files, functions and the advice API every part codes against |
| [06_eval_and_overfitting.md](06_eval_and_overfitting.md) | How each model is tested and kept from overfitting |
| [07_action_plan.md](07_action_plan.md) | Timeline, commands, go/no-go checks |
| [08_checkpoints_and_monitoring.md](08_checkpoints_and_monitoring.md) | Supervisor, saved models, alerts, automatic resets |
| [REVIEW.md](REVIEW.md) | What the independent plan review and code review found, and what we did about each point |
| [review_independent.md](review_independent.md) | The independent plan review itself |

## Decisions (2026-11-08)

1. **Four models, each with one job.**
   - The hand model gives every opponent's grid and feeds RL.
   - The next-move model (one network, cards optional) gives Base advice, the opponents' next move, and the prior of
     the personal model.
   - The RL policy gives "what you should do".
   - The personal model predicts you, and each opponent as watched at this table.
2. **Opponent boxes show Base ("people like this") vs "this player"** (the personal model applied to that opponent).
   RL is never put in an opponent's seat.
3. **One 13×13 grid.** Each opponent has a colour, and every cell shows that opponent's probability of holding the
   hand.
   - Shading uses the probability per combination, so offsuit cells (12 combos) don't look 3× hotter than suited
     ones (4 combos).
   - No Base/RL toggle, because RL has no hand predictions of its own.
   - At showdown the true hands are outlined.
4. **The next-move model learns from every decision in the corpus.** Actions are always visible, so there is no
   showdown filter. Cards are used only from the Pluribus logs, where every hole card is known.
5. **RL is tested against human data.**
   - Training opponents: scripted bots, plus "pro" bots cloned from the 13 professionals in the Pluribus logs, plus
     past copies of itself.
   - Model selection: Pluribus-cloned bots and four new rule bots, never seen in training, scored per hand on paired
     deals.
6. **Two matched RL runs** (same seed and opponents), one with the predicted-hand inputs and one without. This
   measures whether predicting hands helps. v2 continues on Eddie as insurance.
7. **Always keep a working demo**, and improve it by swapping model files. Every model file is optional; the screen
   says "No model loaded".

## Status

| Piece | State | File |
|---|---|---|
| Hand model | Trained. Unbiased Pluribus test: +0.31 nats over card counting | `data/rl/hand_range.npz` |
| Next-move model | Trained (3.24M decisions, 3 seeds). Test gain over frequency baselines: +0.12 nats (online, public), +0.40 nats (Pluribus logs, with cards; top-1 81%) | `data/rl/next_move.npz` |
| RL v2 (old opponents) | Stopped at update 56; update-50 export frozen in `data/rl/v2_u50/` | — |
| RL v3 with / without predicted hands | Ready to start (07) | `data/rl/runs/v3-*` |
| Screen, decision log, personal model | To build (04) | — |
