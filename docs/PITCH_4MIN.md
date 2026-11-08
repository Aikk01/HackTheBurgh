# gg ez poker — the 4-minute pitch, four speakers

Target: 3:50 spoken, 10 s spare. About 140 words per speaker. The live table is on screen the whole time
(`python -m pokersim serve --port 8765`, then http://localhost:8765, a hand already dealt to the flop with the
hole-card cam **off**). Speaker 3 drives the mouse. Every number below is in `DEMO.md` § Numbers to quote,
`docs/slides/SOURCES.md` or `NOTES-market.md`.

---

## Speaker 1 · 0:00–0:55 · What it is

*(Stand by the screen. Point at the table.)*

Poker is the purest problem of deciding under hidden information against people. You never see their cards, and
you rarely see your own leaks.

This is **gg ez poker**. You sit at a six-handed table against the four classic types of player, mined from ten
thousand real online regulars. Everything you'd normally have to guess is on the screen:

- above each player, what they'll do next;
- beside each player, the cards they're likely holding, from two different models;
- at the bottom, what a coach trained by reinforcement learning would do, and what your bluff is worth.

Nothing on screen ever sees a hidden card. And every number we're about to show you was measured on hands and
players no model was trained on.

*(Hand over: "Here's what's under it.")*

## Speaker 2 · 0:55–1:55 · The models and the evidence

We trained on **21.6 million real hands** from five poker sites, plus the ten thousand hands the Pluribus AI played
against professionals, where every card is known. We split by player, kept thirty-one real players out of everything,
and tested on them.

Four models, one question each:

- **Next move:** what will this player do? When it says 75%, it happens 76% of the time, over 670,000 decisions.
- **Hand model:** what are they holding? The real hand is in our top ten guesses 16% of the time; card counting
  alone gets 8%. After the flop, 29% versus 5%.
- **Personal model:** how does *this* person differ? A Bayesian tilt from their own moves. Sharper for 28 of 30
  unseen players.
- **RL coach:** what should you do? Trained with PPO, tested *once* on sealed tables we registered before seeing
  any result: **plus 156 big blinds per hundred hands** against a maths bot, same cards.

*(Hand over: "And then we added a market.")*

## Speaker 3 · 1:55–2:55 · Live ML: poker as a market

*(Mouse: click the cyan **Live ML** grid beside the player who just bet. The big view opens: Base on the left,
Live ML on the right. Hover one square.)*

Every bet in poker is an order. A big bet on the turn tells you something about their cards, the same way a big
buy order tells you something about a stock.

So we built a second, real-time model. It holds a probability for **all 1,326 two-card hands** an opponent could
have, and after every action it does a Bayesian update: a model trained on showdown hands says how likely each
hand was to make that move, and all 1,326 get reweighted. *(Point.)* Same hand, same square in both grids, and you
can see where the two models disagree.

From that range we get a **price**: your chance of winning, right now. *(Point at "you win".)* Price above the pot
odds, call; below, fold.

Is the price honest? On held-out tables it's better calibrated than assuming a random hand: Brier score 0.183 against
0.194, with a bootstrap interval that excludes zero. And it's fast: a typical update takes **0.3 milliseconds**.

*(Close the view. Hand over.)*

## Speaker 4 · 2:55–3:50 · What we got wrong, and where this goes

Three things we found the hard way:

- **Folds never reveal cards.** Of 4,156 bets where the defender's cards were later shown, zero were folds. The
  data hides exactly the hands that fold.
- **Only winners show.** When one player shows at showdown, they won 2,833 times out of 2,833, because losers muck.
  Our first evaluation used those hands and made our model look worse than the baseline. We now test only on hands
  where both players showed.
- **Reinforcement learning left alone overfits.** Two runs peaked early and fell on new opponents. We say so.

The recipe is general: a population model of the situation, a Bayesian tilt per person, a calibrated price built
on both, and a policy tested on counterparties it never saw. That's a market maker reading order flow. That's fraud
detection. That's any decision against people you can't fully see.

Poker Mirror shows you the hidden. The market prices it. Thank you.

---

## Timing and backup

| Speaker | Ends at | If running long, cut |
|---|---|---|
| 1 | 0:55 | the three bullets → one sentence: "what they'll do, what they hold, what you should do" |
| 2 | 1:55 | the personal-model bullet |
| 3 | 2:55 | the calibration sentence (keep the 0.3 ms) |
| 4 | 3:50 | the RL overfitting bullet |

- Rehearse the hand-overs; they cost 5 s each if fumbled.
- Before you start: a hand on the flop with at least two bots still in, so Speaker 3 has a cyan grid to click.
  If the hand ends early, press **N** for the next hand while Speaker 2 talks.
- Likely questions: see `DEMO.md` § Questions the judges asked. For the market model: "heads-up only?" — yes,
  trained on heads-up postflop pots; at the six-handed table each opponent is treated as heads-up against the
  field, which we call out as an approximation.
- Do not claim the call/fold rule makes money. On showdown-only data the difference is not significant.
