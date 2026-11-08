# Poker Mirror: the deck, slide by slide

Five slides for a 4–5 minute pitch to quant-firm judges (HRT, G-Research). Each slide has a one-line message, the
content, the visual to use, and what to say. Every number is sourced in [SOURCES.md](SOURCES.md); images are in
[img/](img/). Keep the slides sparse: the live demo carries the detail.

---

## Slide 1 · Introduction: what Poker Mirror is, and how it fits the challenge

**Message:** a poker training simulator that makes the hidden visible, with every prediction measured on data it never saw.

**Content**
- **The problem:** poker is hidden information plus people with habits. You never see their cards, and you rarely
  see your own leaks.
- **Poker Mirror:** you play six-handed against the four classic player types (mined from real regulars). The screen
  shows four things:
  - each player's next move and likely hand;
  - their habits in plain words;
  - what a bluff is worth;
  - what a coach trained by reinforcement learning would do.
- **How it fits the tracks:**

| Track | What we show |
|---|---|
| **HRT: Best use of data for predictions and decision making** | Predictions made before every move and checked; a bluff check that prices a decision; an RL coach tested once on sealed tables |
| **Make the invisible visible** | Hidden cards as a live 13×13 range, habits in plain words, a hole-card cam that the models never see, a Shark whose read of you is on screen |
| **G-Research (real time, if last year's brief holds)** | Every model updates move by move at the table |

**Visual:** `img/table_holecard_cam.png` (the table, with every card shown to the audience and tagged "hidden from
the models").

**Say:** "Poker is the purest problem of deciding under hidden information against people. Poker Mirror shows you
what's hidden: their likely cards, their habits, what your bluff is worth. Every number on screen was measured on
data no model saw."

---

## Slide 2 · Motivation: what our data analysis showed, and why we built this

**Message:** the data told us *the situation* explains most of what people do, but *the person* adds a small, real
edge, and that edge matters most against distinctive players, which is exactly what a mirror should show.

**Content: what we found first** (heads-up river decisions in 21.6 million real hands)
- **Folds never reveal cards.** Of 4,156 offers where the defender's cards were later shown, 0 were folds. The data
  silently hides the hands that fold: the first "invisible" thing.
- **The situation predicts about 10× more than the person.** Bet size, board and betting history beat the average
  player by +0.025 nats; the player's own history adds +0.002 on top (157,926 decisions, both with intervals above
  zero).
- **The person still adds a real edge.** +0.0028 nats, 95% CI [+0.0002, +0.0060].
- **Player "types" alone add nothing** once you know the person (the interval spans zero).
- **Pluribus confirms the cards matter.** Weaker hands fold more: average equity is 0.51 for a fold, 0.74 for a call
  and 0.84 for a raise.
- **Exploiting people pays only against distinctive players.** A bot tuned to one opponent wins +8.0 chips per spot
  against a frequent folder and +4.6 against a frequent caller, but loses 2.8 against an average player.

**Why we pursued it:**
1. Model the situation first (people in general).
2. Tilt it per person with a principled Bayesian update.
3. Show the hidden things (ranges, habits, leaks).
4. Test everything on unseen data.

The person-specific edge is largest exactly where a human has a leak, so we built a mirror for leaks: yours and your
opponents'.

**Visual:** a small table of the findings above. Optional: a bar of "situation vs person" information gain.

**Say:** "Our first finding was humbling: the situation explains ten times more than the person. But the person adds
a real edge, and it is biggest against players with a leak. So we built a mirror for leaks."

---

## Slide 3 · Method and sources

**Message:** real data at scale, strict splits, and four models that each answer one question.

**Content**
- **Data:**
  - **Online hands:** 21.6 million from PokerStars, iPoker, PartyPoker, Full Tilt and Absolute (5–6 players,
    no-limit).
  - **Pluribus hands:** the 10,000 the Pluribus AI played against professionals, with every card known. They are
    the unbiased test for hand guessing.
- **Splits:** by player and by whole held-out files. 31 real players are kept out of everything: 30 for testing the
  personal model, 1 for the replay.
- **Models:**

| Model | Question | Learned from |
|---|---|---|
| Next move ("people in general") | What will this player do next? | Every decision in the data |
| Hand model | What are they likely holding (169 types)? | Hands shown at showdown; tested on Pluribus hands |
| Personal model ("this player", "you") | What does *this* person do differently? | Their own moves, live (Bayesian update) |
| RL coach | What should you do? | Millions of simulated hands (PPO) |

- **The table's players:**
  - **Pro:** the next-move model of the Pluribus pros.
  - **Shark, Gambler, Sticky, Rock:** the four classic types, mined from 10,213 online regulars (the most extreme
    15% of each type) and fitted until their VPIP, PFR and AF match. Sticky, for example: 39% / 10% / 1.7 vs real
    42% / 11% / 1.6.
- **Compute:** the University of Edinburgh's Eddie cluster (32-core CPU nodes) and one RTX 5090 PC.

**Visual:** the four-model table, and the data box (21.6M + 10k).

**Say:** "Four models, each answering one question, each tested on players and hands it never saw."

---

## Slide 4 · Technical detail: PPO, the Bayesian personal model, and how we trained and tested

**Message:** each piece is a standard, well-understood method, combined so that each makes the others better, and
tested honestly.

**Content**
- **The RL coach (PPO):**
  - **Inputs:** 193 numbers, including each opponent's predicted range (a 9-number summary from the hand model) and
    their style.
  - **Network:** 512-512-256. **Outputs:** 7 actions (fold, call, and raise min, ½ pot, pot, 2× pot or all-in).
  - **Reward:** big blinds won per hand.
  - **Why PPO:** Pluribus's own method needed about 12,400 core-hours; PPO is stable on noisy rewards and learns a
    mixed strategy on CPUs.
  - **Opponents in training:** 30% human-like (the pros' model with style tilts), 45% rule bots, 25% earlier copies
    of itself.
  - **Imitation start:** the network first copies the Base (Pluribus) model, agreeing with it 88% of the time, then
    PPO improves it.
- **Bayesian personal model** (Bayes' Bluff, Southey et al. 2005):
  - It starts as "people in general".
  - In each situation (street × facing a bet), the player's own fold, call and raise counts pull the prediction
    away from the population, as a Dirichlet update with a prior worth 8 moves.
  - It uses public actions only, never hidden cards.
- **Hand model and width:** a 169-way guess from public actions. Each player's range width is learned by maximum
  likelihood from their showdowns.
- **Bluff check:**
  - Copy the hand and add your bet.
  - Ask each remaining player's own model for their fold chance, and multiply.
  - A two-number calibration, fitted on one simulated session and scored on another.
- **The Shark:** before betting, it compares how often the players would all fold by their own models against
  people in general, and shifts its raising (×2.5, capped). One level of reading, and only it adapts, so there are
  no loops.

**Flow chart: training and evaluation** (render the Mermaid in [SOURCES.md](SOURCES.md) § Flow chart):
1. 21.6M hands + 10k Pluribus hands, split by player and held-out files.
2. Supervised models (next move, hand model), then calibration checks on held-out players.
3. Imitation start (copy Base), then PPO against human-like players, rule bots and self-copies.
4. Every 50 updates: held-out tables (opponents never trained on), keeping each run's best version.
5. A sealed test, registered in git before any result, run once on new opponents.
6. Ship the coach chosen by the pre-registered rule.

**Visual:** the flow chart, plus `img/your_move_bluff_check.png` (coaches and the bluff check).

**Say:** "Standard methods, combined: the coach reads the hand model's predictions, the bluff check reads the
personal models, and everything goes through held-out and sealed tests."

---

## Slide 5 · Findings, and how the approach generalises

**Message:** the predictions are calibrated, the coach is measured on sealed tables, and the same recipe works
wherever decisions depend on reading people under hidden information.

**Content: findings** (all on data or opponents no model was trained on)

| Claim | Result |
|---|---|
| Next move is calibrated | says 75% → happens 76% (670,615 decisions) |
| The person adds to the situation | +1.3 ± 1.2 points of accuracy; sharper probabilities for 28 of 30 unseen players |
| Hand model beats card counting | real hand in our top 10: 16% vs 8% (after the flop 29% vs 5%) |
| Hand zones are honest | likeliest 50 / 80 / 90% hold the real hand 48 / 79 / 90% |
| Bluff check is right | "everyone folds" said 30%, happened 28%; each player said 68%, folded 68% |
| RL coach wins | **+156 ± 15 bb/100** vs a maths bot, same cards (sealed tables, 30,000 hands each) |
| RL adds to copying Pluribus | **+154 ± 14** over the same network without RL |
| Our hand predictions help the coach | **+82 ± 14** when switched off |
| The Shark learns a leak in 50 hands | heads-up bets into a "fit or fold" player 26% → 52% (31% without reading) |

**What surprised us:**
- **RL left alone over-fits.** Two runs from the imitation start peaked early and then fell on new opponents
  (+150 → +38, +121 → −11), while their scores against their own training bots kept rising.
- **A penalty that keeps the coach close to Pluribus stops the drift,** but wins less on the sealed tables.
- **Opponent-style inputs added nothing measurable** (−1 ± 14). We say so.

**Generalisation:** the recipe is (1) a population model of the situation, (2) a Bayesian per-person tilt, (3) a
calibrated decision check built on (1) and (2), and (4) a policy started from expert imitation and tested on sealed
counterparties. It transfers to:
- **trading and market making:** predict a counterparty's next order from the market state, tilt it per counterparty,
  price a quote by its fill probability;
- **negotiation and auctions:** bidders' habits, and what a bid is worth given their likely responses;
- **security and fraud:** an adversary's next move from population behaviour plus their own history;
- **any training tool for human decisions:** show people their own leaks, and an opponent that exploits them.

**Caveats (one line):** RL is tested against bots built from real players, not live people; this is a training and
review tool, not for real-money play.

**Visual:** the findings table, plus `img/shark_end_screen.png` or `img/scorecard.png`.

**Say:** "Calibrated predictions, a coach measured once on sealed tables, and an honest list of what didn't work. The
recipe works anywhere you decide against people you can't fully see."
