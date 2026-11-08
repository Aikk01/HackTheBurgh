# How we trained Poker Mirror's models, and how we tested them

Poker Mirror uses four models. Three learn from real poker data and one, the RL coach, learns by playing. This page
says what each was trained on, why, and how we checked it works. For more detail see
[06_eval_and_overfitting.md](06_eval_and_overfitting.md) and [TRAIN_PROGRESS.md](TRAIN_PROGRESS.md).

## The data

- **21.6 million real hands** from online poker sites (the public `poker_hands` dataset). We use the 5- and 6-player
  no-limit hold'em hands.
- **10,000 hands that the Pluribus AI played against professionals.** Every player's cards are known in these, which
  makes them the fair test for guessing hands.
- **Online hands only show cards at showdown.** Hands that reach showdown are not typical, so we never judge the hand
  model on them alone: we also test it on the Pluribus hands, where every card is known.
- **Nothing is tested on data it learned from.** We split by player, keep whole files aside for testing, and keep 31 real
  players out of everything: one for the demo replay and 30 for testing the personal model.

## The three models that learn from data

| Model | What it predicts | Trained on | Why |
|---|---|---|---|
| **Next move** ("people in general") | What a player does next: fold, call or raise, and how much | Every decision in the data | A baseline for what people normally do |
| **Hand model** | Each opponent's likely starting hand, as 169 hand types | Hands where the cards were shown | Shows the hidden cards as a range |
| **Personal model** ("this player", "you") | One person's next move | Nothing beforehand: it starts as "people in general" and shifts towards what this person actually does, move by move | People have habits, and a good player notices them |

The personal model uses the method of the "Bayes' Bluff" paper (Southey et al., 2005). Each of a player's moves nudges
it a little, so a few moves change little and a few hundred change a lot.

## The RL coach

**What it does:** suggests your move: fold, call, or raise by one of five sizes (min, ½ pot, pot, 2× pot, all-in).

**What it sees:** 193 numbers about your cards, the board, the pot and bets, positions, **each opponent's predicted
hand range from the hand model**, and each opponent's style (full list below).

**The network:** a small neural network (three layers of 512, 512 and 256 units).

**How it learns: PPO (proximal policy optimisation).** It plays millions of hands, and after each batch it shifts its
choices towards the moves that won chips. We chose PPO because:
- **Pluribus's own method was out of reach.** It took about 12,400 processor-hours of self-play and careful game
  simplifications; we had a few hours.
- **PPO is stable when rewards are noisy.** Poker results swing wildly from hand to hand, and PPO limits how far each
  update can move the policy.
- **It learns a mixed strategy.** It can do different things in the same spot, which poker needs so you are not
  predictable.
- **It runs on ordinary processors** with a small network.

**Who it plays against while training:** at each seat:
- **about 30%:** human-like players made from the next-move model of the professionals in the Pluribus games, each
  nudged to a different style;
- **about 45%:** rule-based bots: a maths bot, a caller, a maniac, a random player, and tight and loose aggressive bots;
- **25%:** earlier copies of itself. Training against data-driven, human-like players is the point: it should learn where people go wrong, not
just play itself.

**Two ways to start:**
1. **From scratch:** random moves at first.
2. **From an imitation of Pluribus:** the network first learns to copy the Base model (which copies Pluribus) until it
   agrees with it 88% of the time, then PPO improves on that.
   - Two of these runs add a **leash**: a penalty for drifting away from the Pluribus copy (weight 0.1 or 0.3), an idea
     from VPT, AlphaStar and piKL.

**Where it ran:** the University of Edinburgh's Eddie cluster (32-processor nodes) and one PC with an RTX 5090, for 1
to 3.5 hours per run.

## The RL policy in detail

**The shipped coach** is run `v3-bc` at update 50 (`data/rl/policy.npz` and `policy.json`).
- **Training:** it started from the Pluribus copy, then trained by PPO for 20 minutes: 1.5 million hands, 3.3 million
  decisions.
- **At the table:** it runs as plain numpy on the laptop, no GPU, in well under a second per move.

**Its 193 inputs:**

| Group | Numbers | What they say |
|---|---|---|
| Your hand | 4 | Strength of your two cards, suited, a pair, the top card |
| Street and position | 10 | Preflop / flop / turn / river; your seat counted from the button |
| The board | 11 | The best hand you have made so far, and flush and straight draws |
| Your chances | 5 | Chance of winning against random hands, and **against the hands the hand model predicts** |
| Money and betting | 23 | Pot, stacks, the price to call, raises so far, pot odds, players still to act |
| Each of the 5 opponents | 28 each | Stack, bet and status; their style as watched at the table; **a 9-number summary of their predicted range** |

**Its outputs:**
- **A probability for each of 7 moves:** fold, check/call, and raise min, ½ pot, pot, 2× pot or all-in.
  - On screen, these are the RL coach's fold/call/raise bar.
  - Its likeliest raise size is tagged on the size buttons.
- **An estimate of how many chips the spot is worth**, used only in training.

**Its reward:** the big blinds won or lost in each hand.

**Training settings** (PPO with GAE):

| Setting | Value |
|---|---|
| Decisions per update | 65,536 |
| Passes over each batch | 4 |
| Minibatch size | 8,192 |
| PPO clip | 0.2 |
| GAE λ | 0.95 |
| Learning rate | 3e-4, lowered over the run |
| Exploration bonus (entropy) | 0.02, lowered over the run (0.005 for the leashed runs) |
| Copies of itself | One saved every 25 updates; they take 25% of opponent seats |

**Every run, and how each did:**

| Run | Start | Predicted hands | Leash | Kept version | Held-out | Sealed test |
|---|---|---|---|---|---|---|
| **v3-bc (shipped)** | Pluribus copy | on | — | update 50 | +150 | **+156 ± 15** |
| v3-bc-s2 | Pluribus copy | on | — | update 100 | +121 | +202 ± 18 |
| v3-bc-kl10 | Pluribus copy | on | 0.1 | update 100 | +123 | +79 ± 12 |
| v3-bc-kl30 | Pluribus copy | on | 0.3 | update 100 | +87 | +45 ± 11 |
| v3-noranges-s2 | scratch | off | — | update 150 | +186 | +115 ± 16 |
| v3-ranges-s2 | scratch | on | — | update 100 | +110 | +75 ± 14 |
| v3-noranges | scratch | off | — | update 300 | +96 | +58 ± 16 |
| v3-ranges | scratch | on | — | update 150 | +94 | +49 ± 17 |
| v2-eddie | scratch; rule-based opponents only | on | — | update 150 | +73 | +18 ± 15 |
| imitation-only | Pluribus copy, no RL | on | — | — | — | +2 ± 10 |

v2-eddie, the one run that never met human-like opponents in training, did worst of the RL runs on the sealed tables.
With a single run that is a hint, not a proof.

**How to run it again:**
```
# Pluribus copy (scripts/bc_init.py: 300,000 decisions), then PPO from it, on Eddie;
# add --kl-bc 0.1 --kl-bc-ref <run>/bc_init.pt for a leash
qsub eddie/rl_bc.sh data/rl/runs/<name> <end HH:MM> [seed] [trainer flags]

# PPO from scratch
python scripts/rl_supervise.py --run-dir data/rl/runs/<name> --hours 3 -- --mix v3 --seed 2

# the sealed test
python scripts/sealed_eval.py --policy <name>=<run>/export_heldout ... --hands 30000
```

## How we tested it

Results are in **big blinds won per 100 hands (bb/100)**, compared with a simple maths bot playing the **same cards
from the same seat**. Sharing the cards removes most of the luck.

**1. Held-out tables, during training.** Every 50 updates each run plays tables it never trained against: Pluribus
copies, plus four rule bots it never met. Each run keeps its best version by this score. Because we pick the best,
this score flatters the winner a little.

**2. Switching inputs off.** In the same test, the coach plays again with the predicted hands blanked out, and again
with the opponent styles blanked out. The difference says what each input is worth.

**3. The sealed test, run once at the end.**
- **Written down before any result.** The opponents, the hand counts and the questions are in the git history.
- **New opponents:** three new rule bots (Bully, Tourist, Ambusher) and human-like bots pushed outside every style
  used in training.
- **What it compared:** every run's best version, plus the Pluribus copy with no RL, all on the same deals, 30,000
  hands per table.
- **It picks nothing:** the shipped coach was not chosen by its sealed score.

## What we found (sealed test)

| Coach | Against the maths bot | Better than the Pluribus copy by |
|---|---|---|
| Pluribus copy, no RL | +2 ± 10 | — |
| **Imitation start + PPO (shipped)** | **+156 ± 15** | **+154 ± 14** |
| Same method, second seed | +202 ± 18 | +200 ± 17 |
| Best run from scratch | +115 ± 16 | |
| Imitation start + leash (0.1 / 0.3) | +79 / +45 | |

- **RL adds a lot to copying Pluribus.** The copy alone only breaks even on new opponents, and PPO turns it into about
  +150.
- **Starting from the copy beats starting from scratch.** Both seeds did better than every run from scratch.
- **The predicted hands help the shipped coach: +82 ± 14** when they are switched off. They help the leashed coaches a
  little, and the coaches trained from scratch not at all.
- **The leash keeps play closer to Pluribus but wins less** against these opponents. Exploiting them paid more.
- **Left alone too long, RL over-fits.** After about 100 updates, both free runs from the copy kept winning more
  against their training bots while losing ground on new ones. Keeping each run's best held-out version avoided this.
- **The opponent-style inputs did not help reliably.** We report that as is.

## What these results do not show

- **Every RL result is against bots, not people.** The bots are built from real players' decisions, but they are
  still bots.
- **30,000 hands per table gives about ±15 bb/100:** enough to separate the shipped coach from the best run from
  scratch (41 ± 16), not to rank runs that are a few points apart.
- **The coach does not adapt during play.** It is a fixed policy: it does not notice an opponent exploiting it (the
  demo's Shark does exactly that). It only reacts through its inputs, each opponent's predicted range and style, and
  the style inputs measured no reliable effect. Training it against table-reading opponents is the next step.
- **The demo table's players are the same kind of bot the coach trained against**, nudged a bit further, so the demo
  is not a test. The sealed test is.
- **Poker sites ban real-time help.** Poker Mirror is a training and analysis tool, not something to use while
  playing online.
