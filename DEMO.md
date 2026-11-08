# Poker Mirror: the demo, for the presenter

Poker Mirror is a **training simulator that holds up a mirror**. You play against the four classic types of player, it
learns everyone's habits (yours included) and shows them, it prices your bluffs, and a coach trained by reinforcement
learning suggests a better move. Nothing on screen ever sees a hidden card.

Run it: `python -m pokersim serve --port 8765`, then open http://localhost:8765/.

## What is on the screen

| Where | What it shows | How it is made |
|---|---|---|
| Each seat | The player's name and type (tight or loose, aggressive or passive), and a small fold / call / raise bar for their next move | The next-move model (21.6M real hands). A second bar, "This player", appears when their own model disagrees with people in general |
| Middle of the table | One line per move: did we read it (✓/✗), and how much it moved our guess of their hand | |
| Right panel | The selected player's likely starting hands, 13×13; taller cells are likelier | The hand model, from public actions and the board |
| Right panel | Their habit in plain words ("calls 2.3× as often as most people facing a bet") | Their own model, learned at this table |
| Bottom left | Two coaches: **Base** copies Pluribus; **RL** was trained against players like these | |
| Bottom middle | The model of **you**: its guess of your move (shown after you act), how often it has been right, your habit | |
| Bottom right | Bet-size buttons and the **bluff check**: "if you raise to X with nothing: everyone folds Y%, needed to break even Z%, so a bluff wins or loses N chips" | Each player's own model, on a copy of the hand with your bet in |

**The players:**
- **Pro** plays like the human professionals of the Pluribus games (tight aggressive).
- The other four are the four classic types, **mined from 10,213 real online regulars**. The core of each type was kept:
  - **Shark:** tight aggressive (1,395 players);
  - **Gambler:** loose aggressive (237);
  - **Sticky:** loose passive (1,949);
  - **Rock:** tight passive (350).
- Each plays its cards through the pros' model, with its type's real habits in each of 24 situations (street, facing a bet or not, position). Fitted until their stats match:

  | Bot | VPIP | PFR | AF |
  |---|---|---|---|
  | Shark | 19% (real 17%) | 16% (real 14%) | 4.2 (real 3.9) |
  | Sticky | 34% (real 36%) | 12% (real 12%) | 1.9 (real 1.8) |

**The Shark reads the table.** Before each move it asks the players' models how likely they all are to fold if it bets
about the pot, compared with people in general. If they fold more, it bluffs more; if they call more, it bluffs less.
Its seat says so: "Reads the table: they fold 60% (people 45%) → bluffs more".
- It reads what players *do*, never what they think it will do.
- Only the Shark adapts, and its shift is capped, so there are no loops.

**Hole-card cam** (header toggle, like on TV): shows every player's cards to the audience, see-through and tagged "hidden
from the models". The grid outlines the real hand live ("our #3 guess of 169, one of our top guesses"). It changes the
display only. Turn it on for judges, off for real play.

## The 4-minute flow

1. **Table (1 min).** Deal and play a hand with the hole-card cam on. Point at one read: "Rock: Fold 90%, and you can see
   he holds 7♣2♦."
2. **A decision (1 min).** Click a bet size and read the bluff check aloud. If the coaches disagree, read the note.
3. **"Watch the model learn a player" → "The Shark learns you" (1 min, about 10 s of replay).**
   - Your seat has a common human leak: it gives up to almost any bet after the flop. Nobody is told.
   - The first hands: the Shark's seat says "plays its normal game", and its read of you is people in general.
   - The fast-forward to hand 50: the model of you learns the leak ("you fold 86% to a bet after the flop; people
     52%").
   - The next hands: the Shark's seat says "bluffs more"; it bets, you fold.
   - The end screen:
     - you fold 86%;
     - the Shark bets into you heads-up 26% → 52% (hands 1–10 → 51–60), against 31% for the same Shark not reading
       you;
     - measured over 100 sittings of 60 hands, each with and without its read;
     - no chip loss is claimed (10 hands is too few).
   - **Now you play:** the Shark keeps its read of you until you change.
   - The older "A player like you" replay (the learning curve) is the second option on the same card.
4. **(Optional) "A player like you" (1 min).**
   - A simulated player who calls a lot plays 500 hands in your seat, in about 10 seconds, while every model starts
     knowing nothing.
   - It ends on: model of you **78%** vs people in general **68%** (last 100 hands). The five players' models match
     people in general (70% vs 70%): realistic regulars differ in how often they do things, not in their most likely
     move.
   - Press **Now you play**: all those models carry over. Call a lot, and watch it predict you.
5. **"How good are the guesses?" (1 min).** The strongest screen: lead with it and do not rush it. Every number sits next to its baseline,
   on data no model was trained on.

Both mock judges said the same thing: **lead with the scorecard and the bluff check, not the table.**

## Numbers to quote

| Claim | Number | Tested on |
|---|---|---|
| Next move is calibrated | says 75% → happens 76% | 670,615 decisions by unseen online players |
| A player's own model beats people in general | +1.3 ± 1.2 points; sharper probabilities for 28 of 30 players | 30 real players no model saw (17,400 decisions) |
| Hand model beats card counting | real hand in our top 10: 16% vs 8% (after the flop 29% vs 5%) | 44,036 guesses on unseen Pluribus hands, every card known |
| Hand zones are honest | likeliest 50 / 80 / 90% hold it 48 / 79 / 90% of the time | this table's players, fresh hands |
| Bluff check is right | "everyone folds" said 30%, happened 28%; each player said 68%, folded 68% | 871 bets really made at this table, a session it was not fitted on |
| RL coach wins | **+156 ± 15** bb/100 against a maths bot, same cards | sealed tables: opponents it never trained on or was picked on, 30,000 hands each |
| RL adds to copying Pluribus | **+154 ± 14** over the same network copying Pluribus without RL | same |
| Our hand predictions help the coach | **+82 ± 14** when switched off | same |

## The Shark, measured

The Shark reacts to the gap between how often the players still in would fold by their own models and how often people
in general would, multiplied by 2.5 and capped at ±2 on its raise logit, so it pounces on early evidence. With the
leaky player above, over 100 sittings of 60 hands (`scripts/build_sim_shark.py`):

| Hands 51–60 | Shark reading you | Same Shark, not reading |
|---|---|---|
| Bets when heads-up with you after the flop | 52% (46 spots); 26% in hands 1–10 | 31% (61 spots) |
| You fold to its bets | 83% | 70% |
| Chips you lose to it per sitting | 3.0 ± 1.4 | 3.2 ± 1.3 (no measurable difference in 10 hands) |

With a milder leak, or with the Shark reacting one-for-one to the read, the difference was within noise in 60 hands.
That is why the Shark pounces, and why the claim is about what it does, not about chips.

## Scenarios worth showing (pick one, about 30 s)

- **The Shark reads you.** Fold to a few bets. The Shark's seat flips to "bluffs more", and your habit row shows the
  leak it found. Then call it down.
- **The mirror catches the Shark.** As it bluffs more, its "This player" bar drifts away from people in general.

## Questions the judges asked, and our answers

- **"Where does modelling one opponent make money?"** In the bluff check: its fold chances come from each player's own
  model, and they are calibrated (said 68%, folded 68%). The RL coach's opponent-style inputs add nothing measurable
  (−1 ± 14), and we say so on the scorecard.
- **"Is 78% vs 68% from one simulated player?"** Yes, a clearly loose one, to show the mechanism. On 30 real players the
  gain is small but real: +1.3 ± 1.2 points.
- **"Does the coach notice it is being exploited?"** No. The coach is a fixed policy: it does not learn during play.
  - Passively, it randomises its moves and reads each opponent's predicted range.
  - Actively, *you* adapt: the mirror shows the leak and the Shark's read.
  - Training it against table-reading opponents is the next step; we kept it out to avoid two adaptive players chasing
    each other.
- **"Why PPO and not Pluribus's method?"** Pluribus took about 12,400 processor-hours of self-play; we had hours. PPO is
  stable on noisy rewards and learns a mixed strategy. Starting it from a copy of Pluribus and training it against
  human-like players is what made it work. Details: [train/TRAINING.md](train/TRAINING.md).

## What we do not claim

- **The RL coach is tested against bots, not people.** The bots are built from real players' decisions, but they are
  still bots.
- **The demo table is not a test.** The sealed test is.
- **No use while playing online.** Real-time assistance is banned on poker sites. This is for practice and review.
