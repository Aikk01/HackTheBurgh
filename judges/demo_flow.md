# What the team shows a judge (4–5 minutes)

Shared by every mock judge in this folder. The judge watches; the presenter drives.

1. **The table (about 1 minute).** "You" plus five players, each a model of real human professionals nudged into one
   of the four classic types and named after it: Tight Aggressive, Loose Aggressive, Loose Passive, Tight Passive,
   and the Shark (tight aggressive, reads the table). The presenter deals and plays a hand.
   - Above each opponent: the predicted next move. A second row, "This player", appears only when the model of that
     player disagrees with people in general.
   - Right panel: the opponent's likely starting hands on a 13×13 grid. It changes as they act.
   - Centre: one line per move. Did we read it (✓/✗), and how much did it move our guess?
   - "Hole-card cam" (header toggle): the audience sees every player's cards, see-through and tagged "hidden from the
     models"; the grid outlines the real hand live. Display only.
   - The Shark reads the table: before its move it asks the players' models how likely they all fold to a pot-sized
     bet against people in general, and bluffs more or less. Its seat shows the read.
2. **A decision (about 1 minute).**
   - Two coaches as fold/call/raise bars: Base copies Pluribus, RL was trained against players like these. When
     they disagree, a note says why.
   - Size buttons, and the **bluff check**: "if you bet X, they all fold Y% of the time, it needs Z%".
3. **Showdown (about 20 s).** Where the real hand sat in our guess ("our #12 guess of 169").
   - If the presenter went against the RL coach, the **"If you had followed the coach"** box appears until the next
     hand. It shows what each move would have won from that spot (same cards, 150 replays each, the coach finishing
     the hand) and what the presenter's move cost against the coach's own mix, with ±, or "too close to call".
4. **"The Shark learns you" (about 1 minute, a 30-second replay).**
   - A player with a common leak ("fit or fold": gives up to almost any bet after the flop) sits in the presenter's
     seat for 60 hands.
   - The Shark reads every player's model before it bets. At first its seat says "plays its normal game"; by hand 50
     the model of you shows the leak (folds 86% to a bet after the flop, people 52%), and its seat says "bluffs more".
   - Measured over 100 sittings, with and without its read: it bets into you heads-up 26% → 52% (31% when not
     reading).
   - "Now you play": the Shark keeps its read until you change.
5. **"How good are the guesses?" (about 30 s).** Every number next to its baseline, on data no model was trained on:
   next move, hand grid, bluff check, and the RL coach on sealed tables.

Then one or two questions.
