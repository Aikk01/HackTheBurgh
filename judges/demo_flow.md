# What the team shows a judge (4–5 minutes)

Shared by every mock judge in this folder. The judge watches; the presenter drives.

1. **The table (about 1 minute).** "You" plus five players, each a model of real human professionals nudged into one
   of the four classic types: Pro and Shark (tight aggressive), Gambler (loose aggressive), Sticky (loose passive),
   Rock (tight passive). The presenter deals and plays a hand.
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
4. **"A player like you" (about 1 minute).** In about 10 seconds, a simulated human-like player who calls a lot plays
   500 hands in the presenter's seat, and every model starts knowing nothing: the model of you and the models of the
   five players. Over the last 100 hands the model of you is right 81% of the time against 70% for people in general,
   and the models of the players 72% against 68%. "Now you play": the presenter takes the seat with all of them
   carried over and plays like that player.
5. **"How good are the guesses?" (about 30 s).** Every number next to its baseline, on data no model was trained on:
   next move, hand grid, bluff check, and the RL coach on sealed tables.

Then one or two questions.
