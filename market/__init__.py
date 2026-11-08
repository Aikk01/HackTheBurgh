"""market - "poker as a market".

A Bayesian posterior over the opponent's 1,326 hole-card combos, updated action by
action, turned into a live win-probability "price" for the tracked player and a
call/fold decision against the pot-odds price.  See NOTES.md.

Card encoding is shared with ``pokersim.cards``: ``rank * 4 + suit`` (0 = 2c ... 51 = As).
"""
