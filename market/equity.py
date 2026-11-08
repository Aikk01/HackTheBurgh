"""Hero equity against every one of the 1,326 opponent combos on a given board.

``equity_vector(hero, board)`` returns e (1326,) where e[i] = P(hero wins | opponent holds
combo i), ties counted as 1/2, averaged over the remaining runout:

  * river: exact (one comparison)
  * turn:  exact enumeration of all river cards
  * flop:  ``n_runouts`` (300) sampled (turn, river) pairs, seeded -> reproducible

The board changes only on a deal, so e is computed once per street; every action on that
street just reweights it:  price = posterior . e  (a 1,326-length dot product).
Combos blocked by hero/board get e = 0 and must carry zero posterior weight.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from .cards import N_COMBOS, available
from .fasteval import combo_board_values, eval_sets

FLOP_RUNOUTS = 300


def equity_vector(hero: Sequence[int], board: Sequence[int], n_runouts: int = FLOP_RUNOUTS,
                  seed: int = 0) -> np.ndarray:
    """Fully vectorised: all runouts x all 1,326 combos evaluated in one table lookup."""
    hero, board = list(hero), list(board)
    dead = hero + board
    deck = np.array([c for c in range(52) if c not in dead], dtype=np.int64)
    need = 5 - len(board)
    rng = np.random.default_rng(seed)
    if need == 0:
        runouts = np.zeros((1, 0), dtype=np.int64)
    elif need == 1:
        runouts = deck[:, None]
    elif need == 2:
        i, j = np.triu_indices(len(deck), k=1)
        pick = rng.choice(len(i), size=min(n_runouts, len(i)), replace=False)
        runouts = np.column_stack([deck[i[pick]], deck[j[pick]]])
    else:  # preflop / partial boards (tests only): plain Monte Carlo
        runouts = np.array([rng.choice(deck, size=need, replace=False) for _ in range(n_runouts)])
    boards = np.hstack([np.broadcast_to(np.array(board, dtype=np.int64), (len(runouts), len(board))), runouts])
    opp = combo_board_values(boards)                                         # (R, 1326), -1 = blocked
    me = eval_sets(np.hstack([boards, np.broadcast_to(np.array(hero), (len(boards), 2))]))[:, None]
    valid = opp >= 0
    win = np.where(valid, (me > opp) + 0.5 * (me == opp), 0.0)
    den = valid.sum(0)
    out = np.divide(win.sum(0), den, out=np.zeros(N_COMBOS), where=den > 0)
    return out * available(dead)


def equity(hero, board, weights, **kw) -> float:
    """Hero equity vs a weighted range (weights need not be normalised)."""
    e = equity_vector(hero, board, **kw)
    w = weights * available(list(hero) + list(board))
    return float(w @ e / w.sum())


@lru_cache(maxsize=256)
def _cached(hero: tuple, board: tuple, n_runouts: int, seed: int) -> np.ndarray:
    e = equity_vector(list(hero), list(board), n_runouts=n_runouts, seed=seed)
    e.flags.writeable = False
    return e


def equity_vector_cached(hero, board, n_runouts: int = FLOP_RUNOUTS, seed: int = 0) -> np.ndarray:
    return _cached(tuple(hero), tuple(board), n_runouts, seed)
