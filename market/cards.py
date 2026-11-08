"""The 1,326-combo universe, card-removal masks and fast rank tables.

Combos are indexed 0..1325 in lexicographic order of (c1, c2) with c1 < c2, using the
``pokersim.cards`` encoding (card = rank * 4 + suit).
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import eval7
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
N_COMBOS = 1326

COMBOS = np.array(list(combinations(range(52), 2)), dtype=np.int16)          # (1326, 2)
CARD_MASK = np.zeros((N_COMBOS, 52), dtype=bool)                               # combo uses card?
CARD_MASK[np.arange(N_COMBOS), COMBOS[:, 0]] = True
CARD_MASK[np.arange(N_COMBOS), COMBOS[:, 1]] = True
COMBO_INDEX = {(int(a), int(b)): i for i, (a, b) in enumerate(COMBOS)}
COMBO_RANKS = COMBOS >> 2                                                      # (1326, 2)
COMBO_SUITS = COMBOS & 3

_E7 = [eval7.Card(RANKS[c >> 2] + SUITS[c & 3]) for c in range(52)]
_E7_COMBOS = [(_E7[a], _E7[b]) for a, b in COMBOS.tolist()]


def card_from_str(s: str) -> int:
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def card_str(c: int) -> str:
    return RANKS[c >> 2] + SUITS[c & 3]


def combo_index(a: int, b: int) -> int:
    return COMBO_INDEX[(a, b) if a < b else (b, a)]


def combo_str(i: int) -> str:
    a, b = COMBOS[i]
    return card_str(int(b)) + card_str(int(a))


def available(dead: Iterable[int]) -> np.ndarray:
    """Bool mask of combos that use none of the ``dead`` cards (card removal / blockers)."""
    dead = list(dead)
    if not dead:
        return np.ones(N_COMBOS, dtype=bool)
    return ~CARD_MASK[:, dead].any(axis=1)


def board_ranks(board: Sequence[int]) -> np.ndarray:
    """eval7 hand value (higher is better) of every combo + ``board`` (3-5 cards).

    Combos that collide with the board get -1.  Vectorised table lookup (market.fasteval).
    """
    from .fasteval import combo_board_values
    return combo_board_values(np.asarray([board]))[0]


def board_ranks_reference(board: Sequence[int]) -> np.ndarray:
    """Slow reference implementation (one eval7 call per combo) - used by the tests."""
    b = [_E7[c] for c in board]
    out = np.empty(N_COMBOS, dtype=np.int64)
    ev = eval7.evaluate
    for i, (x, y) in enumerate(_E7_COMBOS):
        out[i] = ev([x, y, *b])
    out[~available(board)] = -1
    return out


def hand_value(hole: Sequence[int], board: Sequence[int]) -> int:
    return eval7.evaluate([_E7[c] for c in (*hole, *board)])
