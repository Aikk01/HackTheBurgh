"""Vectorised exact hand evaluation (same values as ``eval7.evaluate``), no per-combo loops.

For 5-7 cards the best hand is max(non-flush value, flush value):
  * non-flush value depends only on the rank multiset -> precomputed for all 73,775 multisets
    of size 5/6/7 (<= 4 per rank), keyed by sum(5**rank), looked up with ``np.searchsorted``.
  * flush value depends only on the 13-bit rank mask of the flush suit (>= 5 cards of a suit)
    -> table of 8,192 entries.
Tables are built once with eval7 (~0.3 s) and cached to ``market/eval_tables.npz``.

``combo_board_values(boards)`` evaluates all 1,326 combos on R boards at once -> (R, 1326).
"""
from __future__ import annotations

import os
from itertools import combinations_with_replacement

import eval7
import numpy as np

from .cards import CARD_MASK, COMBO_RANKS, COMBO_SUITS, N_COMBOS, RANKS

POW5 = 5 ** np.arange(13, dtype=np.int64)
_PATH = os.path.join(os.path.dirname(__file__), "eval_tables.npz")


def _build():
    keys, vals = [], []
    for k in (5, 6, 7):
        for ms in combinations_with_replacement(range(13), k):
            counts = np.bincount(ms, minlength=13)
            if counts.max() > 4:
                continue
            # distinct suits per rank, globally balanced -> never 5 of one suit (no flush)
            cards, off = [], 0
            for r in range(13):
                for i in range(counts[r]):
                    cards.append(eval7.Card(RANKS[r] + "cdhs"[(off + i) % 4]))
                off += counts[r]
            keys.append(int(POW5 @ counts))
            vals.append(eval7.evaluate(cards))
    keys, vals = np.array(keys, dtype=np.int64), np.array(vals, dtype=np.int64)
    o = np.argsort(keys)
    flush = np.full(1 << 13, -1, dtype=np.int64)
    for m in range(1 << 13):
        n = bin(m).count("1")
        if 5 <= n <= 7:
            flush[m] = eval7.evaluate([eval7.Card(RANKS[r] + "s") for r in range(13) if m >> r & 1])
    return keys[o], vals[o], flush


if os.path.exists(_PATH):
    _z = np.load(_PATH)
    KEYS, VALS, FLUSH = _z["keys"], _z["vals"], _z["flush"]
else:
    KEYS, VALS, FLUSH = _build()
    np.savez_compressed(_PATH, keys=KEYS, vals=VALS, flush=FLUSH)

_CKEY = POW5[COMBO_RANKS[:, 0]] + POW5[COMBO_RANKS[:, 1]]                       # (1326,)
_CSUIT_BITS = np.zeros((4, N_COMBOS), dtype=np.int64)                          # rank bits per suit
_CSUIT_CNT = np.zeros((4, N_COMBOS), dtype=np.int64)
for j in (0, 1):
    np.bitwise_or.at(_CSUIT_BITS, (COMBO_SUITS[:, j], np.arange(N_COMBOS)), 1 << COMBO_RANKS[:, j].astype(np.int64))
    np.add.at(_CSUIT_CNT, (COMBO_SUITS[:, j], np.arange(N_COMBOS)), 1)


def _lookup(key: np.ndarray) -> np.ndarray:
    idx = np.minimum(np.searchsorted(KEYS, key), len(KEYS) - 1)
    return VALS[idx]


def eval_sets(cards: np.ndarray) -> np.ndarray:
    """Values of N card sets: ``cards`` (N, k) ints, 5 <= k <= 7, no duplicates."""
    cards = np.asarray(cards, dtype=np.int64)
    r, s = cards >> 2, cards & 3
    val = _lookup(POW5[r].sum(1))
    for suit in range(4):
        m = s == suit
        cnt = m.sum(1)
        bits = np.where(m, 1 << r, 0).sum(1)
        val = np.maximum(val, np.where(cnt >= 5, FLUSH[bits], -1))
    return val


def combo_board_values(boards: np.ndarray) -> np.ndarray:
    """(R, b) boards with 3 <= b <= 5 -> (R, 1326) values of every combo + board.

    Combos that share a card with their board get -1.
    """
    boards = np.atleast_2d(np.asarray(boards, dtype=np.int64))
    r, s = boards >> 2, boards & 3
    val = _lookup(POW5[r].sum(1)[:, None] + _CKEY[None, :])
    if boards.shape[1] >= 3:
        for suit in range(4):
            m = s == suit
            bcnt = m.sum(1)
            if not (bcnt >= 3).any():          # board needs >= 3 of a suit for any flush
                continue
            bbits = np.where(m, 1 << r, 0).sum(1)
            cnt = bcnt[:, None] + _CSUIT_CNT[suit][None, :]
            bits = bbits[:, None] | _CSUIT_BITS[suit][None, :]
            val = np.maximum(val, np.where(cnt >= 5, FLUSH[bits], -1))
    blocked = CARD_MASK.T[boards].any(1)                                         # (R, 1326)
    return np.where(blocked, -1, val)
