"""Position- and role-based preflop ranges (the prior over the opponent's 1,326 combos).

Combos are ordered by pre-flop strength = all-in equity against a random hand
(eval7 Monte Carlo, computed once and cached).  A range is "the top X% of combos",
with soft edges (logistic, width TAU) and, for flat-callers, the very top removed
(those hands would usually have re-raised) down to a floor weight.

The percentages are documented assumptions for 2009 PokerStars 200NL 6-max - not fitted.
"""
from __future__ import annotations

import os

import eval7
import numpy as np

from .cards import COMBO_RANKS, COMBO_SUITS, N_COMBOS, RANKS, available

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CACHE = os.path.join(ROOT, "market", "preflop_strength.npy")
TAU = 0.03

# (role, position) -> (top fraction played, top fraction removed, floor weight for removed part)
OPEN = {"UTG": 0.14, "HJ": 0.18, "CO": 0.25, "BTN": 0.40, "SB": 0.35, "BB": 0.15}
RANGES = {
    ("agg", 1): lambda pos: (OPEN.get(pos, 0.2), 0.0, 1.0),
    ("call", 1): lambda pos: (0.40, 0.05, 0.3) if pos == "BB" else (0.15 if pos == "SB" else 0.12, 0.04, 0.3),
    ("agg", 2): lambda pos: (0.06, 0.0, 1.0),
    ("call", 2): lambda pos: (0.12, 0.03, 0.3),
    ("limp", 0): lambda pos: (1.0, 0.08, 0.5) if pos == "BB" else (0.50, 0.08, 0.3),
}


def _class_key(i: int) -> tuple:
    r1, r2 = sorted(COMBO_RANKS[i], reverse=True)
    return int(r1), int(r2), bool(COMBO_SUITS[i, 0] == COMBO_SUITS[i, 1])


def _compute_strength(iters: int = 30000) -> np.ndarray:
    rnd = eval7.HandRange(",".join(["22+", "A2s+", "K2s+", "Q2s+", "J2s+", "T2s+", "92s+", "82s+", "72s+",
                                    "62s+", "52s+", "42s+", "32s", "A2o+", "K2o+", "Q2o+", "J2o+", "T2o+",
                                    "92o+", "82o+", "72o+", "62o+", "52o+", "42o+", "32o"]))
    eq = {}
    out = np.empty(N_COMBOS)
    for i in range(N_COMBOS):
        k = _class_key(i)
        if k not in eq:
            r1, r2, suited = k
            s2 = "c" if suited else "d"
            hand = [eval7.Card(RANKS[r1] + "c"), eval7.Card(RANKS[r2] + s2)]
            eq[k] = eval7.py_hand_vs_range_monte_carlo(hand, rnd, [], iters)
        out[i] = eq[k]
    return out


def preflop_equity() -> np.ndarray:
    if os.path.exists(_CACHE):
        return np.load(_CACHE)
    e = _compute_strength()
    np.save(_CACHE, e)
    return e


_EQ = preflop_equity()
# top-fraction rank: 0 = best combo, 1 = worst (combo-count weighted, mid-rank for ties)
_order = np.argsort(-_EQ, kind="stable")
_v = np.sort(-_EQ)
TOP = (np.searchsorted(_v, -_EQ, "left") + 0.5 * (np.searchsorted(_v, -_EQ, "right")
                                                  - np.searchsorted(_v, -_EQ, "left"))) / N_COMBOS


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def range_weights(position: str, role: str, n_raises: int) -> np.ndarray:
    key = (role, min(n_raises, 2) if role != "limp" else 0)
    top, cut, floor = RANGES.get(key, lambda p: (0.5, 0.0, 1.0))(position)
    w = _sig((top - TOP) / TAU)
    if cut > 0:
        w = w * (floor + (1 - floor) * _sig((TOP - cut) / TAU))
    return w


def prior(position: str, role: str, n_raises: int, dead=()) -> np.ndarray:
    """Normalised prior over the 1,326 combos with card removal for ``dead`` cards."""
    w = range_weights(position, role, n_raises) * available(dead)
    return w / w.sum()


def uniform(dead=()) -> np.ndarray:
    w = available(dead).astype(float)
    return w / w.sum()
