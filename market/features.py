"""Vectorised combo features (all 1,326 combos at once) and action-context features.

Combo features on a board (computed once per board, then cached):
  * strength percentile: share of board-compatible combos this combo beats (ties = 1/2)
  * flush draw, open-ended / double-gutter straight draw, gutshot (flop/turn only)
Context features: street, position, bet faced, pot odds, SPR, preflop role.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Sequence, Tuple

import numpy as np

from .cards import COMBO_RANKS, COMBO_SUITS, N_COMBOS, available, board_ranks

# STRAIGHT[m] = rank bitmask m (bit r = rank r present, r=0 is a deuce) contains a straight
_m = np.arange(1 << 13)
STRAIGHT = np.zeros(1 << 13, dtype=bool)
for hi in range(4, 13):
    need = 0b11111 << (hi - 4)
    STRAIGHT |= (_m & need) == need
STRAIGHT |= (_m & 0b1000000001111) == 0b1000000001111  # wheel A-2-3-4-5
del _m

_COMBO_RMASK = (1 << COMBO_RANKS[:, 0].astype(np.int64)) | (1 << COMBO_RANKS[:, 1].astype(np.int64))
_COMBO_SUITCNT = np.zeros((N_COMBOS, 4), dtype=np.int8)
np.add.at(_COMBO_SUITCNT, (np.arange(N_COMBOS), COMBO_SUITS[:, 0]), 1)
np.add.at(_COMBO_SUITCNT, (np.arange(N_COMBOS), COMBO_SUITS[:, 1]), 1)

COMBO_FEATURES = ["strength", "strength2", "strength3", "flush_draw", "oesd", "gutshot"]


def percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of ``values`` among ``valid`` entries (invalid -> 0)."""
    v = np.sort(values[valid])
    lo = np.searchsorted(v, values, side="left")
    hi = np.searchsorted(v, values, side="right")
    pct = (lo + 0.5 * (hi - lo)) / max(len(v), 1)
    return np.where(valid, pct, 0.0)


@lru_cache(maxsize=4096)
def board_table(board: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    """All per-combo quantities for one board, computed once and cached."""
    ranks = board_ranks(board)
    valid = available(board)
    s = percentile(ranks, valid)
    out = {"ranks": ranks, "valid": valid, "strength": s}
    if len(board) < 5:
        bsuit = np.bincount(np.array(board) & 3, minlength=4)
        tot = _COMBO_SUITCNT + bsuit
        made_flush = (tot >= 5).any(1)
        fd = ((tot == 4) & (_COMBO_SUITCNT > 0)).any(1) & ~made_flush
        bmask = 0
        for c in board:
            bmask |= 1 << (c >> 2)
        cm = _COMBO_RMASK | bmask
        made = STRAIGHT[cm]
        outs = np.zeros(N_COMBOS, dtype=np.int8)
        board_outs = 0
        for r in range(13):
            outs += STRAIGHT[cm | (1 << r)] & ~made
            board_outs += int(STRAIGHT[bmask | (1 << r)])
        live = ~made & (outs > board_outs)        # the hole cards must add to the draw
        out["flush_draw"] = fd.astype(float)
        out["oesd"] = (live & (outs >= 2)).astype(float)
        out["gutshot"] = (live & (outs == 1)).astype(float)
    else:
        z = np.zeros(N_COMBOS)
        out["flush_draw"] = out["oesd"] = out["gutshot"] = z
    return out


def combo_matrix(board: Sequence[int]) -> np.ndarray:
    """(1326, len(COMBO_FEATURES)) combo feature matrix for ``board``."""
    t = board_table(tuple(board))
    s = t["strength"]
    return np.column_stack([s, s ** 2, s ** 3, t["flush_draw"], t["oesd"], t["gutshot"]])


CONTEXT_FEATURES = ["bias", "turn", "river", "ip", "faced_frac", "pot_odds", "log_spr",
                    "pre_agg", "pre_call", "three_bet_pot", "opp_aggr_street"]


def context_vector(street: str, ip: bool, faced_frac: float, pot_odds: float, spr: float,
                   pre_role: str, pre_raises: int, opp_aggr_street: int) -> np.ndarray:
    return np.array([
        1.0, street == "turn", street == "river", float(ip),
        min(faced_frac, 3.0), pot_odds, np.log1p(min(spr, 50.0)),
        pre_role == "agg", pre_role == "call", pre_raises >= 2, min(opp_aggr_street, 3),
    ], dtype=float)


def design(combo_feats: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """Combo x context interactions: (n, Fc) & (Fx,) or (n, Fx) -> (n, Fc * Fx + Fx).

    Context-only columns are included so the model fits marginal action frequencies; they
    cancel when the likelihood is normalised over combos, so ranking power comes from the
    interaction block.
    """
    if ctx.ndim == 1:
        ctx = np.broadcast_to(ctx, (combo_feats.shape[0], ctx.shape[0]))
    inter = (combo_feats[:, :, None] * ctx[:, None, :]).reshape(combo_feats.shape[0], -1)
    return np.hstack([inter, ctx])
