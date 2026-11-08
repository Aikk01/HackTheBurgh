"""Streaming "market" for one tracked player in one heads-up postflop hand.

Feed it the hand's events in order; after every event it emits one JSON-friendly dict:
posterior over the opponent's 1,326 combos, the tracked player's win-probability "price",
the pot-odds price and (when the tracked player faces a bet) a call/fold decision.

Bayes: posterior_t(c) ∝ posterior_{t-1}(c) · P(action_t | c, context_t)   (opponent actions only)
Board deals apply card removal; the tracked player's own actions do not change the opponent range.
"""
from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional

import numpy as np

from .cards import COMBO_RANKS, COMBO_SUITS, N_COMBOS, RANKS, available, card_str, combo_str
from .equity import FLOP_RUNOUTS
from .equity import equity_vector_cached as equity_vector
from .likelihood import LikelihoodModel, walk
from .prior import prior, uniform

# 13x13 hand grid: row/col 0 = Ace ... 12 = Deuce.  Pairs on the diagonal, suited above it
# (row = high card, col = low card), offsuit below it (row = low card, col = high card).
_hi = 12 - COMBO_RANKS.max(1)
_lo = 12 - COMBO_RANKS.min(1)
_suited = COMBO_SUITS[:, 0] == COMBO_SUITS[:, 1]
GRID_CELL = np.where(_suited, _hi * 13 + _lo, _lo * 13 + _hi).astype(np.int64)     # (1326,) -> 0..168
GRID_LABELS = [[(RANKS[12 - i] + RANKS[12 - j] + ("" if i == j else ("s" if j > i else "o")))
                if i <= j else RANKS[12 - j] + RANKS[12 - i] + "o" for j in range(13)] for i in range(13)]
COMBO_LABELS = [combo_str(i) for i in range(N_COMBOS)]
del _hi, _lo, _suited


def range_summary(post: np.ndarray, top: int = 8) -> Dict:
    """Viz-friendly summary of a posterior: 13x13 mass grid, top combos, effective size."""
    grid = np.bincount(GRID_CELL, weights=post, minlength=169).reshape(13, 13)
    nz = post[post > 0]
    order = np.argsort(-post)[:top]
    return {
        "grid": np.round(grid, 6).tolist(),
        "top_combos": [{"combo": COMBO_LABELS[i], "p": round(float(post[i]), 6)} for i in order],
        "effective_combos": float(np.exp(-(nz * np.log(nz)).sum())),
    }


class MarketTracker:
    def __init__(self, hand: Dict, tracked: int, model: Optional[LikelihoodModel], mode: str = "model",
                 n_runouts: int = FLOP_RUNOUTS, seed: int = 0):
        """mode: 'model' (prior + likelihood updates) | 'prior' (prior only) | 'uniform' (baseline)."""
        self.hand, self.me, self.opp = hand, tracked, 1 - tracked
        self.hero = hand["hole"][tracked]
        assert self.hero is not None, "tracked player's cards must be known"
        self.model, self.mode = model, mode
        self.n_runouts, self.seed = n_runouts, seed
        if mode == "uniform":
            self.post = uniform(self.hero)
        else:
            self.post = prior(hand["positions"][self.opp], hand["pre_role"][self.opp], hand["pre_raises"], self.hero)
        self.board: List[int] = []
        self.eq = None

    # -------------------------------------------------------------------------------------
    def _set_board(self, board: List[int]) -> None:
        if board == self.board:
            return
        self.board = list(board)
        self.post = self.post * available(self.hero + self.board)
        self.post /= self.post.sum()
        self.eq = equity_vector(self.hero, self.board, n_runouts=self.n_runouts, seed=self.seed)

    def price(self) -> float:
        return float(self.post @ self.eq)

    def _event(self, street: str, action: Optional[Dict], t0: float, include_posterior: bool) -> Dict:
        a = action or {}
        facing = bool(a) and a["actor"] == self.me and a["to_call"] > 0
        price = self.price()
        pot_odds = a.get("pot_odds", 0.0) if facing else None
        self.seq = getattr(self, "seq", -1) + 1
        ev = {
            "type": "event", "seq": self.seq, "mode": self.mode,
            "hand_id": self.hand["id"], "street": street, "board": [card_str(c) for c in self.board],
            "tracked": {"position": self.hand["positions"][self.me], "cards": [card_str(c) for c in self.hero]},
            "action": None if not a else {
                "actor": "tracked" if a["actor"] == self.me else "opponent", "kind": a["kind"],
                "size_frac_pot": round(a["frac"], 4), "pot": a["pot"], "to_call": a["to_call"]},
            "price": price,
            "pot_odds_price": pot_odds,
            "decision": (("call" if price > pot_odds else "fold") if facing else None),
            "latency_ms": (time.perf_counter() - t0) * 1e3,
        }
        if include_posterior:
            ev["posterior"] = np.round(self.post, 7).tolist()
            ev.update(range_summary(self.post))
        return ev

    def run(self, include_posterior: bool = False) -> Iterator[Dict]:
        """Replay the hand.  For decision points the event is emitted *before* the tracked
        player's action (that is when the decision is made); for opponent actions *after*
        the Bayesian update."""
        street = None
        for _, a, ctx, board in walk(self.hand):
            t0 = time.perf_counter()
            if a["street"] != street:
                street = a["street"]
                self._set_board(board)
                yield self._event(street, None, t0, include_posterior)
                t0 = time.perf_counter()
            if a["actor"] == self.me:
                yield self._event(street, a, t0, include_posterior)
                continue
            if self.mode == "model":
                L = self.model.likelihood(a, board, ctx)
                if L is not None:
                    p = self.post * L
                    s = p.sum()
                    if s > 0:
                        self.post = p / s
            yield self._event(street, a, t0, include_posterior)
