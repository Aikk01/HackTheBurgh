"""River equity, strength buckets, the range readout and the EV advisor (PLAN.md section 3).

Everything here is exact enumeration on a complete five-card board, so no
Monte-Carlo noise enters the demo.  All values are in opening-pot units,
matching the RL reward in :mod:`pokersim.river_env`.
"""
from __future__ import annotations

import bisect
import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import cards as C
from . import evaluator as E
from .mirror_types import BET_SIZE, BUCKETS, OPENING_POT, RAISE_TO, RESPONSES, bucket_of

Holding = Tuple[int, int]


def _norm(h: Iterable[int]) -> Holding:
    a, b = sorted(h)
    return (a, b)


class BoardRanker:
    """Scores every two-card holding once for a given river board.

    Equity of a holding is the fraction of the other legal holdings it beats
    (ties count half).  Card removal is handled exactly: holdings sharing a
    card with the hand are excluded from its opponents.
    """

    def __init__(self, board: Sequence[int]):
        if len(board) != 5 or any(c is None for c in board):
            raise ValueError("BoardRanker needs a complete five-card board")
        self.board = list(board)
        remaining = [c for c in C.full_deck() if c not in self.board]
        self.holdings: List[Holding] = [_norm(h) for h in itertools.combinations(remaining, 2)]
        self.score: Dict[Holding, int] = {h: E.evaluate(list(h) + self.board) for h in self.holdings}
        self.sorted_scores = sorted(self.score.values())
        self.by_card: Dict[int, List[Holding]] = {}
        for h in self.holdings:
            self.by_card.setdefault(h[0], []).append(h)
            self.by_card.setdefault(h[1], []).append(h)
        self.n = len(self.holdings)  # 1081
        self._equity_cache: Dict[Holding, float] = {}

    def opponents(self, hand: Holding) -> int:
        return self.n - len(self.by_card[hand[0]]) - len(self.by_card[hand[1]]) + 1  # 990

    def equity(self, hand: Sequence[int]) -> float:
        """Equity of ``hand`` against all holdings that do not share a card with it."""
        h = _norm(hand)
        if h in self._equity_cache:
            return self._equity_cache[h]
        s = self.score[h] if h in self.score else E.evaluate(list(h) + self.board)
        below = bisect.bisect_left(self.sorted_scores, s)
        equal = bisect.bisect_right(self.sorted_scores, s) - below
        # remove holdings that share a card with the hand
        shared = set(self.by_card.get(h[0], [])) | set(self.by_card.get(h[1], []))
        for o in shared:
            so = self.score[o]
            if so < s:
                below -= 1
            elif so == s:
                equal -= 1
        opponents = self.n - len(shared)
        eq = (below + 0.5 * equal) / opponents if opponents else 0.0
        self._equity_cache[h] = eq
        return eq

    def bucket(self, hand: Sequence[int]) -> str:
        return bucket_of(self.equity(hand))

    def all_buckets(self) -> Dict[Holding, str]:
        return {h: self.bucket(h) for h in self.holdings}

    def equity_vs_range(self, hero: Sequence[int], weights: Dict[Holding, float]) -> float:
        """Hero equity against a weighted range (holdings sharing a card with hero are skipped)."""
        hs = set(hero)
        hero_score = E.evaluate(list(hero) + self.board)
        num = den = 0.0
        for h, w in weights.items():
            if w <= 0 or h[0] in hs or h[1] in hs:
                continue
            s = self.score[h]
            num += w * (1.0 if hero_score > s else 0.5 if hero_score == s else 0.0)
            den += w
        return num / den if den else 0.0


_ranker_cache: Dict[Tuple[int, ...], BoardRanker] = {}


def ranker_for(board: Sequence[int]) -> BoardRanker:
    key = tuple(sorted(board))
    r = _ranker_cache.get(key)
    if r is None:
        if len(_ranker_cache) > 256:
            _ranker_cache.clear()
        r = BoardRanker(board)
        _ranker_cache[key] = r
    return r


def parse(cards: Sequence[str]) -> List[int]:
    out = []
    for c in cards:
        v = C.parse_card(c)
        if v is None:
            raise ValueError(f"unknown card {c!r}")
        out.append(v)
    return out


# ----------------------------------------------------------------- range readout
StrengthTable = Dict[str, Dict[str, Dict[str, float]]]  # context -> bucket -> response -> prob


def range_readout(board: Sequence[str], own_cards: Sequence[str], context: str, response: str,
                  strength_table: StrengthTable) -> Dict[str, object]:
    """Posterior over the defender's holdings after their response.

    ``w(D) ∝ R[context][bucket(D)][response]`` over holdings that avoid the
    bettor's cards.  Returns bucket masses, a 13x13 grid by starting-hand
    class and the bettor's own cells.  Bucket-level resolution only.
    """
    b = parse(board)
    own = parse(own_cards)
    r = ranker_for(b)
    table = strength_table.get(context) or {}
    own_set = set(own)
    weights: Dict[Holding, float] = {}
    prior: Dict[Holding, float] = {}
    for h in r.holdings:
        if h[0] in own_set or h[1] in own_set:
            continue
        bk = r.bucket(h)
        lik = float((table.get(bk) or {}).get(response, 0.0))
        prior[h] = 1.0
        weights[h] = lik
    total = sum(weights.values())
    if total <= 0:
        weights = dict(prior)
        total = sum(weights.values())
    for h in weights:
        weights[h] /= total
    buckets = {bk: 0.0 for bk in BUCKETS}
    prior_buckets = {bk: 0.0 for bk in BUCKETS}
    n_prior = len(prior)
    for h, w in weights.items():
        buckets[r.bucket(h)] += w
        prior_buckets[r.bucket(h)] += 1.0 / n_prior
    grid: Dict[str, float] = {}
    for h, w in weights.items():
        grid[hand_class(h)] = grid.get(hand_class(h), 0.0) + w
    max_cell = max(grid.values()) if grid else 1.0
    return {
        "buckets": buckets,
        "prior_buckets": prior_buckets,
        "grid": grid,
        "grid_max": max_cell,
        "own_cells": [hand_class(_norm(own))] if len(own) == 2 else [],
        "holdings": len(weights),
        "context": context,
        "response": response,
        "resolution": "three strength buckets",
    }


def hand_class(h: Holding) -> str:
    """Starting-hand class such as 'AKs', 'T9o', '77'."""
    a, b = sorted(h, reverse=True)
    ra, rb = C.RANK_CHARS[C.rank_of(a)], C.RANK_CHARS[C.rank_of(b)]
    if C.rank_of(a) == C.rank_of(b):
        return ra + rb
    return ra + rb + ("s" if C.suit_of(a) == C.suit_of(b) else "o")


def range_weights(board: Sequence[int], own: Sequence[int], context: str, response: str,
                  strength_table: StrengthTable) -> Dict[Holding, float]:
    """Unnormalised posterior weights over the defender's holdings after ``response``.

    Empty (all zero) when the table gives the response no likelihood anywhere;
    callers decide whether that means "no information" (readout) or "cannot
    happen" (EV advisor)."""
    r = ranker_for(board)
    table = strength_table.get(context) or {}
    own_set = set(own)
    out: Dict[Holding, float] = {}
    for h in r.holdings:
        if h[0] in own_set or h[1] in own_set:
            continue
        out[h] = float((table.get(r.bucket(h)) or {}).get(response, 0.0))
    return out


# ----------------------------------------------------------------- EV advisor
def ev_bet(p_fold: float, p_call: float, p_raise: float, eq_call: float, eq_raise: float) -> float:
    """Bettor EV of betting half pot, in opening-pot units (PLAN.md section 3)."""
    return p_fold + p_call * (2 * eq_call - 0.5) + p_raise * max(-0.5, 3 * eq_raise - 1)


def advise(board: Sequence[str], own_cards: Sequence[str], context: str, probs: Dict[str, float],
           strength_table: StrengthTable, fold_interval: Optional[Tuple[float, float]] = None) -> Dict[str, object]:
    """Compare checking back with betting half pot for the bettor."""
    b = parse(board)
    own = parse(own_cards)
    r = ranker_for(b)
    eq0 = r.equity(own)
    eq_c = r.equity_vs_range(own, range_weights(b, own, context, "call", strength_table))
    eq_r = r.equity_vs_range(own, range_weights(b, own, context, "raise", strength_table))
    p_f, p_c, p_r = (float(probs.get(k, 0.0)) for k in RESPONSES)
    ev_check = eq0
    ev_b = ev_bet(p_f, p_c, p_r, eq_c, eq_r)
    band = None
    if fold_interval is not None:
        lo, hi = fold_interval
        rest = max(1e-9, p_c + p_r)
        band = [ev_bet(x, p_c * (1 - x) / rest, p_r * (1 - x) / rest, eq_c, eq_r) for x in (lo, hi)]
        band = [min(band), max(band)]
    break_even = 1.0 / 3.0  # pure bluff
    pure_bluff = eq_c <= 1e-9 and eq_r <= 1e-9
    reason = (f"Betting wins the pot outright when the defender folds ({p_f:.0%} forecast). "
              + (f"A pure bluff breaks even at a {break_even:.0%} fold rate. " if pure_bluff else
                 f"When called you win {eq_c:.0%} of the time against the calling range. ")
              + ("Bet." if ev_b > ev_check else "Check."))
    return {
        "ev_check": round(ev_check, 4),
        "ev_bet": round(ev_b, 4),
        "ev_bet_band": [round(x, 4) for x in band] if band else None,
        "recommend": "bet" if ev_b > ev_check else "check",
        "equity": {"vs_random": round(eq0, 4), "vs_call_range": round(eq_c, 4), "vs_raise_range": round(eq_r, 4)},
        "probs": {"fold": p_f, "call": p_c, "raise": p_r},
        "break_even_fold": break_even,
        "pure_bluff": pure_bluff,
        "units": f"opening pot = {OPENING_POT} chips, bet = {BET_SIZE}, raise to {RAISE_TO}",
        "reason": reason,
    }


def implied_marginal(strength_table: StrengthTable, context: str, bucket_shares: Dict[str, float]) -> Dict[str, float]:
    """The response distribution implied by R and a bucket distribution (diagnostic)."""
    out = {a: 0.0 for a in RESPONSES}
    table = strength_table.get(context) or {}
    for bk, share in bucket_shares.items():
        for a in RESPONSES:
            out[a] += share * float((table.get(bk) or {}).get(a, 0.0))
    return out


# ----------------------------------------------------------------- checks / CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """``mirror-check-ranges``: unit-level sanity checks of the readout and the advisor."""
    import argparse
    import random
    import time

    ap = argparse.ArgumentParser(prog="pokersim mirror-check-ranges")
    ap.add_argument("--boards", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    ok = True
    for _ in range(args.boards):
        deck = C.full_deck()
        rng.shuffle(deck)
        board = deck[:5]
        t = time.perf_counter()
        r = BoardRanker(board)
        dt = time.perf_counter() - t
        # brute force equity for three random hands
        for h in (deck[5:7], deck[7:9], deck[9:11]):
            hs = E.evaluate(list(h) + board)
            wins = ties = n = 0
            for o in itertools.combinations([c for c in deck[5:] if c not in h], 2):
                n += 1
                so = E.evaluate(list(o) + board)
                wins += hs > so
                ties += hs == so
            brute = (wins + 0.5 * ties) / n
            fast = r.equity(h)
            if abs(brute - fast) > 1e-12:
                ok = False
                print("MISMATCH", C.cards_str(board), C.cards_str(h), brute, fast)
        print(f"board {C.cards_str(board)}: ranked {r.n} holdings in {dt * 1000:.0f} ms; brute-force equities match")
    flat = {c: {bk: {"fold": 0.5, "call": 0.4, "raise": 0.1} for bk in BUCKETS} for c in ("checked_through",)}
    rd = range_readout(["Kc", "9d", "7h", "4s", "3c"], ["6h", "2s"], "checked_through", "call", flat)
    s = sum(rd["buckets"].values())
    print("readout weights sum to", round(s, 9), "| holdings", rd["holdings"])
    ok = ok and abs(s - 1) < 1e-9 and rd["holdings"] == 990
    # Pure bluff: only strong hands continue, so 6h2s never wins when called or raised.
    only_strong = {"checked_through": {"weak": {"fold": 1.0, "call": 0.0, "raise": 0.0},
                                       "medium": {"fold": 1.0, "call": 0.0, "raise": 0.0},
                                       "strong": {"fold": 0.0, "call": 1.0, "raise": 0.0}}}
    adv = advise(["Kc", "9d", "7h", "4s", "3c"], ["6h", "2s"], "checked_through", {"fold": 0.6, "call": 0.35, "raise": 0.05},
                 only_strong)
    print("pure bluff 6h2s on Kc9d7h4s3c vs strong-only continuing range: ev_bet", adv["ev_bet"], "(expect 1.5*0.6-0.5 = 0.4)")
    ok = ok and abs(adv["ev_bet"] - 0.4) < 1e-9 and adv["pure_bluff"]
    # General formula check against the computed equities
    adv2 = advise(["Kc", "9d", "7h", "4s", "3c"], ["Kh", "Qd"], "checked_through", {"fold": 0.5, "call": 0.4, "raise": 0.1}, flat)
    e = adv2["equity"]
    expect = ev_bet(0.5, 0.4, 0.1, e["vs_call_range"], e["vs_raise_range"])
    print("KQ on the same board: ev_check", adv2["ev_check"], "ev_bet", adv2["ev_bet"], "formula", round(expect, 4))
    ok = ok and abs(adv2["ev_bet"] - expect) < 1e-3 and adv2["ev_check"] == round(e["vs_random"], 4)
    print("mirror-check-ranges:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
