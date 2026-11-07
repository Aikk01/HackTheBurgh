"""Monte Carlo equity estimation and pre-flop hand strength heuristics."""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence

from . import cards as C
from . import evaluator as E


def estimate_equity(hole: Sequence[int], board: Sequence[Optional[int]], n_opponents: int, *,
                    iterations: int = 400, short_deck: bool = False, omaha: bool = False,
                    rng: Optional[random.Random] = None, dead: Iterable[int] = ()) -> Dict[str, float]:
    """Estimate win/tie probability of ``hole`` against ``n_opponents`` random hands.

    Returns ``{"win": p, "tie": p, "equity": win + tie / (n+1)-ish share, "iterations": n}``.
    """
    rng = rng or random.Random()
    known_board = [c for c in board if c is not None]
    if n_opponents <= 0:
        return {"win": 1.0, "tie": 0.0, "equity": 1.0, "iterations": 0}
    deck = C.full_deck(short_deck)
    used = set(hole) | set(known_board) | set(dead)
    deck = [c for c in deck if c not in used]
    hole_n = 4 if omaha else 2
    need_board = 5 - len(known_board)
    need = n_opponents * hole_n + need_board
    if need > len(deck):
        raise ValueError("not enough cards in the deck")
    wins = ties = 0
    equity = 0.0
    hole_l = list(hole)
    for _ in range(iterations):
        draw = rng.sample(deck, need)
        full_board = known_board + draw[:need_board]
        if omaha:
            hero = E.evaluate_omaha(hole_l, full_board, short_deck)
        else:
            hero = E.evaluate(hole_l + full_board, short_deck)
        best_opp = -1
        n_tied = 0
        pos = need_board
        for _o in range(n_opponents):
            opp_hole = draw[pos:pos + hole_n]
            pos += hole_n
            if omaha:
                s = E.evaluate_omaha(opp_hole, full_board, short_deck)
            else:
                s = E.evaluate(opp_hole + full_board, short_deck)
            if s > best_opp:
                best_opp = s
                n_tied = 1
            elif s == best_opp:
                n_tied += 1
        if hero > best_opp:
            wins += 1
            equity += 1.0
        elif hero == best_opp:
            ties += 1
            equity += 1.0 / (n_tied + 1)
    return {"win": wins / iterations, "tie": ties / iterations, "equity": equity / iterations,
            "iterations": iterations}


def chen_score(hole: Sequence[int]) -> float:
    """Bill Chen's pre-flop point system for two hole cards (higher is better; AA = 20)."""
    if len(hole) != 2:
        return 0.0
    r1, r2 = sorted((C.rank_of(hole[0]), C.rank_of(hole[1])), reverse=True)
    suited = C.suit_of(hole[0]) == C.suit_of(hole[1])
    high_points = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}
    score = high_points.get(r1, (r1 + 2) / 2.0)
    if r1 == r2:
        score = max(score * 2, 5.0)
    if suited:
        score += 2
    gap = r1 - r2
    if r1 != r2:
        if gap == 1:
            score -= 0  # connectors
        elif gap == 2:
            score -= 1
        elif gap == 3:
            score -= 2
        elif gap == 4:
            score -= 4
        else:
            score -= 5
        if gap <= 2 and r1 < 10:  # straight bonus for small connectors / one-gappers
            score += 1
    return round(score * 2) / 2


def preflop_strength(hole: Sequence[int]) -> float:
    """Chen score normalised to roughly 0..1."""
    return max(0.0, min(1.0, chen_score(hole) / 20.0))
