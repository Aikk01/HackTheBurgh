"""Hand evaluation for hold'em style games.

``evaluate`` scores any 1..7 known cards; higher is better.  Scores are only
comparable within the same deck type.  ``describe`` turns a score back into a
readable name such as ``"Two Pair, Aces and Eights"``.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Sequence

from .cards import RANK_NAMES, RANK_NAMES_PLURAL

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)

CATEGORY_NAMES = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
                  "Flush", "Full House", "Four of a Kind", "Straight Flush"]

# Category strength order.  In short-deck (6+) hold'em a flush beats a full house.
_ORDER_STD = list(range(9))
_ORDER_SHORT = list(range(9))
_ORDER_SHORT[FLUSH], _ORDER_SHORT[FULL_HOUSE] = _ORDER_SHORT[FULL_HOUSE], _ORDER_SHORT[FLUSH]

_SHIFT = 20  # 5 ranks x 4 bits of tie-break information below the category


def _pack(ranks: Sequence[int]) -> int:
    v = 0
    for i in range(5):
        v = (v << 4) | (ranks[i] if i < len(ranks) else 0)
    return v


def _unpack(v: int) -> List[int]:
    return [(v >> (4 * (4 - i))) & 15 for i in range(5)]


def _straight_high(mask: int, short_deck: bool) -> int:
    """Highest rank of a straight contained in ``mask`` (bit r = rank r), or -1."""
    for high in range(12, 3, -1):
        need = 0b11111 << (high - 4)
        if mask & need == need:
            return high
    if short_deck:
        wheel = (1 << 12) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)  # A6789
        if mask & wheel == wheel:
            return 7
    else:
        wheel = (1 << 12) | 0b1111  # A2345
        if mask & wheel == wheel:
            return 3
    return -1


def _top(mask: int, n: int, exclude: Sequence[int] = ()) -> List[int]:
    out = []
    for r in range(12, -1, -1):
        if mask & (1 << r) and r not in exclude:
            out.append(r)
            if len(out) == n:
                break
    return out


def evaluate(cards: Sequence[int], short_deck: bool = False) -> int:
    """Score the best 5-card hand within ``cards`` (1 to 7 known cards)."""
    order = _ORDER_SHORT if short_deck else _ORDER_STD
    counts = [0] * 13
    suit_counts = [0, 0, 0, 0]
    suit_masks = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        r = c >> 2
        s = c & 3
        counts[r] += 1
        suit_counts[s] += 1
        suit_masks[s] |= 1 << r
        mask |= 1 << r

    flush_suit = -1
    for s in range(4):
        if suit_counts[s] >= 5:
            flush_suit = s
            break
    if flush_suit >= 0:
        sf = _straight_high(suit_masks[flush_suit], short_deck)
        if sf >= 0:
            return (order[STRAIGHT_FLUSH] << _SHIFT) | _pack([sf])

    quads = [r for r in range(12, -1, -1) if counts[r] == 4]
    if quads:
        kicker = _top(mask, 1, exclude=quads)
        return (order[QUADS] << _SHIFT) | _pack([quads[0]] + kicker)

    trips = [r for r in range(12, -1, -1) if counts[r] == 3]
    pairs = [r for r in range(12, -1, -1) if counts[r] == 2]

    best = -1
    if trips and (len(trips) >= 2 or pairs):
        candidates = []
        if len(trips) >= 2:
            candidates.append(trips[1])
        if pairs:
            candidates.append(pairs[0])
        best = (order[FULL_HOUSE] << _SHIFT) | _pack([trips[0], max(candidates)])
    if flush_suit >= 0:
        sc = (order[FLUSH] << _SHIFT) | _pack(_top(suit_masks[flush_suit], 5))
        best = max(best, sc)
    if best >= 0:
        return best

    st = _straight_high(mask, short_deck)
    if st >= 0:
        return (order[STRAIGHT] << _SHIFT) | _pack([st])
    if trips:
        return (order[TRIPS] << _SHIFT) | _pack([trips[0]] + _top(mask, 2, exclude=trips))
    if len(pairs) >= 2:
        return (order[TWO_PAIR] << _SHIFT) | _pack([pairs[0], pairs[1]] + _top(mask, 1, exclude=pairs[:2]))
    if pairs:
        return (order[PAIR] << _SHIFT) | _pack([pairs[0]] + _top(mask, 3, exclude=pairs))
    return (order[HIGH_CARD] << _SHIFT) | _pack(_top(mask, 5))


def category_of(score: int, short_deck: bool = False) -> int:
    order = _ORDER_SHORT if short_deck else _ORDER_STD
    return order.index(score >> _SHIFT)


def describe(score: int, short_deck: bool = False) -> str:
    cat = category_of(score, short_deck)
    r = _unpack(score & ((1 << _SHIFT) - 1))
    if cat == STRAIGHT_FLUSH:
        return "Royal Flush" if r[0] == 12 else f"Straight Flush, {RANK_NAMES[r[0]]} high"
    if cat == QUADS:
        return f"Four of a Kind, {RANK_NAMES_PLURAL[r[0]]}"
    if cat == FULL_HOUSE:
        return f"Full House, {RANK_NAMES_PLURAL[r[0]]} full of {RANK_NAMES_PLURAL[r[1]]}"
    if cat == FLUSH:
        return f"Flush, {RANK_NAMES[r[0]]} high"
    if cat == STRAIGHT:
        return f"Straight, {RANK_NAMES[r[0]]} high"
    if cat == TRIPS:
        return f"Three of a Kind, {RANK_NAMES_PLURAL[r[0]]}"
    if cat == TWO_PAIR:
        return f"Two Pair, {RANK_NAMES_PLURAL[r[0]]} and {RANK_NAMES_PLURAL[r[1]]}"
    if cat == PAIR:
        return f"Pair of {RANK_NAMES_PLURAL[r[0]]}"
    return f"{RANK_NAMES[r[0]]} high"


def evaluate_omaha(hole: Sequence[int], board: Sequence[int], short_deck: bool = False) -> int:
    """Omaha: exactly two hole cards and three board cards."""
    best = -1
    for h in combinations(hole, 2):
        for b in combinations(board, 3):
            best = max(best, evaluate(list(h) + list(b), short_deck))
    return best


def best_hand(hole: Sequence[Optional[int]], board: Sequence[Optional[int]],
              omaha: bool = False, short_deck: bool = False) -> Optional[int]:
    """Score for a player's hand given the known cards; ``None`` if unknown cards
    make the evaluation impossible."""
    if not hole or any(c is None for c in hole):
        return None
    known_board = [c for c in board if c is not None]
    if omaha:
        if len(known_board) < 3:
            return None
        return evaluate_omaha(list(hole), known_board, short_deck)  # type: ignore[arg-type]
    return evaluate(list(hole) + known_board, short_deck)  # type: ignore[arg-type]
