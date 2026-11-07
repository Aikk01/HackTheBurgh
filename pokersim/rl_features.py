"""Observation encoding and action mapping for the RL advisor (shared by training and live play).

The policy sees what a player at the table could see: its own cards, the board, the betting, stacks
and positions, and for every opponent the style estimate built from their past actions at this table
(:mod:`pokersim.opponent_model`).  The style block is what lets the policy learn to play differently
against a tight folder than against a loose caller.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cards as C
from . import evaluator as E
from .equity import estimate_equity
from .mirror_data import STYLE_FEATURES
from .opponent_model import TYPES, estimate_style, preflop_percentile

ACTIONS = ["fold", "call", "raise_min", "raise_half", "raise_pot", "raise_2pot", "all_in"]
ACTIONS_JA = ["フォールド", "チェック/コール", "最小レイズ", "1/2ポット", "ポット", "2倍ポット", "オールイン"]
N_ACTIONS = len(ACTIONS)
MAX_OPP = 5
EQ_ITERS = 48
STREETS = ["preflop", "flop", "turn", "river"]
N_TYPES = len(TYPES["names"])

HERO_DIM = 4 + 4 + 6 + 9 + 2 + 3 + 4 + 6 + 9 + 4  # cards, street, position, made hand, draws, equity, money, betting, helpers, behind
OPP_DIM = 9 + len(STYLE_FEATURES) + 1 + N_TYPES
OBS_DIM = HERO_DIM + MAX_OPP * OPP_DIM


def _draws(hole: Sequence[int], board: Sequence[int]) -> Tuple[float, float]:
    cards = list(hole) + list(board)
    suits = [C.suit_of(c) for c in cards]
    flush_draw = any(suits.count(s) == 4 and any(C.suit_of(h) == s for h in hole) for s in range(4))
    ranks = {C.rank_of(c) for c in cards}
    if 12 in ranks:
        ranks.add(-1)  # the wheel ace
    straight_draw = any(sum(1 for r in range(lo, lo + 5) if r in ranks) == 4 for lo in range(-1, 9))
    return float(flush_draw), float(straight_draw)


def _bb(x: Any, bb: float) -> float:
    return float(x or 0.0) / bb


def encode(obs: Dict[str, Any], styles: Dict[int, Dict[str, Any]], rng: Optional[random.Random] = None,
           eq_iters: int = EQ_ITERS) -> np.ndarray:
    """Feature vector for the player whose observation ``obs`` is.  ``styles`` maps player index -> style estimate."""
    rng = rng or random.Random()
    bb = float(obs.get("min_bet") or 1.0) or 1.0
    hero = obs["hero"]
    n = len(obs["players"])
    hole = [C.parse_card(c) for c in hero["cards"]]
    hole = [c for c in hole if c is not None]
    board = [c for c in (C.parse_card(x) for x in obs["board"]) if c is not None]
    street = obs["street"]
    la = obs.get("legal_actions") or {}
    pot = float(obs["total_pot"])
    to_call = float(la.get("call") or 0.0)
    active_opp = [p for p in obs["players"] if not p["folded"] and not p["is_hero"]]
    hist = obs.get("history") or []

    f: List[float] = []
    # cards (4)
    pct = preflop_percentile().get(tuple(sorted(hole)), 0.5) if len(hole) == 2 else 0.5
    r = sorted((C.rank_of(c) for c in hole), reverse=True) if len(hole) == 2 else [0, 0]
    f += [1.0 - pct, float(len(hole) == 2 and C.suit_of(hole[0]) == C.suit_of(hole[1])), float(r[0] == r[1]), r[0] / 12.0]
    # street (4) + board size handled by street; position (6)
    f += [float(street == s) for s in STREETS]
    pos = [0.0] * 6
    pos[min(5, n - 1 - hero["index"])] = 1.0  # 0 = button, 1 = cutoff, ... counted back from the button
    f += pos
    # made hand on the board (9) and draws (2)
    cat = [0.0] * 9
    if len(board) >= 3 and len(hole) == 2:
        cat[E.category_of(E.evaluate(hole + board))] = 1.0
    f += cat
    f += list(_draws(hole, board)) if len(board) in (3, 4) and len(hole) == 2 else [0.0, 0.0]
    # equity vs random hands of the live opponents (3)
    eq = float(estimate_equity(hole, board, max(1, len(active_opp)), iterations=eq_iters, rng=rng)["equity"]) if len(hole) == 2 else 0.0
    f += [eq, 1.0 / (len(active_opp) + 1), len(active_opp) / 5.0]
    # money (4)
    stack = float(hero["stack"] if hero["stack"] is not None else 100 * bb)
    f += [min(pot / bb, 400) / 100.0, min(to_call / bb, 400) / 100.0, to_call / (pot + to_call) if to_call > 0 else 0.0,
          min(stack / max(pot, bb), 20.0) / 20.0]
    # betting so far (6)
    this_street = [h for h in hist if h.get("street") == street and h.get("player") is not None]
    raises_now = sum(1 for h in this_street if h["action"] in ("bet", "raise"))
    hero_pre_aggr = any(h.get("player") == hero["index"] and h["street"] == "preflop" and h["action"] in ("bet", "raise") for h in hist)
    hero_aggr_now = any(h.get("player") == hero["index"] and h["action"] in ("bet", "raise") for h in this_street)
    pre_raises = sum(1 for h in hist if h.get("street") == "preflop" and h["action"] in ("bet", "raise"))
    f += [min(raises_now, 4) / 4.0, float(hero_pre_aggr), float(hero_aggr_now), min(pre_raises, 4) / 4.0,
          min(stack / bb, 400) / 100.0, _bb(hero.get("contributed"), bb) / 100.0]
    # pot-odds style helpers (9): equity minus price, and the raise sizes relative to stack
    price = to_call / (pot + to_call) if to_call > 0 else 0.0
    f += [eq - price, float(la.get("check", False)), float(bool(la.get("raise"))),
          min(float(la.get("min_raise_to") or 0) / max(stack + float(hero.get("bet") or 0), bb), 1.0),
          float(la.get("call_is_all_in", False)), float(n) / 6.0, float(street == "river" and to_call > 0),
          float(len(active_opp) == 1), float(to_call == 0 and raises_now == 0)]
    # opponents still to act behind us this street (4)
    behind = [p for p in active_opp if (p["index"] - hero["index"]) % n > 0 and not p["all_in"]]
    f += [len(behind) / 5.0, float(any(p["all_in"] for p in active_opp)),
          max((_bb(p["bet"], bb) for p in active_opp), default=0.0) / 100.0,
          sum(_bb(p["contributed"], bb) for p in active_opp) / 100.0]
    assert len(f) == HERO_DIM, (len(f), HERO_DIM)

    # opponents, in seat order after the hero (5 x OPP_DIM)
    others = sorted((p for p in obs["players"] if not p["is_hero"]), key=lambda p: (p["index"] - hero["index"]) % n)
    for k in range(MAX_OPP):
        if k >= len(others):
            f += [0.0] * OPP_DIM
            continue
        p = others[k]
        acts = [h for h in hist if h.get("player") == p["index"]]
        st = styles.get(p["index"]) or estimate_style({}, 0)
        f += [1.0, float(p["folded"]), float(p["all_in"]),
              min(_bb(p["stack"] if p["stack"] is not None else 100 * bb, bb), 400) / 100.0,
              _bb(p["bet"], bb) / 100.0, _bb(p["contributed"], bb) / 100.0,
              float(any(h["street"] == "preflop" and h["action"] in ("bet", "raise") for h in acts)),
              min(sum(1 for h in acts if h["street"] != "preflop" and h["action"] in ("bet", "raise")), 4) / 4.0,
              float(any(h["street"] == street and h["action"] in ("bet", "raise") for h in acts))]
        f += [st["rates"][x] for x in STYLE_FEATURES]
        f += [min(st["hands"], 300) / 300.0]
        f += [st["type_posterior"].get(t, 0.0) for t in TYPES["names"]]
    v = np.asarray(f, dtype=np.float32)
    assert v.shape[0] == OBS_DIM
    return v


def action_targets(obs: Dict[str, Any]) -> Tuple[np.ndarray, List[Optional[Dict[str, Any]]]]:
    """Legal-action mask and the engine action for each of the :data:`ACTIONS`."""
    la = obs.get("legal_actions") or {}
    mask = np.zeros(N_ACTIONS, dtype=bool)
    out: List[Optional[Dict[str, Any]]] = [None] * N_ACTIONS
    to_call = float(la.get("call") or 0.0)
    if la.get("fold") and to_call > 0:
        mask[0], out[0] = True, {"action": "fold"}
    mask[1], out[1] = True, {"action": "call" if to_call > 0 else "check"}
    if la.get("raise"):
        lo, hi = float(la["min_raise_to"]), float(la["max_raise_to"])
        pot = float(obs["total_pot"])
        hero_bet = float(obs["hero"]["bet"] or 0.0)
        unit = float(obs.get("chip_unit") or 0.01)
        kind = la.get("raise_kind", "raise")

        def rnd(x: float) -> float:
            if unit >= 1:
                x = round(x / unit) * unit
            return round(min(max(x, lo), hi), 2)

        targets = {2: lo, 3: hero_bet + to_call + 0.5 * (pot + to_call), 4: hero_bet + to_call + 1.0 * (pot + to_call),
                   5: hero_bet + to_call + 2.0 * (pot + to_call), 6: hi}
        seen = set()
        for a in (6, 2, 3, 4, 5):  # all-in and min first so duplicates fold into them
            t = rnd(targets[a])
            if t in seen or (a in (3, 4, 5) and (t <= lo or t >= hi)):
                continue
            seen.add(t)
            mask[a], out[a] = True, {"action": kind, "amount": t}
    return mask, out


def styles_for_hand(tracker: Any, players: Sequence[Any]) -> Dict[int, Dict[str, Any]]:
    """Style estimate for every player index of a hand, from a :class:`StyleTracker`."""
    return {p.index: tracker.estimate(p.name) for p in players}
