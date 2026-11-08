"""Real-time ML ranges for the Poker Mirror table.

Runs the market model (prior x action likelihoods over 1,326 combos) on a live ``Hand.observation``:
one posterior per opponent still in the hand, rebuilt from the public action history on every
call, plus the viewer's win probability ("price") against each posterior after the flop.

The model was trained on heads-up postflop pots.  At a full table each opponent is treated as if
heads-up against the field: the context of each of their actions (bet faced, pot odds, SPR,
position, preflop role, aggression this street) is read off the shared history.  Documented
approximation - see NOTES-market.md.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .cards import card_from_str
from .equity import equity_vector_cached
from .features import context_vector
from .likelihood import MODEL_PATH, LikelihoodModel
from .prior import available, prior
from .tracker import range_summary

SHIPPED = os.path.join(os.path.dirname(__file__), "model", "likelihood.pkl")
_MODEL: Optional[LikelihoodModel] = None
_COMBOS = np.array([[6 if r == c else 4 if r < c else 12 for c in range(13)] for r in range(13)], dtype=float)
RANKS = "AKQJT98765432"
CLASS_NAMES = [[RANKS[r] + RANKS[c] if r == c else RANKS[r] + RANKS[c] + "s" if r < c else RANKS[c] + RANKS[r] + "o"
                for c in range(13)] for r in range(13)]


def model() -> Optional[LikelihoodModel]:
    global _MODEL
    if _MODEL is None:
        for path in (MODEL_PATH, SHIPPED):
            if os.path.exists(path):
                _MODEL = LikelihoodModel.load(path)
                break
    return _MODEL


def loaded() -> bool:
    return model() is not None


def _kind(action: str, to_call: float) -> str:
    if action == "all_in":
        return "raise" if to_call > 0 else "bet"
    return action


def ranges_for_table(obs: Dict[str, Any], opponents: List[int]) -> Dict[int, Dict[str, Any]]:
    """{player index: {"hands": {grid, density, top}, "price": float|None, "effective_combos": float}}."""
    m = model()
    if m is None or not opponents:
        return {}
    n = len(obs["players"])
    pos = {p["index"]: p["position"] for p in obs["players"]}
    hero = [card_from_str(c) for c in obs["hero"].get("cards") or [] if c and "?" not in c]
    blinds = list(obs.get("blinds") or [0.0] * n)
    antes = list(obs.get("antes") or [0.0] * n)

    # ---- replay the public history, collecting each opponent's postflop actions with their context
    pot = float(sum(blinds) + sum(antes))
    street = "preflop"
    board: List[int] = []
    to = {i: float(blinds[i]) for i in range(n)}            # bet so far this street
    put = {i: float(blinds[i] + antes[i]) for i in range(n)}  # chips put in so far (for stacks at the time)
    folded: set = set()
    aggr = {i: 0 for i in range(n)}                           # bets/raises this street
    pre_raises, last_raiser = 0, None
    todo: Dict[int, List] = {j: [] for j in opponents}
    stack_now = {p["index"]: (p["stack"] if p["stack"] is not None else 1e9) for p in obs["players"]}
    contributed_now = {p["index"]: float(p["contributed"]) for p in obs["players"]}
    boards_by_street: Dict[str, List[int]] = {}

    for h in obs.get("history") or []:
        if h["action"] == "board":
            street = h["street"]
            board = [card_from_str(c) for c in h["cards"]] if len(h["cards"]) >= 3 else board + [card_from_str(c) for c in h["cards"]]
            boards_by_street[street] = list(board)
            to = {i: 0.0 for i in range(n)}
            aggr = {i: 0 for i in range(n)}
            continue
        j, act = h["player"], h["action"]
        amount = float(h.get("amount") or 0.0)
        max_to = max(to.values())
        to_call = max(0.0, max_to - to[j])
        if street == "preflop":
            if act in ("bet", "raise", "all_in") and amount > 0:
                pre_raises += 1
                last_raiser = j
        elif j in todo and act != "fold":
            kind = _kind(act, to_call)
            stack_then = stack_now[j] + (contributed_now[j] - put[j])
            eff = min(stack_then, max(0.0, *(stack_now[k] + contributed_now[k] - put[k] for k in range(n) if k != j and k not in folded)))
            live_order = [i for i in range(n) if i not in folded]
            ip = live_order and j == live_order[-1]
            pre_role = "agg" if j == last_raiser else ("call" if pre_raises else "limp")
            ctx = context_vector(street, ip=bool(ip), faced_frac=(to_call / max(pot - to_call, 1e-9)) if to_call > 0 else 0.0,
                                 pot_odds=(to_call / (pot + to_call)) if to_call > 0 else 0.0, spr=eff / pot if pot > 0 else 0.0,
                                 pre_role=pre_role, pre_raises=pre_raises,
                                 opp_aggr_street=sum(v for k, v in aggr.items() if k != j))
            a = {"kind": kind, "to_call": to_call, "frac": amount / pot if pot > 0 else 0.0,
                 "all_in": amount >= stack_then - 1e-9 and amount > 0}
            todo[j].append((a, list(boards_by_street.get(street, board)), ctx))
        if act == "fold":
            folded.add(j)
        else:
            to[j] = float(h.get("to") if h.get("to") is not None else to[j] + amount)
            put[j] += amount
            pot += amount
            if act in ("bet", "raise", "all_in") and amount > 0:
                aggr[j] += 1

    # ---- one posterior per opponent
    dead = hero + board
    eq = equity_vector_cached(hero, board) if len(hero) == 2 and len(board) >= 3 else None
    out: Dict[int, Dict[str, Any]] = {}
    for j in opponents:
        role = "agg" if j == last_raiser else ("call" if pre_raises else "limp")
        post = prior(pos[j], role, pre_raises, dead)
        for a, b, ctx in todo[j]:
            L = m.likelihood(a, b, ctx)
            if L is not None:
                p = post * L
                s = p.sum()
                if s > 0:
                    post = p / s
        post = post * available(dead)
        post /= post.sum()
        s = range_summary(post, top=3)
        grid = np.array(s["grid"])
        order = np.argsort(-grid.ravel())[:3]
        out[j] = {"hands": {"grid": grid.tolist(), "density": np.round(grid / _COMBOS, 6).tolist(),
                            "top": [{"hand": CLASS_NAMES[k // 13][k % 13], "p": round(float(grid.ravel()[k]), 4)} for k in order]},
                  "price": float(post @ eq) if eq is not None else None,
                  "effective_combos": round(s["effective_combos"]),
                  "updates": len(todo[j])}
    return out
