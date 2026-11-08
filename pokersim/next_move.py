"""Next-move model: P(player's next action) over the RL advisor's 7 action classes, learned from recorded hands.

One network, two uses:

* **public** (``with_cards=False``): only what the whole table sees.  Learned from every decision in the corpus
  (online hands of 5-6 players, no showdown filter, plus every seat of the Pluribus logs).  Used for the opponents'
  "next move" box and as the population prior of the personal model.
* **with cards** (``with_cards=True``): also the player's own hole cards.  Learned only from the Pluribus logs, where
  every hole card is known (13 professionals + Pluribus, no showdown selection).  ``pluribus=True`` asks how Pluribus
  would play ("what you should do", Base row); ``pluribus=False`` how the professionals would (the "pro" bots and the
  personal-model prior).

Features of the player ``j`` are computed from any observation of the table (:func:`decision_features`), so the same
code serves the player's own decision (training) and an opponent seen from the hero's seat (screen).  Inference is
NumPy only (``data/rl/next_move.npz``); training is ``scripts/train_next_move.py``.
"""
from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cards as C
from . import evaluator as E
from . import hand_range as HR
from .rl_features import ACTIONS, N_ACTIONS, action_targets

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rl", "next_move.npz")
DECISION_DIM = 8
PUBLIC_DIM = HR.CTX_DIM + DECISION_DIM
CARD_DIM = HR.N_CLASSES + 9 + 2 + 1
FLAG_DIM = 3  # has_cards, is_pluribus, from_pluribus_logs
IN_DIM = PUBLIC_DIM + CARD_DIM + FLAG_DIM


def _lg(x: float) -> float:
    return math.log1p(max(x, 0.0)) / 7.0


def facing(obs: Dict[str, Any], j: int) -> Tuple[float, float, float]:
    """(to_call, stack, max_bet) for player ``j`` at the current state of the table."""
    p = obs["players"][j]
    bb = float(obs.get("min_bet") or 1.0) or 1.0
    max_bet = max(float(q.get("bet") or 0.0) for q in obs["players"])
    stack = float(p["stack"]) if p.get("stack") is not None else 100.0 * bb
    to_call = min(max(0.0, max_bet - float(p.get("bet") or 0.0)), stack)
    return to_call, stack, max_bet


def decision_features(obs: Dict[str, Any], j: int) -> List[float]:
    bb = float(obs.get("min_bet") or 1.0) or 1.0
    pot = float(obs.get("total_pot") or 0.0)
    to_call, stack, _ = facing(obs, j)
    street = obs["street"]
    raises_now = sum(1 for h in obs.get("history") or [] if h.get("player") is not None and h.get("street") == street
                     and h.get("action") in ("bet", "raise"))
    live = [q for q in obs["players"] if not q["folded"] and not q["all_in"]]
    return [to_call / (pot + to_call) if to_call > 0 else 0.0, _lg(to_call / bb), min(stack / max(pot, bb), 20.0) / 20.0,
            _lg(stack / bb), float(to_call <= 0), float(stack > to_call), len(live) / 6.0, min(raises_now, 3) / 3.0]


def public_features(obs: Dict[str, Any], j: int) -> np.ndarray:
    return np.concatenate([HR.context_features(obs, j), np.asarray(decision_features(obs, j), np.float32)])


_PCT: Optional[Dict[Tuple[int, int], float]] = None


def card_features(hole: Sequence[int], board: Sequence[int]) -> np.ndarray:
    global _PCT
    v = np.zeros(CARD_DIM, np.float32)
    if len(hole) != 2:
        return v
    v[HR.class_of(hole[0], hole[1])] = 1.0
    if len(board) >= 3:
        v[HR.N_CLASSES + E.category_of(E.evaluate(list(hole) + list(board)))] = 1.0
        if len(board) < 5:
            suits = [C.suit_of(c) for c in list(hole) + list(board)]
            v[HR.N_CLASSES + 9] = float(any(suits.count(s) == 4 and any(C.suit_of(h) == s for h in hole) for s in range(4)))
            ranks = {C.rank_of(c) for c in list(hole) + list(board)} | ({-1} if any(C.rank_of(c) == 12 for c in list(hole) + list(board)) else set())
            v[HR.N_CLASSES + 10] = float(any(sum(1 for r in range(lo, lo + 5) if r in ranks) == 4 for lo in range(-1, 9)))
    if _PCT is None:
        from .opponent_model import preflop_percentile
        _PCT = preflop_percentile()
    v[HR.N_CLASSES + 11] = 1.0 - _PCT.get(tuple(sorted(hole)), 0.5)
    return v


def features(obs: Dict[str, Any], j: int, hole: Optional[Sequence[int]] = None, pluribus: bool = False,
             from_logs: bool = False) -> np.ndarray:
    board = [c for c in (C.parse_card(x) for x in obs.get("board") or []) if c is not None]
    has = hole is not None and len(hole) == 2
    cf = card_features(hole, board) if has else np.zeros(CARD_DIM, np.float32)
    flags = np.asarray([float(has), float(pluribus and has), float(from_logs)], np.float32)
    return np.concatenate([public_features(obs, j), cf, flags]).astype(np.float32)


def hypothetical_mask(obs: Dict[str, Any], j: int) -> Tuple[np.ndarray, List[Optional[Dict[str, Any]]]]:
    """Legal-action mask of player ``j`` if they had to act now (their own observation has none until their turn)."""
    to_call, stack, max_bet = facing(obs, j)
    p = obs["players"][j]
    bb = float(obs.get("min_bet") or 1.0) or 1.0
    bet = float(p.get("bet") or 0.0)
    la = {"fold": to_call > 0, "check": to_call <= 0, "call": to_call, "raise": stack > to_call,
          "min_raise_to": min(max_bet + max(bb, max_bet), bet + stack), "max_raise_to": bet + stack,
          "raise_kind": "raise" if max_bet > 0 else "bet"}
    pseudo = dict(obs)
    pseudo["legal_actions"] = la
    pseudo["hero"] = {"index": j, "name": p["name"], "position": p.get("position"), "cards": [], "stack": stack,
                      "bet": bet, "contributed": p.get("contributed")}
    return action_targets(pseudo)


def label_of(obs: Dict[str, Any], actual: Dict[str, Any]) -> Optional[int]:
    """Map a recorded action onto the 7 classes (bet sizes to the nearest legal size, in log space)."""
    a = actual.get("action")
    if a == "fold":
        return 0
    if a in ("check", "call"):
        return 1
    if a not in ("bet", "raise"):
        return None
    mask, targets = action_targets(obs)
    to = float(actual.get("to") or 0.0)
    best, bd = None, 1e9
    for i in range(2, N_ACTIONS):
        if mask[i] and targets[i] and targets[i].get("amount"):
            d = abs(math.log(max(to, 1e-6)) - math.log(float(targets[i]["amount"])))
            if d < bd:
                best, bd = i, d
    return best


# ----------------------------------------------------------------- inference
class NextMoveModel:
    def __init__(self, path: str = MODEL_PATH):
        z = np.load(path)
        self.w = {k: z[k] for k in z.files}
        self.n = sum(1 for k in self.w if k.startswith("l") and k.endswith(".w"))

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = (np.atleast_2d(x) - self.w["mean"]) / self.w["std"]
        for i in range(self.n):
            h = h @ self.w[f"l{i}.w"] + self.w[f"l{i}.b"]
            if i < self.n - 1:
                h = np.maximum(h, 0.0)
        return h

    def probs(self, x: np.ndarray, mask: np.ndarray, temp: float = 1.0) -> np.ndarray:
        lg = self.logits(x)[0] / max(temp, 1e-3)
        lg = np.where(mask, lg, -1e9)
        e = np.exp(lg - lg.max())
        return e / e.sum()


_MODEL: Optional[NextMoveModel] = None


def model() -> Optional[NextMoveModel]:
    global _MODEL
    if _MODEL is None and os.path.exists(MODEL_PATH):
        _MODEL = NextMoveModel(MODEL_PATH)
    return _MODEL


def predict(obs: Dict[str, Any], j: int, with_cards: bool = False, pluribus: bool = False) -> Optional[np.ndarray]:
    """P(next action of player ``j``) over :data:`ACTIONS`.  ``with_cards`` only for the observing player (j == hero)."""
    m = model()
    if m is None:
        return None
    hero = obs["hero"]["index"]
    hole = None
    if with_cards:
        if j != hero:
            raise ValueError("with_cards=True is only allowed for the player whose observation this is")
        hole = [c for c in (C.parse_card(x) for x in obs["hero"].get("cards") or []) if c is not None]
    mask = action_targets(obs)[0] if (j == hero and obs.get("legal_actions")) else hypothetical_mask(obs, j)[0]
    # card rows were all Pluribus-log rows in training (from_logs=1); public rows predict the online population
    return m.probs(features(obs, j, hole, pluribus=pluribus, from_logs=hole is not None), mask)


def sample_action(obs: Dict[str, Any], rng: random.Random, pluribus: bool = False, temp: float = 1.0,
                  tilt: Sequence[float] = (0.0, 0.0, 0.0)) -> Dict[str, Any]:
    """Act as a bot that plays like the Pluribus-log players (with its own cards).  ``tilt`` shifts fold/call/raise."""
    m = model()
    mask, targets = action_targets(obs)
    if m is None:
        return targets[1] or {"action": "call"}
    j = obs["hero"]["index"]
    hole = [c for c in (C.parse_card(x) for x in obs["hero"].get("cards") or []) if c is not None]
    lg = m.logits(features(obs, j, hole, pluribus=pluribus, from_logs=True))[0] / max(temp, 1e-3)
    lg = lg + np.asarray([tilt[0], tilt[1]] + [tilt[2]] * (N_ACTIONS - 2))
    lg = np.where(mask, lg, -1e9)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    i = rng.choices(range(N_ACTIONS), weights=p)[0]
    return targets[i] or {"action": "call"}
