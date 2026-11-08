"""Action likelihood model P(action bucket | combo features, context).

Two multinomial logistic regressions on combo x context interaction features:

  * "open"   - no bet to face:  check | bet_small (<= 55% pot) | bet_big (> 55% pot or all-in)
  * "facing" - facing a bet:    call | raise
                                (fold is not learnable from showdown-only data: players who
                                 reached showdown never folded.  A fold ends the hand, so the
                                 posterior never needs it.)

Trained ONLY on actions of players whose hole cards became known at showdown.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cards import combo_index
from .features import combo_matrix, context_vector, design

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "data", "processed", "likelihood.pkl")
BIG_BET = 0.55
EPS = 0.05          # likelihood floor: L = EPS + (1 - EPS) * p  (robustness to model error)
CLASSES = {"open": ["check", "bet_small", "bet_big"], "facing": ["call", "raise"]}


def bucket(a: Dict) -> Optional[Tuple[str, int]]:
    """(model, class index) for an action dict; None for folds."""
    if a["kind"] == "fold":
        return None
    if a["to_call"] <= 0:
        if a["kind"] == "check":
            return "open", 0
        return "open", 2 if (a["frac"] > BIG_BET or a["all_in"]) else 1
    return "facing", 0 if a["kind"] == "call" else 1


def walk(hand: Dict) -> Iterator[Tuple[int, Dict, np.ndarray, List[int]]]:
    """Yield (action index, action, context vector of the actor, board) for postflop actions."""
    street, aggr = None, [0, 0]
    for i, a in enumerate(hand["actions"]):
        if a["street"] != street:
            street, aggr = a["street"], [0, 0]
        p = a["actor"]
        ctx = context_vector(a["street"], ip=(p == 1), faced_frac=a["faced_frac"] if a["to_call"] > 0 else 0.0,
                             pot_odds=a["pot_odds"], spr=a["spr"], pre_role=hand["pre_role"][p],
                             pre_raises=hand["pre_raises"], opp_aggr_street=aggr[1 - p])
        yield i, a, ctx, hand["boards"][a["street"]]
        if a["kind"] in ("bet", "raise"):
            aggr[p] += 1


def training_rows(hands: List[Dict]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Design matrices from showdown hands with known cards: {model: (X, y, hand_row_id)}."""
    rows: Dict[str, Tuple[list, list, list]] = {k: ([], [], []) for k in CLASSES}
    for hi, h in enumerate(hands):
        if not h["showdown"]:
            continue
        for _, a, ctx, board in walk(h):
            hole = h["hole"][a["actor"]]
            b = bucket(a)
            if hole is None or b is None:
                continue
            cf = combo_matrix(board)[combo_index(*hole)][None, :]
            rows[b[0]][0].append(design(cf, ctx)[0])
            rows[b[0]][1].append(b[1])
            rows[b[0]][2].append(hi)
    return {k: (np.array(x), np.array(y), np.array(g)) for k, (x, y, g) in rows.items()}


class LikelihoodModel:
    def __init__(self, C: float = 1.0, eps: float = EPS):
        self.C, self.eps = C, eps
        self.models = {}

    def fit(self, hands: List[Dict], seed: int = 0) -> "LikelihoodModel":
        for k, (X, y, _) in training_rows(hands).items():
            m = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=3000, random_state=seed))
            m.fit(X, y)
            self.models[k] = m
        return self

    def proba(self, model: str, board, ctx: np.ndarray) -> np.ndarray:
        """(1326, n_classes) action probabilities for every combo."""
        return self.models[model].predict_proba(design(combo_matrix(board), ctx))

    def likelihood(self, a: Dict, board, ctx: np.ndarray) -> Optional[np.ndarray]:
        """Floored P(observed action | combo) for all 1,326 combos, or None (no update)."""
        b = bucket(a)
        if b is None:
            return None
        p = self.proba(b[0], board, ctx)[:, b[1]]
        return self.eps + (1 - self.eps) * p

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "LikelihoodModel":
        with open(path, "rb") as f:
            return pickle.load(f)
