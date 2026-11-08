"""Personal model: what *this* player will do next, for you and for every opponent at the table.

The prior is the next-move model's public prediction (``pokersim.next_move``: what people in general do here, without
the player's cards).  Each player's own choices then shift it, per situation bucket (street x facing a bet) and action
group (fold / call / raise), by Dirichlet shrinkage::

    p(a) = p_prior(a) * [(n_g + alpha * q_g) / (N + alpha)] / q_g

where ``q_g`` is the prior's share of group ``g`` and ``n_g / N`` the player's own counts in that bucket.  With no data
this is the prior; with many decisions it follows the player's own frequencies.  Every prediction is made *before*
the action and scored after it, next to the "usual move" baseline (the player's most frequent group in that bucket
so far).  Human players' counts and decision logs persist in ``data/players/<name>/``.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .rl_features import ACTIONS, N_ACTIONS

PLAYERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "players")
ALPHA = 5.0
GROUPS = ("fold", "call", "raise")
GROUP_OF = np.array([0, 1, 2, 2, 2, 2, 2])


def group_of_action(action: Dict[str, Any]) -> int:
    a = action.get("action")
    return 0 if a == "fold" else 1 if a in ("check", "call") else 2


def bucket_of(obs: Dict[str, Any], j: int) -> str:
    from .next_move import facing

    to_call, _, _ = facing(obs, j)
    return f"{obs['street']}|{'bet' if to_call > 0 else 'free'}"


class PlayerModel:
    def __init__(self, name: str, persist: bool = False):
        self.name = name
        self.persist = persist
        self.counts: Dict[str, List[float]] = {}
        self.right = self.total = self.baseline_right = 0
        self.last: Optional[Dict[str, Any]] = None
        if persist:
            self._load()

    # ------------------------------------------------------------ persistence (humans only)
    def _dir(self) -> str:
        return os.path.join(PLAYERS_DIR, re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name)[:64] or "player")

    def _load(self) -> None:
        try:
            with open(os.path.join(self._dir(), "model.json"), encoding="utf-8") as f:
                d = json.load(f)
            self.counts = d.get("counts", {})
            self.right, self.total, self.baseline_right = d.get("right", 0), d.get("total", 0), d.get("baseline_right", 0)
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        os.makedirs(self._dir(), exist_ok=True)
        tmp = os.path.join(self._dir(), "model.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"counts": self.counts, "right": self.right, "total": self.total,
                       "baseline_right": self.baseline_right}, f)
        os.replace(tmp, os.path.join(self._dir(), "model.json"))

    def log(self, row: Dict[str, Any]) -> None:
        if self.persist:
            os.makedirs(self._dir(), exist_ok=True)
            with open(os.path.join(self._dir(), "decisions.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=float) + "\n")

    # ------------------------------------------------------------ model
    def predict(self, obs: Dict[str, Any], j: int) -> Tuple[np.ndarray, str]:
        """Probabilities over :data:`ACTIONS` for player ``j`` (public information only) and the prior's source."""
        from .next_move import hypothetical_mask, predict as nm_predict
        from .rl_features import action_targets

        mask = action_targets(obs)[0] if (obs["hero"]["index"] == j and obs.get("legal_actions")) else hypothetical_mask(obs, j)[0]
        prior = None
        try:
            prior = nm_predict(obs, j, with_cards=False)
        except Exception:  # noqa: BLE001
            prior = None
        source = "next_move"
        if prior is None:
            prior = mask / mask.sum()
            source = "uniform"
        n = np.asarray(self.counts.get(bucket_of(obs, j), [0.0, 0.0, 0.0]))
        q = np.array([prior[GROUP_OF == g].sum() for g in range(3)])
        p = prior.copy()
        for g in range(3):
            if q[g] > 1e-9:
                p[GROUP_OF == g] *= ((n[g] + ALPHA * q[g]) / (n.sum() + ALPHA)) / q[g]
        p = p * mask
        return p / max(p.sum(), 1e-12), source

    def usual_group(self, obs: Dict[str, Any], j: int) -> Optional[int]:
        n = self.counts.get(bucket_of(obs, j))
        return int(np.argmax(n)) if n and sum(n) > 0 else None

    def observe(self, obs: Dict[str, Any], j: int, pred: np.ndarray, action: Dict[str, Any]) -> Dict[str, Any]:
        """Score the prediction made before the action, then learn from the action."""
        g_true = group_of_action(action)
        g_pred = int(GROUP_OF[int(np.argmax(pred))])
        usual = self.usual_group(obs, j)
        self.total += 1
        self.right += int(g_pred == g_true)
        self.baseline_right += int(usual is not None and usual == g_true)
        b = bucket_of(obs, j)
        n = self.counts.setdefault(b, [0.0, 0.0, 0.0])
        n[g_true] += 1.0
        group_p = [float(pred[GROUP_OF == g].sum()) for g in range(3)]
        self.last = {"predicted": GROUPS[g_pred], "p": round(group_p[g_pred], 3), "actual": GROUPS[g_true],
                     "right": g_pred == g_true, "probs": {GROUPS[g]: round(group_p[g], 3) for g in range(3)}}
        if self.persist:
            self._save()
        return self.last

    def score(self) -> Dict[str, int]:
        return {"right": self.right, "total": self.total, "baseline_right": self.baseline_right}


class TableWatch:
    """Keeps one PlayerModel per player name at a table; humans persist, bots live in memory."""

    def __init__(self) -> None:
        self.models: Dict[str, PlayerModel] = {}
        self.lock = threading.Lock()

    def model(self, name: str, human: bool = False) -> PlayerModel:
        with self.lock:
            m = self.models.get(name)
            if m is None:
                m = self.models[name] = PlayerModel(name, persist=human)
            return m

    def before(self, hand: Any, actor: int, human: bool) -> Tuple[Dict[str, Any], np.ndarray, str]:
        obs = hand.observation(actor)
        m = self.model(hand.players[actor].name, human)
        pred, src = m.predict(obs, actor)
        return obs, pred, src

    def after(self, hand: Any, actor: int, human: bool, obs: Dict[str, Any], pred: np.ndarray, action: Dict[str, Any],
              extra: Optional[Dict[str, Any]] = None) -> None:
        m = self.model(hand.players[actor].name, human)
        last = m.observe(obs, actor, pred, action)
        if human:
            m.log({"t": time.time(), "hand_id": obs.get("hand_id"), "street": obs["street"],
                   "position": obs["hero"].get("position"), "cards": obs["hero"].get("cards"), "board": obs.get("board"),
                   "pot": obs.get("total_pot"), "to_call": (obs.get("legal_actions") or {}).get("call"),
                   "history": obs.get("history"), "you_pred": last, "action": action, **(extra or {})})
