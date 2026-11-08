"""Personal model: what *this* player will do next, for you and for every opponent at the table.

The prior is the next-move model's public prediction (``pokersim.next_move``: what people in general do here, without
the player's cards).  Each player's own choices then tilt it, per situation bucket (street x facing a bet) and action
group (fold / call / raise), by a shrunk observed-to-expected ratio::

    p(a) ∝ p_prior(a) * (n_g + alpha) / (e_g + alpha)

where ``n_g`` counts how often the player chose group ``g`` in that bucket and ``e_g`` how often the prior expected it
(the sum of the prior's group-``g`` probability over the same decisions).  "Calls 1.3x more than people do" tilts every
situation while keeping what is specific to it; with no data the ratio is 1 and this is the prior.  Every prediction is made *before*
the action and scored after it, next to the "usual move" baseline (the player's most frequent group in that bucket
so far).  Human players' counts and decision logs persist in ``data/players/<name>/``.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .rl_features import ACTIONS, N_ACTIONS

# test and review servers point this elsewhere so they never touch real players' histories
PLAYERS_DIR = os.environ.get("POKERSIM_PLAYERS_DIR") or os.path.join(os.path.dirname(__file__), "..", "data", "players")
ALPHA = 8.0
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
        self.right = self.total = self.baseline_right = self.people_right = 0
        self.warm_hands = 0  # earlier hands the counts were seeded from (Lab bots), not scored here
        self.last: Optional[Dict[str, Any]] = None
        # where this player and the coach (RL) disagree: "bucket|you|coach" -> count, and decisions per bucket
        self.gaps: Dict[str, int] = {}
        self.gap_totals: Dict[str, int] = {}
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
            self.gaps, self.gap_totals = d.get("gaps", {}), d.get("gap_totals", {})
            self.right, self.total, self.baseline_right = d.get("right", 0), d.get("total", 0), d.get("baseline_right", 0)
            self.people_right = d.get("people_right", 0)
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        os.makedirs(self._dir(), exist_ok=True)
        tmp = os.path.join(self._dir(), "model.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"counts": self.counts, "right": self.right, "total": self.total,
                       "baseline_right": self.baseline_right, "people_right": self.people_right, "gaps": self.gaps,
                       "gap_totals": self.gap_totals}, f)
        os.replace(tmp, os.path.join(self._dir(), "model.json"))

    def log(self, row: Dict[str, Any]) -> None:
        if self.persist:
            os.makedirs(self._dir(), exist_ok=True)
            with open(os.path.join(self._dir(), "decisions.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=float) + "\n")

    # ------------------------------------------------------------ model
    def _prior(self, obs: Dict[str, Any], j: int) -> Tuple[np.ndarray, str, np.ndarray]:
        from .next_move import hypothetical_mask, predict as nm_predict
        from .rl_features import action_targets

        mask = action_targets(obs)[0] if (obs["hero"]["index"] == j and obs.get("legal_actions")) else hypothetical_mask(obs, j)[0]
        try:
            prior = nm_predict(obs, j, with_cards=False)
        except Exception:  # noqa: BLE001
            prior = None
        if prior is None:
            return mask / mask.sum(), "uniform", mask
        return prior, "next_move", mask

    def predict(self, obs: Dict[str, Any], j: int) -> Tuple[np.ndarray, str]:
        """Probabilities over :data:`ACTIONS` for player ``j`` (public information only) and the prior's source."""

        prior, source, mask = self._prior(obs, j)
        c = self.counts.get(bucket_of(obs, j), [0.0] * 6)
        n, e = np.asarray(c[:3]), np.asarray(c[3:6] if len(c) >= 6 else [0.0] * 3)
        ratio = (n + ALPHA) / (e + ALPHA)
        p = prior * ratio[GROUP_OF] * mask
        return p / max(p.sum(), 1e-12), source

    def usual_group(self, obs: Dict[str, Any], j: int) -> Optional[int]:
        n = self.counts.get(bucket_of(obs, j))
        return int(np.argmax(n[:3])) if n and sum(n[:3]) > 0 else None

    def observe(self, obs: Dict[str, Any], j: int, pred: np.ndarray, action: Dict[str, Any],
                prior: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Score the prediction made before the action, then learn from the action (``prior``: the prior used)."""
        g_true = group_of_action(action)
        g_pred = int(GROUP_OF[int(np.argmax(pred))])
        usual = self.usual_group(obs, j)
        self.total += 1
        self.right += int(g_pred == g_true)
        self.baseline_right += int(usual is not None and usual == g_true)
        b = bucket_of(obs, j)
        n = self.counts.setdefault(b, [0.0] * 6)
        if len(n) < 6:
            n.extend([0.0] * (6 - len(n)))
        n[g_true] += 1.0
        if prior is None:
            prior = self._prior(obs, j)[0]
        self.people_right += int(int(GROUP_OF[int(np.argmax(prior))]) == g_true)
        for g in range(3):
            n[3 + g] += float(prior[GROUP_OF == g].sum())
        group_p = [float(pred[GROUP_OF == g].sum()) for g in range(3)]
        self.last = {"predicted": GROUPS[g_pred], "p": round(group_p[g_pred], 3), "actual": GROUPS[g_true],
                     "right": g_pred == g_true, "probs": {GROUPS[g]: round(group_p[g], 3) for g in range(3)}}
        if self.persist:
            self._save()
        return self.last

    def note_coach(self, obs: Dict[str, Any], j: int, coach_action: Optional[str], action: Dict[str, Any]) -> None:
        """Remember what the coach advised and what the player did, per situation bucket."""
        if not coach_action:
            return
        b = bucket_of(obs, j)
        cg = 0 if coach_action == "fold" else 1 if coach_action in ("check", "call") else 2
        yg = group_of_action(action)
        self.gap_totals[b] = self.gap_totals.get(b, 0) + 1
        if cg != yg:
            k = f"{b}|{GROUPS[yg]}|{GROUPS[cg]}"
            self.gaps[k] = self.gaps.get(k, 0) + 1
        if self.persist:
            self._save()

    def gap_summary(self, min_count: int = 3, top: int = 2) -> List[Dict[str, Any]]:
        """The player's most frequent disagreements with the coach, in plain words."""
        out = []
        for k, c in self.gaps.items():
            if c < min_count:
                continue
            street, facing, you, coach = k.split("|")  # key: "<street>|<bet|free>|<you>|<coach>"
            b = f"{street}|{facing}"
            where = f"{'preflop' if street == 'preflop' else 'on the ' + street}, {'facing a bet' if facing == 'bet' else 'when you can check'}"
            word = (lambda g: "check" if facing == "free" and g == "call" else "bet" if facing == "free" and g == "raise" else g)
            out.append({"where": where, "you": word(you), "coach": word(coach), "times": c, "of": self.gap_totals.get(b, c)})
        out.sort(key=lambda g: (-g["times"], -g["times"] / max(g["of"], 1)))
        return out[:top]

    def habits(self, top: int = 2, min_seen: int = 8) -> List[Dict[str, Any]]:
        """Where this player differs most from people in general: per situation, a move made how many times as often
        (the model's own tilt, shrunk by ALPHA), with the counts behind it, in plain words."""
        out = []
        for b, c in self.counts.items():
            n, e = c[:3], (c[3:6] if len(c) >= 6 else [0.0] * 3)
            seen = sum(n)
            if seen < min_seen:
                continue
            street, facing = b.split("|")
            where = ("Preflop" if street == "preflop" else f"On the {street}") + (", facing a bet" if facing == "bet" else ", when they can check")
            for g in range(3):
                ratio = (n[g] + ALPHA) / (e[g] + ALPHA)
                if abs(math.log(ratio)) < math.log(1.3):
                    continue
                move = ("check" if g == 1 else "bet") if facing == "free" and g > 0 else GROUPS[g]
                often = f"{ratio:.1f}× as often" if ratio >= 1 else f"only {round(100 * ratio)}% as often"
                out.append({"where": where, "move": move, "ratio": round(ratio, 2), "did": int(n[g]), "seen": int(seen),
                            "people": round(e[g], 1), "text": f"{where}: {move}s {often} as most people",
                            "detail": f"{int(n[g])} of {int(seen)} times; most people about {round(e[g])}",
                            "_rank": abs(math.log(ratio)) * math.sqrt(seen)})
        out.sort(key=lambda h: -h["_rank"])
        best, seen_where = [], set()
        for h in out:  # the strongest habit of each situation, so the list says different things
            if h["where"] not in seen_where:
                seen_where.add(h["where"])
                best.append({k: v for k, v in h.items() if k != "_rank"})
        return best[:top]

    def score(self) -> Dict[str, int]:
        return {"right": self.right, "total": self.total, "baseline_right": self.baseline_right,
                "people_right": self.people_right, "warm_hands": self.warm_hands}


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
