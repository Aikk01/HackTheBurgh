"""Model values for the Table Lab page (``web/lab.html``): one call, ``GET /api/lab/predict``.

The page prints "No model loaded" wherever a value below is ``None``.  To light up a slot, make this module
fill its field; the page needs no change.  Response::

    {
      "key": "<hand id>|<events>",
      "models": {"base":     {"actions": bool, "hands": bool},      # base model = trained on historical hands
                 "rl":       {"actions": bool, "hands": bool},      # PPO policy (pokersim.rl_train)
                 "personal": {"actions": bool}},                    # your own model, from your logged decisions
      "you": {"index": int, "your_turn": bool,
              "should": {"base": Pred|None, "rl": Pred|None},       # what you should do
              "would":  {"personal": Pred|None},                    # what you would do
              "accuracy": {"right": int, "total": int, "baseline_right": int}|None},
      "players": [{"index": int, "seat": int, "name": str,
                   "base": {"action": Pred|None, "hands": Hands|None},
                   "rl":   {"action": Pred|None, "hands": Hands|None}}]
    }
    Pred  = {"actions": [{"action": "fold|check|call|bet|raise|all_in", "amount": float|None, "p": float}, ...]}  # top first
    Hands = {"grid": 13 x 13 probabilities (rows/columns A..2, suited above the diagonal), "top": [{"hand": "AKo", "p": 0.09}, ...]}

Wired today: base-model hands (``pokersim.hand_range``, needs ``data/rl/hand_range.npz``) and the RL model's
advice for your own decision (``pokersim.rl_policy``, needs ``data/rl/policy.npz``).  Everything else is None
until its model exists.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np


def _status() -> Dict[str, Dict[str, bool]]:
    from . import hand_range as HR
    from .rl_policy import POLICY_NPZ
    return {"base": {"actions": False, "hands": os.path.exists(HR.MODEL_PATH)},
            "rl": {"actions": os.path.exists(POLICY_NPZ), "hands": False},
            "personal": {"actions": False}}


def _hands(probs: np.ndarray) -> Dict[str, Any]:
    from . import hand_range as HR
    order = np.argsort(-probs)[:6]
    return {"grid": HR.grid(probs), "top": [{"hand": HR.CLASS_NAMES[int(k)], "p": round(float(probs[k]), 4)} for k in order]}


def _rl_pred(rec: Dict[str, Any], obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not rec.get("available"):
        return None
    to_call = float((obs.get("legal_actions") or {}).get("call") or 0.0)
    kind = (obs.get("legal_actions") or {}).get("raise_kind", "raise")
    out: List[Dict[str, Any]] = []
    for p in rec["probs"]:
        a = p["action"]
        if a == "fold":
            name, amt = "fold", None
        elif a == "call":
            name, amt = ("call", to_call) if to_call > 0 else ("check", None)
        elif a == "all_in":
            name, amt = "all_in", p.get("amount")
        else:
            name, amt = kind, p.get("amount")
        out.append({"action": name, "amount": amt, "p": p["p"]})
    return {"actions": out}


def predict(game: Any) -> Dict[str, Any]:
    status = _status()
    with game.lock:
        hand = game.hand
        if hand is None:
            return {"key": None, "models": status, "you": None, "players": []}
        viewer = None
        if len(game.human_seats) == 1:
            viewer = next((p.index for p in hand.players if game._seat_of(p.index) == game.human_seats[0]), None)
        key = f"{hand.hand_id}|{len(hand.events)}"
        players = [{"index": p.index, "seat": game._seat_of(p.index), "name": p.name, "folded": p.folded}
                   for p in hand.players if p.index != viewer]
        your_turn = viewer is not None and hand.status == "betting" and hand.actor == viewer
        obs = hand.observation(viewer) if viewer is not None and hand.config.hole_count == 2 else None
        styles = {p.index: game.styles.estimate(p.name) for p in hand.players}

    ranges: Dict[int, np.ndarray] = {}
    if obs is not None and status["base"]["hands"]:
        from . import hand_range as HR
        try:
            ranges = HR.predict_ranges(obs, [p["index"] for p in players if not p["folded"]])
        except Exception:  # noqa: BLE001 - a broken model file shows as "No model loaded", not a crash
            ranges = {}
    should_rl = None
    if your_turn and status["rl"]["actions"]:
        from .rl_policy import recommend
        try:
            should_rl = _rl_pred(recommend(obs, styles), obs)
        except Exception:  # noqa: BLE001
            should_rl = None

    return {
        "key": key,
        "models": status,
        "you": None if viewer is None else {
            "index": viewer, "your_turn": your_turn,
            "should": {"base": None, "rl": should_rl},
            "would": {"personal": None},
            "accuracy": None,
        },
        "players": [{"index": p["index"], "seat": p["seat"], "name": p["name"],
                     "base": {"action": None, "hands": _hands(ranges[p["index"]]) if p["index"] in ranges else None},
                     "rl": {"action": None, "hands": None}} for p in players],
    }
