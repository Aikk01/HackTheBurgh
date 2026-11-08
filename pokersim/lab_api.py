"""Model values for the Poker Mirror page (``web/lab.html``): one call, ``GET /api/lab/predict``.

The page prints "No model loaded" wherever a value below is ``None``.  Response::

    {
      "key": "<hand id>|<events>",
      "models": {"base":     {"actions": bool, "hands": bool},    # trained on historical hands (next_move, hand_range)
                 "rl":       {"actions": bool},                   # PPO policy (pokersim.rl_train)
                 "personal": {"actions": bool}},                  # per-player models (pokersim.you_model)
      "you": {"index": int, "your_turn": bool,
              "should": {"base": Pred|None, "rl": Pred|None, "rl_value_bb": float|None},   # what you should do
              "would":  {"personal": Pred|None},                  # what you would do (the page reveals it after you act)
              "accuracy": {"right": int, "total": int, "baseline_right": int}|None},
      "players": [{"index": int, "seat": int, "name": str, "folded": bool,
                   "base":        {"action": Pred|None, "hands": Hands|None},   # what people do / likely hands
                   "this_player": {"action": Pred|None, "hands": Hands|None,    # this opponent as watched at the table
                                   "seen": int, "accuracy": {...}}}]
    }
    Pred  = {"actions": [{"action": "fold|check|call|bet|raise|all_in", "amount": float|None, "p": float}, ...]}  # top first
    Hands = {"grid": 13 x 13 probabilities, "density": 13 x 13 per-combination, "top": [{"hand": "AKo", "p": 0.09}, ...],
             "shown": "AKo"|None}

Base "next move" for an opponent uses only public information (never their cards); "what you should do" (Base) uses
your cards and plays like Pluribus; RL is the PPO policy.  ``this_player.hands`` carries a coarse summary of the same
predicted range ("pair+ 35%") rather than a second model.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from . import cards as C

_COMBOS = np.array([[6 if r == c else 4 if r < c else 12 for c in range(13)] for r in range(13)], dtype=float)


def _status() -> Dict[str, Dict[str, bool]]:
    from . import hand_range as HR
    from . import next_move as NM
    from .rl_policy import POLICY_NPZ
    return {"base": {"actions": os.path.exists(NM.MODEL_PATH), "hands": os.path.exists(HR.MODEL_PATH)},
            "rl": {"actions": os.path.exists(POLICY_NPZ)},
            "personal": {"actions": True}}


def _hands(probs: np.ndarray, shown: Optional[str] = None) -> Dict[str, Any]:
    from . import hand_range as HR
    order = np.argsort(-probs)[:6]
    g = np.asarray(HR.grid(probs))
    return {"grid": g.tolist(), "density": np.round(g / _COMBOS, 6).tolist(),
            "top": [{"hand": HR.CLASS_NAMES[int(k)], "p": round(float(probs[k]), 4)} for k in order], "shown": shown}


def _summary(probs: np.ndarray, board: List[int]) -> Dict[str, Any]:
    """Coarse reading of a range: preflop by hand type, after the flop by what it holds on this board."""
    from . import hand_range as HR
    if len(board) >= 3:
        g = HR.class_board_features(board)
        strong = (g[:, 12:16].sum(1) + g[:, 18]) > 0
        pair = ((g[:, 4] + g[:, 6] + g[:, 11] + HR._CLS_PAIR) > 0) & ~strong
        draw = ((g[:, 17] + g[:, 19]) > 0) & ~strong & ~pair
        groups = {"two pair+": strong, "a pair": pair, "a draw": draw, "nothing": ~(strong | pair | draw)}
    else:
        hi, lo = HR._CLS_HI, HR._CLS_LO
        pairs = HR._CLS_PAIR > 0
        ace = (hi == 12) & ~pairs
        broad = (lo >= 8) & ~pairs & ~ace
        suited = (HR._CLS_SUITED > 0) & ~pairs & ~ace & ~broad
        groups = {"pairs": pairs, "ace-x": ace, "broadway": broad, "suited": suited,
                  "other": ~(pairs | ace | broad | suited)}
    mass = sorted(((float(probs[m].sum()), k) for k, m in groups.items()), reverse=True)
    return {"top": [{"hand": k, "p": round(p, 3)} for p, k in mass]}


def _pred(probs: Optional[np.ndarray], targets: List[Optional[Dict[str, Any]]], obs: Dict[str, Any], j: int) -> Optional[Dict[str, Any]]:
    """7-class probabilities -> the page's action list (engine names, raise-to amounts)."""
    if probs is None:
        return None
    from .next_move import facing
    to_call, _, _ = facing(obs, j)
    out = []
    for i in np.argsort(-probs):
        if probs[i] <= 0 or targets[i] is None:
            continue
        t = targets[i]
        if i == 0:
            out.append({"action": "fold", "amount": None, "p": float(probs[i])})
        elif i == 1:
            out.append({"action": "call", "amount": to_call, "p": float(probs[i])} if to_call > 0 else
                       {"action": "check", "amount": None, "p": float(probs[i])})
        else:
            name = "all_in" if i == 6 else t.get("action", "raise")
            out.append({"action": name, "amount": t.get("amount"), "p": float(probs[i])})
    return {"actions": [{**a, "p": round(a["p"], 4)} for a in out]}


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
        over = hand.status != "betting"
        players = []
        for p in hand.players:
            if p.index == viewer:
                continue
            shown = None
            if over and not p.folded and p.shown and not p.mucked:  # only cards revealed at showdown, never mucked ones
                fc = [c for c in p.shown if c is not None]
                if len(fc) == 2:
                    from . import hand_range as HR
                    shown = HR.CLASS_NAMES[HR.class_of(fc[0], fc[1])]
            players.append({"index": p.index, "seat": game._seat_of(p.index), "name": p.name, "folded": p.folded,
                            "shown": shown})
        your_turn = viewer is not None and hand.status == "betting" and hand.actor == viewer
        obs = hand.observation(viewer) if viewer is not None and hand.config.hole_count == 2 else None
        styles = {p.index: game.styles.estimate(p.name) for p in hand.players}
        watch = getattr(game, "watch", None) if status["personal"]["actions"] else None
        models = {p["index"]: watch.model(p["name"]) for p in players} if watch else {}
        you_model = watch.model(hand.players[viewer].name, human=True) if (watch and viewer is not None) else None

    board = [c for c in (C.parse_card(x) for x in (obs or {}).get("board") or []) if c is not None]
    ranges: Dict[int, np.ndarray] = {}
    if obs is not None and status["base"]["hands"]:
        from . import hand_range as HR
        try:
            ranges = HR.predict_ranges(obs, [p["index"] for p in players if not p["folded"]])
        except Exception:  # noqa: BLE001 - a broken model file shows as "No model loaded", not a crash
            ranges = {}

    from .next_move import hypothetical_mask, predict as nm_predict
    rows = []
    for p in players:
        j = p["index"]
        base_act = this_act = None
        if obs is not None and not p["folded"] and not over:
            targets = hypothetical_mask(obs, j)[1]
            if status["base"]["actions"]:
                try:
                    base_act = _pred(nm_predict(obs, j, with_cards=False), targets, obs, j)
                except Exception:  # noqa: BLE001
                    base_act = None
            m = models.get(j)
            if m is not None:
                try:
                    this_act = _pred(m.predict(obs, j)[0], targets, obs, j)
                except Exception:  # noqa: BLE001
                    this_act = None
        hands = _hands(ranges[j], p["shown"]) if j in ranges else None
        m = models.get(j)
        rows.append({"index": j, "seat": p["seat"], "name": p["name"], "folded": p["folded"],
                     "base": {"action": base_act, "hands": hands},
                     "this_player": {"action": this_act, "hands": _summary(ranges[j], board) if j in ranges else None,
                                     "seen": m.total if m else 0, "accuracy": m.score() if m else None}})

    should_base = should_rl = rl_value = would = None
    if your_turn and obs is not None:
        from .rl_features import action_targets
        targets = action_targets(obs)[1]
        if status["base"]["actions"]:
            try:
                should_base = _pred(nm_predict(obs, viewer, with_cards=True, pluribus=True), targets, obs, viewer)
            except Exception:  # noqa: BLE001
                should_base = None
        if status["rl"]["actions"]:
            from .rl_policy import recommend
            try:
                rec = recommend(obs, styles)
                if rec.get("available"):
                    probs = np.zeros(7)
                    from .rl_features import ACTIONS
                    for q in rec["probs"]:
                        probs[ACTIONS.index(q["action"])] = q["p"]
                    should_rl = _pred(probs, targets, obs, viewer)
                    rl_value = rec.get("value_bb")
            except Exception:  # noqa: BLE001
                should_rl = None
        if you_model is not None:
            try:
                would = _pred(you_model.predict(obs, viewer)[0], targets, obs, viewer)
            except Exception:  # noqa: BLE001
                would = None

    out = {
        "key": key,
        "models": status,
        "you": None if viewer is None else {
            "index": viewer, "your_turn": your_turn,
            "should": {"base": should_base, "rl": should_rl, "rl_value_bb": rl_value},
            "would": {"personal": would},
            "accuracy": you_model.score() if you_model is not None and you_model.total else None,
            "coach_gaps": you_model.gap_summary() if you_model is not None else [],
        },
        "players": rows,
    }
    game.last_lab = out
    return out


def scorecard() -> Dict[str, Any]:
    """The "How good are the guesses?" card: results on data no model was trained on, each next to its baseline.

    Sources: data/demo/unseen_players_eval.json (scripts/eval_unseen_players.py), data/demo/calibration.json
    (scripts/calibration.py) and data/rl/sealed_eval.json (scripts/sealed_eval.py; absent until training ends).
    A missing source leaves its row None."""
    root = os.path.join(os.path.dirname(__file__), "..", "data")

    def load(*parts: str) -> Any:
        try:
            with open(os.path.join(root, *parts), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    out: Dict[str, Any] = {"moves": None, "hands": None, "coach": None}
    u, cal = load("demo", "unseen_players_eval.json"), load("demo", "calibration.json")
    if u:
        s = u["summary"]
        out["moves"] = {"ours": s["mean_personal_top1"], "people": s["mean_people_top1"], "usual": s["mean_usual_top1"],
                        "players": s["players"], "decisions": s["decisions"], "helped": s["players_helped"]}
    if cal:
        nm = cal["next_move"]["online"]
        b = max(nm["three_way"]["bins"], key=lambda x: x["n"])  # the most common confidence level
        out["moves"] = {**(out["moves"] or {}), "says": b["pred"], "happens": b["obs"], "calib_n": nm["n"]}
        a = cal["hand_range"]["pluribus"]["all_streets"]
        out["hands"] = {"top10": a["top10"], "top10_counting": a["baseline_top10"], "half": a["coverage"]["50"]["hit"],
                        "n": a["n"]}
    sealed, meta = load("rl", "sealed_eval.json"), load("rl", "policy.json") or {}
    if sealed and sealed.get("policies"):
        run = os.path.basename(str(meta.get("run_dir", "")))
        name = run if run in sealed["policies"] else next(iter(sealed["policies"]))
        row = sealed["policies"][name]
        out["coach"] = {"policy": name, "vs_equity": row["vs_equity_bot"]["all"], "ranges": row["range_inputs_gain"].get("all"),
                        "styles": row["style_inputs_gain"]["all"], "hands": sealed["hands_per_table"]}
    return out
