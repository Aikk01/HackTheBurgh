"""HTTP-facing glue for Poker Mirror.  The routes are registered in :mod:`pokersim.server`.

GET  /api/mirror/state?session=TOKEN      full session state (creates the session)
GET  /api/mirror/profile?session=TOKEN
GET  /api/mirror/compare?session=TOKEN
GET  /api/mirror/exhibit                   filter exhibit (two-shard + full corpus)
GET  /api/mirror/demo_hands                the two opening replays
GET  /api/mirror/evidence                  every evidence panel with presence flags
POST /api/mirror/reset   {session}
POST /api/mirror/drill   {session}
POST /api/mirror/respond {session, decision_id, action}
POST /api/mirror/freeze  {session}
POST /api/mirror/play/new {session, scenario_index?}
POST /api/mirror/play/act {session, action}
POST /api/mirror/train   {session, episodes?, seed?}
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, List, Optional

from . import phh
from .mirror_data import extract_spot
from .mirror_session import MIRROR_DIR, REGISTRY, CONTEXT_LABELS, _load_json
from .replay import replay

_demo_cache: Optional[Dict[str, Any]] = None


def _session(params: Dict[str, Any]):
    return REGISTRY.get(params.get("session") or None)


def demo_hands() -> Dict[str, Any]:
    """The two bundled Absolute Poker hands, replayed, with the decision step located."""
    global _demo_cache
    if _demo_cache is not None:
        return _demo_cache
    path = os.path.join(MIRROR_DIR, "demo_hands.phhs")
    out: List[Dict[str, Any]] = []
    if os.path.exists(path):
        for rec in phh.load_file(path):
            res = replay(rec)
            spot, why = extract_spot(rec)
            river = [d for d in res.decisions if d.street == "river"]
            decision_step = river[2].step_index if len(river) >= 3 else len(res.steps) - 1
            out.append({
                "hand": str(rec.meta.get("hand")), "venue": rec.meta.get("venue"), "stake": spot.stake if spot else None,
                "players": [rec.player_name(i) for i in range(rec.n)], "defender": rec.player_name(0),
                "context": spot.context if spot else None, "context_label": CONTEXT_LABELS.get(spot.context, "") if spot else "",
                "bet": spot.bet if spot else None, "pot_before_bet": spot.pot_before_bet if spot else None,
                "response": spot.response if spot else None, "board": spot.board if spot else [],
                "decision_step": decision_step, "replay": res.to_dict(include_observations=False),
                "reason_if_missing": why,
            })
    _demo_cache = {"hands": out, "source": "data/mirror/demo_hands.phhs", "license": "CC BY 4.0 (takara-ai/poker_hands, Zenodo 13997158)"}
    return _demo_cache


def exhibit() -> Dict[str, Any]:
    ex = _load_json("exhibit.json")
    scan = _load_json("scan_counts.json")
    return {"present": ex is not None, "exhibit": ex, "scan": {k: scan.get(k) for k in ("rows", "prefiltered", "eligible_unique",
            "eligible_modelling", "venues", "responses_unique", "generated", "revision", "manifest")} if scan else None}


def evidence() -> Dict[str, Any]:
    ex = _load_json("exhibit.json")
    scan = _load_json("scan_counts.json")
    pop = _load_json("population.json")
    styles = _load_json("styles.json")
    style_counts = _load_json("style_counts.json")
    ev = _load_json("eval.json")
    bench = _load_json("benchmark.json")
    ranges = _load_json("range_checks.json")
    policy = _load_json("policy_population.json")
    panels = {
        "filter_exhibit": {"present": ex is not None, "data": ex, "scan": scan,
                           "source": "takara-ai/poker_hands @ " + (scan or {}).get("revision", "?")[:10], "generated": (ex or {}).get("generated")},
        "population": {"present": pop is not None, "data": pop, "source": "training manifest", "generated": (pop or {}).get("generated")},
        "forecast_ladder": {"present": ev is not None, "data": ev, "source": "frozen 20% test manifest", "generated": (ev or {}).get("generated")},
        "type_gate": {"present": styles is not None, "data": {k: v for k, v in (styles or {}).items() if k != "map_points"},
                      "style_counts": style_counts, "source": "style table up to each venue cutoff", "generated": (styles or {}).get("generated")},
        "decision_benchmark": {"present": bench is not None, "data": bench, "source": "simulated matched deals", "generated": (bench or {}).get("generated")},
        "range_checks": {"present": ranges is not None, "data": ranges, "source": "held-out sessions / Pluribus / twin", "generated": (ranges or {}).get("generated")},
        "population_bot": {"present": policy is not None, "data": {"meta": (policy or {}).get("meta"), "checkpoints": (policy or {}).get("checkpoints")} if policy else None,
                           "source": "Q-learning vs. the context-only twin", "generated": None},
    }
    return {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "panels": panels}


def get(path: str, qs: Dict[str, str]) -> Any:
    if path == "/api/mirror/state":
        return _session(qs).state()
    if path == "/api/mirror/profile":
        return _session(qs).profile_summary()
    if path == "/api/mirror/compare":
        return _session(qs).compare_bots()
    if path == "/api/mirror/exhibit":
        return exhibit()
    if path == "/api/mirror/demo_hands":
        return demo_hands()
    if path == "/api/mirror/evidence":
        return evidence()
    return None


def post(path: str, body: Dict[str, Any]) -> Any:
    if path == "/api/mirror/reset":
        token = body.get("session") or REGISTRY.get(None).token
        return REGISTRY.reset(token).state()
    s = _session(body)
    if path == "/api/mirror/drill":
        return {"token": s.token, **s.new_drill(), "profile": s.profile_summary()}
    if path == "/api/mirror/respond":
        return {"token": s.token, **s.respond(str(body.get("decision_id")), str(body.get("action", "")))}
    if path == "/api/mirror/freeze":
        return {"token": s.token, "frozen": s.freeze(), "profile": s.profile_summary()}
    if path == "/api/mirror/play/new":
        idx = body.get("scenario_index")
        return {"token": s.token, **s.new_play_spot(int(idx) if idx is not None else None)}
    if path == "/api/mirror/play/act":
        return {"token": s.token, **s.play_act(str(body.get("action", "")))}
    if path == "/api/mirror/train":
        return {"token": s.token, "training": s.start_training(int(body.get("episodes", 20000)), int(body.get("seed", 7)))}
    return None
