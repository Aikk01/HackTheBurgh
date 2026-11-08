"""Hand histories -> clean heads-up-from-the-flop action sequences.

Keeps only hands that (a) come from a 6-seat table with >= 3 players dealt in and
(b) have exactly two players left when the flop is dealt.  Everything else is dropped.

Each kept hand becomes a plain dict (pickle/JSON friendly)::

    {
      "id", "table", "day", "source",
      "players": [name_a, name_b],            # the two flop players, OOP first
      "positions": ["BB", "BTN"],             # 6-max position names (pokersim engine)
      "pre_role": ["call", "agg"],            # agg | call | limp  (see preflop_roles)
      "pre_raises": 1,                        # number of preflop raises (1 = SRP, 2 = 3-bet pot)
      "hole": [[c, c] | None, ...],           # known only if shown at showdown
      "boards": {"flop": [...], "turn": [...], "river": [...]},
      "actions": [ {street, actor (0 = OOP, 1 = IP), kind, to, added, pot, to_call,
                    eff_stack, frac, faced_frac, pot_odds, spr} ... ],   # postflop only
      "showdown": bool,                       # reached showdown (no postflop fold)
      "share": [s_a, s_b] | None,             # fraction of the pot won (ties = 0.5)
      "bb": big blind,
    }
"""
from __future__ import annotations

import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

from pokersim import phh
from pokersim.replay import replay

from .cards import card_from_str

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data", "cache")
PROCESSED = os.path.join(ROOT, "data", "processed")
STREETS = ("flop", "turn", "river")


def _quick_filter(rec: phh.HandRecord) -> bool:
    """Cheap pre-filter on the raw action strings (avoids replaying 60% of hands)."""
    if rec.variant != "NT" or rec.seat_count != 6:
        return False
    n = sum(1 for a in rec.actions if a.startswith("d dh"))
    if n < 3:
        return False
    folded = set()
    for a in rec.actions:
        t = a.split()
        if t[0] == "d" and t[1] == "db":
            return n - len(folded) == 2
        if t[0] != "d" and t[1] == "f":
            folded.add(t[0])
    return False


def preflop_roles(history: List[Dict[str, Any]], live: List[int]) -> tuple[List[str], int]:
    """Role of each live player from the preflop action history.

    agg  = made the last preflop raise
    call = voluntarily called a raise (or is the BB facing a raise)
    limp = unraised pot (limped / checked option)
    """
    pre = [h for h in history if h["street"] == "preflop" and h["player"] is not None]
    raises = [h for h in pre if h["action"] in ("raise", "bet")]
    if not raises:
        return ["limp", "limp"], 0
    last = raises[-1]["player"]
    return ["agg" if p == last else "call" for p in live], len(raises)


def extract(rec: phh.HandRecord) -> Optional[Dict[str, Any]]:
    if not _quick_filter(rec):
        return None
    try:
        res = replay(rec, snapshots=False, observations=True)
    except Exception:
        return None
    if any("negative blind" in w for w in res.warnings) or not res.complete:
        return None
    post = [d for d in res.decisions if d.street != "preflop"]
    if not post:
        return None   # everyone all-in preflop
    o0 = post[0].observation
    live = sorted(p["index"] for p in o0["players"] if not p["folded"])
    if len(live) != 2:
        return None
    local = {g: i for i, g in enumerate(live)}  # global index -> 0 (OOP) / 1 (IP)
    roles, n_raises = preflop_roles(o0["history"], live)

    boards: Dict[str, List[int]] = {}
    actions = []
    for d in post:
        o = d.observation
        boards.setdefault(d.street, [card_from_str(c) for c in o["board"]])
        me = o["hero"]
        opp = next(p for p in o["players"] if p["index"] != me["index"] and not p["folded"])
        pot = float(o["total_pot"])
        to_call = float(o["to_call"])
        kind = d.actual["action"]
        to = d.actual.get("to")
        added = (float(to) - float(me["bet"])) if kind in ("bet", "raise") and to is not None else (
            to_call if kind == "call" else 0.0)
        stack = me["stack"] if me["stack"] is not None else 1e9
        ostack = opp["stack"] if opp["stack"] is not None else 1e9
        eff = min(stack, ostack + float(opp["bet"]) - float(me["bet"]))
        actions.append({
            "street": d.street, "actor": local[d.player], "kind": kind,
            "to": to, "added": added, "pot": pot, "to_call": to_call, "eff_stack": eff,
            "frac": added / pot if pot > 0 else 0.0,                    # bet size / pot (before it)
            "faced_frac": to_call / max(pot - to_call, 1e-9),          # size of the bet being faced
            "pot_odds": to_call / (pot + to_call) if to_call > 0 else 0.0,
            "spr": eff / pot if pot > 0 else 0.0,
            "all_in": added >= stack - 1e-9 and added > 0,
        })
    # the river (or turn) may have been dealt with no action (all-in): take the final board
    final_board = [c for c in res.hand.board if c is not None]
    for s, n in zip(STREETS, (3, 4, 5)):
        if len(final_board) >= n:
            boards.setdefault(s, final_board[:n])

    hole: List[Optional[List[int]]] = []
    for g in live:
        cards = res.hand.players[g].final_cards()
        hole.append([int(c) for c in cards] if cards and all(c is not None for c in cards) else None)

    showdown = not any(a["kind"] == "fold" for a in actions) and len(final_board) == 5
    share = None
    pay = [res.hand.players[g].payout for g in live]
    if sum(pay) > 0:
        share = [p / sum(pay) for p in pay]
    return {
        "id": rec.meta.get("hand"), "table": rec.meta.get("table"), "day": rec.meta.get("day"),
        "source": os.path.basename(rec.source or ""),
        "players": [rec.player_name(g) for g in live],
        "positions": [o0["players"][g]["position"] for g in live],
        "pre_role": roles, "pre_raises": n_raises, "hole": hole, "boards": boards,
        "actions": actions, "showdown": showdown, "share": share,
        "bb": max(rec.blinds) if rec.blinds else 1.0,
    }


def _parse_file(path: str) -> List[Dict[str, Any]]:
    out = []
    for rec in phh.load_file(path):
        rec.source = path
        h = extract(rec)
        if h is not None:
            out.append(h)
    return out


def parse_files(paths: Iterable[str], workers: int = 8) -> List[Dict[str, Any]]:
    paths = sorted(paths)
    with ProcessPoolExecutor(workers) as ex:
        return [h for chunk in ex.map(_parse_file, paths) for h in chunk]


def load_hands(pattern: str = "handhq/PS-*200NLH*/2/*.phhs", refresh: bool = False) -> List[Dict[str, Any]]:
    """Parse (or load the cached parse of) all downloaded files matching ``pattern``."""
    os.makedirs(PROCESSED, exist_ok=True)
    out = os.path.join(PROCESSED, "hu_flop_hands.pkl")
    paths = glob.glob(os.path.join(CACHE, pattern))
    if os.path.exists(out) and not refresh:
        with open(out, "rb") as f:
            cached = pickle.load(f)
        if cached["n_files"] == len(paths):
            return cached["hands"]
    hands = parse_files(paths)
    with open(out, "wb") as f:
        pickle.dump({"n_files": len(paths), "hands": hands}, f)
    return hands


if __name__ == "__main__":
    import collections
    import sys

    H = load_hands(refresh="--refresh" in sys.argv)
    sd = [h for h in H if h["showdown"]]
    print("HU-from-flop hands:", len(H))
    print("showdown:", len(sd), "| >=1 hole known:", sum(any(x is not None for x in h["hole"]) for h in sd),
          "| both known:", sum(all(x is not None for x in h["hole"]) for h in sd))
    print("tables:", len({h["table"] for h in H}), "players:", len({p for h in H for p in h["players"]}))
    print("positions:", collections.Counter(tuple(h["positions"]) for h in H).most_common(5))
    print("roles:", collections.Counter(tuple(h["pre_role"]) for h in H).most_common(5))
    print("actions:", collections.Counter(a["kind"] for h in H for a in h["actions"]))
