"""Fit the Poker Mirror's four type bots to the real regulars of each type (data/demo/player_types.json).

    python scripts/fit_types.py [--hands 3000] [--rounds 5]   # -> data/demo/type_tilts.json

Each bot is the card-aware model of the professionals with a tilt per situation (street x facing a bet x position) on
its fold / call / raise logits.  The tilts are fitted by simulation: play a table of the four types (plus a plain Pro
and a stand-in), measure each bot's action rates per situation, and move each tilt by the log of target over actual,
damped, for a few rounds, until the rates match the type's real regulars.

"Good at what they do": the online hands carry no results, so skill is set here.  For each candidate temperature
(how precisely a bot plays its cards) the tilts are refitted, so the style stays the same, and each type keeps the
temperature at which it won most at this table.  The output records each bot's VPIP / PFR / AF next to its real
type's, and its win rate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pokersim import agents as AG  # noqa: E402
from pokersim.agents import make_agent, safe_act  # noqa: E402
from pokersim.engine import IllegalAction, Seat, Table  # noqa: E402

TYPES = json.load(open(os.path.join(ROOT, "data", "demo", "player_types.json")))
OUT = os.path.join(ROOT, "data", "demo", "type_tilts.json")
CODES = {"tight aggressive": "tag", "loose aggressive": "lag", "loose passive": "lp", "tight passive": "tp"}
NAMES = list(CODES)
A = {"fold": 0, "check": 1, "call": 1}


def target(name):
    c = np.asarray(TYPES["counts"][name], float) + 1.0  # [street][facing][position][action], smoothed
    return c / c.sum(-1, keepdims=True), np.asarray(TYPES["counts"][name], float).sum(-1)


def play(tilts, temps, hands, seed):
    """One table: a stand-in Pro, a plain Pro, and the four types.  Returns per-type action counts per situation,
    tracker stats, and chips won."""
    AG._TYPE_TABLE["t"] = {"types": {n: {"tilt": tilts[n].tolist(), "temp": temps[n]} for n in NAMES}}
    rng = random.Random(seed)
    specs = [("pro", "stand-in"), ("pro", "Pro")] + [(f"type:style={CODES[n]}", n) for n in NAMES]
    agents, seats = {}, []
    for i, (spec, name) in enumerate(specs):
        ag = make_agent(spec, seed=rng.randrange(1 << 30))
        ag.name = name
        agents[i + 1] = ag
        seats.append(Seat(seat_no=i + 1, name=name, stack=200.0, agent=ag, kind=ag.kind, buy_in=200.0))
    table = Table(seats, small_blind=1.0, big_blind=2.0, rng=random.Random(rng.randrange(1 << 30)))
    counts = {n: np.zeros((4, 2, 3, 3)) for n in NAMES}
    track = {n: np.zeros(5) for n in NAMES + ["Pro"]}  # hands, vpip, pfr, postflop bets+raises, postflop calls
    won = {n: 0.0 for n in NAMES + ["Pro", "stand-in"]}
    streets = ["preflop", "flop", "turn", "river"]
    for _ in range(hands):
        for s in table.seats:
            s.stack = 200.0
        hand = table.new_hand()
        n = len(hand.players)
        vp, pr = set(), set()
        guard = 0
        while hand.status == "betting" and hand.actor is not None and guard < 400:
            guard += 1
            actor = hand.actor
            obs = hand.observation(actor)
            name = hand.players[actor].name
            action = safe_act(agents[table.seat_for_player(actor).seat_no], obs)
            kind = action["action"]
            g = 0 if kind == "fold" else 1 if kind in ("check", "call") else 2
            st = streets.index(obs["street"]) if obs["street"] in streets else 3
            facing = 1 if float((obs.get("legal_actions") or {}).get("call") or 0) > 0 else 0
            if name in counts:
                pos = 0 if actor < 2 else 2 if actor >= n - 2 else 1
                counts[name][st, facing, pos, g] += 1
            if name in track:
                if st == 0 and g == 2:
                    vp.add(name); pr.add(name)
                elif st == 0 and g == 1 and facing:
                    vp.add(name)
                elif st > 0 and g == 2:
                    track[name][3] += 1
                elif st > 0 and g == 1 and facing:
                    track[name][4] += 1
            try:
                hand.act(actor, kind, action.get("amount"))
            except IllegalAction:
                hand.act(actor, "call")
        if hand.status not in ("finished", "incomplete"):
            hand.finalize()
        for p in hand.players:
            if p.name in won:
                won[p.name] += (p.stack - 200.0) / 2.0
            if p.name in track:
                track[p.name][0] += 1
                track[p.name][1] += p.name in vp
                track[p.name][2] += p.name in pr
        table.finish_hand()
    return counts, track, {k: 100.0 * v / hands for k, v in won.items()}


def stats(t):
    return {"vpip": round(t[1] / max(t[0], 1), 3), "pfr": round(t[2] / max(t[0], 1), 3), "af": round(t[3] / max(t[4], 1), 2)}


def fit(temps, a, seed0):
    tilts = {n: np.zeros((4, 2, 3, 3)) for n in NAMES}
    for r in range(a.rounds):
        counts, track, won = play(tilts, temps, a.hands, seed0 + r)
        for n in NAMES:
            tgt, seen = target(n)
            c = counts[n] + 1.0
            act = c / c.sum(-1, keepdims=True)
            step = 0.7 * np.log(tgt / act)
            step -= step.mean(-1, keepdims=True)
            mask = (counts[n].sum(-1, keepdims=True) >= 20) & (seen[..., None] >= 200)  # only situations seen enough
            tilts[n] = np.clip(tilts[n] + np.where(mask, step, 0.0), -4.0, 4.0)
        print(f"  temp {temps[NAMES[0]]} round {r + 1}: " + " · ".join(f"{n.split()[0][0]}{n.split()[1][0]} {stats(track[n])}" for n in NAMES), flush=True)
    return tilts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--temps", default="1.0,0.6")
    ap.add_argument("--eval-hands", type=int, default=8000)
    a = ap.parse_args()
    res = {}
    for k, T in enumerate(float(x) for x in a.temps.split(",")):
        temps = {n: T for n in NAMES}
        tilts = fit(temps, a, 1000 * (k + 1))
        _, track, won = play(tilts, temps, a.eval_hands, 99_000 + k)
        res[T] = (tilts, track, won)
        print(f"temp {T}: won bb/100 " + " · ".join(f"{n}: {won[n]:+.0f}" for n in NAMES) + f" · Pro {won['Pro']:+.0f}", flush=True)
    out = {"source": "data/demo/player_types.json", "hands_per_round": a.hands, "rounds": a.rounds, "eval_hands": a.eval_hands,
           "note": "won: bb/100 at a table of the four types, a plain Pro and a stand-in Pro; same session for all",
           "types": {}}
    for n in NAMES:
        best = max(res, key=lambda T: res[T][2][n])
        tilts, track, won = res[best]
        out["types"][n] = {"temp": best, "tilt": np.round(tilts[n], 3).tolist(), "bot": stats(track[n]),
                           "real": {k: TYPES["types"][n][k] for k in ("vpip", "pfr", "af")},
                           "won_bb100": {str(T): round(res[T][2][n], 1) for T in res}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for n, v in out["types"].items():
        print(f"{n:17s} temp {v['temp']}: bot {v['bot']} · real {v['real']} · won {v['won_bb100']}")


if __name__ == "__main__":
    main()
