"""How well calibrated is the hand grid on the demo table's bots (not on the human hands it was trained on)?

    python scripts/grid_on_bots.py [--hands 600]   # -> data/demo/grid_on_bots.json

Plays the Lab's default table (a stand-in in the human seat), and at every decision asks the hand model for each
opponent's 169-class guess from the stand-in's point of view, exactly as the screen does.  Since every card is known
in simulation, it records whether the real hand was inside the likeliest 50 / 80 / 90% of the guess, per opponent
and street (the same measure as scripts/calibration.py on Pluribus hands: 47 / 78 / 89%).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pokersim import cards as C  # noqa: E402
from pokersim import hand_range as HR  # noqa: E402
from pokersim.agents import make_agent, safe_act  # noqa: E402
from pokersim.engine import IllegalAction, Seat, Table  # noqa: E402

OUT = os.path.join(ROOT, "data", "demo", "grid_on_bots.json")
LAB_TABLE = ["tag", "pro", "lag", "equity", "station"]  # web/lab/lab.js
LEVELS = (0.5, 0.8, 0.9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--widen", action="store_true", help="apply each bot's range width learned in data/demo/warm_reads.json")
    a = ap.parse_args()
    width = {}
    if a.widen:  # widths from the warm-up hands (another seed), so this measures them out of sample
        from pokersim.you_model import PlayerModel
        warm = json.load(open(os.path.join(ROOT, "data", "demo", "warm_reads.json")))
        for spec, obs_ in warm["range_obs"].items():
            m = PlayerModel(spec)
            m.range_obs = obs_
            width[make_agent(spec).label] = m.range_width()
    rng = random.Random(a.seed)
    specs = ["pro"] + LAB_TABLE
    agents, seats = {}, []
    for i, spec in enumerate(specs):
        ag = make_agent(spec, seed=rng.randrange(1 << 30))
        ag.name = "stand-in" if i == 0 else ag.label
        agents[i + 1] = ag
        seats.append(Seat(seat_no=i + 1, name=ag.name, stack=200.0, agent=ag, kind=ag.kind, buy_in=200.0))
    table = Table(seats, small_blind=1.0, big_blind=2.0, rng=random.Random(rng.randrange(1 << 30)))
    hits = defaultdict(lambda: np.zeros(len(LEVELS) + 1))  # (who, street) -> [inside 50, 80, 90, n]
    for _ in range(a.hands):
        for s in table.seats:
            s.stack = 200.0
        hand = table.new_hand()
        me = next(i for i, p in enumerate(hand.players) if p.name == "stand-in")
        guard = 0
        while hand.status == "betting" and hand.actor is not None and guard < 400:
            guard += 1
            actor = hand.actor
            if actor != me:  # read the table from the stand-in's seat, as the screen does for you
                obs = hand.observation(me)
                opp = [p["index"] for p in obs["players"] if p["index"] != me and not p["folded"]]
                dead = [c for c in hand.board if c is not None] + [c for c in hand.players[me].hole if c is not None]
                for j, probs in HR.predict_ranges(obs, opp).items():
                    probs = HR.widen(probs, width.get(hand.players[j].name, 0.0), dead)
                    hole = hand.players[j].hole
                    if not hole or None in hole:
                        continue
                    k = HR.class_of(hole[0], hole[1])
                    order = np.argsort(-probs)
                    cum = np.cumsum(probs[order])
                    mass_to_true = cum[int(np.nonzero(order == k)[0][0])]
                    row = hits[(hand.players[j].name, obs["street"])]
                    row[:len(LEVELS)] += [mass_to_true <= lv + 1e-9 for lv in LEVELS]
                    row[-1] += 1
            obs_a = hand.observation(actor)
            action = safe_act(agents[table.seat_for_player(actor).seat_no], obs_a)
            try:
                hand.act(actor, action["action"], action.get("amount"))
            except IllegalAction:
                hand.act(actor, "call")
        if hand.status not in ("finished", "incomplete"):
            hand.finalize()
        table.finish_hand()

    def summ(rows):
        tot = np.sum(rows, axis=0) if rows else np.zeros(len(LEVELS) + 1)
        n = int(tot[-1])
        return {"n": n, **{f"inside_{int(lv * 100)}": round(float(tot[i] / max(n, 1)), 3) for i, lv in enumerate(LEVELS)}}

    who = sorted({w for w, _ in hits})
    out = {"hands": a.hands, "table": LAB_TABLE, "levels": list(LEVELS), "widths": width,
           "reference": "Pluribus test hands (humans and Pluribus): inside 50/80/90% = 0.47 / 0.78 / 0.89",
           "all": summ(list(hits.values())),
           "by_player": {w: summ([v for (x, _), v in hits.items() if x == w]) for w in who},
           "by_street": {s: summ([v for (_, x), v in hits.items() if x == s]) for s in ("preflop", "flop", "turn", "river")}}
    with open(OUT.replace(".json", "_widened.json") if a.widen else OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("all", "by_player", "by_street")}, indent=1))


if __name__ == "__main__":
    main()
