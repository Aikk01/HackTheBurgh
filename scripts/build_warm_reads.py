"""Warm start for the Table Lab's opponent reads: the default table's bots, watched for a few hundred earlier hands.

    python scripts/build_warm_reads.py [--hands 400]   # -> data/demo/warm_reads.json

At a new Lab table every bot's personal model ("This player") starts from these counts instead of from nothing, so its
read differs from "People" from the first hand, as a poker tracker's would after a few hundred hands with that player.
Only the counts carry over; the accuracy shown at the table counts live predictions only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pokersim.agents import make_agent, safe_act  # noqa: E402
from pokersim.engine import IllegalAction, Seat, Table  # noqa: E402
from pokersim.you_model import TableWatch  # noqa: E402

OUT = os.path.join(ROOT, "data", "demo", "warm_reads.json")
LAB_TABLE = ["tag", "pro", "lag", "equity", "station"]  # the Lab's default opponents (web/lab/lab.js)
STAND_IN = "pro"  # plays the human's seat; not recorded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    specs = [STAND_IN] + LAB_TABLE
    agents, seats = {}, []
    for i, spec in enumerate(specs):
        agent = make_agent(spec, seed=rng.randrange(1 << 30))
        agent.name = "stand-in" if i == 0 else spec
        agents[i + 1] = agent
        seats.append(Seat(seat_no=i + 1, name=agent.name, stack=200.0, agent=agent, kind=agent.kind, buy_in=200.0))
    table = Table(seats, small_blind=1.0, big_blind=2.0, rng=random.Random(rng.randrange(1 << 30)))
    watch = TableWatch()
    t0 = time.time()
    for _ in range(a.hands):
        for s in table.seats:
            s.stack = 200.0
        hand = table.new_hand()
        guard = 0
        while hand.status == "betting" and hand.actor is not None and guard < 400:
            guard += 1
            actor = hand.actor
            obs, pred, _ = watch.before(hand, actor, human=False)
            action = safe_act(agents[table.seat_for_player(actor).seat_no], obs)
            try:
                result = hand.act(actor, action["action"], action.get("amount"))
            except IllegalAction:
                result = hand.act(actor, "call")
            watch.after(hand, actor, False, obs, pred, result)
        if hand.status not in ("finished", "incomplete"):
            hand.finalize()
        table.finish_hand()
    out = {"hands": a.hands, "table": LAB_TABLE,
           "note": "personal-model counts after watching the Lab's default bots; live accuracy starts from zero",
           "by_agent": {spec: watch.model(spec).counts for spec in LAB_TABLE}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), default=lambda x: round(float(x), 3))
    for spec in LAB_TABLE:
        m = watch.model(spec)
        print(f"{spec:8s} {m.total:5d} moves · right {m.right / max(m.total, 1):.0%} (people {m.people_right / max(m.total, 1):.0%})")
    print(f"{a.hands} hands in {time.time() - t0:.0f} s -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
