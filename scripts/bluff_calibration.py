"""Is the bluff check right?  Make the bet it prices and count how often everyone really folds.

    python scripts/bluff_calibration.py [--hands 1500]   # -> data/demo/bluff_calibration.json

Plays the Poker Mirror default table (a stand-in in your seat, the bots warm-started from data/demo/warm_reads.json
as on screen, their reads learning as they go).  At a random half of the stand-in's turns where a bet is allowed, it
asks lab_api.bluff for one of the size buttons (min, 1/2 pot, pot, 2x pot), then really makes that bet and lets the
bots answer.  Recorded per probe: the predicted "they all fold" against whether they all did, and per opponent the
predicted fold chance against what they did (only when everyone before them folded, which is what the check assumes).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pokersim import lab_api  # noqa: E402
from pokersim.agents import make_agent, safe_act  # noqa: E402
from pokersim.engine import IllegalAction, Seat, Table  # noqa: E402
from pokersim.lab_api import MIRROR_TABLE  # noqa: E402
from pokersim.you_model import TableWatch  # noqa: E402

OUT = os.path.join(ROOT, "data", "demo", "bluff_calibration.json")
BINS = [0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0001]


class Stand:
    """The parts of GameSession that lab_api.bluff reads."""

    def __init__(self, watch):
        self.lock = threading.Lock()
        self.hand = None
        self.human_seats = [1]
        self.watch = watch

    def _seat_of(self, i):
        return self.hand.players[i].seat if hasattr(self.hand.players[i], "seat") else i + 1


def sizes(obs):
    la = obs["legal_actions"]
    pot, to_call, mine = float(obs["total_pot"]), float(la.get("call") or 0), float(obs["hero"]["bet"] or 0)
    lo, hi = float(la["min_raise_to"]), float(la["max_raise_to"])
    rnd = lambda x: min(hi, max(lo, round(x)))  # noqa: E731 - the same buttons as web/lab/lab.js renderSizes
    return {"min": lo, "half": rnd(mine + to_call + 0.5 * (pot + to_call)), "pot": rnd(mine + to_call + pot + to_call),
            "2x": rnd(mine + to_call + 2 * (pot + to_call))}


def table_of(bins_of, pred, real):
    pred, real = np.asarray(pred), np.asarray(real)
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        k = (pred >= lo) & (pred < hi)
        if k.sum():
            rows.append({"from": lo, "to": min(hi, 1.0), "n": int(k.sum()), "said": round(float(pred[k].mean()), 3),
                         "happened": round(float(real[k].mean()), 3)})
    w = np.array([r["n"] for r in rows], float)
    ece = float(np.sum(w * np.abs([r["said"] - r["happened"] for r in rows])) / max(w.sum(), 1))
    return {"n": int(len(pred)), "said": round(float(pred.mean()), 3) if len(pred) else None,
            "happened": round(float(real.mean()), 3) if len(real) else None, "ece": round(ece, 3), "bins": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--warm", default=os.path.join(ROOT, "data", "demo", "warm_reads.json"), help="its table is the one played")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fit", action="store_true", help="score the raw fold chances and fit the correction -> data/demo/bluff_fit.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    warm = json.load(open(a.warm))
    watch = TableWatch()
    specs = ["pro"] + warm["table"]
    agents, seats = {}, []
    for i, spec in enumerate(specs):
        ag = make_agent(spec, seed=rng.randrange(1 << 30))
        ag.name = "stand-in" if i == 0 else dict(MIRROR_TABLE).get(spec, ag.label)
        if i:  # as the server does for a new Mirror table
            m = watch.model(ag.name)
            m.counts = json.loads(json.dumps(warm["by_agent"][spec]))
            m.warm_hands = int(warm["hands"])
            m.range_obs = [list(x) for x in warm["range_obs"][spec]]
        agents[i + 1] = ag
        seats.append(Seat(seat_no=i + 1, name=ag.name, stack=200.0, agent=ag, kind=ag.kind, buy_in=200.0))
    table = Table(seats, small_blind=1.0, big_blind=2.0, rng=random.Random(rng.randrange(1 << 30)))
    stand = Stand(watch)
    allp, allr, by_size, opp_p, opp_r, by_opp, pays = [], [], {}, [], [], {}, []
    for _ in range(a.hands):
        for s in table.seats:
            s.stack = 200.0
        hand = table.new_hand()
        stand.hand = hand
        me = next(i for i, p in enumerate(hand.players) if p.name == "stand-in")
        stand.human_seats = [stand._seat_of(me)]
        probe, guard = None, 0
        while hand.status == "betting" and hand.actor is not None and guard < 400:
            guard += 1
            actor = hand.actor
            obs, pred, _ = watch.before(hand, actor, human=False)
            if actor == me and probe is None and (obs.get("legal_actions") or {}).get("raise") and rng.random() < 0.5:
                name, amt = rng.choice(list(sizes(obs).items()))
                b = lab_api.bluff(stand, amt, raw=a.fit)
                if b and b["players"]:
                    probe = {"size": name, "pred": b["all_fold"], "rows": b["players"], "seen": [], "worth": b["worth"],
                             "pot": b["pot"], "cost": b["cost"]}
                    action = {"action": obs["legal_actions"].get("raise_kind", "raise"), "amount": b["amount"]}
                else:
                    action = safe_act(agents[table.seat_for_player(actor).seat_no], obs)
            else:
                action = safe_act(agents[table.seat_for_player(actor).seat_no], obs)
            if probe is not None and actor != me and not probe.get("over"):
                probe["seen"].append((actor, action["action"] == "fold"))
            if probe is not None and actor == me and probe["seen"]:
                probe["over"] = True  # action came back to us: someone did not fold
            try:
                result = hand.act(actor, action["action"], action.get("amount"))
            except IllegalAction:
                result = hand.act(actor, "call")
                if probe is not None and probe["seen"] and probe["seen"][-1][0] == actor:
                    probe["seen"][-1] = (actor, False)
            watch.after(hand, actor, False, obs, pred, result)
            if probe is not None and probe["seen"] and not probe["seen"][-1][1]:
                probe["over"] = True
        if hand.status not in ("finished", "incomplete"):
            hand.finalize()
        table.finish_hand()
        if probe is None:
            continue
        folded = {j: f for j, f in probe["seen"]}
        everyone = all(folded.get(r["index"], False) for r in probe["rows"])
        allp.append(probe["pred"])
        allr.append(float(everyone))
        by_size.setdefault(probe["size"], ([], []))
        by_size[probe["size"]][0].append(probe["pred"])
        by_size[probe["size"]][1].append(float(everyone))
        pays.append((probe["worth"], everyone * probe["pot"] - (1 - everyone) * probe["cost"]))
        for r in probe["rows"]:  # each opponent's fold chance, scored while everyone before them had folded
            if r["index"] not in folded:
                break
            opp_p.append(r["fold"])
            opp_r.append(float(folded[r["index"]]))
            by_opp.setdefault(r["name"], ([], []))
            by_opp[r["name"]][0].append(r["fold"])
            by_opp[r["name"]][1].append(float(folded[r["index"]]))
            if not folded[r["index"]]:
                break
    pw = np.array(pays, float)
    out = {"hands": a.hands, "table": warm["table"], "seed": a.seed,
           "all_fold": table_of(BINS, allp, allr),
           "each_opponent": table_of(BINS, opp_p, opp_r),
           "by_size": {k: table_of(BINS, *v) for k, v in by_size.items()},
           "by_opponent": {k: table_of(BINS, *v) for k, v in by_opp.items()},
           "worth": {"said": round(float(pw[:, 0].mean()), 2), "happened_bluff_only": round(float(pw[:, 1].mean()), 2),
                     "said_pays": int((pw[:, 0] > 0).sum()), "n": int(len(pw))} if len(pw) else None}
    if a.fit:  # logistic recalibration of each player's fold chance: logit -> a * logit + b
        x = np.log(np.clip(opp_p, 1e-4, 1 - 1e-4) / (1 - np.clip(opp_p, 1e-4, 1 - 1e-4)))
        y, w = np.asarray(opp_r), np.array([1.0, 0.0])
        for _ in range(50):  # Newton steps
            q = 1 / (1 + np.exp(-(w[0] * x + w[1])))
            g = np.array([np.sum((q - y) * x), np.sum(q - y)])
            hs = q * (1 - q)
            H = np.array([[np.sum(hs * x * x), np.sum(hs * x)], [np.sum(hs * x), np.sum(hs)]]) + 1e-6 * np.eye(2)
            w = w - np.linalg.solve(H, g)
        fit = {"a": round(float(w[0]), 4), "b": round(float(w[1]), 4), "seed": a.seed, "hands": a.hands, "n": len(opp_p),
               "table": warm["table"], "note": "fitted on this session only; data/demo/bluff_calibration.json scores it on another"}
        with open(os.path.join(ROOT, "data", "demo", "bluff_fit.json"), "w", encoding="utf-8") as f:
            json.dump(fit, f, indent=1)
        print("fit", fit)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("all_fold", "each_opponent", "worth")}, indent=1))
    for k, v in out["by_opponent"].items():
        print(f"{k:12s} n {v['n']:4d} said {v['said']:.2f} happened {v['happened']:.2f} ece {v['ece']:.3f}")


if __name__ == "__main__":
    main()
