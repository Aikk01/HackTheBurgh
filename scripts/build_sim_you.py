"""A player like you, simulated: 500 hands of a human-like player with a hidden style in your seat (for the replay).

    python scripts/build_sim_you.py [--style calls|raises|folds] [--hands 500]   # -> data/demo/sim_you_<style>.json

The player in your seat is the same next-move model of human professionals as the rest of the Poker Mirror table, tilted
towards one instinct that no model is told.  The model of you starts knowing nothing about it, as it would about you.
The replay shows its guesses on the first hands (it guesses like people in general, and misses), then on the last ones
(it has learned the player), and the learning curve over all of them.  The page then hands the learned model to your
live table ("Now you play"): play in that style and watch it follow you.

Every model value comes from the same code as the live table (scripts/build_replay.py run_hand); the hands themselves
are dealt and played here by the engine.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import build_replay as BR  # noqa: E402
from pokersim import you_model as Y  # noqa: E402
from pokersim.agents import make_agent, safe_act  # noqa: E402
from pokersim.engine import IllegalAction, Seat, Table  # noqa: E402
from pokersim.lab_api import MIRROR_TABLE  # noqa: E402
from pokersim.simulate import hand_to_record  # noqa: E402

STYLES = {"calls": ("pro:tilt_c=3.0", "calls a lot"), "raises": ("pro:tilt_r=3.0", "raises a lot"),
          "folds": ("pro:tilt_f=3.0", "folds a lot")}
ME = "You"


def deal(spec: str, n: int, seed: int):
    rng = random.Random(seed)
    agents, seats = {}, []
    for i, (sp, name) in enumerate([(spec, ME)] + MIRROR_TABLE):
        ag = make_agent(sp, seed=rng.randrange(1 << 30))
        ag.name = name
        agents[i + 1] = ag
        seats.append(Seat(seat_no=i + 1, name=name, stack=200.0, agent=ag, kind=ag.kind, buy_in=200.0))
    table = Table(seats, small_blind=1.0, big_blind=2.0, rng=random.Random(rng.randrange(1 << 30)))
    recs = []
    for _ in range(n):
        for s in table.seats:
            s.stack = 200.0
        hand = table.new_hand()
        guard = 0
        while hand.status == "betting" and hand.actor is not None and guard < 400:
            guard += 1
            actor = hand.actor
            action = safe_act(agents[table.seat_for_player(actor).seat_no], hand.observation(actor))
            try:
                hand.act(actor, action["action"], action.get("amount"))
            except IllegalAction:
                hand.act(actor, "call")
        if hand.status not in ("finished", "incomplete"):
            hand.finalize()
        recs.append(hand_to_record(hand))
        table.finish_hand()
    return recs


def stand():
    """A fresh stand-in table where every model starts knowing nothing: the model of you and those of the five
    players (the page hands all of them to the live table at the end)."""
    Y.PLAYERS_DIR = tempfile.mkdtemp(prefix="sim_you_players_")
    return BR.Stand()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="calls", choices=sorted(STYLES))
    ap.add_argument("--hands", type=int, default=500)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    spec, text = STYLES[a.style]
    BR.HIM, BR.MINE = ME, True
    recs = deal(spec, a.hands, a.seed)
    n = len(recs)

    # pass 1 (no frames): how each hand went for the model of you
    probe, per_hand = stand(), {}
    for i, rec in enumerate(recs):
        _, c = BR.run_hand(_copy(rec), probe, ME, i + 1, False)
        per_hand[i] = c
    early = [i for i in range(min(20, n)) if any(r[0] == 0 for r in per_hand[i]) and len(per_hand[i]) >= 2][:3] or \
        [i for i in range(min(20, n)) if per_hand[i]][:3]
    late = [i for i in range(max(0, n - 100), n) if any(r[0] == 1 and r[1] == 0 for r in per_hand[i])
            and BR.shown_cards(recs[i], BR.his_index(recs[i], ME))]
    late = [late[j] for j in np.linspace(0, len(late) - 1, min(3, len(late))).round().astype(int)] if late else []
    pick = {i: "early" for i in early} | {i: "late" for i in late}

    # pass 2: the real run, from knowing nothing, with frames for the picked hands
    st = stand()
    curve, showcase, opp = [], [], []
    for i, rec in enumerate(recs):
        frames, c = BR.run_hand(_copy(rec), st, ME, i + 1, i in pick, opp_curve=opp)
        curve += [[i + 1] + row for row in c]
        if i in pick:
            showcase.append({"hand_no": i + 1, "phase": pick[i], "frames": frames})
    C_ = np.array([r[1:] for r in curve], float)
    first = np.array([r[1:] for r in curve if r[0] <= 20], float)
    last = np.array([r[1:] for r in curve if r[0] > n - 100], float)
    facing = np.array([r[1:] for r in curve if r[4]], float)  # facing a bet, where a style shows most
    facing_last = np.array([r[1:] for r in curve if r[4] and r[0] > n - 100], float)
    me = st.watch.model(ME, True)

    def acc(x):
        return {"personal": float(x[:, 0].mean()), "people": float(x[:, 1].mean()), "usual": float(x[:, 2].mean()),
                "decisions": int(len(x))} if len(x) else None

    def acc2(rows):  # the other players: [hand, their model right, people right]
        x = np.array([r[1:] for r in rows], float)
        return {"personal": float(x[:, 0].mean()), "people": float(x[:, 1].mean()), "decisions": int(len(x))} if len(x) else None

    out = {"player": {"label": "A player like you", "venue": "simulated", "hands": n, "decisions": len(curve),
                      "mine": True, "sim": True, "style": a.style, "style_text": text, "spec": spec,
                      "picked": "first hands (it knows nothing yet), then last hands (it has learned the player)",
                      "note": "simulated with the next-move model of human professionals, tilted to one style no model is told"},
           "habits": me.habits(top=3), "curve": curve,
           "summary": {**acc(C_), "first20": acc(first), "last100": acc(last), "facing": acc(facing),
                       "facing_last100": acc(facing_last),
                       "opp_first20": acc2([r for r in opp if r[0] <= 20]), "opp_last100": acc2([r for r in opp if r[0] > n - 100])},
           "opp_curve": opp,
           "opponents": {name: {"counts": st.watch.model(name).counts, "habits": st.watch.model(name).habits(top=1)}
                         for _, name in MIRROR_TABLE},
           "showcase_rule": "up to 3 of the first 10 hands where the model of you missed, then up to 3 of the last 100 "
                            "where it was right and people in general were not",
           "hands": showcase, "counts": me.counts}
    path = os.path.join(ROOT, "data", "demo", f"sim_you_{a.style}.json")
    s = json.dumps(out, separators=(",", ":"), default=lambda x: round(float(x), 4))
    with open(path + ".tmp", "w") as f:
        f.write(s)
    os.replace(path + ".tmp", path)
    f10, l100 = out["summary"]["first20"], out["summary"]["last100"]
    o20, o100 = out["summary"]["opp_first20"], out["summary"]["opp_last100"]
    print(f"  the other players: first 20 hands theirs {o20['personal']:.0%} people {o20['people']:.0%} · "
          f"last 100 theirs {o100['personal']:.0%} people {o100['people']:.0%}")
    print(f"{a.style}: {n} hands, {len(curve)} decisions, {len(showcase)} showcase hands, {len(s) / 1e6:.1f} MB; "
          f"first 20 hands: you {f10['personal']:.0%} people {f10['people']:.0%} · last 100: you {l100['personal']:.0%} "
          f"people {l100['people']:.0%} · all: you {C_[:, 0].mean():.1%} people {C_[:, 1].mean():.1%}")
    fl = out["summary"]["facing_last100"]
    if fl:
        print(f"  facing a bet, last 100 hands: you {fl['personal']:.0%} people {fl['people']:.0%} ({fl['decisions']} decisions)")
    for h in out["habits"]:
        print("  habit:", h["text"])


def _copy(rec):
    import copy

    return copy.deepcopy(rec)  # run_hand rewrites the record's actions


if __name__ == "__main__":
    main()
