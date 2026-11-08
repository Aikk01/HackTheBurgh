"""The four classic player types, mined from the real online hands (for the Poker Mirror's bots).

    python scripts/mine_types.py [--min-hands 1000] [--workers 14]   # -> data/demo/player_types.json

Pass 1 measures every online player's style the way poker trackers do, from public actions only:
* VPIP: share of hands where they put money in voluntarily before the flop (a big-blind check does not count);
* PFR: share of hands where they raised before the flop;
* AF: after the flop, bets and raises per call.
Regulars (``--min-hands`` or more) are sorted into the four types by VPIP (loose or tight) and an aggression score
(PFR / VPIP and AF, each scaled by its median).  Only the core of each type counts: players in the outer ``--core``
share on both scores (30% by default); the borderline middle belongs to no type.

Pass 2 counts, for each type and for all regulars together, what they do in each situation: street x facing a bet or
not x position (blinds / early / late), as fold / check-call / bet-raise.  scripts/fit_types.py turns these into bots.
The online hands carry no results, so nothing here says who wins.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pokersim.mirror_data import N_SHARDS, iter_rows, shard_path  # noqa: E402

OUT = os.path.join(ROOT, "data", "demo", "player_types.json")
COLS = ["variant", "venue", "players", "actions"]
TYPES = ["tight aggressive", "loose aggressive", "loose passive", "tight passive"]


def posgroup(i: int, n: int) -> int:
    """0 blinds, 1 early, 2 late; PHH lists players from the small blind, the button last."""
    if i < 2:
        return 0
    return 2 if i >= n - 2 else 1


def walk(row):
    """Yield (player index, street, facing a bet, action 'f'/'c'/'r') for every decision of one hand."""
    n = len(row["players"])
    street, bets = 0, 1  # preflop the big blind is the bet to face
    for a in row["actions"]:
        parts = a.split()
        if parts[0] == "d":
            if len(parts) > 1 and parts[1] == "db":
                street, bets = street + 1, 0
            continue
        if len(parts) < 2 or not parts[0].startswith("p"):
            continue
        i = int(parts[0][1:]) - 1
        if not 0 <= i < n:
            continue
        v = parts[1]
        if v == "f":
            yield i, street, 1, "f"
        elif v == "cc":
            facing = 1 if bets > 0 and not (street == 0 and i == 1 and bets == 1) else 0  # the big blind's check
            yield i, street, facing, "c"
        elif v == "cbr":
            facing = 1 if bets > 0 and not (street == 0 and i == 1 and bets == 1) else 0
            yield i, street, facing, "r"
            bets += 1


def ok(row) -> bool:
    return row.get("variant") == "NT" and bool(row.get("venue")) and len(row.get("players") or []) in (5, 6)


def pass1(i):
    st = defaultdict(lambda: [0, 0, 0, 0, 0])  # hands, vpip, pfr, postflop bets+raises, postflop calls
    for row in iter_rows(shard_path(i), columns=COLS):
        if not ok(row):
            continue
        names, v = row["players"], row["venue"]
        vp, pr = set(), set()
        for j, street, facing, act in walk(row):
            s = st[(v, names[j])]
            if street == 0:
                if act == "r":
                    vp.add(j); pr.add(j)
                elif act == "c" and facing:
                    vp.add(j)
            elif act == "r":
                s[3] += 1
            elif act == "c" and facing:
                s[4] += 1
        for j, name in enumerate(names):
            s = st[(v, name)]
            s[0] += 1
            s[1] += j in vp
            s[2] += j in pr
    return {f"{v}\t{n}": x for (v, n), x in st.items()}


def pass2(job):
    i, members = job  # members: "venue\tname" -> type index
    counts = np.zeros((len(TYPES) + 1, 4, 2, 3, 3))  # type (last = all regulars), street, facing, position, action
    A = {"f": 0, "c": 1, "r": 2}
    for row in iter_rows(shard_path(i), columns=COLS):
        if not ok(row):
            continue
        names, v, n = row["players"], row["venue"], len(row["players"])
        for j, street, facing, act in walk(row):
            t = members.get(f"{v}\t{names[j]}")
            if t is None:
                continue
            counts[t, street, facing, posgroup(j, n), A[act]] += 1
            counts[-1, street, facing, posgroup(j, n), A[act]] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hands", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--core", type=float, default=0.3, help="share of regulars at each end of both scores")
    a = ap.parse_args()
    stats = defaultdict(lambda: np.zeros(5))
    with Pool(a.workers) as pool:
        for part in pool.imap_unordered(pass1, range(N_SHARDS)):
            for k, x in part.items():
                stats[k] += x
    reg = {k: x for k, x in stats.items() if x[0] >= a.min_hands}
    K = list(reg)
    X = np.array([reg[k] for k in K])
    vpip, pfr = X[:, 1] / X[:, 0], X[:, 2] / X[:, 0]
    af = X[:, 3] / np.maximum(X[:, 4], 1)
    ratio = pfr / np.maximum(vpip, 1e-6)
    aggr = ratio / np.median(ratio) + af / np.median(af)  # 2 at the median
    lo_v, hi_v = np.quantile(vpip, [a.core, 1 - a.core])
    lo_a, hi_a = np.quantile(aggr, [a.core, 1 - a.core])
    loose, tight = vpip >= hi_v, vpip <= lo_v
    aggressive, passive = aggr >= hi_a, aggr <= lo_a
    t = np.full(len(K), -1)
    t[tight & aggressive], t[loose & aggressive], t[loose & passive], t[tight & passive] = 0, 1, 2, 3
    members = {K[i]: int(t[i]) for i in range(len(K)) if t[i] >= 0}
    with Pool(a.workers) as pool:
        counts = sum(pool.imap_unordered(pass2, [(i, members) for i in range(N_SHARDS)]))

    def summary(sel):
        return {"players": int(sel.sum()), "hands": int(X[sel, 0].sum()),
                "vpip": round(float(np.median(vpip[sel])), 3), "pfr": round(float(np.median(pfr[sel])), 3),
                "af": round(float(np.median(af[sel])), 2)}

    out = {"min_hands": a.min_hands, "regulars": len(K), "core": a.core,
           "split": {"vpip": [round(float(lo_v), 3), round(float(hi_v), 3)], "aggression": [round(float(lo_a), 3), round(float(hi_a), 3)]},
           "types": {name: summary(t == k) for k, name in enumerate(TYPES)},
           "all": summary(np.ones(len(K), bool)),
           "axes": {"street": ["preflop", "flop", "turn", "river"], "facing": ["free", "bet"],
                    "position": ["blinds", "early", "late"], "action": ["fold", "check/call", "bet/raise"]},
           "counts": {name: counts[k].tolist() for k, name in enumerate(TYPES)}, "counts_all": counts[-1].tolist()}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"{len(K)} regulars ({a.min_hands}+ hands) of {len(stats)} players")
    for name in TYPES + ["all"]:
        s = out["types"].get(name, out["all"])
        print(f"  {name:17s} {s['players']:5d} players {s['hands']:9d} hands · VPIP {s['vpip']:.0%} PFR {s['pfr']:.0%} AF {s['af']:.2f}")


if __name__ == "__main__":
    main()
