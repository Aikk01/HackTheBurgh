"""Score the personal model on real humans that no model has seen: predict each of their decisions before it happens.

    python scripts/eval_unseen_players.py [--players 30] [--workers 2]   # -> data/demo/unseen_players_eval.json

Players: the online players with the most 5-6-max hands among those with no hand touched by the hand-model or
next-move extraction (same conservative reconstruction as scripts/reserve_demo_player.py; iPoker excluded).
For every decision of each player, in time order: the personal model (pokersim.you_model) and the people-in-general
model (pokersim.next_move, public view) predict fold / call / raise; then the personal model learns the true action.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from reserve_demo_player import scan  # noqa: E402
from pokersim.mirror_data import N_SHARDS, iter_rows, row_to_record, shard_path  # noqa: E402

OUT = os.path.join(ROOT, "data", "demo", "unseen_players_eval.json")


def collect(args):
    i, wanted = args
    got = defaultdict(list)
    for row in iter_rows(shard_path(i), columns=None):
        if row.get("variant") != "NT" or len(row.get("players") or []) not in (5, 6):
            continue
        v = row.get("venue") or ""
        for p in row.get("players") or []:
            if (v, p) in wanted:
                rec = row_to_record(row)
                if rec is not None:
                    got[(v, p)].append(rec)
    return got


def evaluate(key_recs):
    from pokersim import next_move as NM, you_model as Y
    from pokersim.replay import replay
    (venue, name), recs = key_recs
    m = Y.PlayerModel(name, persist=False)
    rows = []
    for rec in recs:
        try:
            r = replay(rec, snapshots=False, observations=True)
        except Exception:
            continue
        for d in r.decisions:
            if r.hand.players[d.player].name != name or not d.observation.get("legal_actions"):
                continue
            obs = d.observation
            prior = NM.predict(obs, d.player, with_cards=False)
            pred, _ = m.predict(obs, d.player)
            g = Y.group_of_action(d.actual)
            usual = m.usual_group(obs, d.player)
            rows.append((int(Y.GROUP_OF[int(np.argmax(pred))]) == g, int(Y.GROUP_OF[int(np.argmax(prior))]) == g, usual == g,
                         -np.log(max(pred[Y.GROUP_OF == g].sum(), 1e-9)), -np.log(max(prior[Y.GROUP_OF == g].sum(), 1e-9))))
            m.observe(obs, d.player, pred, d.actual, prior=prior)
    R = np.array(rows, float)
    if not len(R):
        return None
    late = R[len(R) // 2:]  # second half: after the model has seen the first half
    return {"venue": venue, "player": name, "decisions": len(R),
            "personal_top1": R[:, 0].mean(), "people_top1": R[:, 1].mean(), "usual_top1": R[:, 2].mean(),
            "personal_logloss": R[:, 3].mean(), "people_logloss": R[:, 4].mean(),
            "late_gain_nats": (late[:, 4] - late[:, 3]).mean(), "late_personal_top1": late[:, 0].mean(),
            "late_people_top1": late[:, 1].mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    hands, touched = Counter(), Counter()
    with Pool(a.workers) as pool:
        for h, t in pool.imap_unordered(scan, range(N_SHARDS)):
            hands.update(h)
            touched.update(t)
    clean = sorted(((n, k) for k, n in hands.items() if touched[k] == 0 and n >= 150 and k[0] != "iPoker Network"), reverse=True)
    wanted = {k for _, k in clean[:a.players]}
    print(f"{len(clean)} untouched players; evaluating {len(wanted)}", flush=True)
    recs = defaultdict(list)
    with Pool(a.workers) as pool:
        for got in pool.imap_unordered(collect, [(i, wanted) for i in range(N_SHARDS)]):
            for k, v in got.items():
                recs[k].extend(v)
    with Pool(a.workers) as pool:
        res = [r for r in pool.map(evaluate, list(recs.items())) if r]
    d = np.array([r["late_gain_nats"] for r in res])
    summary = {"players": len(res), "decisions": int(sum(r["decisions"] for r in res)),
               "mean_personal_top1": float(np.mean([r["personal_top1"] for r in res])),
               "mean_people_top1": float(np.mean([r["people_top1"] for r in res])),
               "mean_usual_top1": float(np.mean([r["usual_top1"] for r in res])),
               "late_gain_nats_mean": float(d.mean()), "late_gain_ci95": float(1.96 * d.std() / np.sqrt(len(d))),
               "players_helped": int((d > 0).sum())}
    json.dump({"summary": summary, "per_player": res}, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
