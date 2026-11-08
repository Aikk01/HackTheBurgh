"""Reserve one real human from the corpus whom no model has trained or been evaluated on: the demo's unseen player.

    python scripts/reserve_demo_player.py [--workers 2]   # -> data/demo/reserved_player.json + data/demo/reserved_hands.phhs

A hand is "touched" if either extraction could have used it:
* hand model (scripts/train_hand_range.py): online hands with known cards, in shard order, up to its per-shard cap;
* next-move model (scripts/train_next_move.py): the same random acceptance sequence (seeded per shard) up to its cap.
Both are reconstructed conservatively (300 extra hands past each cap).  The reserved player is the online player with
the most 5-6-player hold'em hands among those with **no** touched hand in any shard.  Both extraction scripts skip any
hand containing the reserved player from now on (pokersim.demo_player.is_reserved).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pokersim.mirror_data import N_SHARDS, iter_rows, shard_path  # noqa: E402

HAND_RANGE_CAP, NEXT_MOVE_CAP, ACCEPT, MARGIN = 15000, 15000, 0.06, 300
OUT_DIR = os.path.join(ROOT, "data", "demo")


def _known(actions) -> bool:
    for a in actions:
        if a.startswith("d dh") and "?" not in a:
            return True
        if " sm " in a and not a.endswith(" sm") and not a.endswith(" -"):
            return True
    return False


def scan(i):
    rng = random.Random(1000 + i)
    hr_count = nm_count = 0
    hands, touched = Counter(), Counter()
    for row in iter_rows(shard_path(i), columns=["variant", "actions", "players", "source_file", "venue"]):
        src = row.get("source_file") or ""
        if src.startswith("pluribus/") or row.get("variant") != "NT":
            continue
        acts = row.get("actions") or []
        players = row.get("players") or []
        venue = row.get("venue") or ""
        hit = False
        if len(acts) > 1 and _known(acts) and hr_count < HAND_RANGE_CAP + MARGIN:
            hr_count += 1
            hit = True
        if len(players) in (5, 6):
            if nm_count < NEXT_MOVE_CAP + MARGIN and rng.random() <= ACCEPT:
                nm_count += 1
                hit = True
            for p in players:
                hands[(venue, p)] += 1
        if hit:
            for p in players:
                touched[(venue, p)] += 1
    return hands, touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-hands", type=int, default=150)
    ap.add_argument("--exclude-venue", default="iPoker Network", help="iPoker records stacks as inf: its hands cannot be replayed")
    a = ap.parse_args()
    hands, touched = Counter(), Counter()
    with Pool(a.workers) as pool:
        for k, (h, t) in enumerate(pool.imap_unordered(scan, range(N_SHARDS))):
            hands.update(h)
            touched.update(t)
            print(f"{k + 1}/{N_SHARDS} shards", flush=True)
    clean = [(n, key) for key, n in hands.items() if touched[key] == 0 and n >= a.min_hands and key[0] != a.exclude_venue]
    clean.sort(reverse=True)
    print(f"{len(hands):,} online players in 5-6-max hands; {len(clean):,} untouched with >= {a.min_hands} hands")
    if not clean:
        raise SystemExit("no untouched player found")
    n, (venue, name) = clean[0]
    # collect their hands
    from pokersim.mirror_data import row_to_record
    from pokersim.phh import dump_hands

    recs = []
    for i in range(N_SHARDS):
        for row in iter_rows(shard_path(i), columns=None):
            if (row.get("venue") or "") == venue and name in (row.get("players") or []) and row.get("variant") == "NT" \
                    and len(row.get("players") or []) in (5, 6):
                rec = row_to_record(row)
                if rec is not None:
                    recs.append(rec)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "reserved_hands.phhs"), "w") as f:
        f.write(dump_hands(recs))
    info = {"venue": venue, "player": name, "hands_5_6_max": n, "hands_written": len(recs),
            "touched_by_training_or_eval": 0,
            "rule": "no hand of this player was available to the hand-model or next-move extraction (conservative "
                    "reconstruction, 300-hand margin past each cap); both scripts now skip any hand containing them",
            "note": "population-level statistics from the Mirror pipeline (styles.json type centres over 48k players, "
                    "population.json river counts) were computed over the whole corpus and may include this player "
                    "as 1 of tens of thousands; no model was fitted on this player's own decisions",
            "runner_up_candidates": [{"venue": v, "player": p, "hands": c} for c, (v, p) in clean[1:6]]}
    with open(os.path.join(OUT_DIR, "reserved_player.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: info[k] for k in ("venue", "player", "hands_5_6_max", "hands_written")}))


if __name__ == "__main__":
    main()
