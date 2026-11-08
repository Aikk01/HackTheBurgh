"""Write every 5-6-player no-limit hand of one online player to a .phhs file (for the replay).

    python scripts/extract_player_hands.py --venue "Absolute Poker" --player NX+SmeQ+96PcDBG5fFPKZw \
        --out data/demo/replay_hands.phhs [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pokersim.mirror_data import N_SHARDS, iter_rows, row_to_record, shard_path  # noqa: E402
from pokersim.phh import dump_hands  # noqa: E402


def collect(job):
    i, venue, name = job
    out = []
    for row in iter_rows(shard_path(i), columns=None):
        players = row.get("players") or []
        if (row.get("venue") or "") == venue and name in players and row.get("variant") == "NT" and len(players) in (5, 6):
            rec = row_to_record(row)
            if rec is not None:
                out.append(rec)
    return i, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True)
    ap.add_argument("--player", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    parts = {}
    with Pool(a.workers) as pool:
        for i, recs in pool.imap_unordered(collect, [(i, a.venue, a.player) for i in range(N_SHARDS)]):
            parts[i] = recs
            print(f"shard {i:2d}: {len(recs)} hands", flush=True)
    recs = [r for i in sorted(parts) for r in parts[i]]  # shard order is time order
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(dump_hands(recs))
    print(f"{len(recs)} hands -> {a.out}")


if __name__ == "__main__":
    main()
