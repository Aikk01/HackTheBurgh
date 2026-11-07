"""Replay every hand under one or more paths and report engine consistency.

Usage: python scripts/validate_dataset.py PATH [PATH ...] [--limit N] [--verbose]

For hands that record ``finishing_stacks`` the computed stacks are compared
exactly; for hands that record ``winnings`` the winner set is compared.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import phh  # noqa: E402
from pokersim.engine import UnsupportedVariant  # noqa: E402
from pokersim.replay import replay  # noqa: E402


def iter_paths(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith((".phh", ".phhs")):
                        yield os.path.join(root, f)
        else:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--limit", type=int, default=None, help="max hands per file")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show-mismatch", type=int, default=3, help="print this many mismatching hands")
    args = ap.parse_args()

    stats = collections.Counter()
    warn_types = collections.Counter()
    shown = 0
    t0 = time.perf_counter()
    for path in iter_paths(args.paths):
        try:
            records = phh.load_file(path, limit=args.limit)
        except Exception as exc:  # noqa: BLE001
            stats["files_failed"] += 1
            print(f"PARSE FAIL {path}: {exc}")
            continue
        stats["files"] += 1
        for rec in records:
            stats["hands"] += 1
            try:
                res = replay(rec, snapshots=False, observations=False)
            except UnsupportedVariant:
                stats["unsupported_variant"] += 1
                continue
            except Exception as exc:  # noqa: BLE001
                stats["engine_error"] += 1
                if shown < args.show_mismatch:
                    shown += 1
                    print(f"ENGINE ERROR {rec.label}: {exc!r}")
                continue
            if res.complete:
                stats["complete"] += 1
            else:
                stats["incomplete_or_unresolved"] += 1
            for w in res.warnings:
                key = w.split(":")[0]
                key = " ".join(key.split()[:6])
                warn_types[key] += 1
            if res.warnings:
                stats["hands_with_warnings"] += 1
            if res.stacks_match is True:
                stats["stacks_match"] += 1
            elif res.stacks_match is False:
                stats["stacks_mismatch"] += 1
                if shown < args.show_mismatch:
                    shown += 1
                    print(f"\nMISMATCH {rec.label}: diffs {res.stack_diffs}")
                    print("  actions:", rec.actions)
                    print("  computed:", [p.stack for p in res.hand.players])
                    print("  recorded:", rec.finishing_stacks)
                    for w in res.warnings:
                        print("  warning:", w)
            if rec.winnings is not None and res.hand.status == "finished":
                hinted = {i for i, w in enumerate(rec.winnings) if w > 0}
                ours = {p.index for p in res.hand.players if p.payout > 0}
                if hinted:
                    if hinted == ours:
                        stats["winners_match"] += 1
                    else:
                        stats["winners_mismatch"] += 1
                        if args.verbose and shown < args.show_mismatch + 5:
                            shown += 1
                            print(f"\nWINNER MISMATCH {rec.label}: recorded {sorted(hinted)} computed {sorted(ours)}")
                            print("  actions:", rec.actions)
                            for w in res.warnings:
                                print("  warning:", w)
    dt = time.perf_counter() - t0
    print("\n=== summary ===")
    for k, v in sorted(stats.items()):
        print(f"{k:28s} {v}")
    print(f"{'seconds':28s} {dt:.2f}")
    if warn_types:
        print("\n=== warning types ===")
        for k, v in warn_types.most_common(15):
            print(f"{v:6d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
