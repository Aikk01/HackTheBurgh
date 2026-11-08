"""The live RL coach against the best held-out export of any run; swap only when asked.

    python scripts/swap_coach.py            # report: the live coach and the best candidate
    python scripts/swap_coach.py --swap     # make the best candidate live (if it is better), rebuild the replays
    python scripts/swap_coach.py --replay   # rebuild your own replay (build_replay.py --me You)

``--swap`` archives the candidate in ``data/rl/coaches/<run>-upd<N>/`` and replaces ``data/rl/policy.*`` atomically;
the server reloads the coach when the file changes, so no restart.  It then rebuilds the unseen-player replay (its
coach advice comes from the live coach) and your own replay.
The held-out score picks the model, so it is biased upwards; the sealed test (train/06) is the final word.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RL = os.path.join(ROOT, "data", "rl")
PLAYERS = os.environ.get("POKERSIM_PLAYERS_DIR") or os.path.join(ROOT, "data", "players")


def say(msg: str) -> None:
    print(f"{_dt.datetime.now():%H:%M} [coach] {msg}", flush=True)


def score(path: str):
    try:
        m = json.load(open(path))
        return float((m.get("eval") or {}).get("heldout_score", m.get("heldout_score"))), int(m.get("updates") or 0), m
    except (OSError, ValueError, TypeError):
        return None


def candidates():
    for pj in glob.glob(os.path.join(RL, "remote", "*", "export_heldout", "policy.json")) + \
            glob.glob(os.path.join(RL, "runs", "*", "export_heldout", "policy.json")):
        s = score(pj)
        if s is not None and s[2].get("opponent_mix"):
            yield s[0], s[1], os.path.basename(os.path.dirname(os.path.dirname(pj))), os.path.dirname(pj)


def build(args) -> None:
    subprocess.run(["nice", "-n", "10", sys.executable, os.path.join(ROOT, "scripts", "build_replay.py")] + args,
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def your_replay() -> None:
    hands = os.path.join(PLAYERS, "You", "hands.phhs")
    n = sum(1 for line in open(hands) if line.startswith("[")) if os.path.exists(hands) else 0
    if n:
        build(["--me", "You"])
    say(f"REPLAY your replay rebuilt from {n} hands" if n else "REPLAY no hands of yours saved yet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    if a.replay:
        your_replay()
        return
    live = score(os.path.join(RL, "policy.json"))
    live_score = live[0] if live else float("-inf")
    best = max(candidates(), default=None)
    say(f"live: {os.path.basename(str((live[2] if live else {}).get('run_dir', '?')))} update {live[1] if live else '?'} "
        f"(held-out {live_score:+.1f})")
    if best:
        say(f"best candidate: {best[2]} update {best[1]} (held-out {best[0]:+.1f})")
    if a.swap and best and best[0] > live_score + 1e-9:
        sc, upd, run, src = best
        dest = os.path.join(RL, "coaches", f"{run}-upd{upd}")
        os.makedirs(dest, exist_ok=True)
        for f in ("policy.json", "policy.npz"):  # the npz last: its change is what the server reloads on
            shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
            shutil.copy2(os.path.join(src, f), os.path.join(RL, f + ".tmp"))
            os.replace(os.path.join(RL, f + ".tmp"), os.path.join(RL, f))
        say(f"SWAP live coach -> {run} update {upd} (held-out {sc:+.1f}, was {live_score:+.1f})")
        build([])
        your_replay()


if __name__ == "__main__":
    main()
