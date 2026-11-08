"""Sealed final test of the RL coach: opponents no run trained against and no run was picked on.

    python scripts/sealed_eval.py --policy v3-ranges=data/rl/remote/v3-ranges/export_heldout \
        --policy v3-noranges=data/rl/remote/v3-noranges/export_heldout [--hands 10000] [--workers 15]
    # -> data/rl/sealed_eval.json

Registered in train/06_eval_and_overfitting.md before any result existed (see git history).  Training opponents are
the equity / station / maniac / random / tag / lag bots and the "pro" next-move bots with temp 0.8-1.2 and tilts
within ±0.5; the held-out tables that pick each run's model use the "plur" bots at temp 0.9-1.2 and the nit / trapper
/ minraiser / potodds rule bots.  The sealed tables use neither:

* ``sealed_styles``: next-move bots pushed outside every style used before (tilts of 1.2, temperatures 0.6 and 1.5);
* ``sealed_rules``: three rule bots written for this test (overbettor, limper, checkraiser);
* ``sealed_mix``: both.

Every hero plays the same deals (same seeds, deterministic button), so all comparisons are paired per hand:
each policy against the equity bot, against its own copy with blank range inputs ("are the predicted hands worth
anything?") and with blank opponent-style inputs ("is watching the table worth anything?"), and policies against
each other.  The 95% intervals are over hands.  This script picks nothing: the shipped model is chosen by the
held-out tables, as before.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "data", "rl", "sealed_eval.json")
SEED0 = 700_000  # never used before: in-training evaluations start at 10_000, final evaluations at 50_000
BASE = "equity:iterations=40"
SEALED_LINEUPS: Dict[str, List[str]] = {
    "sealed_styles": ["pro:temp=1.0,tilt_c=1.2", "pro:temp=1.0,tilt_r=1.2", "plur:temp=1.0,tilt_f=1.2", "plur:temp=1.5",
                      "pro:temp=0.6"],
    "sealed_rules": ["overbettor", "limper", "checkraiser", "limper", "overbettor"],
    "sealed_mix": ["pro:temp=1.0,tilt_c=1.2", "limper", "plur:temp=1.5", "overbettor", "checkraiser"],
}


def _hero(spec: Dict[str, Any], rng: random.Random, policies: Dict[str, Any]):
    """``spec``: {"bot": spec} or {"policy": name, "path": dir, "use_ranges": bool, "blank": "" | "ranges" | "styles"}."""
    import numpy as np

    from pokersim.agents import make_agent, safe_act
    from pokersim.rl_features import action_targets, encode
    from pokersim.rl_policy import NumpyPolicy

    if "bot" in spec:
        agent = make_agent(spec["bot"], seed=rng.randrange(1 << 30))
        return lambda obs, styles: (safe_act(agent, obs), None)
    if spec["path"] not in policies:
        with np.load(os.path.join(spec["path"], "policy.npz")) as z:
            policies[spec["path"]] = NumpyPolicy({k: z[k] for k in z.files})
    policy = policies[spec["path"]]
    use_ranges = spec["use_ranges"] and spec["blank"] != "ranges"

    def fn(obs, styles):
        x = encode(obs, {} if spec["blank"] == "styles" else styles, rng=rng, use_ranges=use_ranges)
        mask, targets = action_targets(obs)
        logits, _ = policy.forward(x[None, :])
        a = int(np.argmax(np.where(mask, logits[0], -1e9)))
        return targets[a] or {"action": "call"}, None
    return fn


def _job(job: Dict[str, Any]) -> Dict[str, Any]:
    from pokersim.rl_train import Session, play_hand

    policies = _job.policies = getattr(_job, "policies", {})
    hero = _hero(job["spec"], random.Random(job["seed"]), policies)
    s = Session(random.Random(job["seed"]), 0.0, [], lineup=SEALED_LINEUPS[job["lineup"]], hands=job["hands"])
    drng = random.Random(job["seed"] + 1)
    nets = []
    while s.hands_left > 0:
        net_bb, _ = play_hand(s, hero, {}, drng, deep_random=False)
        nets.append(net_bb)
    return {"key": job["key"], "lineup": job["lineup"], "chunk": job["chunk"], "nets": nets}


def _init() -> None:
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"


def ci(xs: List[float]) -> Tuple[float, float]:
    """Mean and 95% half-width, in bb/100."""
    n = len(xs)
    if n < 2:
        return (100.0 * xs[0] if xs else 0.0), float("nan")
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return round(100.0 * m, 1), round(100.0 * 1.96 * sd / math.sqrt(n), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, help="name=export_dir (holding policy.npz and policy.json)")
    ap.add_argument("--hands", type=int, default=10_000, help="hands per table and hero")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed0", type=int, default=SEED0, help="only to smoke-test the pipeline on other deals; the registered test uses the default")
    a = ap.parse_args()
    _init()

    heroes: Dict[str, Dict[str, Any]] = {"equity_bot": {"bot": BASE}}
    cands: Dict[str, Dict[str, Any]] = {}
    for item in a.policy:
        name, path = item.split("=", 1)
        meta = json.load(open(os.path.join(path, "policy.json")))
        use_ranges = bool(meta.get("use_ranges", True))
        cands[name] = {"path": path, "use_ranges": use_ranges, "updates": meta.get("updates"),
                       "heldout_score": (meta.get("eval") or {}).get("heldout_score", meta.get("heldout_score"))}
        for blank in ("", "styles") + (("ranges",) if use_ranges else ()):
            heroes[f"{name}{'/no_' + blank if blank else ''}"] = {"policy": name, "path": path, "use_ranges": use_ranges,
                                                                    "blank": blank}

    chunk = max(50, a.hands // a.chunks)
    jobs = [{"key": key, "spec": spec, "lineup": lineup, "chunk": c, "hands": chunk, "seed": a.seed0 + 97 * c + 1009 * li}
            for li, lineup in enumerate(SEALED_LINEUPS) for c in range(a.chunks) for key, spec in heroes.items()]
    print(f"{len(cands)} policies, {len(heroes)} heroes, {len(jobs)} jobs of {chunk} hands, {a.workers} workers", flush=True)
    t0 = time.time()
    got: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            got.setdefault((r["key"], r["lineup"]), {})[r["chunk"]] = r["nets"]
            if i % max(1, len(jobs) // 10) == 0:
                print(f"  {i}/{len(jobs)} jobs, {time.time() - t0:.0f} s", flush=True)

    def series(key: str, lineup: str) -> List[float]:
        d = got.get((key, lineup), {})
        return [x for c in sorted(d) for x in d[c]]

    def paired(k1: str, k2: str, lineups=tuple(SEALED_LINEUPS)) -> Tuple[float, float]:
        diffs = []
        for lineup in lineups:
            x, y = series(k1, lineup), series(k2, lineup)
            diffs += [x[i] - y[i] for i in range(min(len(x), len(y)))]
        return ci(diffs)

    res: Dict[str, Any] = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "hands_per_table": a.hands,
                           "seed0": a.seed0, "lineups": SEALED_LINEUPS, "unit": "bb/100, paired per hand, 95% CI",
                           "seconds": round(time.time() - t0), "policies": {}}
    for name, c in cands.items():
        row: Dict[str, Any] = {**c, "vs_equity_bot": {}, "range_inputs_gain": {}, "style_inputs_gain": {}}
        for lineup in list(SEALED_LINEUPS) + ["all"]:
            ls = tuple(SEALED_LINEUPS) if lineup == "all" else (lineup,)
            row["vs_equity_bot"][lineup] = paired(name, "equity_bot", ls)
            row["style_inputs_gain"][lineup] = paired(name, f"{name}/no_styles", ls)
            if c["use_ranges"]:
                row["range_inputs_gain"][lineup] = paired(name, f"{name}/no_ranges", ls)
        res["policies"][name] = row
    names = list(cands)
    res["head_to_head"] = {f"{x} - {y}": paired(x, y) for i, x in enumerate(names) for y in names[i + 1:]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    print(f"\nSealed test, {a.hands} hands per table, {res['seconds']} s  (bb/100, paired, 95% CI)")
    for name, row in res["policies"].items():
        v = row["vs_equity_bot"]
        print(f"  {name:16s} vs equity bot: all {v['all'][0]:+.1f}±{v['all'][1]:.0f}  " +
              "  ".join(f"{l.replace('sealed_', '')} {v[l][0]:+.1f}±{v[l][1]:.0f}" for l in SEALED_LINEUPS) +
              f"  | style inputs {row['style_inputs_gain']['all'][0]:+.1f}±{row['style_inputs_gain']['all'][1]:.0f}" +
              (f"  | range inputs {row['range_inputs_gain']['all'][0]:+.1f}±{row['range_inputs_gain']['all'][1]:.0f}"
               if row["use_ranges"] else ""))
    for k, (m, h) in res["head_to_head"].items():
        print(f"  {k}: {m:+.1f}±{h:.0f}")
    print(f"-> {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
