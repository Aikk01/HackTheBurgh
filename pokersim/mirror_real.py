"""Phase 6 · RL benchmark on real players (CHECKLIST.md 6.6).

For each player in ``data/mirror/real_players.json`` the bettor tuned to that player's twin is compared with the
population bettor, both playing against the player's own profile:

* probes are the player's earliest recorded decisions, in time order (0, 4, 16 or 32 of them);
* the "true" profile comes from the decisions after the largest probe block, so nothing is scored on answers the
  tuned bettor has seen;
* each bettor plays the exact best response to its own model (Q* on the training scenario pool).  Q-learning
  reaches the same choices (oracle checks in Phase 3); ``--qlearning-sample`` re-checks a few players;
* value = exact expected chips over the test scenario pool against the true profile, so there is no dealing noise;
  the interval comes from resampling players.

How a profile depends on hand strength is not observable for real players (folds never show cards), so both the
twins and the true profile use the same strength table (Pluribus-calibrated when available).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mirror_data import ROOT, SMALL_OUT
from .mirror_model import PersonalModel, PopulationModel, StrengthTable, Twin
from .mirror_types import BUCKETS, CONTEXTS, RESPONSES
from .river_env import scenario_pool

REAL_PLAYERS = os.path.join(SMALL_OUT, "real_players.json")
POPULATION_WIDE = os.path.join(SMALL_OUT, "population_wide.json")
STRENGTH_PLURIBUS = os.path.join(SMALL_OUT, "strength_pluribus.json")
OUT = os.path.join(SMALL_OUT, "benchmark_real.json")
CTX_OF = {"c": 0, "b": 1, "d": 2}
RESP_OF = {"f": "fold", "c": "call", "r": "raise"}


def decode(sequence: str) -> List[Tuple[str, str]]:
    return [(CONTEXTS[CTX_OF[sequence[i]]], RESP_OF[sequence[i + 1]]) for i in range(0, len(sequence), 2)]


def strength_table(population: PopulationModel, path: str = STRENGTH_PLURIBUS) -> StrengthTable:
    """Pluribus-calibrated strength table if 6.7 produced one, otherwise the synthetic default."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("use"):
            st = StrengthTable.from_calibration(population, d["fit_counts"], sessions=0, prior_weight=d.get("prior_weight", 8.0))
            st.sources = {"calibration_sessions": 0,
                          "prior": f"Pluribus hands ({d.get('fit_spots')} fitted, every card known) + default"}
            return st
    return StrengthTable.default(population)


def pool_arrays(pool) -> Dict[str, np.ndarray]:
    from .mirror_rl import showdown_value

    return {"ctx": np.array([CONTEXTS.index(s.context) for s in pool]), "db": np.array([BUCKETS.index(s.defender_bucket) for s in pool]),
            "bb": np.array([BUCKETS.index(s.bettor_bucket) for s in pool]), "w": np.array([showdown_value(s) for s in pool])}


def response_table(twin: Twin) -> np.ndarray:
    st = twin.profile.strength
    return np.array([[[st[c][b][a] for a in RESPONSES] for b in BUCKETS] for c in CONTEXTS])


def q_star_fast(R: np.ndarray, P: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Same values as :func:`mirror_rl.q_star`, vectorised: (q1[ctx, bucket, check|bet], q2[ctx, bucket, fold|call])."""
    p = R[P["ctx"], P["db"]]
    w, key = P["w"], (P["ctx"], P["bb"])
    w2 = np.zeros((3, 3))
    a2 = np.zeros((3, 3, 2))
    np.add.at(w2, key, p[:, 2])
    np.add.at(a2[..., 0], key, p[:, 2] * -0.5)
    np.add.at(a2[..., 1], key, p[:, 2] * (3 * w - 1))
    q2 = np.where(w2[..., None] > 0, a2 / np.maximum(w2[..., None], 1e-300), -0.5)
    v2 = q2.max(-1)
    cnt = np.zeros((3, 3))
    a1 = np.zeros((3, 3, 2))
    np.add.at(cnt, key, 1)
    np.add.at(a1[..., 0], key, w)
    np.add.at(a1[..., 1], key, p[:, 0] + p[:, 1] * (2 * w - 0.5) + p[:, 2] * v2[key])
    q1 = np.where(cnt[..., None] > 0, a1 / np.maximum(cnt[..., None], 1), 0.0)
    return q1, q2


def greedy(q: np.ndarray) -> np.ndarray:
    """Index of the better action; ties go to check / fold, as in mirror_rl.TIE_BREAK."""
    return (q[..., 1] > q[..., 0]).astype(int)


def policy_values(a1: np.ndarray, a2: np.ndarray, R: np.ndarray, P: Dict[str, np.ndarray]) -> np.ndarray:
    """Exact expected value (pot units) of each test scenario for a policy against response table ``R``."""
    p = R[P["ctx"], P["db"]]
    w, key = P["w"], (P["ctx"], P["bb"])
    v_raise = np.where(a2[key] == 1, 3 * w - 1, -0.5)
    bet = p[:, 0] + p[:, 1] * (2 * w - 0.5) + p[:, 2] * v_raise
    return np.where(a1[key] == 1, bet, w)


def twin_from(population: PopulationModel, s: float, decisions: Sequence[Tuple[str, str]], strength: StrengthTable,
              pid: str) -> Twin:
    pm = PersonalModel(population, s, player_id=pid)
    for c, a in decisions:
        pm.observe(c, a)
    return Twin(pm.freeze(strength))


def _interval(values: np.ndarray, bootstrap: int = 2000, seed: int = 1) -> List[float]:
    if len(values) == 0:
        return [0.0, 0.0]
    idx = np.random.default_rng(seed).integers(0, len(values), size=(bootstrap, len(values)))
    m = np.sort(values[idx].mean(1))
    return [round(float(m[int(0.025 * bootstrap)]), 3), round(float(m[int(0.975 * bootstrap) - 1]), 3)]


def _qlearn_job(args: Tuple[Dict[str, Any], str, float, float, int, Tuple[int, int], int]) -> Dict[str, Any]:
    from .mirror_rl import train

    player, population_path, s, truth_s, k, pools, episodes = args
    population = PopulationModel.load(population_path)
    strength = strength_table(population)
    dec = decode(player["sequence"])
    cand = twin_from(population, s, dec[:k], strength, player["id"])
    truth = twin_from(population, truth_s, dec[k:], strength, player["id"])
    train_pool, test_pool = scenario_pool(pools[0], 1), scenario_pool(pools[1], 3)
    Ptr, Pte = pool_arrays(train_pool), pool_arrays(test_pool)
    q1, q2 = q_star_fast(response_table(cand), Ptr)
    pol = train(cand, train_pool, episodes=episodes, seed=17)
    ql1 = np.array([[pol.q[f"1|{c}|{b}"] for b in BUCKETS] for c in CONTEXTS])
    ql2 = np.array([[pol.q[f"2|{c}|{b}"] for b in BUCKETS] for c in CONTEXTS])
    Rt = response_table(truth)
    v_star = policy_values(greedy(q1), greedy(q2), Rt, Pte).mean()
    v_q = policy_values(greedy(ql1), greedy(ql2), Rt, Pte).mean()
    return {"id": player["id"], "probes": k, "stage1_cells_agree": int((greedy(ql1) == greedy(q1)).sum()), "cells": 9,
            "value_gap_chips": round(float((v_q - v_star) * 100), 3)}


def _summary_row(per_player: List[Dict[str, Any]], n: int) -> Dict[str, Any]:
    g = np.array([r["gain"][str(n)] for r in per_player])
    ch = np.array([r["changed"][str(n)] for r in per_player])
    if not len(g):
        return {"probes": n, "players": 0}
    return {"probes": n, "players": int(len(g)), "mean_gain_chips": round(float(g.mean()), 3), "ci95": _interval(g),
            "median": round(float(np.median(g)), 3), "p10": round(float(np.percentile(g, 10)), 3),
            "p90": round(float(np.percentile(g, 90)), 3), "share_helped": round(float((g > 0.05).mean()), 3),
            "share_hurt": round(float((g < -0.05).mean()), 3), "decisions_changed": round(float(ch.mean()), 4)}


def run_real_benchmark(players_path: str = REAL_PLAYERS, population_path: str = POPULATION_WIDE,
                       probes: Sequence[int] = (0, 4, 16, 32), truth_strength: float = 8.0, prior_strength: Optional[float] = None,
                       min_truth: int = 24, pools: Tuple[int, int] = (4000, 2000), qlearning_sample: int = 0,
                       episodes: int = 20000, workers: Optional[int] = None, out: Optional[str] = OUT) -> Dict[str, Any]:
    t0 = time.perf_counter()
    population = PopulationModel.load(population_path)
    with open(players_path, encoding="utf-8") as f:
        doc = json.load(f)
    if prior_strength is None:
        ev = os.path.join(os.path.dirname(players_path), "eval_wide.json")
        prior_strength = float(json.load(open(ev, encoding="utf-8"))["prior_strength"]) if os.path.exists(ev) else 24.0
    strength = strength_table(population)
    train_pool, test_pool = scenario_pool(pools[0], 1), scenario_pool(pools[1], 3)
    Ptr, Pte = pool_arrays(train_pool), pool_arrays(test_pool)
    q1b, q2b = q_star_fast(response_table(twin_from(population, prior_strength, [], strength, "population")), Ptr)
    a1b, a2b = greedy(q1b), greedy(q2b)
    kmax = max(probes)
    key = (Pte["ctx"], Pte["bb"])
    per_player: List[Dict[str, Any]] = []
    for pl in doc["players"]:
        dec = decode(pl["sequence"])
        if len(dec) < kmax + min_truth:
            continue
        later = dec[kmax:]
        Rt = response_table(twin_from(population, truth_strength, later, strength, pl["id"]))
        vb = policy_values(a1b, a2b, Rt, Pte)
        rec = {"id": pl["id"], "venue": pl["venue"], "decisions": len(dec), "baseline_chips": round(float(vb.mean() * 100), 3),
               "later_fold_rate": round(sum(a == "fold" for _c, a in later) / len(later), 3), "gain": {}, "changed": {}}
        for n in probes:
            q1, q2 = q_star_fast(response_table(twin_from(population, prior_strength, dec[:n], strength, pl["id"])), Ptr)
            a1, a2 = greedy(q1), greedy(q2)
            rec["gain"][str(n)] = round(float((policy_values(a1, a2, Rt, Pte) - vb).mean() * 100), 3)
            rec["changed"][str(n)] = round(float((a1[key] != a1b[key]).mean()), 4)
        per_player.append(rec)
    groups = {"folds under 45%": (0.0, 0.45), "folds 45-65%": (0.45, 0.65), "folds over 65%": (0.65, 1.01)}
    by_group = {}
    for name, (lo, hi) in groups.items():
        g = np.array([r["gain"][str(kmax)] for r in per_player if lo <= r["later_fold_rate"] < hi])
        by_group[name] = {"players": int(len(g)), "mean_gain_chips": round(float(g.mean()), 3) if len(g) else None, "ci95": _interval(g)}
    qcheck = None
    if qlearning_sample:
        import multiprocessing as mp

        sample = [p for p in sorted(doc["players"], key=lambda p: -p["decisions"]) if p["decisions"] >= kmax + min_truth][:qlearning_sample]
        jobs = [(p, population_path, prior_strength, truth_strength, kmax, pools, episodes) for p in sample]
        with mp.Pool(max(1, min(len(jobs), workers or 8))) as procs:
            res = procs.map(_qlearn_job, jobs)
        qcheck = {"players": res, "episodes": episodes, "cells_agree": sum(r["stage1_cells_agree"] for r in res),
                  "cells": 9 * len(res), "mean_value_gap_chips": round(float(np.mean([r["value_gap_chips"] for r in res])), 3) if res else None}
    result = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "players": len(per_player),
              "probes": list(probes), "prior_strength": prior_strength,
              "truth": {"prior_strength": truth_strength, "from": f"each player's decisions after the first {kmax}", "min_decisions": min_truth},
              "strength_table": strength.sources, "pools": {"train": pools[0], "test": pools[1]},
              "rows": [_summary_row(per_player, n) for n in probes], "by_true_fold_rate": by_group,
              "zero_probe_identical": all(r["gain"]["0"] == 0 and r["changed"]["0"] == 0 for r in per_player) if 0 in probes else None,
              "baseline_chips_mean": round(float(np.mean([r["baseline_chips"] for r in per_player])), 3) if per_player else None,
              "qlearning_check": qcheck, "per_player": per_player, "seconds": round(time.perf_counter() - t0, 1),
              "label": ("real players' recorded answers; exact best responses on simulated river scenarios; gain = chips per decision "
                        "of the bettor tuned to the player's twin minus the population bettor (pot = 100), 95% intervals over players"),
              "units": "chips per decision, opening pot = 100"}
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pokersim mirror-real", description="Bettors tuned to real players vs the population bettor")
    ap.add_argument("--probes", default="0,4,16,32")
    ap.add_argument("--truth-strength", type=float, default=8.0)
    ap.add_argument("--min-truth", type=int, default=24)
    ap.add_argument("--qlearning-sample", type=int, default=0, help="re-check this many players with Q-learning")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--players", default=REAL_PLAYERS)
    ap.add_argument("--population", default=POPULATION_WIDE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    res = run_real_benchmark(args.players, args.population, [int(x) for x in args.probes.split(",")], args.truth_strength,
                             min_truth=args.min_truth, qlearning_sample=args.qlearning_sample, episodes=args.episodes,
                             workers=args.workers, out=args.out)
    print(json.dumps({k: res[k] for k in ("players", "rows", "by_true_fold_rate", "zero_probe_identical", "qlearning_check", "seconds")}, indent=1))
    return 0
