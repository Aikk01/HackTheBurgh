"""Tabular Q-learning bettor, the exact Q* oracle and the paired benchmark (PLAN.md sections 4 and 6).

States are ``(stage, context, bettor bucket)``: at most 18 of them, two legal
actions each.  Values are in opening-pot units, like the environment reward.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import cards as C
from . import evaluator as E
from .mirror_model import FrozenProfile, PopulationModel, Twin, population_twin
from .mirror_types import BUCKET_CUTS, BUCKETS, CONTEXTS, RESPONSES
from .river_env import ENV_VERSION, RiverResponseEnv, Scenario, ScriptedDefender, StrengthDefender, scenario_pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POLICY_VERSION = "qtable-v1"
ACTION_ENCODING = {"1": ["check_back", "bet_half"], "2": ["fold", "call"]}
TIE_BREAK = {1: 0, 2: 0}  # check at the opening, fold facing a raise


def state_key(stage: int, context: str, bucket: str) -> str:
    return f"{stage}|{context}|{bucket}"


def env_key(env: RiverResponseEnv) -> str:
    return state_key(*env.state_key())


def all_states() -> List[str]:
    return [state_key(s, c, b) for s in (1, 2) for c in CONTEXTS for b in BUCKETS]


@dataclass
class Policy:
    q: Dict[str, List[float]]
    visits: Dict[str, List[int]]
    meta: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def blank(cls, meta: Optional[Dict[str, Any]] = None) -> "Policy":
        return cls({s: [0.0, 0.0] for s in all_states()}, {s: [0, 0] for s in all_states()}, dict(meta or {}))

    def greedy(self, key: str) -> int:
        q = self.q[key]
        stage = int(key.split("|")[0])
        if q[0] == q[1]:
            return TIE_BREAK[stage]
        return 0 if q[0] > q[1] else 1

    def action_name(self, key: str, action: int) -> str:
        return ACTION_ENCODING[key.split("|")[0]][action]

    def to_dict(self) -> Dict[str, Any]:
        return {"version": POLICY_VERSION, "env_version": ENV_VERSION, "action_encoding": ACTION_ENCODING,
                "bucket_cuts": list(BUCKET_CUTS), "q": self.q, "visits": self.visits, "meta": self.meta,
                "checkpoints": self.checkpoints}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Policy":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("env_version") != ENV_VERSION or d.get("version") != POLICY_VERSION:
            raise ValueError(f"incompatible policy ({d.get('version')}, {d.get('env_version')}); expected "
                             f"{POLICY_VERSION}/{ENV_VERSION}")
        return cls(d["q"], d["visits"], d.get("meta", {}), d.get("checkpoints", []))

    def chart(self) -> Dict[str, Any]:
        """Three-by-three policy chart per stage: action, both values, visits."""
        out: Dict[str, Any] = {}
        for stage in (1, 2):
            out[str(stage)] = {}
            for c in CONTEXTS:
                out[str(stage)][c] = {}
                for b in BUCKETS:
                    k = state_key(stage, c, b)
                    a = self.greedy(k)
                    out[str(stage)][c][b] = {"action": self.action_name(k, a), "q": [round(x, 4) for x in self.q[k]],
                                             "visits": list(self.visits[k])}
        return out


# ----------------------------------------------------------------- defenders from specs
def make_defender(spec: str, population: Optional[PopulationModel] = None) -> Any:
    """``population`` | ``profile:<path>`` | ``scripted:f,c,r`` | ``caller`` | ``folder`` | ``strength``."""
    population = population or PopulationModel.load()
    if spec == "population":
        return population_twin(population)
    if spec.startswith("profile:"):
        return Twin(FrozenProfile.load(spec.split(":", 1)[1]))
    if spec.startswith("scripted:"):
        f, c, r = (float(x) for x in spec.split(":", 1)[1].split(","))
        return ScriptedDefender(f, c, r, name=spec)
    if spec == "caller":
        return ScriptedDefender(0.15, 0.75, 0.10, name="frequent caller")
    if spec == "folder":
        return ScriptedDefender(0.65, 0.30, 0.05, name="frequent folder")
    if spec == "strength":
        table = {c: {"weak": {"fold": 0.85, "call": 0.13, "raise": 0.02}, "medium": {"fold": 0.45, "call": 0.50, "raise": 0.05},
                     "strong": {"fold": 0.02, "call": 0.48, "raise": 0.50}} for c in CONTEXTS}
        return StrengthDefender(table, name="strength-dependent raiser")
    raise ValueError(f"unknown defender spec {spec!r}")


# ----------------------------------------------------------------- training
def _epsilon(episode: int, episodes: int, final: float = 0.05) -> float:
    cut = 0.8 * episodes
    if episode >= cut:
        return final
    return 1.0 + (final - 1.0) * (episode / cut)


def evaluate_greedy(policy: Policy, defender: Any, pool: Sequence[Scenario], seed: int, episodes: Optional[int] = None) -> Dict[str, Any]:
    """Deterministic greedy play on fixed seeds; returns mean reward and action frequencies."""
    env = RiverResponseEnv(defender, pool, seed=seed, opponent_seed=seed + 1)
    n = episodes or len(pool)
    total = 0.0
    freq: Dict[str, Dict[str, int]] = {}
    for i in range(n):
        env.reset(options={"scenario": pool[i % len(pool)]})
        done = False
        reward = 0.0
        while not done:
            k = env_key(env)
            a = policy.greedy(k)
            f = freq.setdefault(k, {"check_back": 0, "bet_half": 0, "fold": 0, "call": 0})
            f[policy.action_name(k, a)] += 1
            _, reward, done, _, _ = env.step(a)
        total += reward
    return {"mean_reward": total / n, "episodes": n, "action_frequencies": freq}


def train(defender: Any, pool: Sequence[Scenario], *, episodes: int = 20000, seed: int = 7,
          checkpoints: Sequence[int] = (1000, 5000, 10000, 20000), validation_pool: Optional[Sequence[Scenario]] = None,
          validation_episodes: int = 1000, alpha_exponent: float = 1.0, epsilon_final: float = 0.05,
          progress: Optional[Callable[[int, int], None]] = None, meta: Optional[Dict[str, Any]] = None) -> Policy:
    """Masked epsilon-greedy tabular Q-learning (Watkins) with gamma = 1.

    ``alpha = visits ** -alpha_exponent``.  PLAN.md starts from 0.6; that schedule
    averages only the last ~100 visits of a cell at 20k episodes and missed the
    0.10 oracle tolerance, so the default is plain averaging (1.0), which is the
    minimum-variance consistent estimator in this stationary two-stage problem.
    """
    policy = Policy.blank(meta)
    policy.meta.update({"episodes": episodes, "seed": seed, "alpha_exponent": alpha_exponent, "epsilon_final": epsilon_final,
                        "defender": getattr(defender, "describe", lambda: {"kind": "unknown"})(),
                        "profile_hash": getattr(getattr(defender, "profile", None), "hash", None),
                        "pool_size": len(pool), "env_version": ENV_VERSION})
    env = RiverResponseEnv(defender, pool, seed=seed, opponent_seed=seed + 1)
    explore = random.Random(seed + 2)
    t0 = time.perf_counter()
    for ep in range(episodes):
        eps = _epsilon(ep, episodes, epsilon_final)
        env.reset()
        key = env_key(env)
        done = False
        while not done:
            if explore.random() < eps:
                a = explore.randrange(2)
            else:
                a = policy.greedy(key)
            _, r, done, _, _ = env.step(a)
            policy.visits[key][a] += 1
            alpha = policy.visits[key][a] ** (-alpha_exponent)
            if done:
                target = r
            else:
                nxt = env_key(env)
                target = r + max(policy.q[nxt])
            policy.q[key][a] += alpha * (target - policy.q[key][a])
            if not done:
                key = env_key(env)
        if (ep + 1) in checkpoints or ep + 1 == episodes:
            cp: Dict[str, Any] = {"episode": ep + 1, "seconds": round(time.perf_counter() - t0, 1),
                                  "visits": {k: list(v) for k, v in policy.visits.items()}}
            if validation_pool:
                val = evaluate_greedy(policy, defender, validation_pool, seed=99, episodes=validation_episodes)
                cp["validation_mean_reward"] = round(val["mean_reward"], 4)
                cp["greedy"] = {k: policy.action_name(k, policy.greedy(k)) for k in all_states()}
            policy.checkpoints.append(cp)
        if progress and ((ep + 1) % 500 == 0 or ep + 1 == episodes):
            progress(ep + 1, episodes)
    policy.meta["train_seconds"] = round(time.perf_counter() - t0, 1)
    return policy


# ----------------------------------------------------------------- the Q* oracle
def showdown_value(scenario: Scenario) -> float:
    """1 if the bettor wins at showdown, 0.5 on a tie, 0 otherwise (both hands are known)."""
    board = [C.parse_card(c) for c in scenario.board]
    b = E.evaluate([C.parse_card(c) for c in scenario.bettor_cards] + board)  # type: ignore[arg-type]
    d = E.evaluate([C.parse_card(c) for c in scenario.defender_cards] + board)  # type: ignore[arg-type]
    return 1.0 if b > d else 0.5 if b == d else 0.0


def response_probs(defender: Any, context: str, bucket: str) -> Dict[str, float]:
    if hasattr(defender, "profile"):
        return defender.profile.strength[context][bucket]
    if hasattr(defender, "table"):
        return defender.table[context][bucket]
    if hasattr(defender, "probs"):
        return defender.probs
    raise ValueError("defender exposes no response probabilities")


def q_star(defender: Any, pool: Sequence[Scenario]) -> Dict[str, List[float]]:
    """Exact per-cell values: average over the pool of the defender-exact EV of each action."""
    acc: Dict[str, List[float]] = {s: [0.0, 0.0] for s in all_states()}
    cnt: Dict[str, int] = {s: 0 for s in all_states()}
    # stage 2 first: value of fold / call given a raise happened, weighted by P(raise | scenario)
    w2: Dict[str, float] = {}
    for sc in pool:
        p = response_probs(defender, sc.context, sc.defender_bucket)
        w = showdown_value(sc)
        k2 = state_key(2, sc.context, sc.bettor_bucket)
        pr = p.get("raise", 0.0)
        acc[k2][0] += pr * (-0.5)
        acc[k2][1] += pr * (3 * w - 1)
        w2[k2] = w2.get(k2, 0.0) + pr
    for k2, weight in w2.items():
        if weight > 0:
            acc[k2] = [acc[k2][0] / weight, acc[k2][1] / weight]
    for sc in pool:
        p = response_probs(defender, sc.context, sc.defender_bucket)
        w = showdown_value(sc)
        k1 = state_key(1, sc.context, sc.bettor_bucket)
        k2 = state_key(2, sc.context, sc.bettor_bucket)
        v2 = max(acc[k2]) if w2.get(k2, 0) > 0 else -0.5
        acc[k1][0] += w
        acc[k1][1] += p.get("fold", 0.0) * 1.0 + p.get("call", 0.0) * (2 * w - 0.5) + p.get("raise", 0.0) * v2
        cnt[k1] += 1
    out: Dict[str, List[float]] = {}
    for k in all_states():
        if k.startswith("1|"):
            out[k] = [acc[k][0] / cnt[k], acc[k][1] / cnt[k]] if cnt[k] else [0.0, 0.0]
        else:
            out[k] = list(acc[k]) if w2.get(k, 0) > 0 else [-0.5, -0.5]
    return out


def compare_with_oracle(policy: Policy, oracle: Dict[str, List[float]], min_visits: int = 100,
                        tolerance: float = 0.10) -> Dict[str, Any]:
    cells = []
    failures = 0
    for k in all_states():
        visits = policy.visits[k]
        q = policy.q[k]
        qs = oracle[k]
        supported = min(visits) >= min_visits
        diff = [abs(q[0] - qs[0]), abs(q[1] - qs[1])]
        greedy_learned = policy.greedy(k)
        greedy_star = 0 if qs[0] >= qs[1] else 1
        near_tie = abs(qs[0] - qs[1]) < tolerance
        ok = (not supported) or (max(diff) <= tolerance and (greedy_learned == greedy_star or near_tie))
        if supported and not ok:
            failures += 1
        cells.append({"state": k, "learned": [round(x, 4) for x in q], "q_star": [round(x, 4) for x in qs],
                      "visits": list(visits), "supported": supported, "max_abs_diff": round(max(diff), 4),
                      "greedy_learned": policy.action_name(k, greedy_learned), "greedy_star": policy.action_name(k, greedy_star),
                      "near_tie": near_tie, "ok": ok})
    return {"cells": cells, "failures": failures, "min_visits": min_visits, "tolerance": tolerance,
            "pass": failures == 0}


# ----------------------------------------------------------------- paired benchmark
def play_greedy(policy: Policy, env: RiverResponseEnv, scenario: Scenario, opp_seed: int) -> Tuple[float, List[str]]:
    env.reset(options={"scenario": scenario})
    env.opp_rng = random.Random(opp_seed)
    done = False
    reward = 0.0
    actions: List[str] = []
    while not done:
        k = env_key(env)
        a = policy.greedy(k)
        actions.append(policy.action_name(k, a))
        _, reward, done, _, _ = env.step(a)
    return reward, actions


def matched_benchmark(baseline: Policy, candidate: Policy, defender: Any, pool: Sequence[Scenario], *, pairs: int = 1000,
                      seed: int = 17, bootstrap: int = 2000) -> Dict[str, Any]:
    """Both policies face the same deals and the same defender random draws."""
    env_b = RiverResponseEnv(defender, pool, seed=seed)
    env_c = RiverResponseEnv(defender, pool, seed=seed)
    rng = random.Random(seed)
    diffs: List[float] = []
    changed = 0
    total_b = total_c = 0.0
    for i in range(pairs):
        sc = pool[i % len(pool)]
        opp_seed = rng.randrange(1 << 30)
        rb, ab = play_greedy(baseline, env_b, sc, opp_seed)
        rc, ac = play_greedy(candidate, env_c, sc, opp_seed)
        diffs.append(rc - rb)
        changed += ab != ac
        total_b += rb
        total_c += rc
    mean = sum(diffs) / pairs
    boot = random.Random(seed + 1)
    means = []
    for _ in range(bootstrap):
        s = 0.0
        for _ in range(pairs):
            s += diffs[boot.randrange(pairs)]
        means.append(s / pairs)
    means.sort()
    lo, hi = means[int(0.025 * bootstrap)], means[int(0.975 * bootstrap) - 1]
    return {"pairs": pairs, "seed": seed, "defender": getattr(defender, "describe", lambda: {})(),
            "baseline_mean": round(total_b / pairs, 4), "candidate_mean": round(total_c / pairs, 4),
            "mean_diff_pot_units": round(mean, 4), "mean_diff_chips": round(mean * 100, 2),
            "ci95_chips": [round(lo * 100, 2), round(hi * 100, 2)], "fraction_changed": round(changed / pairs, 4),
            "units": "opening pot = 100 chips", "label": "simulated, conditional on these river scenarios"}


# ----------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pokersim mirror-train|mirror-evaluate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--profile", default="population", help="population | profile:<path> | caller | folder | strength | scripted:f,c,r")
    t.add_argument("--episodes", type=int, default=20000)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--out", required=True)
    t.add_argument("--pool-size", type=int, default=4000)
    t.add_argument("--pool-seed", type=int, default=1)
    t.add_argument("--validation-seed", type=int, default=2)
    t.add_argument("--alpha-exponent", type=float, default=1.0)
    b = sub.add_parser("benchmark")
    b.add_argument("--baseline", default=None,
                   help="policy file to compare against; default: train the population bot with the same seed")
    b.add_argument("--out", default=os.path.join(ROOT, "data", "mirror", "benchmark.json"))
    b.add_argument("--episodes", type=int, default=20000)
    b.add_argument("--pairs", type=int, default=1000)
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--opponents", default="caller,folder,population,strength")
    b.add_argument("--probes", default="0,4,16,64")
    sf = sub.add_parser("sufficiency")
    sf.add_argument("--seeds", default="7,8,9")
    sf.add_argument("--episodes", type=int, default=20000)
    sf.add_argument("--out", default=os.path.join(ROOT, "data", "mirror", "training_sufficiency.json"))
    e = sub.add_parser("evaluate")
    e.add_argument("--baseline", required=True)
    e.add_argument("--candidate", required=True)
    e.add_argument("--defender", default="population")
    e.add_argument("--seed", type=int, default=17)
    e.add_argument("--pairs", type=int, default=1000)
    e.add_argument("--pool-size", type=int, default=2000)
    e.add_argument("--pool-seed", type=int, default=3)
    e.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        defender = make_defender(args.profile)
        pool = scenario_pool(args.pool_size, args.pool_seed)
        val = scenario_pool(1000, args.validation_seed)
        print(f"training against {defender.describe()} on {len(pool)} scenarios ...", flush=True)
        policy = train(defender, pool, episodes=args.episodes, seed=args.seed, validation_pool=val,
                       alpha_exponent=args.alpha_exponent,
                       progress=lambda d, n: print(f"\r{d}/{n}", end="", flush=True))
        print()
        finish_policy(policy, defender, pool, val)
        policy.save(args.out)
        cmp = policy.meta["oracle_detail"]
        print(json.dumps({"out": args.out, "train_seconds": policy.meta["train_seconds"],
                          "last_checkpoint": policy.checkpoints[-1].get("validation_mean_reward"),
                          "oracle_pass": cmp["pass"], "oracle_failures": cmp["failures"]}, indent=1))
        for cell in cmp["cells"]:
            if cell["supported"]:
                flag = "ok " if cell["ok"] else "BAD"
                print(f"  {flag} {cell['state']:28s} learned {cell['learned']} q* {cell['q_star']} visits {cell['visits']}")
        return 0 if cmp["pass"] else 1
    if args.cmd == "sufficiency":
        res = run_sufficiency([int(s) for s in args.seeds.split(",")], args.episodes, out=args.out)
        print(json.dumps({k: res[k] for k in ("stage1_agreement", "disagreements_are_near_ties", "all_oracle_pass")}, indent=1))
        for c in res["cells"]:
            print(f"  {'ok  ' if c['agree'] else 'DIFF'} {c['state']:28s} {c['greedy_by_seed']} "
                  f"q* gap {c['q_star_gap']} min visits {c['min_visits']}")
        return 0
    if args.cmd == "benchmark":
        res = run_benchmark(args.baseline, opponents=args.opponents.split(","), probes=[int(x) for x in args.probes.split(",")],
                            pairs=args.pairs, episodes=args.episodes, seed=args.seed, workers=args.workers, out=args.out)
        for row in res["rows"]:
            print(f"  {row['opponent']:22s} probes {row['probes']:3d}  diff {row['mean_diff_chips']:+7.2f} chips/spot  "
                  f"CI {row['ci95_chips']}  changed {row['fraction_changed']:.0%}")
        print("control (population vs population, other seed):", res["control"]["mean_diff_chips"], res["control"]["ci95_chips"])
        return 0
    base = Policy.load(args.baseline)
    cand = Policy.load(args.candidate)
    defender = make_defender(args.defender)
    pool = scenario_pool(args.pool_size, args.pool_seed)
    res = matched_benchmark(base, cand, defender, pool, pairs=args.pairs, seed=args.seed)
    print(json.dumps(res, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


# ----------------------------------------------------------------- full pipeline helpers
def finish_policy(policy: Policy, defender: Any, train_pool: Sequence[Scenario], validation_pool: Sequence[Scenario]) -> None:
    """Attach the oracle comparison (on the deals the learner saw) and Q* on validation deals."""
    oracle_train = q_star(defender, train_pool)
    cmp = compare_with_oracle(policy, oracle_train)
    policy.meta["oracle_check"] = {"pass": cmp["pass"], "failures": cmp["failures"], "pool": "training deals",
                                   "tolerance": cmp["tolerance"], "min_visits": cmp["min_visits"]}
    policy.meta["oracle_detail"] = cmp
    policy.meta["q_star_training"] = oracle_train
    policy.meta["q_star_validation"] = q_star(defender, validation_pool)


def simulate_probe_block(defender: Any, n: int, seed: int) -> Tuple[Any, Any]:
    """A fixed block of ``n`` observed responses from ``defender`` for the personal learner."""
    from .mirror_model import PersonalModel, StrengthTable
    population = PopulationModel.load()
    personal = PersonalModel(population, player_id=f"probe:{getattr(defender, 'name', 'x')}")
    strength = StrengthTable.default(population)
    rng = random.Random(seed)
    for i in range(n):
        context = CONTEXTS[i % 3]
        bucket = BUCKETS[rng.randrange(3)]
        resp = defender.respond(context, bucket, rng)
        personal.observe(context, resp)
        strength.observe(context, bucket, resp)
    return personal, strength


def _benchmark_job(args: Tuple[str, int, int, int, int, Tuple[int, int]]) -> Dict[str, Any]:
    opponent, probes, episodes, seed, pairs, (train_n, val_n) = args
    defender = make_defender(opponent)
    personal, strength = simulate_probe_block(defender, probes, seed=seed + probes)
    twin = Twin(personal.freeze(strength))
    pool = scenario_pool(train_n, 1)
    val = scenario_pool(val_n, 2)
    policy = train(twin, pool, episodes=episodes, seed=seed, validation_pool=val, checkpoints=(episodes,),
                   meta={"arm": "personal", "opponent": opponent, "probes": probes})
    finish_policy(policy, twin, pool, val)
    path = os.path.join(ROOT, "data", "cache", "mirror", "policies", f"bench_{opponent}_{probes}_{seed}.json")
    policy.save(path)
    return {"opponent": opponent, "probes": probes, "policy_path": path, "profile_hash": twin.profile.hash,
            "oracle_pass": policy.meta["oracle_check"]["pass"], "twin_public": twin.profile.public}


def run_benchmark(baseline_path: Optional[str], *, opponents: Sequence[str], probes: Sequence[int], pairs: int = 1000,
                  episodes: int = 20000, seed: int = 17, workers: Optional[int] = None, out: Optional[str] = None,
                  pool_sizes: Tuple[int, int, int] = (4000, 1000, 2000)) -> Dict[str, Any]:
    """PLAN.md section 6 decision-value benchmark: personalised vs population bettor per opponent and probe budget.

    By default the population baseline is trained here with the candidates' seed, so at 0 probes the
    two bettors are identical and every difference at 4/16/64 probes comes from what was learned about
    the opponent.  (A baseline trained with another seed mixes in training noise: see ``control``.)
    """
    import datetime as _dt
    import multiprocessing as mp

    train_n, val_n, test_n = pool_sizes
    # Warm the cached scenario pools here; parallel workers would otherwise race to create the files.
    train_pool = scenario_pool(train_n, 1)
    val_pool = scenario_pool(val_n, 2)
    test_pool = scenario_pool(test_n, 3)
    pop_def = make_defender("population")
    if baseline_path:
        baseline = Policy.load(baseline_path)
        baseline_note = f"loaded from {baseline_path} (seed {baseline.meta.get('seed')})"
    else:
        baseline = train(pop_def, train_pool, episodes=episodes, seed=seed, validation_pool=val_pool, checkpoints=(episodes,),
                         meta={"arm": "population"})
        baseline_note = f"population twin trained here with the candidates' seed {seed}"
    jobs = [(o, p, episodes, seed, pairs, (train_n, val_n)) for o in opponents for p in probes]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with mp.Pool(workers) as procs:
        trained = procs.map(_benchmark_job, jobs)
    rows = []
    for tr in trained:
        defender = make_defender(tr["opponent"])
        cand = Policy.load(tr["policy_path"])
        res = matched_benchmark(baseline, cand, defender, test_pool, pairs=pairs, seed=seed)
        rows.append({"opponent": tr["opponent"], "probes": tr["probes"], **{k: res[k] for k in ("baseline_mean", "candidate_mean",
                     "mean_diff_chips", "ci95_chips", "fraction_changed", "pairs")}, "oracle_pass": tr["oracle_pass"],
                     "twin_public_fold": {c: round(v["fold"], 3) for c, v in tr["twin_public"].items()}})
    # training-variation control: the same population twin trained with another seed
    other = train(pop_def, train_pool, episodes=episodes, seed=seed + 101, validation_pool=val_pool, checkpoints=(episodes,))
    control = matched_benchmark(baseline, other, pop_def, test_pool, pairs=pairs, seed=seed)
    matched = baseline_path is None
    result = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "baseline": baseline_note, "matched_seed": matched,
              "zero_probe_identical": all(r["fraction_changed"] == 0 for r in rows if r["probes"] == 0) if matched else None,
              "episodes": episodes, "pool_sizes": list(pool_sizes),
              "pairs": pairs, "seed": seed, "opponents": list(opponents), "probes": list(probes), "rows": rows,
              "control": {k: control[k] for k in ("mean_diff_chips", "ci95_chips", "fraction_changed")},
              "label": "synthetic opponents; simulated river scenarios; chips per opportunity with paired 95% bootstrap intervals",
              "units": "opening pot = 100 chips"}
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return result


# ----------------------------------------------------------------- training sufficiency (PLAN.md section 4, run 3)
def _sufficiency_job(args: Tuple[int, int]) -> Dict[str, Any]:
    seed, episodes = args
    defender = make_defender("population")
    pool = scenario_pool(4000, 1)
    val = scenario_pool(1000, 2)
    pol = train(defender, pool, episodes=episodes, seed=seed, validation_pool=val)
    finish_policy(pol, defender, pool, val)
    return {"seed": seed, "train_seconds": pol.meta.get("train_seconds"),
            "checkpoints": [{"episode": c["episode"], "validation_mean_reward": c.get("validation_mean_reward")} for c in pol.checkpoints],
            "greedy": {k: pol.action_name(k, pol.greedy(k)) for k in all_states()},
            "visits": {k: list(v) for k, v in pol.visits.items()},
            "oracle": dict(pol.meta["oracle_check"])}


def run_sufficiency(seeds: Sequence[int] = (7, 8, 9), episodes: int = 20000, out: Optional[str] = None,
                    workers: Optional[int] = None) -> Dict[str, Any]:
    """Three training seeds against the population twin: checkpoints, greedy agreement, support per action."""
    import datetime as _dt
    import multiprocessing as mp

    pool = scenario_pool(4000, 1)
    scenario_pool(1000, 2)
    oracle = q_star(make_defender("population"), pool)
    with mp.Pool(workers or len(seeds)) as procs:
        runs = procs.map(_sufficiency_job, [(s, episodes) for s in seeds])
    cells = []
    for k in all_states():
        actions = [r["greedy"][k] for r in runs]
        qs = oracle[k]
        min_visits = min(min(r["visits"][k]) for r in runs)
        cells.append({"state": k, "greedy_by_seed": actions, "agree": len(set(actions)) == 1,
                      "q_star": [round(x, 4) for x in qs], "q_star_gap": round(abs(qs[0] - qs[1]), 4),
                      "min_visits": min_visits, "supported": min_visits >= 100})
    stage1 = [c for c in cells if c["state"].startswith("1|")]
    result = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "seeds": list(seeds), "episodes": episodes,
              "runs": [{k: v for k, v in r.items() if k != "visits"} for r in runs], "cells": cells,
              "stage1_agreement": round(sum(c["agree"] for c in stage1) / len(stage1), 4),
              "disagreements_are_near_ties": all(c["q_star_gap"] < 0.05 for c in cells if not c["agree"] and c["supported"]),
              "all_oracle_pass": all(r["oracle"]["pass"] for r in runs),
              "label": "population twin; Q* on the training deals; a disagreement on a near-tie cell costs almost nothing"}
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return result


if __name__ == "__main__":
    import sys

    sys.exit(main())
