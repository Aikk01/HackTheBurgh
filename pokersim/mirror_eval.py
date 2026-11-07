"""Held-out scoring of the forecasters and the Pluribus range check (PLAN.md section 6).

Forecasters, all scored on the frozen 20% test manifest, decision by decision:

* ``pooled``      - one fold/call/raise distribution for everything;
* ``population``  - the public-context model;
* ``type``        - the mixture-of-types prior (only when a type prior exists);
* ``personal``    - population prior plus the player's *training* responses.

Standard library only, so the demo can regenerate it offline.
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .mirror_data import LARGE_OUT, SMALL_OUT, RiverSpot, load_spots
from .mirror_model import PersonalModel, PopulationModel, TypePrior
from .mirror_types import CONTEXTS, RESPONSES

BINS: Sequence[Tuple[int, Optional[int]]] = ((0, 0), (1, 3), (4, 15), (16, 63), (64, None))
RESPONSIBILITIES_PATH = os.path.join(LARGE_OUT, "style_responsibilities.json")


def bin_label(n: int) -> str:
    for lo, hi in BINS:
        if n >= lo and (hi is None or n <= hi):
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi is not None else f"{lo}+")
    return "?"


def load_manifest(path: Optional[str] = None) -> Dict[Tuple[str, str], str]:
    import csv

    path = path or os.path.join(LARGE_OUT, "manifest.csv")
    out: Dict[Tuple[str, str], str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["venue"], row["hand_id"])] = row["split"]
    return out


def modelling_spots(spots: Iterable[RiverSpot], split: Dict[Tuple[str, str], str]) -> Tuple[List[RiverSpot], List[RiverSpot]]:
    train, test = [], []
    for s in spots:
        sp = split.get(s.key)
        if sp == "train":
            train.append(s)
        elif sp == "test":
            test.append(s)
    key = lambda s: (s.timestamp or "", s.hand_id)  # noqa: E731
    return sorted(train, key=key), sorted(test, key=key)


def log_loss(probs: Dict[str, float], actual: str, floor: float = 1e-6) -> float:
    return -math.log(max(probs.get(actual, 0.0), floor))


def brier_fold(probs: Dict[str, float], actual: str) -> float:
    return (probs.get("fold", 0.0) - (1.0 if actual == "fold" else 0.0)) ** 2


def forecast_ladder(train: Sequence[RiverSpot], test: Sequence[RiverSpot], population: PopulationModel, *,
                    type_prior: Optional[TypePrior] = None,
                    responsibilities: Optional[Dict[Tuple[str, str], Sequence[float]]] = None,
                    prior_strength: float = 8.0, online: bool = False, bootstrap: int = 1000, seed: int = 1,
                    exclude_sources: Sequence[str] = ()) -> Dict[str, Any]:
    """Score the forecasters on every test decision; bootstrap paired differences over players."""
    # training histories per player
    hist: Dict[Tuple[str, str], PersonalModel] = {}
    hist_type: Dict[Tuple[str, str], PersonalModel] = {}
    for s in train:
        k = (s.venue, s.defender)
        if k not in hist:
            hist[k] = PersonalModel(population, prior_strength, player_id=s.defender)
            if type_prior is not None:
                pi0 = responsibilities.get(k) if responsibilities else None
                hist_type[k] = PersonalModel(population, prior_strength, type_prior, pi0=pi0, player_id=s.defender)
        hist[k].observe(s.context, s.response)
        if type_prior is not None:
            hist_type[k].observe(s.context, s.response)
    pooled = population.q_pooled
    per_player: Dict[Tuple[str, str], Dict[str, List[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    rows: List[Dict[str, Any]] = []
    for s in test:
        if any(src in s.source_file for src in exclude_sources):
            continue
        k = (s.venue, s.defender)
        pm = hist.get(k)
        n_before = pm.observations() if pm else 0
        if pm is None:
            pm = PersonalModel(population, prior_strength, player_id=s.defender)
            hist[k] = pm
        p_pool = pooled
        p_pop = population.probs(s.context)
        p_personal = pm.probs(s.context)
        p_type = None
        if type_prior is not None:
            tm = hist_type.get(k)
            if tm is None:
                pi0 = responsibilities.get(k) if responsibilities else None
                tm = PersonalModel(population, prior_strength, type_prior, pi0=pi0, player_id=s.defender)
                hist_type[k] = tm
            p_type = tm.probs(s.context)
        row = {"player": k, "bin": bin_label(n_before), "n_before": n_before, "style_known": bool(responsibilities and k in responsibilities),
               "actual": s.response, "context": s.context,
               "ll": {"pooled": log_loss(p_pool, s.response), "population": log_loss(p_pop, s.response),
                      "personal": log_loss(p_personal, s.response), **({"type": log_loss(p_type, s.response)} if p_type else {})},
               "brier": {"pooled": brier_fold(p_pool, s.response), "population": brier_fold(p_pop, s.response),
                         "personal": brier_fold(p_personal, s.response), **({"type": brier_fold(p_type, s.response)} if p_type else {})}}
        rows.append(row)
        for arm, v in row["ll"].items():
            per_player[k][arm].append(v)
        if online:
            pm.observe(s.context, s.response)
            if type_prior is not None:
                hist_type[k].observe(s.context, s.response)
    arms = ["pooled", "population", "personal"] + (["type"] if type_prior is not None else [])
    ladder = []
    for lo, hi in BINS:
        label = bin_label(lo)
        sub = [r for r in rows if r["bin"] == label]
        if not sub:
            continue
        ladder.append({"bin": label, "n": len(sub),
                       "log_loss": {a: sum(r["ll"][a] for r in sub) / len(sub) for a in arms if all(a in r["ll"] for r in sub)},
                       "brier": {a: sum(r["brier"][a] for r in sub) / len(sub) for a in arms if all(a in r["brier"] for r in sub)}})
    overall = {"n": len(rows), "players": len(per_player),
               "log_loss": {a: sum(r["ll"][a] for r in rows) / len(rows) for a in arms} if rows else {},
               "brier": {a: sum(r["brier"][a] for r in rows) / len(rows) for a in arms} if rows else {}}
    # paired bootstrap over players: lift of each arm versus the population model
    rng = random.Random(seed)
    players = list(per_player)
    paired: Dict[str, Any] = {}
    for arm in arms:
        if arm == "population":
            continue
        diffs = []
        for _ in range(bootstrap):
            tot = 0.0
            cnt = 0
            for _ in range(len(players)):
                k = players[rng.randrange(len(players))]
                d = per_player[k]
                if arm not in d:
                    continue
                tot += sum(d["population"]) - sum(d[arm])
                cnt += len(d[arm])
            diffs.append(tot / cnt if cnt else 0.0)
        diffs.sort()
        mean = (overall["log_loss"]["population"] - overall["log_loss"][arm]) if rows else 0.0
        paired[arm] = {"lift_vs_population": round(mean, 5), "ci95": [round(diffs[int(0.025 * bootstrap)], 5), round(diffs[int(0.975 * bootstrap) - 1], 5)],
                       "better": mean > 0 and diffs[int(0.025 * bootstrap)] > 0}
    known = [r for r in rows if r["n_before"] > 0]
    new = [r for r in rows if r["n_before"] == 0]
    return {"ladder": ladder, "overall": overall, "paired_lift_log_loss": paired, "online": online,
            "known_players": {"n": len(known), "log_loss": {a: sum(r["ll"][a] for r in known) / len(known) for a in arms} if known else {}},
            "new_players": {"n": len(new), "log_loss": {a: sum(r["ll"][a] for r in new) / len(new) for a in arms} if new else {}},
            "style_known_decisions": sum(1 for r in rows if r["style_known"]), "prior_strength": prior_strength}


def choose_prior_strength(train: Sequence[RiverSpot], population: PopulationModel, candidates: Sequence[float] = (2, 8, 24),
                          windows: int = 4) -> Dict[str, Any]:
    """Forward-in-time validation inside the training 80%: pick the prior weight by pre-action log loss."""
    by_time = sorted(train, key=lambda s: (s.timestamp or "", s.hand_id))
    n = len(by_time)
    scores: Dict[float, List[float]] = {c: [] for c in candidates}
    for w in range(1, windows + 1):
        cut = int(n * w / (windows + 1))
        fit, val = by_time[:cut], by_time[cut:int(n * (w + 1) / (windows + 1))]
        pop_w = PopulationModel.from_spots(fit)
        for c in candidates:
            res = forecast_ladder(fit, val, pop_w, prior_strength=c, bootstrap=1)
            scores[c].append(res["overall"]["log_loss"]["personal"])
    mean = {c: sum(v) / len(v) for c, v in scores.items()}
    best = min(mean, key=mean.get)
    return {"candidates": {str(c): round(m, 5) for c, m in mean.items()}, "chosen": best, "windows": windows}


# ----------------------------------------------------------------- Pluribus range check (extras)
def pluribus_spots(shard_path: str) -> List[Dict[str, Any]]:
    """Heads-up river check -> bet -> response spots from the Pluribus rows, all cards known."""
    from . import cards as C
    from .mirror_advisor import ranker_for
    from .mirror_data import iter_rows, row_to_record
    from .replay import replay

    out = []
    for row in iter_rows(shard_path):
        sf = row.get("source_file") or ""
        if not sf.startswith("pluribus/") or row.get("variant") != "NT":
            continue
        rec = row_to_record(row)
        if rec is None:
            continue
        try:
            res = replay(rec, snapshots=False, observations=True)
        except Exception:  # noqa: BLE001
            continue
        river = [d for d in res.decisions if d.street == "river"]
        if len(river) < 3:
            continue
        d0, d1, d2 = river[0], river[1], river[2]
        if d0.observation["num_active"] != 2 or d0.actual["action"] != "check" or d1.actual["action"] != "bet" or d2.player != d0.player:
            continue
        bet = float(d1.actual["to"] or 0)
        pot_before = float(d2.observation["total_pot"]) - bet
        if pot_before <= 0:
            continue
        defender = res.hand.players[d2.player]
        cards = defender.final_cards()
        board = res.hand.board
        if any(c is None for c in cards) or any(c is None for c in board) or len(board) != 5:
            continue
        r = ranker_for(board)
        eq = r.equity(cards)  # type: ignore[arg-type]
        out.append({"hand": sf, "response": d2.actual["action"], "ratio": bet / pot_before, "defender_equity": eq,
                    "bucket": r.bucket(cards), "defender": defender.name})  # type: ignore[arg-type]
    return out


def pluribus_check(spots: Sequence[Dict[str, Any]], bootstrap: int = 2000, seed: int = 1) -> Dict[str, Any]:
    rng = random.Random(seed)
    by_resp: Dict[str, List[float]] = collections.defaultdict(list)
    for s in spots:
        by_resp[s["response"]].append(s["defender_equity"])
    stats = {}
    for resp, xs in by_resp.items():
        means = []
        for _ in range(bootstrap):
            means.append(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs))
        means.sort()
        stats[resp] = {"n": len(xs), "mean_equity": round(sum(xs) / len(xs), 4),
                       "ci95": [round(means[int(0.025 * bootstrap)], 4), round(means[int(0.975 * bootstrap) - 1], 4)],
                       "buckets": dict(collections.Counter(s["bucket"] for s in spots if s["response"] == resp))}
    order = [stats[r]["mean_equity"] for r in ("fold", "call", "raise") if r in stats]
    return {"spots": len(spots), "by_response": stats, "fold_below_call": ("fold" in stats and "call" in stats and stats["fold"]["mean_equity"] < stats["call"]["mean_equity"]),
            "call_below_raise": ("call" in stats and "raise" in stats and stats["call"]["mean_equity"] < stats["raise"]["mean_equity"]),
            "monotone": order == sorted(order), "half_pot_band": sum(1 for s in spots if 0.4 <= s["ratio"] <= 0.6),
            "label": "directional check: every card known, no showdown selection; intervals are wide"}


# ----------------------------------------------------------------- range readout checks
def synthetic_readout_check(n: int = 1500, seed: int = 5) -> Dict[str, Any]:
    """Unit-level check: with responses drawn from the strength table itself, the readout's
    posterior mass on the true bucket must beat the response-blind prior (log score)."""
    from .mirror_advisor import range_readout
    from .mirror_model import StrengthTable
    from .river_env import scenario_pool

    population = PopulationModel.load()
    table = StrengthTable.default(population).table_for_profile(None)
    pool = scenario_pool(2000, 7)[:n]
    rng = random.Random(seed)
    tot_post = tot_prior = 0.0
    strong_by_resp: Dict[str, List[float]] = collections.defaultdict(list)
    hits = 0
    for sc in pool:
        probs = table[sc.context][sc.defender_bucket]
        u, acc, resp = rng.random(), 0.0, "raise"
        for a in RESPONSES:
            acc += probs[a]
            if u < acc:
                resp = a
                break
        rd = range_readout(sc.board, sc.bettor_cards, sc.context, resp, table)
        p_true = rd["buckets"][sc.defender_bucket]
        p_prior = rd["prior_buckets"][sc.defender_bucket]
        tot_post += math.log(max(p_true, 1e-9))
        tot_prior += math.log(max(p_prior, 1e-9))
        strong_by_resp[resp].append(rd["buckets"]["strong"])
        hits += max(rd["buckets"], key=rd["buckets"].get) == sc.defender_bucket
    return {"n": n, "mean_log_score_readout": round(tot_post / n, 4), "mean_log_score_blind": round(tot_prior / n, 4),
            "improvement": round((tot_post - tot_prior) / n, 4), "top_bucket_accuracy": round(hits / n, 4),
            "mean_posterior_strong_by_response": {k: round(sum(v) / len(v), 4) for k, v in strong_by_resp.items()},
            "label": "twin's own responses: tests the Bayes code, not human behaviour"}


def write_range_checks(out: str, eval_path: Optional[str] = None) -> Dict[str, Any]:
    pl = None
    eval_path = eval_path or os.path.join(SMALL_OUT, "eval.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            pl = json.load(f).get("pluribus")
    result = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
              "held_out_sessions": {"sessions": 0, "note": "no volunteer sessions recorded yet; nothing claimed about humans"},
              "pluribus": pl, "twin_unit_check": synthetic_readout_check()}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    return result


# ----------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pokersim mirror-eval")
    ap.add_argument("--range-checks", action="store_true", help="only write data/mirror/range_checks.json")
    ap.add_argument("--spots", default=os.path.join(LARGE_OUT, "spots.jsonl"))
    ap.add_argument("--manifest", default=os.path.join(LARGE_OUT, "manifest.csv"))
    ap.add_argument("--styles", default=os.path.join(SMALL_OUT, "styles.json"))
    ap.add_argument("--out", default=os.path.join(SMALL_OUT, "eval.json"))
    ap.add_argument("--pluribus-shard", default=None, help="parquet shard holding the Pluribus rows (shard 0)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.range_checks:
        res = write_range_checks(os.path.join(SMALL_OUT, "range_checks.json"), args.out if args.out.endswith("eval.json") else None)
        print(json.dumps({k: res[k] for k in ("twin_unit_check", "pluribus")}, indent=1))
        return 0
    spots = load_spots(args.spots)
    split = load_manifest(args.manifest)
    train, test = modelling_spots(spots, split)
    population = PopulationModel.from_spots(train)
    choice = choose_prior_strength(train, population)
    s = float(choice["chosen"])
    type_prior = None
    resp = None
    if os.path.exists(args.styles):
        with open(args.styles, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("type_prior"):
            type_prior = TypePrior.from_dict(st["type_prior"])
            rows = st.get("responsibilities")
            if rows is None and os.path.exists(RESPONSIBILITIES_PATH):
                with open(RESPONSIBILITIES_PATH, encoding="utf-8") as f:
                    rows = json.load(f)
            resp = {(r["venue"], r["handle"]): r["responsibilities"] for r in rows or []}
    frozen = forecast_ladder(train, test, population, type_prior=type_prior, responsibilities=resp, prior_strength=s, bootstrap=args.bootstrap)
    online = forecast_ladder(train, test, population, type_prior=type_prior, responsibilities=resp, prior_strength=s, online=True, bootstrap=args.bootstrap)
    outside = forecast_ladder(train, test, population, type_prior=type_prior, responsibilities=resp, prior_strength=s, bootstrap=args.bootstrap,
                              exclude_sources=("00035-of-00037", "00036-of-00037"))
    result = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "train": len(train), "test": len(test),
              "prior_strength_selection": choice, "frozen": frozen, "online_adaptation": online, "outside_shards_35_36": outside,
              "ladder": frozen["ladder"], "type_prior_used": type_prior is not None,
              "label": "historical, frozen 20% test; personal counts fitted only on earlier training hands"}
    if args.pluribus_shard and os.path.exists(args.pluribus_shard):
        result["pluribus"] = pluribus_check(pluribus_spots(args.pluribus_shard))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"train": len(train), "test": len(test), "prior_strength": s, "overall": frozen["overall"],
                      "paired": frozen["paired_lift_log_loss"], "ladder": frozen["ladder"]}, indent=1))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
