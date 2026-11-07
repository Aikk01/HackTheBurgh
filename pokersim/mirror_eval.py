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

from .mirror_data import LARGE_OUT, SMALL_OUT, RiverSpot, _last_aggressor, _role, load_spots
from .mirror_model import PersonalModel, PopulationModel, StrengthTable, TypePrior
from .mirror_types import BUCKETS, CONTEXTS, RESPONSES

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
    def _paired(a: str, b: str, keep: Any = None) -> Dict[str, Any]:
        """Lift of arm ``a`` over arm ``b`` (b's log loss minus a's), bootstrapped over players."""
        per: Dict[Any, List[float]] = collections.defaultdict(lambda: [0.0, 0])
        for r in rows:
            if (keep is None or keep(r)) and a in r["ll"] and b in r["ll"]:
                per[r["player"]][0] += r["ll"][b] - r["ll"][a]
                per[r["player"]][1] += 1
        players = list(per)
        n = sum(v[1] for v in per.values())
        if not n:
            return {"n": 0, "players": 0, "lift": None, "ci95": None, "better": False}
        draws = []
        for _ in range(bootstrap):
            tot = cnt = 0.0
            for _ in range(len(players)):
                v = per[players[rng.randrange(len(players))]]
                tot += v[0]
                cnt += v[1]
            draws.append(tot / cnt if cnt else 0.0)
        draws.sort()
        lo, hi = draws[int(0.025 * bootstrap)], draws[int(0.975 * bootstrap) - 1]
        return {"n": n, "players": len(players), "lift": round(sum(v[0] for v in per.values()) / n, 5),
                "ci95": [round(lo, 5), round(hi, 5)], "better": lo > 0}

    gate_comparisons = None
    if type_prior is not None:
        gate_comparisons = {
            "type_vs_population_zero_history_style_known": _paired("type", "population", lambda r: r["style_known"] and r["n_before"] == 0),
            "type_vs_personal_all": _paired("type", "personal"),
        }
    known = [r for r in rows if r["n_before"] > 0]
    new = [r for r in rows if r["n_before"] == 0]
    return {"ladder": ladder, "overall": overall, "paired_lift_log_loss": paired, "online": online, "gate_comparisons": gate_comparisons,
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
        avoid = {c for c in res.hand.players[d1.player].final_cards() if c is not None}
        shares = collections.Counter(b for h, b in r.all_buckets().items() if not (set(h) & avoid))
        total = sum(shares.values()) or 1
        context = _role(_last_aggressor(d2.observation["history"], "turn"), d2.player, d1.player)
        out.append({"hand": sf, "hand_index": rec.index, "response": d2.actual["action"], "ratio": bet / pot_before,
                    "defender_equity": eq, "bucket": r.bucket(cards), "defender": defender.name,  # type: ignore[arg-type]
                    "context": context, "bucket_shares": {b: shares[b] / total for b in BUCKETS}})
    return out


STRENGTH_PLURIBUS = os.path.join(SMALL_OUT, "strength_pluribus.json")


def _readout_log_score(table: Dict[str, Dict[str, Dict[str, float]]], spot: Dict[str, Any]) -> Tuple[float, bool]:
    """log P(true bucket | answer) under the bucket-level readout, and whether the top bucket is the true one."""
    shares, c, a = spot["bucket_shares"], spot["context"], spot["response"]
    w = {b: shares[b] * table[c][b][a] for b in BUCKETS}
    z = sum(w.values()) or 1.0
    return math.log(max(w[spot["bucket"]] / z, 1e-9)), max(w, key=w.get) == spot["bucket"]


def strength_from_pluribus(spots: Sequence[Dict[str, Any]], population: PopulationModel, prior_weight: float = 8.0,
                           bootstrap: int = 2000, seed: int = 1, out: Optional[str] = STRENGTH_PLURIBUS) -> Dict[str, Any]:
    """Fit how each hand-strength bucket answers on half of the Pluribus spots (every card known) and check the
    range readout on the other half.  Replaces the volunteer calibration sessions; used only if it beats the default."""
    import hashlib

    usable = [sp for sp in spots if sp.get("context") in CONTEXTS and sp.get("bucket_shares")]
    half = lambda sp: int(hashlib.sha1(f"{sp['hand']}|{sp.get('hand_index')}".encode()).hexdigest(), 16) % 2  # noqa: E731
    fit = [sp for sp in usable if half(sp) == 0]
    check = [sp for sp in usable if half(sp) == 1]
    counts = {c: {b: {a: 0 for a in RESPONSES} for b in BUCKETS} for c in CONTEXTS}
    for sp in fit:
        counts[sp["context"]][sp["bucket"]][sp["response"]] += 1
    default = StrengthTable.default(population).prior
    calibrated = StrengthTable.from_calibration(population, counts, sessions=0, prior_weight=prior_weight).prior
    d_scores = [_readout_log_score(default, sp) for sp in check]
    c_scores = [_readout_log_score(calibrated, sp) for sp in check]
    diffs = [c[0] - d[0] for c, d in zip(c_scores, d_scores)]
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(bootstrap)) if diffs else [0.0]
    lo, hi = means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    result = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"), "spots": len(usable), "fit_spots": len(fit),
        "check_spots": len(check), "fit_counts": counts, "prior_weight": prior_weight,
        "check": {"log_score_default": round(sum(d[0] for d in d_scores) / max(len(d_scores), 1), 4),
                  "log_score_calibrated": round(sum(c[0] for c in c_scores) / max(len(c_scores), 1), 4),
                  "improvement": round(mean_diff, 4), "ci95": [round(lo, 4), round(hi, 4)],
                  "top_bucket_right_default": round(sum(d[1] for d in d_scores) / max(len(d_scores), 1), 3),
                  "top_bucket_right_calibrated": round(sum(c[1] for c in c_scores) / max(len(c_scores), 1), 3)},
        "use": bool(diffs) and lo > 0,
        "rule": "used by the twins only if the held-out half's readout log score improves with a 95% interval above zero",
        "label": "Pluribus hands (the bot and the pros), every card known; any bet size; split by a hash of the hand",
    }
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return result


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
    held = {"sessions": 0, "note": "no held-out volunteer sessions recorded yet; nothing claimed about humans"}
    sess_path = os.path.join(SMALL_OUT, "sessions_eval.json")
    if os.path.exists(sess_path):
        with open(sess_path, encoding="utf-8") as f:
            se = json.load(f)
        if se.get("heldout_readout", {}).get("decisions"):
            held = {"sessions": se["heldout_readout"]["sessions"], **se["heldout_readout"]}
    result = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
              "held_out_sessions": held, "pluribus": pl, "twin_unit_check": synthetic_readout_check()}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    return result


# ----------------------------------------------------------------- volunteer sessions
SESSIONS_DIR = os.path.join(LARGE_OUT, "sessions")


def load_session_logs(sessions_dir: str = SESSIONS_DIR) -> Dict[str, List[Dict[str, Any]]]:
    """Drill decisions per session token, in the order they were made."""
    import glob

    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(sessions_dir, "*.jsonl"))):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("type") == "drill" and r.get("decision", {}).get("actual"):
                        rows.append(r)
        if rows:
            out[os.path.splitext(os.path.basename(path))[0]] = rows
    return out


def session_counts(sessions_dir: str = SESSIONS_DIR) -> Dict[str, Dict[str, int]]:
    """Independent sessions and decisions per tag (a session counts under the tag of each decision)."""
    counts: Dict[str, Dict[str, Any]] = {}
    for token, rows in load_session_logs(sessions_dir).items():
        for r in rows:
            c = counts.setdefault(r["tag"], {"sessions": set(), "decisions": 0})
            c["sessions"].add(token)
            c["decisions"] += 1
    return {t: {"sessions": len(c["sessions"]), "decisions": c["decisions"]} for t, c in counts.items()}


def _bootstrap_mean(values: Sequence[float], n: int = 2000, seed: int = 1) -> Optional[List[float]]:
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(n))
    return [round(means[int(0.025 * n)], 5), round(means[int(0.975 * n) - 1], 5)]


def evaluate_sessions(sessions_dir: str = SESSIONS_DIR, out_dir: str = SMALL_OUT,
                      population: Optional[PopulationModel] = None) -> Dict[str, Any]:
    """Fit the strength calibration on calibration sessions and score held-out sessions (PLAN.md sections 3 and 6).

    Only aggregates are written; raw decisions stay in ``sessions_dir``.
    """
    from .mirror_advisor import range_readout
    from .mirror_model import StrengthTable
    from .mirror_types import BUCKETS

    logs = load_session_logs(sessions_dir)
    population = population or PopulationModel.load()
    counts = {c: {b: {a: 0 for a in RESPONSES} for b in BUCKETS} for c in CONTEXTS}
    cal_sessions = set()
    cal_decisions = 0
    for token, rows in logs.items():
        for r in rows:
            d = r["decision"]
            if r["tag"] == "calibration" and d.get("defender_strength") in BUCKETS:
                counts[d["context"]][d["defender_strength"]][d["actual"]] += 1
                cal_sessions.add(token)
                cal_decisions += 1
    calibration = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "sessions": len(cal_sessions),
                   "decisions": cal_decisions, "counts": counts,
                   "label": "development sessions only; fitted before any held-out session is scored"}
    if cal_sessions:
        table = StrengthTable.from_calibration(population, counts, len(cal_sessions)).table_for_profile(None)
    else:
        table = StrengthTable.default(population).table_for_profile(None)
    # held-out sessions: committed personal forecast vs the population forecast logged at the same moment
    per_session_diff: List[float] = []
    ll = {"population": 0.0, "personal": 0.0}
    brier = {"population": 0.0, "personal": 0.0}
    n = 0
    readout_post = readout_prior = 0.0
    readout_n = 0
    held_sessions = set()
    for token, rows in logs.items():
        s_ll = {"population": 0.0, "personal": 0.0}
        s_n = 0
        for r in rows:
            if r["tag"] != "heldout":
                continue
            d = r["decision"]
            actual = d["actual"]
            p_pers = (d.get("prediction") or {}).get("probs")
            p_pop = (d.get("extra") or {}).get("population_prediction")
            if not p_pers or not p_pop:
                continue
            held_sessions.add(token)
            for arm, p in (("population", p_pop), ("personal", p_pers)):
                v = log_loss(p, actual)
                ll[arm] += v
                s_ll[arm] += v
                brier[arm] += brier_fold(p, actual)
            n += 1
            s_n += 1
            outcome = (d.get("extra") or {}).get("outcome") or {}
            if outcome.get("bettor_cards") and d.get("defender_strength") in BUCKETS:
                rd = range_readout(d["board"], outcome["bettor_cards"], d["context"], actual, table)
                readout_post += math.log(max(rd["buckets"][d["defender_strength"]], 1e-9))
                readout_prior += math.log(max(rd["prior_buckets"][d["defender_strength"]], 1e-9))
                readout_n += 1
        if s_n:
            per_session_diff.append((s_ll["population"] - s_ll["personal"]) / s_n)
    result = {
        "generated": calibration["generated"], "counts_by_tag": session_counts(sessions_dir), "calibration": {k: v for k, v in calibration.items() if k != "counts"},
        "heldout_forecast": {"sessions": len(held_sessions), "decisions": n,
                             "log_loss": {k: round(v / n, 5) for k, v in ll.items()} if n else None,
                             "brier_fold": {k: round(v / n, 5) for k, v in brier.items()} if n else None,
                             "per_session_lift": [round(x, 5) for x in per_session_diff],
                             "lift_ci95_over_sessions": _bootstrap_mean(per_session_diff),
                             "label": "lift = population log loss minus personal log loss, per decision; positive means knowing the judge helped"},
        "heldout_readout": {"sessions": len(held_sessions), "decisions": readout_n,
                            "mean_log_score_readout": round(readout_post / readout_n, 4) if readout_n else None,
                            "mean_log_score_blind": round(readout_prior / readout_n, 4) if readout_n else None,
                            "improvement": round((readout_post - readout_prior) / readout_n, 4) if readout_n else None,
                            "strength_table": "calibration sessions" if cal_sessions else "synthetic default (no calibration sessions)",
                            "label": "true strength bucket of a held-out human under the readout vs the response-blind prior"},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "strength_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=1)
    with open(os.path.join(out_dir, "sessions_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    return result


# ----------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pokersim mirror-eval")
    ap.add_argument("--range-checks", action="store_true", help="only write data/mirror/range_checks.json")
    ap.add_argument("--sessions", action="store_true",
                    help="fit the strength calibration and score held-out volunteer sessions, then refresh range_checks.json")
    ap.add_argument("--spots", default=os.path.join(LARGE_OUT, "spots.jsonl"))
    ap.add_argument("--manifest", default=os.path.join(LARGE_OUT, "manifest.csv"))
    ap.add_argument("--styles", default=os.path.join(SMALL_OUT, "styles.json"))
    ap.add_argument("--out", default=os.path.join(SMALL_OUT, "eval.json"))
    ap.add_argument("--pluribus-shard", default=None, help="parquet shard holding the Pluribus rows (shard 0)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--pluribus-strength", action="store_true",
                    help="calibrate the strength table on half the Pluribus spots (needs --pluribus-shard)")
    args = ap.parse_args(argv)
    if args.pluribus_strength:
        spots = pluribus_spots(args.pluribus_shard or os.path.join(os.path.dirname(LARGE_OUT), "takara", "train-00000-of-00037.parquet"))
        pop_path = os.path.join(SMALL_OUT, "population_wide.json")
        res = strength_from_pluribus(spots, PopulationModel.load(pop_path if os.path.exists(pop_path) else os.path.join(SMALL_OUT, "population.json")))
        print(json.dumps({k: res[k] for k in ("spots", "fit_spots", "check_spots", "check", "use")}, indent=1))
        return 0
    if args.sessions:
        res = evaluate_sessions()
        write_range_checks(os.path.join(SMALL_OUT, "range_checks.json"))
        print(json.dumps({k: res[k] for k in ("counts_by_tag", "calibration", "heldout_forecast", "heldout_readout")}, indent=1))
        return 0
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
