"""Player types from public style statistics (PLAN.md section 3, type layer).

Offline only (NumPy + scikit-learn).  Produces ``data/mirror/styles.json`` with
the fitted types, the type response tables (a :class:`TypePrior`), the style map
points, and the type-prior gate result from :mod:`pokersim.mirror_eval`.

Reliability note: the plan asks for a split-half correlation over odd and even
hands.  The style scan aggregates counts per player, so the minimum hand count
is chosen with the equivalent binomial reliability
``var_true / (var_true + E[p(1-p)/n]) >= 0.7`` for every feature.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .mirror_data import LARGE_OUT, ROOT, SMALL_OUT, STYLE_FEATURES, load_spots, read_style_table
from .mirror_eval import RESPONSIBILITIES_PATH, forecast_ladder, load_manifest, modelling_spots
from .mirror_model import PopulationModel, TypePrior
from .mirror_types import CONTEXTS, RESPONSES

# joblib (used by scikit-learn) shells out to wmic to count physical cores, which Windows 11 no longer ships.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

SHRINK = 20.0  # pseudo-hands pulling each rate towards its venue mean


def _rates(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        d = {"venue": r["venue"], "handle": r["handle"], "hands": r["hands"]}
        for f in STYLE_FEATURES:
            num, den = r[f"{f}_num"], r[f"{f}_den"]
            d[f] = (num / den) if den else None
            d[f + "_den"] = den
        out.append(d)
    return out


def binomial_reliability(rows: Sequence[Dict[str, Any]], feature: str, n: int) -> float:
    """Reliability of an observed rate from ``n`` trials, estimated from players with >= n trials."""
    xs = [r for r in rows if r[feature] is not None and r[feature + "_den"] >= n]
    if len(xs) < 50:
        return 0.0
    ps = [r[feature] for r in xs]
    mean = sum(ps) / len(ps)
    var_obs = sum((p - mean) ** 2 for p in ps) / (len(ps) - 1)
    noise = sum(p * (1 - p) / r[feature + "_den"] for p, r in zip(ps, xs)) / len(xs)
    var_true = max(var_obs - noise, 1e-9)
    binom_at_n = sum(p * (1 - p) for p in ps) / len(ps) / n
    return var_true / (var_true + binom_at_n)


def choose_n_min(rows: Sequence[Dict[str, Any]], candidates: Sequence[int] = (25, 50, 100, 150, 200, 300, 500),
                 threshold: float = 0.7) -> Dict[str, Any]:
    table: Dict[int, Dict[str, float]] = {}
    for n in candidates:
        table[n] = {f: round(binomial_reliability(rows, f, n), 3) for f in STYLE_FEATURES}
    usable = [f for f in STYLE_FEATURES if any(table[n][f] >= threshold for n in candidates)]
    dropped = [f for f in STYLE_FEATURES if f not in usable]
    chosen = None
    for n in candidates:
        if all(table[n][f] >= threshold for f in usable):
            chosen = n
            break
    return {"reliability": {str(n): v for n, v in table.items()}, "n_min": chosen or candidates[-1], "features": usable,
            "dropped": dropped, "threshold": threshold}


def feature_matrix(rows: Sequence[Dict[str, Any]], features: Sequence[str], n_min: int):
    import numpy as np

    venue_mean: Dict[Tuple[str, str], float] = {}
    for f in features:
        by_v: Dict[str, List[float]] = {}
        for r in rows:
            if r[f] is not None and r["hands"] >= n_min:
                by_v.setdefault(r["venue"], []).append(r[f])
        for v, xs in by_v.items():
            venue_mean[(v, f)] = sum(xs) / len(xs)
    kept = [r for r in rows if r["hands"] >= n_min and all(r[f] is not None for f in features)]
    X = np.zeros((len(kept), len(features)))
    for i, r in enumerate(kept):
        for j, f in enumerate(features):
            m = venue_mean.get((r["venue"], f), 0.5)
            den = r[f + "_den"]
            p = (r[f] * den + SHRINK * m) / (den + SHRINK)
            p = min(max(p, 0.005), 0.995)
            X[i, j] = math.log(p / (1 - p))
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    return kept, (X - mu) / sd, {"mean": mu.tolist(), "sd": sd.tolist()}


def fit_types(X, k_range: Sequence[int] = range(2, 9), seed: int = 1, bootstrap: int = 20, ari_threshold: float = 0.7,
              k_cap: int = 6) -> Dict[str, Any]:
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    results = []
    for k in k_range:
        gmm = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=3).fit(X)
        base = gmm.predict(X)
        aris = []
        for b in range(bootstrap):
            idx = rng.integers(0, len(X), len(X))
            g2 = GaussianMixture(k, covariance_type="full", random_state=seed + b + 1, n_init=1).fit(X[idx])
            aris.append(adjusted_rand_score(base, g2.predict(X)))
        results.append({"k": k, "bic": float(gmm.bic(X)), "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)), "model": gmm})
    stable = [r for r in results if r["ari_mean"] >= ari_threshold and r["k"] <= k_cap]
    chosen = min(stable, key=lambda r: r["bic"]) if stable else min(results, key=lambda r: r["bic"] if r["k"] <= k_cap else float("inf"))
    return {"chosen": chosen, "table": [{k: v for k, v in r.items() if k != "model"} for r in results], "stable": bool(stable)}


def name_type(centroid: Dict[str, float], ref: Dict[str, float]) -> str:
    loose = centroid.get("vpip", 0) > ref.get("vpip", 0)
    aggressive = centroid.get("af", 0) > ref.get("af", 0) or centroid.get("pfr", 0) > ref.get("pfr", 0)
    return ("loose" if loose else "tight") + "-" + ("aggressive" if aggressive else "passive")


def build(styles_path: str, out_path: str, seed: int = 1, bootstrap: int = 20, k_cap: int = 6,
          spots_path: Optional[str] = None, manifest_path: Optional[str] = None, eval_bootstrap: int = 1000) -> Dict[str, Any]:
    import numpy as np
    from sklearn.decomposition import PCA

    rows = _rates(read_style_table(styles_path))
    rel = choose_n_min(rows)
    features = rel["features"] or list(STYLE_FEATURES)
    kept, X, scaler = feature_matrix(rows, features, rel["n_min"])
    fit = fit_types(X, seed=seed, bootstrap=bootstrap, k_cap=k_cap)
    gmm = fit["chosen"]["model"]
    k = fit["chosen"]["k"]
    resp = gmm.predict_proba(X)
    centroids_std = gmm.means_
    # centroids back in rate units: invert standardisation and logit
    centroids = []
    for row in centroids_std:
        c = {}
        for j, f in enumerate(features):
            z = row[j] * scaler["sd"][j] + scaler["mean"][j]
            c[f] = round(1 / (1 + math.exp(-z)), 4)
        centroids.append(c)
    ref = {f: float(np.median([c[f] for c in centroids])) for f in features}
    names_raw = [name_type(c, ref) for c in centroids]
    names = []
    for i, n in enumerate(names_raw):
        names.append(n if names_raw.count(n) == 1 else f"{n} {names_raw[:i].count(n) + 1}")
    pca = PCA(2, random_state=seed).fit(X)
    coords = pca.transform(X)
    rng = random.Random(seed)
    idx = list(range(len(kept)))
    rng.shuffle(idx)
    sample = idx[:3000]
    map_points = [{"x": round(float(coords[i, 0]), 3), "y": round(float(coords[i, 1]), 3), "type": int(np.argmax(resp[i])),
                   "venue": kept[i]["venue"], "hands": kept[i]["hands"]} for i in sample]
    centroid_xy = pca.transform(centroids_std)
    responsibilities = {(r["venue"], r["handle"]): [float(x) for x in resp[i]] for i, r in enumerate(kept)}
    shares = resp.mean(axis=0).tolist()

    # type response tables from the training spots
    spots_path = spots_path or os.path.join(LARGE_OUT, "spots.jsonl")
    manifest_path = manifest_path or os.path.join(LARGE_OUT, "manifest.csv")
    spots = load_spots(spots_path)
    split = load_manifest(manifest_path)
    train, test = modelling_spots(spots, split)
    population = PopulationModel.from_spots(train)
    s = chosen_prior_strength(train, population)
    soft = [{c: {a: 0.0 for a in RESPONSES} for c in CONTEXTS} for _ in range(k)]
    matched = 0
    for sp in train:
        r = responsibilities.get((sp.venue, sp.defender))
        if r is None:
            continue
        matched += 1
        for t in range(k):
            soft[t][sp.context][sp.response] += r[t]
    q_k = []
    for t in range(k):
        q_t = {}
        for c in CONTEXTS:
            q = population.probs(c)
            tot = sum(soft[t][c].values())
            q_t[c] = {a: (soft[t][c][a] + s * q[a]) / (tot + s) for a in RESPONSES}
        q_k.append(q_t)
    type_prior = TypePrior(names, shares, q_k)

    gate = compute_gate(train, test, population, type_prior, responsibilities, s, bootstrap=eval_bootstrap)
    ladder = gate.pop("ladder")
    result = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"), "k": k, "names": names, "shares": [round(x, 4) for x in shares],
        "centroids": centroids, "features": features, "reliability": rel, "players_used": len(kept), "players_total": len(rows),
        "selection": fit["table"], "stable": fit["stable"], "map_points": map_points,
        "centroid_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in centroid_xy],
        "type_prior": type_prior.to_dict(), "matched_training_offers": matched, "gate": gate, "ladder_with_types": ladder,
        "responsibilities_path": os.path.relpath(RESPONSIBILITIES_PATH, ROOT).replace(os.sep, "/"),
    }
    # Per-player responsibilities are large (one row per player) and keyed by handle: keep them out of git.
    os.makedirs(os.path.dirname(RESPONSIBILITIES_PATH), exist_ok=True)
    with open(RESPONSIBILITIES_PATH, "w", encoding="utf-8") as f:
        json.dump([{"venue": v, "handle": h, "responsibilities": [round(x, 4) for x in r]} for (v, h), r in responsibilities.items()], f)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result


GATE_RULE = ("types ship only if (a) for style-known players with no earlier river decisions the type prior beats the "
             "population model, and (b) the type-plus-history model beats the personal (population-plus-history) model, "
             "each with a 95% bootstrap interval over players above zero")


def chosen_prior_strength(train: Sequence[Any], population: PopulationModel) -> float:
    """The prior weight frozen by forward validation (data/mirror/eval.json), recomputed if absent."""
    path = os.path.join(SMALL_OUT, "eval.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            chosen = (json.load(f).get("prior_strength_selection") or {}).get("chosen")
        if chosen:
            return float(chosen)
    from .mirror_eval import choose_prior_strength
    return float(choose_prior_strength(train, population)["chosen"])


def compute_gate(train: Sequence[Any], test: Sequence[Any], population: PopulationModel, type_prior: TypePrior,
                 responsibilities: Dict[Tuple[str, str], Sequence[float]], prior_strength: float,
                 bootstrap: int = 1000) -> Dict[str, Any]:
    """Does knowing a player's type add anything the live model does not already have?

    The plan's first wording compared the type model with the population model only; since the type model
    also learns from the player's history, that comparison passes on history alone.  Both checks here
    isolate what the types contribute."""
    full = forecast_ladder(train, test, population, type_prior=type_prior, responsibilities=responsibilities,
                           prior_strength=prior_strength, bootstrap=bootstrap)
    outside = forecast_ladder(train, test, population, type_prior=type_prior, responsibilities=responsibilities,
                              prior_strength=prior_strength, bootstrap=bootstrap, exclude_sources=("00035-of-00037", "00036-of-00037"))
    a = full["gate_comparisons"]["type_vs_population_zero_history_style_known"]
    b = full["gate_comparisons"]["type_vs_personal_all"]
    orig = full["paired_lift_log_loss"].get("type", {})
    return {
        "pass": bool(a["better"] and b["better"]), "rule": GATE_RULE, "prior_strength": prior_strength,
        "type_vs_population_zero_history": a, "type_vs_personal": b,
        "original_rule_type_vs_population": {"lift": orig.get("lift_vs_population"), "ci95": orig.get("ci95"), "better": orig.get("better")},
        "outside_shards_35_36": outside["gate_comparisons"],
        "style_known_test_decisions": full["style_known_decisions"], "test_decisions": full["overall"]["n"],
        "lift_vs_population_log_loss": orig.get("lift_vs_population"), "ci95": orig.get("ci95"),
        "ladder": full["ladder"],
    }


def recompute_gate(styles_path: str, bootstrap: int = 1000) -> Dict[str, Any]:
    """Re-run the gate on an existing type fit (no refit), e.g. after the prior weight was selected."""
    from .mirror_eval import RESPONSIBILITIES_PATH as RP

    with open(styles_path, encoding="utf-8") as f:
        st = json.load(f)
    type_prior = TypePrior.from_dict(st["type_prior"])
    with open(RP, encoding="utf-8") as f:
        responsibilities = {(r["venue"], r["handle"]): r["responsibilities"] for r in json.load(f)}
    train, test = modelling_spots(load_spots(os.path.join(LARGE_OUT, "spots.jsonl")), load_manifest(os.path.join(LARGE_OUT, "manifest.csv")))
    population = PopulationModel.from_spots(train)
    gate = compute_gate(train, test, population, type_prior, responsibilities, chosen_prior_strength(train, population), bootstrap)
    st["ladder_with_types"] = gate.pop("ladder")
    st["gate"] = gate
    st["gate_recomputed"] = _dt.datetime.now().isoformat(timespec="seconds")
    with open(styles_path, "w", encoding="utf-8") as f:
        json.dump(st, f)
    return gate


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pokersim mirror-cluster")
    ap.add_argument("--styles", default=os.path.join(LARGE_OUT, "style_table.csv"))
    ap.add_argument("--out", default=os.path.join(SMALL_OUT, "styles.json"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=20)
    ap.add_argument("--k-cap", type=int, default=6)
    ap.add_argument("--gate-only", action="store_true", help="re-run the type-prior gate on the existing fit")
    args = ap.parse_args(argv)
    if args.gate_only:
        gate = recompute_gate(args.out)
        print(json.dumps({k: v for k, v in gate.items() if k not in ("outside_shards_35_36",)}, indent=1))
        return 0
    res = build(args.styles, args.out, seed=args.seed, bootstrap=args.bootstrap, k_cap=args.k_cap)
    print(json.dumps({k: res[k] for k in ("k", "names", "shares", "centroids", "players_used", "players_total", "stable", "gate",
                                          "matched_training_offers")}, indent=1))
    print("reliability:", json.dumps(res["reliability"]["reliability"]), "n_min", res["reliability"]["n_min"], "dropped", res["reliability"]["dropped"])
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
