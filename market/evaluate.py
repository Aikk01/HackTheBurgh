"""Held-out evaluation: calibration of the price and the call/fold rule.

    python -m market.evaluate            # writes figures/*.png and results.json

* Split: by TABLE (one table ~ one session), 80/20, seed 0, fixed before any fitting.
* Methods (same equity engine, different opponent range):
    uniform - card-removed uniform range              (baseline)
    prior   - position/role preflop prior, no updates (ablation)
    model   - prior x action likelihoods              (ours)
* Outcome: tracked player's share of the pot at showdown (1 / 0.5 / 0).
* Metrics: Brier, log-loss, 10-bin reliability; 95% CIs from a bootstrap that resamples
  whole hands (events within a hand are correlated); paired differences vs baseline.
* Decision metric: at every spot where the tracked player faces a bet, call iff price >
  pot-odds price.  Payoff of calling = share * (pot + to_call) - to_call (in big blinds,
  realised showdown result, ignoring later streets); folding = 0.  Reported per 100 decisions.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np

from .likelihood import CLASSES, LikelihoodModel, training_rows
from .parse import PROCESSED, ROOT, load_hands
from .tracker import MarketTracker

FIG = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results.json")
METHODS = ("uniform", "prior", "model")
SEED = 0
N_BOOT = 1000


def split_by_table(hands: List[Dict], test_frac: float = 0.2, seed: int = SEED):
    tables = sorted({h["table"] for h in hands})
    rng = np.random.default_rng(seed)
    test_t = set(rng.choice(tables, size=int(round(test_frac * len(tables))), replace=False).tolist())
    train = [h for h in hands if h["table"] not in test_t]
    test = [h for h in hands if h["table"] in test_t]
    return train, test, sorted(test_t)


def collect(test: List[Dict], model: LikelihoodModel) -> Dict[str, np.ndarray]:
    rows = {k: [] for k in ("hand", "y", "street", "facing", "pot", "to_call", "bb", "lat",
                            *[f"p_{m}" for m in METHODS])}
    for hi, h in enumerate(test):
        # Both hole cards must be known: when only one player shows, it is the winner
        # (the loser mucked) and the label is 1.0 by construction - see NOTES.md.
        if not h["showdown"] or h["share"] is None or not all(h["hole"]):
            continue
        for me in (0, 1):
            if h["hole"][me] is None:
                continue
            # model first: its latency then includes the (uncached) equity computation
            streams = {m: list(MarketTracker(h, me, model, mode=m).run()) for m in ("model", "uniform", "prior")}
            for k, ev in enumerate(streams["model"]):
                a = ev["action"]
                rows["hand"].append(hi)
                rows["y"].append(h["share"][me])
                rows["street"].append(ev["street"])
                rows["facing"].append(ev["decision"] is not None)
                rows["pot"].append(a["pot"] if a else 0.0)
                rows["to_call"].append(a["to_call"] if a else 0.0)
                rows["bb"].append(h["bb"])
                rows["lat"].append(ev["latency_ms"])
                for m in METHODS:
                    rows[f"p_{m}"].append(streams[m][k]["price"])
    return {k: np.array(v) for k, v in rows.items()}


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_bootstrap(values: Dict[str, np.ndarray], groups: np.ndarray, n: int = N_BOOT, seed: int = SEED):
    """Per-hand sums -> resample hands -> mean of each series.  Returns {name: (mean, lo, hi)}."""
    uniq, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv)
    sums = {k: np.bincount(inv, weights=v) for k, v in values.items()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    out = {}
    for k, s in sums.items():
        boots = s[idx].sum(1) / cnt[idx].sum(1)
        out[k] = (float(s.sum() / cnt.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def reliability(p, y, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        out.append({"bin": [float(edges[i]), float(edges[i + 1])], "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()) if m.any() else None,
                    "mean_outcome": float(y[m].mean()) if m.any() else None})
    ece = sum(r["n"] * abs(r["mean_pred"] - r["mean_outcome"]) for r in out if r["n"]) / len(p)
    return out, float(ece)


def action_model_heldout(model: LikelihoodModel, train, test):
    """Held-out log-loss of the action model vs. predicting the training class frequencies."""
    tr, te = training_rows(train), training_rows(test)
    out = {}
    for k in CLASSES:
        X, y, _ = te[k]
        freq = np.bincount(tr[k][1], minlength=len(CLASSES[k])) / len(tr[k][1])
        p = model.models[k].predict_proba(X)
        ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
        ll0 = -np.log(freq[y]).mean()
        out[k] = {"n_test": int(len(y)), "logloss_model": float(ll), "logloss_class_freq": float(ll0),
                  "accuracy": float((p.argmax(1) == y).mean())}
    return out


def plot_reliability(R, rel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"uniform": "#9aa0a6", "prior": "#e8a33d", "model": "#2f6fdf"}
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.5, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.8, label="perfect")
    for m in METHODS:
        pts = [(r["mean_pred"], r["mean_outcome"]) for r in rel[m][0] if r["n"] >= 20]
        x, y = zip(*pts)
        ax.plot(x, y, "o-", c=colors[m], label=f"{m}  (Brier {R[m]['brier'][0]:.4f}, ECE {rel[m][1]:.3f})")
        axh.hist(R["_p"][m], bins=20, range=(0, 1), histtype="step", color=colors[m])
    ax.set_ylabel("observed showdown share")
    ax.set_title("Reliability of the live price (held-out tables)")
    ax.legend(fontsize=8, loc="upper left")
    axh.set_xlabel("predicted win probability (price)")
    axh.set_ylabel("events")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_decisions(D, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["always_call", "uniform", "prior", "model"]
    mean = [D[n][0] for n in names]
    err = [[D[n][0] - D[n][1] for n in names], [D[n][2] - D[n][0] for n in names]]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(names, mean, yerr=err, capsize=4, color=["#cccccc", "#9aa0a6", "#e8a33d", "#2f6fdf"])
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("big blinds / 100 decisions")
    ax.set_title("Call/fold rule vs pot odds (held-out, 95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    hands = load_hands()
    train, test, test_tables = split_by_table(hands)
    os.makedirs(PROCESSED, exist_ok=True)
    with open(os.path.join(PROCESSED, "split.json"), "w") as f:
        json.dump({"seed": SEED, "test_tables": test_tables}, f)
    tr_players = {p for h in train for p in h["players"]}
    te_players = {p for h in test for p in h["players"]}

    model = LikelihoodModel().fit(train, seed=SEED)
    model.save()
    E = collect(test, model)
    y, g = E["y"], E["hand"]

    series = {}
    for m in METHODS:
        series[f"{m}_brier"] = brier(E[f"p_{m}"], y)
        series[f"{m}_logloss"] = logloss(E[f"p_{m}"], y)
    for m in ("prior", "model"):
        series[f"{m}_minus_uniform_brier"] = series[f"{m}_brier"] - series["uniform_brier"]
        series[f"{m}_minus_uniform_logloss"] = series[f"{m}_logloss"] - series["uniform_logloss"]
    series["model_minus_prior_brier"] = series["model_brier"] - series["prior_brier"]
    CI = cluster_bootstrap(series, g)
    R = {m: {"brier": CI[f"{m}_brier"], "logloss": CI[f"{m}_logloss"]} for m in METHODS}
    R["_p"] = {m: E[f"p_{m}"] for m in METHODS}
    rel = {m: reliability(E[f"p_{m}"], y) for m in METHODS}

    # per-street Brier (point estimates)
    by_street = {s: {m: float(series[f"{m}_brier"][E["street"] == s].mean()) for m in METHODS}
                 for s in ("flop", "turn", "river")}

    # decisions
    f = E["facing"]
    pot, tc, bb, yy = E["pot"][f], E["to_call"][f], E["bb"][f], y[f]
    pot_odds = tc / (pot + tc)
    call_pay = (yy * (pot + tc) - tc) / bb
    dec = {"always_call": call_pay * 100}
    for m in METHODS:
        dec[m] = np.where(E[f"p_{m}"][f] > pot_odds, call_pay, 0.0) * 100
    dec["model_minus_uniform"] = dec["model"] - dec["uniform"]
    D = cluster_bootstrap(dec, g[f])

    os.makedirs(FIG, exist_ok=True)
    plot_reliability(R, rel, os.path.join(FIG, "reliability.png"))
    plot_decisions(D, os.path.join(FIG, "decisions.png"))

    results = {
        "data": {"source": "gabbler/PokerData handhq PokerStars 200NL (Jul 2009), 6-seat tables",
                 "hu_from_flop_hands": len(hands), "train_hands": len(train), "test_hands": len(test),
                 "test_tables": len(test_tables), "train_tables": len({h['table'] for h in train}),
                 "test_showdown_hands": int(len(np.unique(g))), "test_events": int(len(y)),
                 "test_decisions": int(f.sum()),
                 "players_in_both_splits_frac_of_test": len(tr_players & te_players) / len(te_players)},
        "split": {"by": "table", "test_frac": 0.2, "seed": SEED},
        "calibration": {m: {"brier": CI[f"{m}_brier"], "logloss": CI[f"{m}_logloss"], "ece": rel[m][1],
                            "reliability": rel[m][0]} for m in METHODS},
        "paired_differences": {k: CI[k] for k in CI if "minus" in k},
        "brier_by_street": by_street,
        "decisions_bb_per_100": {k: D[k] for k in D},
        "action_model_heldout": action_model_heldout(model, train, test),
        "latency_ms_per_event": {"median": float(np.median(E["lat"])), "p95": float(np.percentile(E["lat"], 95)),
                                 "max": float(E["lat"].max())},
        "notes": "(mean, 2.5%, 97.5%) from a hand-level cluster bootstrap, 1000 resamples, seed 0. "
                 "Evaluated on held-out showdown hands where BOTH players' cards are known "
                 "(selection-biased, see NOTES.md).",
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(RESULTS, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({k: results[k] for k in ("data", "paired_differences", "brier_by_street",
                                               "decisions_bb_per_100", "action_model_heldout",
                                               "latency_ms_per_event")}, indent=1))
    for m in METHODS:
        print(m, "brier", R[m]["brier"], "logloss", R[m]["logloss"], "ECE", rel[m][1])


if __name__ == "__main__":
    main()
