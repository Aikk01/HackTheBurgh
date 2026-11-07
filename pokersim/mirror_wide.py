"""Phase 6: every heads-up river decision in the corpus (CHECKLIST.md Phase 6).

Reads the compact table written by ``python -m pokersim mirror-prepare --wide`` and writes
* ``data/mirror/eval_wide.json``    forecast ladder on the frozen test half: pooled, population, situation
                                    (gradient-boosted trees on public features), personal (the player's own
                                    history), situation + personal; player-level bootstrap intervals
* ``data/mirror/bluff_test.json``   a pure bluffer scored by real players' recorded answers
* ``data/mirror/real_players.json`` players with enough decisions for twins and the RL benchmark
* ``data/mirror/neural/neural_dataset.npz``  the dataset for ``scripts/train_neural_forecaster.py`` (GPU)
"""
from __future__ import annotations

import collections
import datetime as _dt
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mirror_data import SIZE_CLASSES, SMALL_OUT, WIDE_META, WIDE_TABLE
from .mirror_types import CONTEXTS, RESPONSES

EVAL_WIDE = os.path.join(SMALL_OUT, "eval_wide.json")
BLUFF_TEST = os.path.join(SMALL_OUT, "bluff_test.json")
REAL_PLAYERS = os.path.join(SMALL_OUT, "real_players.json")
NEURAL_DATASET = os.path.join(SMALL_OUT, "neural", "neural_dataset.npz")
BINS = ((0, 0), (1, 3), (4, 15), (16, 63), (64, None))
ARMS = ("pooled", "population", "situation", "personal", "situation_personal")
CTX_LETTERS, RESP_LETTERS = "cbd", "fcr"


def bin_label(n: int) -> str:
    for lo, hi in BINS:
        if n >= lo and (hi is None or n <= hi):
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi is not None else f"{lo}+")
    return "?"


def iso(ts: int) -> str:
    return _dt.datetime.fromtimestamp(int(ts), _dt.timezone.utc).replace(tzinfo=None).isoformat()


def load_table(path: str = WIDE_TABLE, meta_path: str = WIDE_META) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path) as z:
        t = {k: z[k] for k in z.files}
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return t, meta


def size_class_codes(table_size: np.ndarray) -> np.ndarray:
    return np.where(table_size <= 2, 0, np.where(table_size <= 6, 1, 2)).astype(np.int8)


def board_features(board: np.ndarray) -> Dict[str, np.ndarray]:
    """Texture flags from the five board cards (code = rank * 4 + suit, rank 0 = deuce)."""
    n = board.shape[0]
    b = np.where(board < 0, 0, board).astype(np.int16)
    ranks, suits = b // 4, b % 4
    rows = np.arange(n)
    rc = np.zeros((n, 13), np.int8)
    sc = np.zeros((n, 4), np.int8)
    for j in range(5):
        np.add.at(rc, (rows, ranks[:, j]), 1)
        np.add.at(sc, (rows, suits[:, j]), 1)
    present = np.concatenate([rc[:, 12:13] > 0, rc > 0], axis=1).astype(np.int8)  # the ace also plays low
    window = np.stack([present[:, k:k + 5].sum(1) for k in range(10)], 1).max(1)
    first4 = rc.copy()
    np.add.at(first4, (rows, ranks[:, 4]), -1)
    f32 = np.float32
    return {
        "paired": (rc.max(1) >= 2).astype(f32), "trips": (rc.max(1) >= 3).astype(f32),
        "two_pair_board": ((rc >= 2).sum(1) >= 2).astype(f32),
        "flush3": (sc.max(1) >= 3).astype(f32), "flush4": (sc.max(1) >= 4).astype(f32),
        "straight3": (window >= 3).astype(f32), "straight4": (window >= 4).astype(f32),
        "river_pairs": (first4[rows, ranks[:, 4]] >= 1).astype(f32),
        "river_suit_n": sc[rows, suits[:, 4]].astype(f32) / 5.0,
        "river_over": (ranks[:, 4] > ranks[:, :4].max(1)).astype(f32),
        "river_rank": ranks[:, 4].astype(f32) / 12.0, "high_rank": ranks.max(1).astype(f32) / 12.0,
        "broadway": (ranks >= 8).sum(1).astype(f32) / 5.0,
    }


def situation_matrix(t: Dict[str, np.ndarray], n_venues: int) -> Tuple[np.ndarray, List[str]]:
    """Public features of each spot: what anyone watching the table could know before the response."""
    f32 = np.float32
    cols: Dict[str, np.ndarray] = {}
    for k, c in enumerate(CONTEXTS):
        cols[f"ctx_{c}"] = (t["context"] == k).astype(f32)
    scc = size_class_codes(t["table_size"])
    for k, name in enumerate(SIZE_CLASSES):
        cols[f"table_{name}"] = (scc == k).astype(f32)
    ts = t["table_size"].astype(f32)
    cols["table_size"] = ts / 10.0
    ratio = np.clip(t["ratio"].astype(f32), 0.01, 20.0)
    cols["ratio"] = np.minimum(ratio, 3.0)
    cols["log_ratio"] = np.log(ratio)
    cols["log_spr"] = np.log1p(np.clip(t["spr"], 0, 1e4)).astype(f32)
    cols["log_pot_bb"] = np.log1p(np.clip(t["pot_bb"], 0, 1e5)).astype(f32)
    cols["log_bb"] = np.log(np.clip(t["bb"], 0.01, 1e5)).astype(f32)
    denom = np.maximum(ts - 1, 1)
    cols["def_seat"] = (t["def_idx"] / denom).astype(f32)
    cols["bet_seat"] = (t["bet_idx"] / denom).astype(f32)
    cols["bettor_last"] = (t["bet_idx"] == t["table_size"] - 1).astype(f32)
    cols["flop_players"] = t["flop_players"].astype(f32) / 10.0
    cols["multiway_flop"] = (t["flop_players"] > 2).astype(f32)
    for k, name in enumerate(("none", "bettor", "defender", "other")):
        cols[f"preflop_raiser_{name}"] = (t["pf_aggr"] == k).astype(f32)
    for k, name in enumerate(("checked", "bettor", "defender", "other")):
        cols[f"flop_bet_{name}"] = (t["flop_ctx"] == k).astype(f32)
    for name, v in board_features(t["board"]).items():
        cols[f"board_{name}"] = v
    for k in range(n_venues):
        cols[f"venue_{k}"] = (t["venue"] == k).astype(f32)
    cols["raise_legal"] = t["raise_legal"].astype(f32)
    names = list(cols)
    X = np.empty((len(t["response"]), len(names)), f32)
    for j, k in enumerate(names):
        X[:, j] = cols[k]
    return X, names


# ----------------------------------------------------------------- forecasters
def population_q(t: Dict[str, np.ndarray], rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(context x response) Laplace-smoothed rates from ``rows`` and the raw counts."""
    counts = np.zeros((3, 3))
    np.add.at(counts, (t["context"][rows], t["response"][rows]), 1)
    return (counts + 1) / (counts.sum(1, keepdims=True) + 3), counts


def _player_counts(t: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for p, c, a in zip(t["player"][rows].tolist(), t["context"][rows].tolist(), t["response"][rows].tolist()):
        m = out.get(p)
        if m is None:
            m = out[p] = np.zeros((3, 3))
        m[c, a] += 1
    return out


def personal_forecasts(t: Dict[str, np.ndarray], history_rows: np.ndarray, rows: np.ndarray, q: np.ndarray, s: float,
                       online: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dirichlet forecast per row: prior ``s * q[context]`` plus the player's counts in ``history_rows``.

    ``online`` also adds each forecast row's own answer after it is scored, so later rows see earlier ones
    (a bettor who keeps learning).  Returns (rows in time order, probs, decisions seen before each row)."""
    counts = _player_counts(t, history_rows)
    order = rows[np.argsort(t["seq"][rows], kind="stable")]
    probs = np.empty((len(order), 3))
    n_before = np.empty(len(order), np.int32)
    zero = np.zeros((3, 3))
    for i, (p, c, a) in enumerate(zip(t["player"][order].tolist(), t["context"][order].tolist(), t["response"][order].tolist())):
        m = counts.get(p, zero)
        probs[i] = (s * q[c] + m[c]) / (s + m[c].sum())
        n_before[i] = int(m.sum())
        if online:
            if m is zero:
                m = counts[p] = np.zeros((3, 3))
            m[c, a] += 1
    return order, probs, n_before


def log_loss(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-6))


def choose_prior_strength(t: Dict[str, np.ndarray], train_rows: np.ndarray, candidates: Sequence[float] = (2, 8, 24, 64),
                          windows: int = 4) -> Dict[str, Any]:
    """Forward-in-time validation inside the training half, as for the original cohort."""
    rows = train_rows[np.argsort(t["seq"][train_rows], kind="stable")]
    n = len(rows)
    scores: Dict[float, List[float]] = {c: [] for c in candidates}
    for w in range(1, windows + 1):
        fit = rows[: int(n * w / (windows + 1))]
        val = rows[int(n * w / (windows + 1)): int(n * (w + 1) / (windows + 1))]
        q, _ = population_q(t, fit)
        for c in candidates:
            order, p, _ = personal_forecasts(t, fit, val, q, c, online=False)
            scores[c].append(float(log_loss(p, t["response"][order]).mean()))
    mean = {c: sum(v) / len(v) for c, v in scores.items()}
    best = min(mean, key=mean.get)
    return {"candidates": {str(c): round(m, 5) for c, m in mean.items()}, "chosen": best, "windows": windows}


def fit_situation(X: np.ndarray, y: np.ndarray, rows: np.ndarray, seed: int = 1, max_iter: int = 300):
    """Gradient-boosted trees on public features (CPU)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=seed)
    clf.fit(X[rows], y[rows])
    return clf


def situation_probs(clf, X: np.ndarray, raise_legal: np.ndarray) -> np.ndarray:
    p = np.zeros((len(X), 3))
    p[:, clf.classes_] = clf.predict_proba(X)
    p[~raise_legal.astype(bool), 2] = 0.0
    return p / p.sum(1, keepdims=True)


def combine(situation: np.ndarray, personal: np.ndarray, population: np.ndarray) -> np.ndarray:
    """The situation forecast tilted by how this player deviates from the population in this context."""
    p = situation * (personal / np.maximum(population, 1e-9))
    return p / p.sum(1, keepdims=True)


def player_bootstrap(players: np.ndarray, values: np.ndarray, bootstrap: int = 1000, seed: int = 1,
                     chunk: int = 50) -> Dict[str, Any]:
    """Mean of ``values`` per decision with a 95% interval from resampling whole players."""
    uniq, inv = np.unique(players, return_inverse=True)
    s = np.bincount(inv, weights=values)
    c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, bootstrap, chunk):
        idx = rng.integers(0, len(uniq), size=(min(chunk, bootstrap - start), len(uniq)))
        draws.append(s[idx].sum(1) / c[idx].sum(1))
    d = np.sort(np.concatenate(draws))
    lo, hi = float(d[int(0.025 * bootstrap)]), float(d[int(0.975 * bootstrap) - 1])
    return {"mean": round(float(s.sum() / c.sum()), 5), "ci95": [round(lo, 5), round(hi, 5)],
            "better": lo > 0, "worse": hi < 0, "n": int(c.sum()), "players": int(len(uniq))}


# ----------------------------------------------------------------- evaluation
def _ladder(ll: Dict[str, np.ndarray], n_before: np.ndarray) -> List[Dict[str, Any]]:
    labels = np.array([bin_label(int(n)) for n in n_before])
    out = []
    for lo, _hi in BINS:
        m = labels == bin_label(lo)
        if m.any():
            out.append({"bin": bin_label(lo), "n": int(m.sum()), "log_loss": {a: round(float(v[m].mean()), 5) for a, v in ll.items()}})
    return out


def _means(ll: Dict[str, np.ndarray], m: np.ndarray) -> Dict[str, Any]:
    return {"n": int(m.sum()), "log_loss": {a: round(float(v[m].mean()), 5) for a, v in ll.items()} if m.any() else {}}


def evaluate(t: Dict[str, np.ndarray], meta: Dict[str, Any], *, bootstrap: int = 1000, seed: int = 1, gbm_iter: int = 300):
    """Forecast ladder on the main cohort's frozen test half, frozen and online protocols."""
    t0 = time.perf_counter()
    main = t["in_band"] & t["legal3"]
    train_any = np.flatnonzero(t["split"] == 0)
    train_main = np.flatnonzero(main & (t["split"] == 0))
    test_main = np.flatnonzero(main & (t["split"] == 1))
    test_main = test_main[np.argsort(t["seq"][test_main], kind="stable")]
    q, counts = population_q(t, train_main)
    selection = choose_prior_strength(t, train_main)
    s = float(selection["chosen"])
    X, names = situation_matrix(t, len(meta["venues"]))
    clf = fit_situation(X, t["response"], train_any, seed=seed, max_iter=gbm_iter)
    y, ctx = t["response"][test_main], t["context"][test_main]
    players = t["player"][test_main]
    size = size_class_codes(t["table_size"][test_main])
    pooled = counts.sum(0)
    pooled = (pooled + 1) / (pooled.sum() + 3)
    base = {"pooled": np.tile(pooled, (len(test_main), 1)), "population": q[ctx],
            "situation": situation_probs(clf, X[test_main], t["raise_legal"][test_main])}
    protocols: Dict[str, Any] = {}
    forecasts: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]] = {}
    for proto, online in (("frozen", False), ("online", True)):
        order, p_pers, n_before = personal_forecasts(t, train_main, test_main, q, s, online)
        assert np.array_equal(order, test_main)
        probs = dict(base, personal=p_pers, situation_personal=combine(base["situation"], p_pers, base["population"]))
        forecasts[proto] = (probs, n_before)
        ll = {a: log_loss(probs[a], y) for a in ARMS}
        comps: Dict[str, Any] = {}
        for a, b in (("situation", "population"), ("personal", "population"), ("situation_personal", "situation"),
                     ("situation_personal", "population"), ("pooled", "population")):
            d = ll[b] - ll[a]
            comps[f"{a}_vs_{b}"] = dict(player_bootstrap(players, d, bootstrap, seed), by_table_size={
                SIZE_CLASSES[k]: player_bootstrap(players[size == k], d[size == k], bootstrap, seed) for k in range(3) if (size == k).any()})
        protocols[proto] = {"online": online, "overall": {a: round(float(v.mean()), 5) for a, v in ll.items()},
                            "ladder": _ladder(ll, n_before), "lift_log_loss": comps,
                            "known_players": _means(ll, n_before > 0), "new_players": _means(ll, n_before == 0),
                            "by_table_size": {SIZE_CLASSES[k]: _means(ll, size == k) for k in range(3) if (size == k).any()}}
    result = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "cohort": {"definition": "heads-up river after a check, bet 40-60% of the pot, all three answers legal, any table size",
                   "train": int(len(train_main)), "test": int(len(test_main)), "test_players": int(len(np.unique(players))),
                   "test_by_table_size": {SIZE_CLASSES[k]: int((size == k).sum()) for k in range(3)}},
        "prior_strength_selection": selection, "prior_strength": s,
        "population": {CONTEXTS[c]: {RESPONSES[a]: round(float(q[c, a]), 4) for a in range(3)} for c in range(3)},
        "situation_model": {"kind": "HistGradientBoostingClassifier (CPU)", "features": names, "iterations": int(clf.n_iter_),
                            "train_rows_any_bet_size": int(len(train_any))},
        "protocols": protocols, "seconds": round(time.perf_counter() - t0, 1),
        "label": ("log loss per decision on the frozen test half (lower is better); lifts are the reduction in log loss, "
                  "95% intervals from resampling players"),
    }
    return result, forecasts, (test_main, players, size, X, names)


def bluff_test(t: Dict[str, np.ndarray], test_main: np.ndarray, forecasts: Dict[str, Any], players: np.ndarray,
               size: np.ndarray, bootstrap: int = 1000, seed: int = 1) -> Dict[str, Any]:
    """A pure bluffer (never wins a showdown) bets when the forecast fold rate beats the break-even r / (1 + r).

    The player's recorded answer scores it: a fold wins the 100-chip pot, a call or raise loses the bet.
    The defender only saw public information, so a bluffer in that spot would have got the same answer."""
    fold = t["response"][test_main] == 0
    r = np.clip(t["ratio"][test_main].astype(float), 0.01, 20.0)
    lose = -100.0 * r
    out: Dict[str, Any] = {}
    for proto, (probs, n_before) in forecasts.items():
        pol = {"never": np.zeros(len(fold)), "always": np.where(fold, 100.0, lose), "perfect": np.where(fold, 100.0, 0.0)}
        bets: Dict[str, np.ndarray] = {}
        for a in ("population", "situation", "personal", "situation_personal"):
            bets[a] = probs[a][:, 0] > r / (1 + r)
            pol[a] = np.where(bets[a], np.where(fold, 100.0, lose), 0.0)
        res: Dict[str, Any] = {
            "chips_per_decision": {k: round(float(v.mean()), 3) for k, v in pol.items()},
            "bluff_rate": {k: round(float(b.mean()), 4) for k, b in bets.items()},
            "fold_rate_when_bluffing": {k: round(float(fold[b].mean()), 4) if b.any() else None for k, b in bets.items()},
            "vs_population": {}, "by_history": [], "by_table_size": {}}
        for a in ("situation", "personal", "situation_personal", "always"):
            res["vs_population"][a] = dict(player_bootstrap(players, pol[a] - pol["population"], bootstrap, seed),
                                           decisions_changed=round(float((bets.get(a, np.ones(len(fold), bool)) != bets["population"]).mean()), 4))
        res["personal_on_top_of_situation"] = player_bootstrap(players, pol["situation_personal"] - pol["situation"], bootstrap, seed)
        labels = np.array([bin_label(int(n)) for n in n_before])
        for lo, _hi in BINS:
            m = labels == bin_label(lo)
            if m.any():
                res["by_history"].append({"bin": bin_label(lo), "n": int(m.sum()), **{k: round(float(v[m].mean()), 3) for k, v in pol.items()}})
        for k in range(3):
            m = size == k
            if m.any():
                res["by_table_size"][SIZE_CLASSES[k]] = {"n": int(m.sum()), **{p: round(float(v[m].mean()), 3) for p, v in pol.items()}}
        out[proto] = res
    return {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "decisions": int(len(fold)),
            "players": int(len(np.unique(players))), "protocols": out, "units": "chips per decision, pot = 100 chips",
            "rule": "bet when the forecast fold rate is above r / (1 + r), where r is the bet as a fraction of the pot",
            "label": "real held-out decisions; each bluff is scored by the player's recorded answer"}


# ----------------------------------------------------------------- exports
def anon_id(venue: str, handle: str) -> str:
    return "P-" + hashlib.sha1(f"{venue}|{handle}".encode()).hexdigest()[:6]


def export_real_players(t: Dict[str, np.ndarray], meta: Dict[str, Any], path: str = REAL_PLAYERS, min_decisions: int = 40,
                        cap: int = 400) -> Dict[str, Any]:
    """Players with at least ``min_decisions`` main-cohort decisions, their answers in time order (anonymous IDs)."""
    rows = np.flatnonzero(t["in_band"] & t["legal3"])
    rows = rows[np.argsort(t["seq"][rows], kind="stable")]
    by_player: Dict[int, List[int]] = collections.defaultdict(list)
    for r, p in zip(rows.tolist(), t["player"][rows].tolist()):
        by_player[p].append(r)
    chosen = sorted(((p, rs) for p, rs in by_player.items() if len(rs) >= min_decisions), key=lambda x: (-len(x[1]), x[0]))
    out = []
    for p, rs in chosen[:cap]:
        venue, handle = meta["players"][p]
        ctx, resp = t["context"][rs], t["response"][rs]
        sizes = size_class_codes(t["table_size"][rs])
        out.append({"id": anon_id(venue, handle), "venue": venue, "decisions": len(rs),
                    "test_decisions": int((t["split"][rs] == 1).sum()), "fold_rate": round(float((resp == 0).mean()), 3),
                    "counts": {CONTEXTS[c]: {RESPONSES[a]: int(((ctx == c) & (resp == a)).sum()) for a in range(3)} for c in range(3)},
                    "table_mix": {SIZE_CLASSES[k]: int((sizes == k).sum()) for k in range(3)},
                    "sequence": "".join(CTX_LETTERS[c] + RESP_LETTERS[a] for c, a in zip(ctx.tolist(), resp.tolist())),
                    "first": iso(t["ts"][rs[0]]), "last": iso(t["ts"][rs[-1]])})
    doc = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "min_decisions": min_decisions,
           "eligible_players": len(chosen), "players": out,
           "sequence_code": "two letters per decision in time order: context c/b/d (checked through, bettor bet the turn, "
                            "defender bet the turn), then answer f/c/r",
           "label": "main cohort (40-60% bets, all three answers legal), all table sizes; IDs are hashes of the dataset's "
                    "already anonymised handles"}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return {"eligible_players": len(chosen), "exported": len(out)}


def history_features(t: Dict[str, np.ndarray]) -> np.ndarray:
    """Per decision: the player's earlier answers at any bet size (fold, call, raise), overall then in this context."""
    order = np.argsort(t["seq"], kind="stable")
    out = np.zeros((len(order), 6), np.int16)
    counts: Dict[int, np.ndarray] = {}
    for i, p, c, a in zip(order.tolist(), t["player"][order].tolist(), t["context"][order].tolist(), t["response"][order].tolist()):
        m = counts.get(p)
        if m is None:
            m = counts[p] = np.zeros((3, 3), np.int32)
        out[i, :3] = np.minimum(m.sum(0), 32767)
        out[i, 3:] = np.minimum(m[c], 32767)
        m[c, a] += 1
    return out


def export_neural(t: Dict[str, np.ndarray], X: np.ndarray, names: Sequence[str], test_main: np.ndarray,
                  forecasts: Dict[str, Any], path: str = NEURAL_DATASET) -> Dict[str, Any]:
    """Everything the GPU forecaster needs, with the CPU forecasts for the same test rows as baselines."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    seen = np.unique(t["player"][t["split"] == 0])
    pos = np.clip(np.searchsorted(seen, t["player"]), 0, max(len(seen) - 1, 0))
    emb = np.where(seen[pos] == t["player"], pos + 1, 0).astype(np.int32) if len(seen) else np.zeros(len(X), np.int32)
    pf, po = forecasts["frozen"][0], forecasts["online"][0]
    np.savez_compressed(
        path, X=X.astype(np.float16), feature_names=np.array(list(names)), y=t["response"].astype(np.int8),
        raise_legal=t["raise_legal"].astype(np.int8), split=t["split"].astype(np.int8),
        main=(t["in_band"] & t["legal3"]).astype(np.int8), seq=t["seq"].astype(np.int32), player_index=emb,
        n_players=np.int32(len(seen) + 1), player_raw=t["player"].astype(np.int32), history=history_features(t),
        table_class=size_class_codes(t["table_size"]), test_rows=test_main.astype(np.int32),
        base_population=pf["population"].astype(np.float32), base_situation=pf["situation"].astype(np.float32),
        base_personal_frozen=pf["personal"].astype(np.float32), base_personal_online=po["personal"].astype(np.float32),
        base_combined_frozen=pf["situation_personal"].astype(np.float32),
        base_combined_online=po["situation_personal"].astype(np.float32))
    return {"path": path, "rows": int(len(X)), "features": len(names), "mb": round(os.path.getsize(path) / 1e6, 1)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pokersim mirror-wide", description="Phase 6: forecasts and bluff test on every table size")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--gbm-iter", type=int, default=300)
    ap.add_argument("--min-decisions", type=int, default=40, help="threshold for real_players.json")
    ap.add_argument("--no-neural", action="store_true", help="skip the GPU dataset export")
    ap.add_argument("--table", default=WIDE_TABLE)
    ap.add_argument("--meta", default=WIDE_META)
    ap.add_argument("--out-dir", default=SMALL_OUT)
    args = ap.parse_args(argv)
    t, meta = load_table(args.table, args.meta)
    result, forecasts, (test_main, players, size, X, names) = evaluate(t, meta, bootstrap=args.bootstrap, gbm_iter=args.gbm_iter)
    bluff = bluff_test(t, test_main, forecasts, players, size, bootstrap=args.bootstrap)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, doc in (("eval_wide.json", result), ("bluff_test.json", bluff)):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
    rp = export_real_players(t, meta, os.path.join(args.out_dir, "real_players.json"), args.min_decisions)
    summary = {"cohort": result["cohort"], "prior_strength": result["prior_strength"],
               "frozen_overall": result["protocols"]["frozen"]["overall"],
               "lifts_frozen": {k: (v["mean"], v["ci95"]) for k, v in result["protocols"]["frozen"]["lift_log_loss"].items()},
               "bluff_online": bluff["protocols"]["online"]["chips_per_decision"], "real_players": rp}
    if not args.no_neural:
        summary["neural_dataset"] = export_neural(t, X, names, test_main, forecasts,
                                                  os.path.join(args.out_dir, "neural", "neural_dataset.npz"))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
