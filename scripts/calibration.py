"""Calibration of the Table Lab's two learned models on their held-out test rows -> data/demo/calibration.json.

    nice -n 10 .venv/bin/python scripts/calibration.py [--threads 4]

Answers "when the model says 70%, does it happen 70% of the time?" for exactly what the page shows:

* **Hand model** (``pokersim.hand_range``), test examples of ``data/cache/hand_range`` (Pluribus hands with hand
  number % 5 == 0: every hole card known, no showdown selection; and the online showdown hands of shards 34-36):
  coverage of the smallest set of classes reaching 50 / 80 / 90 % of the predicted mass, top-10 hit rate against the
  card-counting baseline of ``scripts/train_hand_range.py``, and reliability of every (example, class) probability.
* **Next-move model** (``pokersim.next_move``), public view as the page computes it for an opponent
  (``next_move.predict(..., with_cards=False)``: no cards, from_logs flag 0), test rows of ``data/cache/next_move``:
  reliability and ECE of the top action and its % as ``lab_api._pred`` lists it, and of the fold / call / raise sum.

Where something is clearly miscalibrated a single temperature is fitted on the **validation** rows (never on test) and
the test numbers are given before and after.  No model file is changed.  Inference is the repo's NumPy code
(``RangeModel.logits``, ``NextMoveModel.logits``); the numbers of the existing test reports are reproduced first.
"""
from __future__ import annotations

import argparse
import os
import sys

if __name__ == "__main__":  # BLAS threads before NumPy loads: this machine also serves the demo
    _t = "4"
    for _i, _a in enumerate(sys.argv):
        if _a == "--threads" and _i + 1 < len(sys.argv):
            _t = sys.argv[_i + 1]
        elif _a.startswith("--threads="):
            _t = _a.split("=", 1)[1]
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, _t)

import datetime as _dt  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pokersim import hand_range as H  # noqa: E402
from pokersim import next_move as NM  # noqa: E402
from pokersim.rl_features import ACTIONS  # noqa: E402

OUT = os.path.join(ROOT, "data", "demo", "calibration.json")
LEVELS = (0.5, 0.8, 0.9)
STATIC_COMBOS = np.array([6 if r == c else 4 if r < c else 12 for r in range(13) for c in range(13)], np.float64)
MISCAL_ECE, MISCAL_COVERAGE = 0.03, 0.05


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


THR = _load_script("train_hand_range")
TNM = _load_script("train_next_move")


def r4(x) -> float:
    return round(float(x), 4)


def pct(x, d=0) -> str:
    return f"{100 * float(x):.{d}f}%"


def golden(f, lo: float, hi: float, it: int = 40) -> float:
    """Minimise a 1-D unimodal f on [lo, hi] (in log space)."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(np.exp(c)), f(np.exp(d))
    for _ in range(it):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(np.exp(d))
    return float(np.exp((a + b) / 2))


# ================================================================= hand model
def hand_probs(z: np.ndarray, loglc: np.ndarray, live: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """RangeModel.predict from stored network logits: p ~ live combos * exp(logit / T); T = 1 is the shipped model."""
    lg = z / temp + loglc
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg) * live
    return p / p.sum(1, keepdims=True)


def hand_inputs(m: H.RangeModel, D, rows: np.ndarray):
    """Network logits (no combo term) and live combos for the given rows, one RangeModel.logits call per board."""
    ctx = D["ctx"][rows].astype(np.float32)
    keys = [THR.board_key(b) for b in D["board"][rows]]
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    z = np.zeros((len(rows), H.N_CLASSES), np.float32)
    for k, idx in groups.items():
        idx = np.asarray(idx)
        g = H.class_board_features(list(k))
        for s in range(0, len(idx), 32768):
            b = idx[s:s + 32768]
            z[b] = m.logits(ctx[b], g)
    lc = np.zeros((len(rows), H.N_CLASSES), np.float32)
    cache = {}
    for i, r in enumerate(rows):
        dead = tuple(sorted([int(x) for x in D["board"][r] if x >= 0] + [int(x) for x in D["hole"][r] if x >= 0]))
        v = cache.get(dead)
        if v is None:
            v = cache[dead] = H.live_combos(list(dead))
        lc[i] = v
    return z, lc, len(groups)


def hand_metrics(p: np.ndarray, y: np.ndarray, lc: np.ndarray) -> dict:
    n = len(y)
    ar = np.arange(n)
    pt = p[ar, y]
    out = {"n": int(n), "nll": r4(-np.log(np.maximum(pt, 1e-12)).mean()), "live_combos": round(float(lc.sum(1).mean()), 1)}
    rank = (p > pt[:, None]).sum(1)
    out["top1"] = r4((rank < 1).mean())
    out["top10"] = r4((rank < 10).mean())
    # card counting (scripts/train_hand_range.py baseline_nll): p = live combos / total; its many ties broken at random
    lt = lc[ar, y]
    gt, eq = (lc > lt[:, None]).sum(1), (lc == lt[:, None]).sum(1)
    out["baseline_nll"] = r4(-np.log(np.maximum(lt / lc.sum(1), 1e-12)).mean())
    out["baseline_top10"] = r4(np.clip((10 - gt) / eq, 0, 1).mean())
    # listed on screen: the top 2 classes and their %
    o2 = np.argsort(-p, 1)[:, :2]
    for j, name in enumerate(("top1_listed", "top2_listed")):
        pj = p[ar, o2[:, j]]
        out[name] = {"claimed": r4(pj.mean()), "observed": r4((o2[:, j] == y).mean())}
    # predicted mass on each kind of hand against how often the real hand is of that kind
    out["hand_type"] = {t: {"pred": r4(p[:, sel].sum(1).mean()), "obs": r4(sel[y].mean())}
                        for t, sel in (("pairs", H._CLS_PAIR > 0), ("suited", H._CLS_SUITED > 0),
                                       ("offsuit", (H._CLS_PAIR == 0) & (H._CLS_SUITED == 0)))}
    # coverage sets: by class mass (the spec: smallest set of classes) and by grid shade (darkest cells first)
    for tag, key in (("coverage", p), ("coverage_by_shade", p / STATIC_COMBOS[None, :])):
        order = np.argsort(-key, 1, kind="stable")
        cum = np.cumsum(np.take_along_axis(p, order, 1), 1)
        ccum = np.cumsum(np.take_along_axis(lc, order, 1), 1)
        pos = np.argmax(order == y[:, None], 1)
        cov = {}
        for lv in LEVELS:
            k = np.minimum((cum < lv - 1e-9).sum(1), H.N_CLASSES - 1)  # index of the class that reaches the level
            cov[f"{int(lv * 100)}"] = {"hit": r4((pos <= k).mean()), "claimed": r4(cum[ar, k].mean()),
                                       "classes": round(float((k + 1).mean()), 1), "combos": round(float(ccum[ar, k].mean()), 1)}
        out[tag] = cov
    return out


def hand_reliability(p: np.ndarray, y: np.ndarray) -> list:
    edges = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 11)])  # 1e-4, 2.5e-4, 6.3e-4, ... 0.4, 1
    edges[-1] = 1.0 + 1e-9
    pv = p.ravel()
    hit = np.zeros_like(p, dtype=bool)
    hit[np.arange(len(y)), y] = True
    hv = hit.ravel()
    keep = pv > 0  # classes with no live combination are impossible, not predictions
    pv, hv = pv[keep], hv[keep]
    b = np.clip(np.searchsorted(edges, pv, side="right") - 1, 0, len(edges) - 2)
    cnt = np.bincount(b, minlength=len(edges) - 1)
    sp = np.bincount(b, weights=pv, minlength=len(edges) - 1)
    sh = np.bincount(b, weights=hv, minlength=len(edges) - 1)
    rows = []
    for i in range(len(edges) - 1):
        if cnt[i] == 0:
            continue
        rows.append({"lo": float(f"{edges[i]:.3g}"), "hi": float(f"{min(edges[i + 1], 1.0):.3g}"), "n": int(cnt[i]),
                     "hits": int(sh[i]), "pred": float(f"{sp[i] / cnt[i]:.4g}"), "obs": float(f"{sh[i] / cnt[i]:.4g}")})
    return rows


def run_hand(log) -> dict:
    t0 = time.time()
    m = H.RangeModel(H.MODEL_PATH)
    with open(os.path.join(ROOT, "data", "rl", "hand_range.json")) as f:
        rep = json.load(f)
    D = THR.load_all()
    n = len(D["label"])
    test = D["test"].astype(bool)
    # validation exactly as in scripts/train_hand_range.py cmd_train (5% of the non-test rows, RandomState(0))
    val = (~test) & (np.random.RandomState(0).rand(n) < 0.05)
    assert n == rep["examples"] and int(val.sum()) == rep["validation"], (n, int(val.sum()))
    rows = np.nonzero(test | val)[0]
    z, lc, nb = hand_inputs(m, D, rows)
    log(f"hand: {n:,} examples, {len(rows):,} test+val rows, {nb:,} boards ({time.time() - t0:.0f} s)")
    is_test, is_val = test[rows], val[rows]
    y = D["label"][rows].astype(np.int64)
    src, street = D["source"][rows].astype(bool), D["street"][rows].astype(np.int64)
    loglc = np.log(np.maximum(lc, 1e-9)).astype(np.float32)
    live = (lc > 0).astype(np.float32)
    p1 = hand_probs(z, loglc, live)

    # the shipped NumPy path (RangeModel.predict) on random test rows must give the same probabilities
    rs = np.random.RandomState(1)
    chk = rs.choice(np.nonzero(is_test)[0], 300, replace=False)
    diff = 0.0
    for i in chk:
        r = rows[i]
        bd = [int(x) for x in D["board"][r] if x >= 0]
        dead = bd + [int(x) for x in D["hole"][r] if x >= 0]
        diff = max(diff, float(np.abs(m.predict(D["ctx"][r].astype(np.float32), bd, dead)[0] - p1[i]).max()))
    out = {"model": "data/rl/hand_range.npz", "check": {"predict_max_abs_diff": float(f"{diff:.2e}")}}

    sets = {"pluribus": is_test & src, "online_showdown": is_test & ~src}
    res = {}
    for name, mk in sets.items():
        sec = {"all_streets": hand_metrics(p1[mk], y[mk], lc[mk])}
        for si, s in enumerate(H.STREETS):
            mm = mk & (street == si)
            if mm.any():
                sec[s] = hand_metrics(p1[mm], y[mm], lc[mm])
        sec["reliability"] = hand_reliability(p1[mk], y[mk])
        res[name] = sec
        a = sec["all_streets"]
        rr = rep["test"][f"{name}/all_streets"]
        out["check"][name] = {"nll": a["nll"], "report_nll": rr["model_nll"], "baseline_nll": a["baseline_nll"],
                              "report_baseline_nll": rr["baseline_nll"], "top10": a["top10"], "report_top10": rr["top10"]}
        assert abs(a["nll"] - rr["model_nll"]) < 2e-3 and abs(a["baseline_nll"] - rr["baseline_nll"]) < 2e-3, out["check"]
        log(f"hand {name}: n {a['n']:,} nll {a['nll']:.4f} (report {rr['model_nll']:.4f}) cov50/80/90 "
            + " / ".join(f"{a['coverage'][k]['hit']:.3f}<-{a['coverage'][k]['claimed']:.3f}" for k in ("50", "80", "90")))

    # miscalibrated? coverage more than 5 points from the mass it claims, on either test population
    worst = max(abs(res[nm]["all_streets"][t][k]["hit"] - res[nm]["all_streets"][t][k]["claimed"])
                for nm in res for t in ("coverage", "coverage_by_shade") for k in ("50", "80", "90"))
    temp = None
    if worst > MISCAL_COVERAGE:
        def val_nll(mask):
            def f(T):
                p = hand_probs(z[mask], loglc[mask], live[mask], T)
                return float(-np.log(np.maximum(p[np.arange(mask.sum()), y[mask]], 1e-12)).mean())
            return f
        T_all = golden(val_nll(is_val), 0.3, 4.0)
        T_plu = golden(val_nll(is_val & src), 0.3, 4.0)
        temp = {"fitted_on": "validation split of scripts/train_hand_range.py (5% of non-test rows, both sources)",
                "applies_to": "network logits only: p ~ live combos * exp(logit / T)",
                "T": round(T_all, 3), "T_if_fitted_on_pluribus_validation_only": round(T_plu, 3),
                "n_validation": int(is_val.sum()), "n_validation_pluribus": int((is_val & src).sum()), "test_after": {}}
        for T, tag in ((T_all, "T"), (T_plu, "T_pluribus")):
            pT = hand_probs(z, loglc, live, T)
            for name, mk in sets.items():
                a = hand_metrics(pT[mk], y[mk], lc[mk])
                temp["test_after"].setdefault(tag, {})[name] = {"nll": a["nll"], "top10": a["top10"],
                                                                "coverage": a["coverage"], "coverage_by_shade": a["coverage_by_shade"]}
        log(f"hand temperature: T {T_all:.3f} (pluribus-only val {T_plu:.3f})")
    out["max_coverage_gap"] = r4(worst)
    out["temperature"] = temp

    def line(name, label):
        a = res[name]["all_streets"]
        c, s = a["coverage"], a["coverage_by_shade"]
        nn = f"n={a['n']:,} {label}"
        return [
            f"real hand inside the model's 50% region {pct(c['50']['hit'])} of the time, 80% region {pct(c['80']['hit'])}, "
            f"90% region {pct(c['90']['hit'])} ({nn})",
            f"darkest grid cells holding 50% of the mass contain the real hand {pct(s['50']['hit'])} of the time "
            f"(avg {s['50']['classes']:.0f} cells, {s['50']['combos']:.0f} of {a['live_combos']:,.0f} live combos)",
            f"real hand among the model's top 10 classes {pct(a['top10'], 1)} of the time vs {pct(a['baseline_top10'], 1)} "
            f"for card counting alone",
            f"top listed hand claims {pct(a['top1_listed']['claimed'], 1)} on average and is right "
            f"{pct(a['top1_listed']['observed'], 1)} of the time",
            "predicted mass on pairs / suited / offsuit " + " / ".join(pct(a["hand_type"][t]["pred"]) for t in ("pairs", "suited", "offsuit"))
            + "; real hands " + " / ".join(pct(a["hand_type"][t]["obs"]) for t in ("pairs", "suited", "offsuit")),
        ]
    for name, label, split in (("pluribus", "Pluribus test examples", "test: Pluribus hands with hand number % 5 == 0 "
                                "(all hole cards known, no showdown selection)"),
                               ("online_showdown", "online showdown test examples",
                                "test: online hands of shards 34-36 that reached showdown (biased towards strong hands)")):
        out[name] = {"split": split, "n": res[name]["all_streets"]["n"], "summary": line(name, label),
                     "units": "coverage: hit = how often the real class is inside the set, claimed = mean predicted mass of "
                              "the set; classes / combos = mean set size (combos = live combinations)",
                     **res[name]}
    log(f"hand done ({time.time() - t0:.0f} s)")
    return out


# ================================================================= next-move model
BINS = np.linspace(0, 1, 11)


def reliability(conf: np.ndarray, hit: np.ndarray) -> dict:
    b = np.clip(np.searchsorted(BINS, conf, side="right") - 1, 0, 9)
    cnt = np.bincount(b, minlength=10)
    sc = np.bincount(b, weights=conf, minlength=10)
    sh = np.bincount(b, weights=hit.astype(np.float64), minlength=10)
    nz = cnt > 0
    ece = float((np.abs(sc[nz] - sh[nz])).sum() / len(conf))
    rows = [{"lo": r4(BINS[i]), "hi": r4(BINS[i + 1]), "n": int(cnt[i]), "pred": r4(sc[i] / cnt[i]), "obs": r4(sh[i] / cnt[i])}
            for i in range(10) if cnt[i] > 0]
    return {"ece": r4(ece), "mean_shown": r4(conf.mean()), "observed": r4(hit.mean()), "bins": rows}


def nm_probs(lg: np.ndarray, mask: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """NextMoveModel.probs for a batch."""
    lg = np.where(mask, lg / max(temp, 1e-3), -1e9)
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def nm_metrics(p: np.ndarray, y: np.ndarray, street: np.ndarray) -> dict:
    n = len(y)
    ar = np.arange(n)
    # lab_api._pred lists the legal classes by probability; the first one is the label and % on screen
    top = np.argmax(p, 1)
    conf = p[ar, top]
    out = {"n": int(n), "nll": r4(-np.log(np.maximum(p[ar, y], 1e-12)).mean()), "top_action": reliability(conf, top == y)}
    p3 = np.stack([p[:, 0], p[:, 1], p[:, 2:].sum(1)], 1)
    y3 = np.minimum(y, 2)
    t3 = np.argmax(p3, 1)
    out["three_way"] = reliability(p3[ar, t3], t3 == y3)
    # per group: P(fold), P(check/call), P(any raise) against how often it happens
    out["three_way"]["per_group"] = {g: {"pred": r4(p3[:, i].mean()), "obs": r4((y3 == i).mean()),
                                         "ece": reliability(p3[:, i], y3 == i)["ece"]}
                                     for i, g in enumerate(("fold", "check_call", "raise"))}
    out["top_action"]["by_street"] = {s: {"n": int((street == si).sum()),
                                          "ece": reliability(conf[street == si], (top == y)[street == si])["ece"]}
                                      for si, s in enumerate(H.STREETS) if (street == si).any()}
    out["top_action"]["shown_class_share"] = {ACTIONS[i]: r4((top == i).mean()) for i in range(7) if (top == i).any()}
    # mean predicted probability of each of the 7 classes against how often it was played
    out["class_share"] = {ACTIONS[i]: {"pred": r4(p[:, i].mean()), "obs": r4((y == i).mean())} for i in range(7)}
    return out


def run_next_move(log) -> dict:
    t0 = time.time()
    m = NM.NextMoveModel(NM.MODEL_PATH)
    with open(os.path.join(ROOT, "data", "rl", "next_move.json")) as f:
        rep = json.load(f)
    D = TNM.load_all()
    assert len(D["label"]) == rep["decisions"], len(D["label"])
    keep = np.nonzero(D["split"] >= 1)[0]  # validation and test rows only (the training rows are not needed)
    D = {k: v[keep] for k, v in D.items()}
    X, sel, kind = TNM.build_inputs(D)
    pub = kind == 0
    X, sel = X[pub], sel[pub]
    y, mask = D["label"][sel].astype(np.int64), D["mask"][sel]
    split, logs, street = D["split"][sel], D["logs"][sel].astype(bool), D["street"][sel].astype(np.int64)
    lg_train = np.concatenate([m.logits(X[s:s + 65536]) for s in range(0, len(X), 65536)])
    log(f"next move: {len(y):,} public val+test rows ({time.time() - t0:.0f} s)")

    # check: NextMoveModel.probs row by row, and the test report (which used the training flags)
    p_tr = nm_probs(lg_train, mask)
    chk = np.random.RandomState(2).choice(np.nonzero(split == 2)[0], 300, replace=False)
    diff = max(float(np.abs(m.probs(X[i], mask[i]) - p_tr[i]).max()) for i in chk)
    # on screen (next_move.predict, with_cards=False) the from_logs flag is always 0: the online population view
    X[:, NM.IN_DIM - 1] = 0.0
    lg_screen = np.concatenate([m.logits(X[s:s + 65536]) for s in range(0, len(X), 65536)])
    del X
    out = {"model": "data/rl/next_move.npz", "check": {"probs_max_abs_diff": float(f"{diff:.2e}")}}
    sets = {"online": (split == 2) & ~logs, "pluribus_logs": (split == 2) & logs}
    for name, rk in (("online", "online_public"), ("pluribus_logs", "pluribus_logs_public")):
        mk = sets[name]
        nll = float(-np.log(p_tr[mk][np.arange(mk.sum()), y[mk]]).mean())
        top1 = float((p_tr[mk].argmax(1) == y[mk]).mean())
        rr = rep["test"][rk]
        out["check"][name] = {"n": int(mk.sum()), "report_n": rr["n"], "nll": r4(nll), "report_nll": rr["model_nll"],
                              "top1": r4(top1), "report_top1": rr["top1"]}
        assert int(mk.sum()) == rr["n"] and abs(nll - rr["model_nll"]) < 2e-3, out["check"]

    p = nm_probs(lg_screen, mask)
    res = {nm_: nm_metrics(p[mk], y[mk], street[mk]) for nm_, mk in sets.items()}
    # the Pluribus-log rows with the flag the model was trained with for them (from_logs = 1), for reference
    mk = sets["pluribus_logs"]
    alt = nm_metrics(p_tr[mk], y[mk], street[mk])
    res["pluribus_logs"]["with_logs_flag"] = {"nll": alt["nll"], "top_action_ece": alt["top_action"]["ece"],
                                              "three_way_ece": alt["three_way"]["ece"]}
    for nm_ in res:
        a = res[nm_]
        log(f"next move {nm_}: n {a['n']:,} top ECE {a['top_action']['ece']:.4f} shown {a['top_action']['mean_shown']:.3f} "
            f"obs {a['top_action']['observed']:.3f} · 3-way ECE {a['three_way']['ece']:.4f}")

    worst = max(max(res[k]["top_action"]["ece"], res[k]["three_way"]["ece"]) for k in res)
    temp = None
    if worst > MISCAL_ECE:
        def val_nll(mk):
            return lambda T: float(-np.log(nm_probs(lg_screen[mk], mask[mk], T)[np.arange(mk.sum()), y[mk]]).mean())
        vmask = split == 1
        T_all = golden(val_nll(vmask), 0.3, 4.0)
        T_on = golden(val_nll(vmask & ~logs), 0.3, 4.0)
        T_pl = golden(val_nll(vmask & logs), 0.3, 4.0)
        temp = {"fitted_on": "validation rows (split 1: online players hashed to validation + Pluribus hands % 10 == 2), "
                             "public view as on screen", "applies_to": "NextMoveModel.probs(temp=T)",
                "T": round(T_all, 3), "T_online_validation_only": round(T_on, 3), "T_pluribus_validation_only": round(T_pl, 3),
                "n_validation": int(vmask.sum()), "n_validation_pluribus": int((vmask & logs).sum()), "test_after": {}}
        for T, tag in ((T_all, "T"), (T_pl, "T_pluribus")):
            pT = nm_probs(lg_screen, mask, T)
            for nm_, mk in sets.items():
                a = nm_metrics(pT[mk], y[mk], street[mk])
                temp["test_after"].setdefault(tag, {})[nm_] = {"nll": a["nll"], "top_action_ece": a["top_action"]["ece"],
                                                               "mean_shown": a["top_action"]["mean_shown"],
                                                               "observed": a["top_action"]["observed"],
                                                               "three_way_ece": a["three_way"]["ece"]}
        log(f"next move temperature: T {T_all:.3f} (online val {T_on:.3f}, pluribus val {T_pl:.3f})")
    out["max_ece"] = r4(worst)
    out["temperature"] = temp

    def line(name, label):
        a = res[name]
        ta, tw = a["top_action"], a["three_way"]

        def at(rel, lo):
            return next((b for b in rel["bins"] if abs(b["lo"] - lo) < 1e-6), None)
        s = [f"fold / call / raise (as the page shows it): the top choice's % averages {pct(tw['mean_shown'])} and it "
             f"happens {pct(tw['observed'])} of the time; ECE {tw['ece']:.3f} (n={a['n']:,} {label})"]
        for lo in (0.9, 0.7):
            b = at(tw, lo)
            if b:
                s.append(f"when it shows {int(lo * 100)}-{int(round(b['hi'] * 100))}% (avg {pct(b['pred'])}), that choice "
                         f"happens {pct(b['obs'])} of the time (n={b['n']:,})")
        b = at(ta, 0.7)
        s.append(f"single action with bet size (7 classes): top % averages {pct(ta['mean_shown'])}, happens "
                 f"{pct(ta['observed'])} of the time; ECE {ta['ece']:.3f}"
                 + (f"; shown 70-80% -> happens {pct(b['obs'])}" if b else ""))
        if "with_logs_flag" in a:
            w = a["with_logs_flag"]
            s.append(f"with the Pluribus-log flag the model was trained with for these rows (not what the page sends): "
                     f"top-move ECE {w['top_action_ece']:.3f}, fold / call / raise ECE {w['three_way_ece']:.3f}, "
                     f"log-loss {w['nll']:.3f} vs {a['nll']:.3f}")
        return s
    for name, label, split_txt in (
            ("online", "online test decisions", "test: online players hashed to test (10%) plus every player of shards 34-36"),
            ("pluribus_logs", "Pluribus-log test decisions", "test: Pluribus-log hands with hash % 10 in (0, 1), every seat")):
        out[name] = {"split": split_txt, "n": res[name]["n"], "summary": line(name, label),
                     "view": "public (no cards, from_logs flag 0), as next_move.predict(with_cards=False) for an opponent",
                     "on_screen": "three_way (web/lab/lab.js grouped(): raise sizes and all-in summed); top_action = the "
                                  "7-class first entry of lab_api._pred",
                     "units": "bins: pred = mean shown probability, obs = how often the shown choice happened; "
                              "ECE over 10 equal-width bins",
                     **res[name]}
    log(f"next move done ({time.time() - t0:.0f} s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="4", help="BLAS threads (the demo runs on this machine)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f} s] {s}", flush=True)  # noqa: E731
    hand = run_hand(log)
    nm = run_next_move(log)
    out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/calibration.py",
           "about": "Calibration on held-out test rows only (never refitted on test). 'claimed' / 'pred' is what the "
                    "model says, 'hit' / 'obs' how often it happened.",
           "hand_range": hand, "next_move": nm, "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    log(f"wrote {a.out} ({os.path.getsize(a.out) / 1024:.0f} KB)")
    for sec in ("hand_range", "next_move"):
        for k, v in out[sec].items():
            if isinstance(v, dict) and "summary" in v:
                print(f"{sec}/{k}:")
                for s in v["summary"]:
                    print(f"  - {s}")


if __name__ == "__main__":
    main()
