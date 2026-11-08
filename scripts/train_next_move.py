"""Train the next-move model (pokersim.next_move) on the Takara corpus.

    python scripts/train_next_move.py extract [--workers 14] [--online-hands 15000]   # -> data/cache/next_move/*.npz
    python scripts/train_next_move.py train [--seeds 1,2,3]                           # -> data/rl/next_move.npz + .json

Rows: every decision of no-limit hold'em hands with 5-6 players.
* Online hands (HandHQ): a random sample per shard, with **no** showdown filter, so folds are represented.  Cards are
  never used for these rows (online cards are only known at showdown, which would remove every fold).
* Pluribus logs: every decision of every seat, with the actor's cards (all known, no selection).

Splits: online rows by player (hash of venue + name: 80% train, 10% validation, 10% test) and shards 34-36 are test
only; Pluribus logs by hand number (% 10: 0-1 test, 2 validation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pokersim import cards as C  # noqa: E402
from pokersim import next_move as NM  # noqa: E402
from pokersim.mirror_data import N_SHARDS, iter_rows, row_to_record, shard_path  # noqa: E402
from pokersim.demo_player import row_has_reserved  # noqa: E402
from pokersim.replay import replay  # noqa: E402
from pokersim.rl_features import action_targets  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache", "next_move")
OUT_NPZ = os.path.join(ROOT, "data", "rl", "next_move.npz")
OUT_JSON = os.path.join(ROOT, "data", "rl", "next_move.json")
TEST_SHARDS = (34, 35, 36)
COLUMNS = ["variant", "actions", "players", "antes", "blinds_or_straddles", "starting_stacks", "finishing_stacks",
           "winnings", "min_bet", "seat_count", "venue", "source_file", "hand_number_in_file", "ante_trimming_status",
           "seats", "small_bet", "big_bet", "bring_in", "time", "day", "month", "year"]


def _h(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def _split_online(venue: str, name: str, shard: int) -> int:
    if shard in TEST_SHARDS:
        return 2
    u = _h(f"{venue}|{name}") % 10
    return 2 if u == 0 else 1 if u == 1 else 0


def _split_pluribus(src: str) -> int:
    u = _h(src) % 10
    return 2 if u <= 1 else 1 if u == 2 else 0


def extract_shard(args):
    i, online_cap, accept = args
    rng = random.Random(1000 + i)
    cols = {k: [] for k in ("pub", "cards", "board", "label", "mask", "pluribus", "logs", "split", "street")}
    online = 0
    for row in iter_rows(shard_path(i), columns=None):
        if row.get("variant") != "NT" or row_has_reserved(row):  # the reserved demo player stays unseen
            continue
        players = row.get("players") or []
        if len(players) not in (5, 6):
            continue
        src = row.get("source_file") or ""
        logs = src.startswith("pluribus/")
        if not logs:
            if online >= online_cap or rng.random() > accept:
                continue
        rec = row_to_record(row)
        if rec is None:
            continue
        try:
            res = replay(rec, snapshots=False, observations=True)
        except Exception:
            continue
        if not logs:
            online += 1
        venue = str(row.get("venue") or src.split("/")[1] if "/" in src else "")
        for d in res.decisions:
            obs = d.observation
            if not obs.get("legal_actions"):
                continue
            y = NM.label_of(obs, d.actual)
            if y is None:
                continue
            j = d.player
            mask, _ = action_targets(obs)
            if not mask[y]:
                continue
            hole = [c for c in (C.parse_card(x) for x in d.cards_at_time) if c is not None] if logs else []
            board = [c for c in (C.parse_card(x) for x in obs.get("board") or []) if c is not None]
            cols["pub"].append(NM.public_features(obs, j))
            cols["cards"].append((hole + [-1, -1])[:2] if len(hole) == 2 else [-1, -1])
            cols["board"].append((board + [-1] * 5)[:5])
            cols["label"].append(y)
            cols["mask"].append(mask)
            cols["pluribus"].append(1 if logs and players[j] == "Pluribus" else 0)
            cols["logs"].append(1 if logs else 0)
            cols["split"].append(_split_pluribus(src) if logs else _split_online(venue, players[j], i))
            cols["street"].append(["preflop", "flop", "turn", "river"].index(obs["street"]))
    os.makedirs(CACHE, exist_ok=True)
    if cols["pub"]:
        np.savez_compressed(os.path.join(CACHE, f"shard{i:02d}.npz"), pub=np.stack(cols["pub"]).astype(np.float16),
                            cards=np.array(cols["cards"], np.int8), board=np.array(cols["board"], np.int8),
                            label=np.array(cols["label"], np.int8), mask=np.array(cols["mask"], bool),
                            pluribus=np.array(cols["pluribus"], np.int8), logs=np.array(cols["logs"], np.int8),
                            split=np.array(cols["split"], np.int8), street=np.array(cols["street"], np.int8))
    return i, online, len(cols["pub"])


def cmd_extract(a):
    t = time.time()
    total = 0
    with Pool(a.workers) as pool:
        for i, nh, ne in pool.imap_unordered(extract_shard, [(i, a.online_hands, a.accept) for i in range(N_SHARDS)]):
            total += ne
            print(f"shard {i:2d}: {nh:6d} online hands, {ne:8d} rows (total {total:,}, {time.time() - t:.0f} s)", flush=True)


# ----------------------------------------------------------------- training
def load_all():
    parts = [np.load(os.path.join(CACHE, f)) for f in sorted(os.listdir(CACHE)) if f.endswith(".npz")]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def build_inputs(D):
    """Public rows for everything; card rows (has_cards=1) additionally for the Pluribus logs."""
    n = len(D["label"])
    pub = D["pub"].astype(np.float32)
    zeros_card = np.zeros((n, NM.CARD_DIM), np.float32)
    flags = np.stack([np.zeros(n), np.zeros(n), D["logs"].astype(np.float32)], 1).astype(np.float32)
    X_pub = np.concatenate([pub, zeros_card, flags], 1)
    idx = np.nonzero(D["logs"] == 1)[0]
    cf = np.zeros((len(idx), NM.CARD_DIM), np.float32)
    for k, i in enumerate(idx):
        hole = [int(c) for c in D["cards"][i] if c >= 0]
        board = [int(c) for c in D["board"][i] if c >= 0]
        cf[k] = NM.card_features(hole, board)
    fl = np.stack([np.ones(len(idx)), D["pluribus"][idx].astype(np.float32), np.ones(len(idx))], 1).astype(np.float32)
    X_card = np.concatenate([pub[idx], cf, fl], 1)
    X = np.concatenate([X_pub, X_card])
    sel = np.concatenate([np.arange(n), idx])
    kind = np.concatenate([np.zeros(n, np.int8), np.ones(len(idx), np.int8)])  # 0 public row, 1 card row
    return X, sel, kind


def cmd_train(a):
    import torch
    import torch.nn as nn

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load_all()
    X, sel, kind = build_inputs(D)
    y = D["label"][sel].astype(np.int64)
    mask = D["mask"][sel]
    split = D["split"][sel]
    logs = D["logs"][sel]
    plur = D["pluribus"][sel]
    street = D["street"][sel]
    print(f"{len(D['label']):,} decisions ({int(D['logs'].sum()):,} from the Pluribus logs) -> {len(y):,} rows; device {dev}",
          flush=True)
    mean = X[split == 0].mean(0)
    std = X[split == 0].std(0) + 1e-3
    mean[NM.PUBLIC_DIM:] = 0.0  # one-hot / flags stay as they are
    std[NM.PUBLIC_DIM:] = 1.0
    T = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=dev)  # noqa: E731
    Xt, yt, Mt = T((X - mean) / std), T(y, torch.long), T(mask, torch.bool)
    # card rows are few (Pluribus logs): weight them up so the with-cards head is learned
    w = np.where(kind == 1, a.card_weight, 1.0).astype(np.float32)
    Wt = T(w)

    def make(seed):
        torch.manual_seed(seed)
        layers, d = [], NM.IN_DIM
        for _ in range(a.layers):
            layers += [nn.Linear(d, a.hidden), nn.ReLU()]
            d = a.hidden
        layers.append(nn.Linear(d, 7))
        return nn.Sequential(*layers).to(dev)

    def logits(net, idx):
        return net(Xt[idx]).masked_fill(~Mt[idx], -1e9)

    def nll(net, rows):
        net.eval()
        out = []
        with torch.no_grad():
            r = T(rows, torch.long)
            for s in range(0, len(r), 262144):
                b = r[s:s + 262144]
                out.append(-torch.log_softmax(logits(net, b), 1).gather(1, yt[b, None]).squeeze(1))
        net.train()
        return torch.cat(out).cpu().numpy() if out else np.zeros(0)

    tr = np.nonzero(split == 0)[0]
    val_pub = np.nonzero((split == 1) & (kind == 0))[0]
    val_card = np.nonzero((split == 1) & (kind == 1))[0]
    results = []
    for seed in a.seeds:
        net = make(seed)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        best, bad, state = 1e9, 0, None
        for ep in range(a.epochs):
            t = time.time()
            perm = T(np.random.RandomState(seed * 100 + ep).permutation(tr), torch.long)
            for s in range(0, len(perm), a.batch):
                b = perm[s:s + a.batch]
                loss = (nn.functional.cross_entropy(logits(net, b), yt[b], reduction="none") * Wt[b]).sum() / Wt[b].sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            vp, vc = float(nll(net, val_pub).mean()), float(nll(net, val_card).mean())
            v = 0.5 * vp + 0.5 * vc  # both heads matter
            print(f"  seed {seed} epoch {ep + 1}: val public {vp:.4f} · val with cards {vc:.4f} ({time.time() - t:.0f} s)", flush=True)
            if v < best - 1e-4:
                best, bad, state = v, 0, {k: x.detach().clone() for k, x in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        net.load_state_dict(state)
        results.append((best, seed, net))
    results.sort(key=lambda r: r[0])
    med = results[len(results) // 2]
    best_val, seed, net = med
    print(f"seeds {[r[1] for r in results]} val {[round(r[0], 4) for r in results]} -> shipping the median, seed {seed}", flush=True)

    # test report, with frequency baselines fitted on the training rows
    def freq_baseline(rows, key_fn):
        counts = {}
        for i in tr:
            k = key_fn(i)
            c = counts.setdefault(k, np.ones(7))
            c[y[i]] += 1
        out = []
        for i in rows:
            c = counts.get(key_fn(i), np.ones(7)) * mask[i]
            out.append(-np.log(c[y[i]] / c.sum()))
        return np.array(out)

    to_call_col = NM.PUBLIC_DIM - NM.DECISION_DIM  # first decision feature: price share
    pos_cols = slice(6, 12)
    pub_key = lambda i: (int(street[i]), X[i, to_call_col] > 0, int(np.argmax(X[i, pos_cols])))  # noqa: E731
    card_key = lambda i: (int(street[i]), X[i, to_call_col] > 0, int(X[i, NM.PUBLIC_DIM + 169 + 11] * 5))  # noqa: E731
    report = {"decisions": int(len(D["label"])), "rows": int(len(y)), "shipped_seed": seed,
              "validation": {str(r[1]): round(r[0], 4) for r in results},
              "units": "mean negative log-likelihood of the recorded action (nats), 7 classes, illegal actions masked",
              "test": {}}
    tr = tr[np.random.RandomState(0).permutation(len(tr))[:400000]]  # baseline counts from a sample
    for name, cond, key in (("online_public", (split == 2) & (kind == 0) & (logs == 0), pub_key),
                            ("pluribus_logs_public", (split == 2) & (kind == 0) & (logs == 1), pub_key),
                            ("pluribus_logs_with_cards", (split == 2) & (kind == 1), card_key),
                            ("pluribus_seat_with_cards", (split == 2) & (kind == 1) & (plur == 1), card_key)):
        rows = np.nonzero(cond)[0]
        if len(rows) == 0:
            continue
        m = nll(net, rows)
        b = freq_baseline(rows, key)
        with torch.no_grad():
            pred = logits(net, T(rows, torch.long)).argmax(1).cpu().numpy()
        share_pred = np.bincount(pred, minlength=7) / len(rows)
        share_true = np.bincount(y[rows], minlength=7) / len(rows)
        report["test"][name] = {"n": int(len(rows)), "model_nll": round(float(m.mean()), 4), "baseline_nll": round(float(b.mean()), 4),
                                "gain_nats": round(float(b.mean() - m.mean()), 4),
                                "gain_ci95": round(float(1.96 * (b - m).std() / np.sqrt(len(rows))), 4),
                                "top1": round(float((pred == y[rows]).mean()), 4),
                                "share_true": [round(float(x), 4) for x in share_true],
                                "share_top1": [round(float(x), 4) for x in share_pred]}
        r_ = report["test"][name]
        print(f"  test {name:26s} n {r_['n']:8d}  model {r_['model_nll']:.3f}  baseline {r_['baseline_nll']:.3f}  "
              f"gain {r_['gain_nats']:+.3f} ± {r_['gain_ci95']:.3f}  top1 {r_['top1']:.3f}", flush=True)
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    lin = sorted({int(k.split(".")[0]) for k in sd})
    w = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    for li, k in enumerate(lin):
        w[f"l{li}.w"], w[f"l{li}.b"] = sd[f"{k}.weight"].T.astype(np.float32), sd[f"{k}.bias"].astype(np.float32)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, **w)
    m = NM.NextMoveModel(OUT_NPZ)
    chk = np.nonzero(split == 2)[0][:256]
    with torch.no_grad():
        ref = torch.softmax(logits(net, T(chk, torch.long)), 1).cpu().numpy()
    got = np.stack([m.probs(X[i], mask[i]) for i in chk])
    report["numpy_port_max_abs_diff"] = float(np.abs(ref - got).max())
    print(f"numpy port max |diff| {report['numpy_port_max_abs_diff']:.1e}", flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {OUT_NPZ} and {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    e.add_argument("--online-hands", type=int, default=15000, help="online hands sampled per shard")
    e.add_argument("--accept", type=float, default=0.06, help="chance of taking an eligible online hand")
    t = sub.add_parser("train")
    t.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3])
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=8192)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--card-weight", type=float, default=5.0)
    a = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
