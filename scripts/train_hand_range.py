"""Train the hand-range predictor (pokersim.hand_range) on the Takara corpus.

    python scripts/train_hand_range.py extract [--workers 14] [--online-hands 30000]   # -> data/cache/hand_range/*.npz
    python scripts/train_hand_range.py train                                           # -> data/rl/hand_range.npz + .json

Examples: at every decision point of a recorded hand, the player to act is the observer; every other player still in
the hand whose hole cards are known at the end becomes a target (features from the observer's observation, label =
the target's 169-class hand).  Pluribus hands show every hole card, so they are free of showdown selection; online
hands only show the cards of players who reached a showdown.  Test data: Pluribus hands with index % 5 == 0 and the
online hands of shards 34-36.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pokersim import hand_range as H  # noqa: E402
from pokersim.mirror_data import N_SHARDS, iter_rows, row_to_record, shard_path  # noqa: E402
from pokersim.replay import replay  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache", "hand_range")
OUT_NPZ = os.path.join(ROOT, "data", "rl", "hand_range.npz")
OUT_JSON = os.path.join(ROOT, "data", "rl", "hand_range.json")
TEST_SHARDS = (34, 35, 36)


def _has_known_cards(actions) -> bool:
    for a in actions:
        if a.startswith("d dh") and "?" not in a:
            return True
        if " sm " in a and not a.endswith(" sm") and not a.endswith(" -"):
            return True
    return False


def _pluribus_number(src: str) -> int:
    """pluribus/<session>/<hand>.phh -> a stable number for the test split."""
    digits = "".join(ch for ch in src if ch.isdigit())
    return int(digits) if digits else 0


def extract_shard(args):
    i, online_cap, per_hand = args
    rng = random.Random(i)
    out = {"ctx": [], "label": [], "board": [], "hole": [], "source": [], "street": [], "test": []}
    online = 0
    n_hands = 0
    for row in iter_rows(shard_path(i), columns=None):
        acts = row.get("actions") or []
        if len(acts) <= 1 or row.get("variant") != "NT" or not _has_known_cards(acts):
            continue
        src = row.get("source_file") or ""
        pluribus = src.startswith("pluribus/")
        if not pluribus:
            if online >= online_cap:
                continue
        rec = row_to_record(row)
        if rec is None:
            continue
        try:
            res = replay(rec, snapshots=False, observations=True)
        except Exception:
            continue
        known = {}
        for p in res.hand.players:
            fc = [c for c in p.final_cards() if c is not None]
            if len(fc) == 2:
                known[p.index] = fc
        if not known:
            continue
        if not pluribus:
            online += 1
        n_hands += 1
        test = (i in TEST_SHARDS) if not pluribus else _pluribus_number(src) % 5 == 0
        dps = [d for d in res.decisions if any(j != d.player for j in known)]
        if not pluribus and len(dps) > per_hand:
            dps = rng.sample(dps, per_hand)
        for d in dps:
            obs = d.observation
            board = [H.C.parse_card(x) for x in obs.get("board") or []]
            board = [c for c in board if c is not None]
            hole = [c for c in (H.C.parse_card(x) for x in obs["hero"].get("cards") or []) if c is not None]
            for j, cards in known.items():
                if j == d.player or obs["players"][j]["folded"]:
                    continue
                out["ctx"].append(H.context_features(obs, j))
                out["label"].append(H.class_of(cards[0], cards[1]))
                out["board"].append((board + [-1] * 5)[:5])
                out["hole"].append((hole + [-1, -1])[:2] if len(hole) == 2 else [-1, -1])
                out["source"].append(1 if pluribus else 0)
                out["street"].append(H.STREETS.index(obs["street"]))
                out["test"].append(1 if test else 0)
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"shard{i:02d}.npz")
    if out["ctx"]:
        np.savez_compressed(path, ctx=np.stack(out["ctx"]).astype(np.float16), label=np.array(out["label"], np.int16),
                            board=np.array(out["board"], np.int8), hole=np.array(out["hole"], np.int8),
                            source=np.array(out["source"], np.int8), street=np.array(out["street"], np.int8),
                            test=np.array(out["test"], np.int8))
    return i, n_hands, len(out["ctx"])


def cmd_extract(a):
    t = time.time()
    jobs = [(i, a.online_hands, a.per_hand) for i in range(N_SHARDS)]
    total = 0
    with Pool(a.workers) as pool:
        for i, nh, ne in pool.imap_unordered(extract_shard, jobs):
            total += ne
            print(f"shard {i:2d}: {nh:6d} hands, {ne:7d} examples  (total {total:,}, {time.time() - t:.0f} s)", flush=True)


# ----------------------------------------------------------------- training
def load_all():
    parts = [np.load(os.path.join(CACHE, f)) for f in sorted(os.listdir(CACHE)) if f.endswith(".npz")]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def board_key(b):
    return tuple(int(x) for x in b if x >= 0)


def cmd_train(a):
    import torch
    import torch.nn as nn

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load_all()
    n = len(D["label"])
    print(f"{n:,} examples · pluribus {int(D['source'].sum()):,} · test {int(D['test'].sum()):,} · device {dev}", flush=True)
    # per-example class features and live-combo counts, cached per (board, dead) set
    boards = {}
    bidx = np.zeros(n, np.int32)
    t = time.time()
    keys = [board_key(b) for b in D["board"]]
    for i, k in enumerate(keys):
        if k not in boards:
            boards[k] = len(boards)
        bidx[i] = boards[k]
    # stored as uint8 (x 12: the two rank columns are k/12, the rest are 0/1) to keep a million boards on the GPU
    G = np.zeros((len(boards), H.N_CLASSES, H.G_DIM), np.uint8)
    for k, v in boards.items():
        G[v] = np.round(H.class_board_features(list(k)) * 12)
    print(f"{len(boards):,} distinct boards ({time.time() - t:.0f} s)", flush=True)
    LC = np.zeros((n, H.N_CLASSES), np.float16)
    t = time.time()
    cache = {}
    for i in range(n):
        dead = tuple(sorted([int(x) for x in D["board"][i] if x >= 0] + [int(x) for x in D["hole"][i] if x >= 0]))
        v = cache.get(dead)
        if v is None:
            v = cache[dead] = H.live_combos(list(dead))
        LC[i] = v
    print(f"live combos ({time.time() - t:.0f} s)", flush=True)

    ctx = D["ctx"].astype(np.float32)
    mean, std = ctx.mean(0), ctx.std(0) + 1e-3
    y = D["label"].astype(np.int64)
    test = D["test"].astype(bool)
    src = D["source"].astype(bool)
    street = D["street"].astype(np.int64)
    # validation = 5% of the training rows, by a fixed hash of the index
    rs = np.random.RandomState(0)
    val = (~test) & (rs.rand(n) < 0.05)
    tr = (~test) & (~val)

    T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    ctx_t, y_t = T((ctx - mean) / std), T(y, torch.long)
    G_t = torch.as_tensor(G, device=dev)
    bidx_t = T(bidx, torch.long)
    loglc_t = T(np.log(np.maximum(LC.astype(np.float32), 1e-9)))
    lcpos_t = T(LC.astype(np.float32) > 0, torch.bool)

    class Net(nn.Module):
        def __init__(self, hidden, k, layers):
            super().__init__()
            mods, d = [], H.CTX_DIM
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.ReLU()]
                d = hidden
            self.trunk = nn.Sequential(*mods)
            self.head_b = nn.Linear(hidden, H.N_CLASSES)
            self.head_u = nn.Linear(hidden, k)
            self.g_w = nn.Parameter(torch.randn(H.G_DIM, k) * 0.05)
            self.emb = nn.Parameter(torch.randn(H.N_CLASSES, k) * 0.05)

        def forward(self, x, g):
            h = self.trunk(x)
            cls = g @ self.g_w + self.emb  # (B, 169, k)
            return self.head_b(h) + torch.einsum("bk,bck->bc", self.head_u(h), cls)

    def batch_logits(net, idx):
        lg = net(ctx_t[idx], G_t[bidx_t[idx]].float() / 12.0) + loglc_t[idx]
        return lg.masked_fill(~lcpos_t[idx], -1e9)

    def evaluate(net, mask):
        net.eval()
        idx_all = torch.nonzero(T(mask, torch.bool)).squeeze(1)
        nll = torch.zeros(len(idx_all), device=dev)
        top1 = torch.zeros(len(idx_all), device=dev)
        top10 = torch.zeros(len(idx_all), device=dev)
        with torch.no_grad():
            for s in range(0, len(idx_all), 65536):
                idx = idx_all[s:s + 65536]
                lg = batch_logits(net, idx)
                lp = torch.log_softmax(lg, 1)
                nll[s:s + 65536] = -lp.gather(1, y_t[idx, None]).squeeze(1)
                rank = (lp > lp.gather(1, y_t[idx, None])).sum(1)
                top1[s:s + 65536] = (rank < 1).float()
                top10[s:s + 65536] = (rank < 10).float()
        net.train()
        return nll.cpu().numpy(), top1.cpu().numpy(), top10.cpu().numpy()

    def baseline_nll(mask):
        lc = LC[mask].astype(np.float64)
        p = lc / lc.sum(1, keepdims=True)
        return -np.log(np.maximum(p[np.arange(mask.sum()), y[mask]], 1e-12))

    best = None
    tr_idx = np.nonzero(tr)[0]
    for cfg in a.configs:
        hidden, k, layers = cfg
        torch.manual_seed(0)
        net = Net(hidden, k, layers).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        best_cfg, bad, state = 1e9, 0, None
        for ep in range(a.epochs):
            t = time.time()
            perm = torch.as_tensor(np.random.RandomState(ep).permutation(tr_idx), device=dev)
            for s in range(0, len(perm), a.batch):
                idx = perm[s:s + a.batch]
                loss = nn.functional.cross_entropy(batch_logits(net, idx), y_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            v = float(evaluate(net, val)[0].mean())
            print(f"  cfg {cfg} epoch {ep + 1}: val nll {v:.4f} ({time.time() - t:.0f} s)", flush=True)
            if v < best_cfg - 1e-4:
                best_cfg, bad = v, 0
                state = {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        net.load_state_dict(state)
        if best is None or best_cfg < best[0]:
            best = (best_cfg, cfg, net)
    val_nll, cfg, net = best
    print(f"chosen {cfg}: val nll {val_nll:.4f}", flush=True)

    report = {"examples": int(n), "train": int(tr.sum()), "validation": int(val.sum()), "test": int(test.sum()),
              "config": {"hidden": cfg[0], "k": cfg[1], "layers": cfg[2]}, "validation_nll": round(val_nll, 4),
              "device": str(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"),
              "units": "mean negative log probability of the true 169-class hand (nats); baseline = card counting only",
              "test": {}}
    nll, t1, t10 = evaluate(net, test)
    for name, m in (("all", np.ones(test.sum(), bool)), ("pluribus", src[test]), ("online_showdown", ~src[test])):
        for si, s in enumerate(["all_streets"] + H.STREETS):
            mm = m if si == 0 else m & (street[test] == si - 1)
            if mm.sum() == 0:
                continue
            base = baseline_nll(_mask_from(test, mm))
            report["test"][f"{name}/{s}"] = {"n": int(mm.sum()), "model_nll": round(float(nll[mm].mean()), 4),
                                              "baseline_nll": round(float(base.mean()), 4),
                                              "gain_nats": round(float(base.mean() - nll[mm].mean()), 4),
                                              "top1": round(float(t1[mm].mean()), 4), "top10": round(float(t10[mm].mean()), 4)}
    for k_, v_ in report["test"].items():
        print(f"  test {k_:28s} n {v_['n']:8d}  model {v_['model_nll']:.3f}  baseline {v_['baseline_nll']:.3f}  "
              f"gain {v_['gain_nats']:+.3f}  top1 {v_['top1']:.3f}  top10 {v_['top10']:.3f}", flush=True)

    sd = {kk: vv.detach().cpu().numpy() for kk, vv in net.state_dict().items()}
    w = {"ctx_mean": mean.astype(np.float32), "ctx_std": std.astype(np.float32), "g_w": sd["g_w"], "emb": sd["emb"],
         "head_b.w": sd["head_b.weight"].T, "head_b.b": sd["head_b.bias"],
         "head_u.w": sd["head_u.weight"].T, "head_u.b": sd["head_u.bias"]}
    li = 0
    for kk in sorted(k_ for k_ in sd if k_.startswith("trunk") and k_.endswith(".weight")):
        pre = kk[:-len(".weight")]
        w[f"trunk{li}.w"], w[f"trunk{li}.b"] = sd[kk].T, sd[pre + ".bias"]
        li += 1
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, **{kk: np.ascontiguousarray(vv, dtype=np.float32) for kk, vv in w.items()})
    # check the NumPy port against torch on a few test rows
    m = H.RangeModel(OUT_NPZ)
    ti = np.nonzero(test)[0][:64]
    with torch.no_grad():
        ref = torch.softmax(batch_logits(net, T(ti, torch.long)), 1).cpu().numpy()
    got = np.stack([m.predict(ctx[i], [int(x) for x in D["board"][i] if x >= 0],
                              [int(x) for x in D["board"][i] if x >= 0] + [int(x) for x in D["hole"][i] if x >= 0])[0] for i in ti])
    report["numpy_port_max_abs_diff"] = float(np.abs(ref - got).max())
    print(f"numpy port max |diff| {report['numpy_port_max_abs_diff']:.2e}", flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {OUT_NPZ} and {OUT_JSON}")


def _mask_from(test, mm):
    full = np.zeros(len(test), bool)
    full[np.nonzero(test)[0][mm]] = True
    return full


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    e.add_argument("--online-hands", type=int, default=30000, help="showdown hands per shard from the online games")
    e.add_argument("--per-hand", type=int, default=4, help="decision points sampled per online hand")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--configs", type=lambda s: [tuple(int(x) for x in c.split("x")) for c in s.split(",")],
                   default=[(256, 32, 2), (512, 64, 3)], help="hidden x k x layers, comma separated")
    a = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
