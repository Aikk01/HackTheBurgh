"""Imitation start for an RL run: a fresh policy network trained to play like the Base model, saved as a checkpoint.

    python scripts/bc_init.py --out data/rl/runs/v3-bc/bc_init.pt [--decisions 300000] [--workers 30] [--no-range-inputs]
    python scripts/rl_supervise.py --run-dir data/rl/runs/v3-bc --hours 3 --first-resume data/rl/runs/v3-bc/bc_init.pt \
        -- --mix v3 --seed 3 --cpu --workers 30

The hero plays the v3 training tables by sampling the Base model's move (the next-move model with its own cards, as
Pluribus played).  Every decision records the network input, the legal actions, the Base model's probabilities over
the 7 actions and, once the hand ends, the hero's result.  The network learns the probabilities (cross-entropy) and
the result (value head), and is saved in the trainer's checkpoint format at update 0, so PPO starts from imitation
instead of from scratch.  The trainer, its opponents and its evaluation are unchanged.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _init() -> None:
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"


def _gen(job):
    """One worker: play training tables as the Base model until ``n`` hero decisions are recorded."""
    wid, n, seed, use_ranges = job
    import numpy as np

    from pokersim import next_move as NM
    from pokersim.rl_features import action_targets, encode
    from pokersim.rl_train import REWARD_SCALE, Session, play_hand

    rng = random.Random(seed * 7919 + wid)

    def hero(obs, styles):
        x = encode(obs, styles, rng=rng, use_ranges=use_ranges)
        mask, targets = action_targets(obs)
        p = NM.predict(obs, obs["hero"]["index"], with_cards=True, pluribus=True)
        if p is None:
            raise SystemExit("data/rl/next_move.npz is missing")
        p = np.where(mask & np.array([t is not None for t in targets]), p, 0.0)
        p = p / p.sum() if p.sum() > 0 else mask / mask.sum()
        a = rng.choices(range(len(p)), weights=p)[0]
        return targets[a] or {"action": "call"}, (x, mask, p)

    xs, ms, ps, vs = [], [], [], []
    sess = None
    while len(xs) < n:
        if sess is None or sess.hands_left <= 0:
            sess = Session(rng, 0.0, [], mix="v3")
        net_bb, steps = play_hand(sess, hero, {}, rng)
        for x, m, p in steps:
            xs.append(x); ms.append(m); ps.append(p); vs.append(net_bb / REWARD_SCALE)
    return (np.stack(xs).astype(np.float32), np.stack(ms), np.stack(ps).astype(np.float32), np.asarray(vs, np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--decisions", type=int, default=300_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4, help="the PPO learning rate stored in the checkpoint's optimiser")
    ap.add_argument("--no-range-inputs", action="store_true")
    a = ap.parse_args()
    _init()
    use_ranges = not a.no_range_inputs
    t0 = time.time()
    per = -(-a.decisions // a.workers)
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        parts = pool.map(_gen, [(w, per, a.seed, use_ranges) for w in range(a.workers)])
    import numpy as np
    import torch

    from pokersim.rl_features import OBS_DIM
    from pokersim.rl_train import HIDDEN, build_net

    X, M, P, V = (np.concatenate([p[i] for p in parts]) for i in range(4))
    print(f"{len(X):,} decisions from the Base model in {time.time() - t0:.0f} s", flush=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_net(torch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(X)
    rs = np.random.RandomState(a.seed)
    order = rs.permutation(n)
    va, tr = order[: n // 20], order[n // 20:]
    T = {k: torch.as_tensor(v, device=dev) for k, v in (("x", X), ("m", M), ("p", P), ("v", V))}

    def losses(idx):
        logits, v = net(T["x"][idx])
        logp = torch.log_softmax(logits.masked_fill(~T["m"][idx], -1e9), -1)
        ce = -(T["p"][idx] * logp).sum(-1).mean()
        return ce, ((v - T["v"][idx]) ** 2).mean(), logp

    floor = float(-(P[va] * np.log(np.clip(P[va], 1e-12, 1))).sum(-1).mean())  # the targets' own entropy
    for ep in range(a.epochs):
        net.train()
        for s in range(0, len(tr), 2048):
            idx = torch.as_tensor(tr[s:s + 2048], device=dev)
            ce, vl, _ = losses(idx)
            opt.zero_grad()
            (ce + 0.5 * vl).backward()
            opt.step()
        rs.shuffle(tr)
        net.eval()
        with torch.no_grad():
            ce, vl, logp = losses(torch.as_tensor(va, device=dev))
            agree = float((logp.argmax(-1).cpu().numpy() == P[va].argmax(-1)).mean())
        print(f"epoch {ep + 1}: validation cross-entropy {float(ce):.4f} (floor {floor:.4f}) · value loss {float(vl):.3f} · "
              f"top choice agrees {agree:.1%}", flush=True)
    report = {"decisions": int(n), "use_ranges": use_ranges, "epochs": a.epochs, "val_cross_entropy": round(float(ce), 4),
              "target_entropy": round(floor, 4), "val_value_loss": round(float(vl), 4), "top_choice_agreement": round(agree, 4),
              "seconds": round(time.time() - t0)}
    net = net.cpu()
    ppo_opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)  # as the trainer builds it
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"net": net.state_dict(), "opt": ppo_opt.state_dict(), "update": 0, "steps": 0, "hands": 0, "seconds": 0.0,
                "snapshots": {}, "best_score": -1e9, "best_heldout": -1e9, "history": [], "args": {"bc_init": report},
                "obs_dim": OBS_DIM, "hidden": list(HIDDEN)}, a.out)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"-> {a.out}  {json.dumps(report)}")


if __name__ == "__main__":
    main()
