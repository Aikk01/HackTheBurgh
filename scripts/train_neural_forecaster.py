"""Neural forecaster for Poker Mirror (Phase 6, GPU track, CHECKLIST.md 6.10).

Reads ``data/mirror/neural/neural_dataset.npz`` (written by ``python -m pokersim mirror-wide``) and trains a small
network on public features of the spot, the player's earlier answers and a learned player embedding.  The
configuration is chosen on the last 10% of the training half (by time); the frozen test half is scored once and
compared with the CPU forecasters on the same decisions, with intervals from resampling players.

    python scripts/train_neural_forecaster.py            # full sweep: 4 configurations x 3 seeds (GPU if present)
    python scripts/train_neural_forecaster.py --quick    # small CPU check of the whole pipeline

Writes ``data/mirror/neural_forecaster.json``; commit that file (it is small) and the Evidence tab shows it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from pokersim.mirror_wide import player_bootstrap  # noqa: E402

DATA = os.path.join(ROOT, "data", "mirror", "neural", "neural_dataset.npz")
OUT = os.path.join(ROOT, "data", "mirror", "neural_forecaster.json")
CONFIGS = [  # (player embedding size, hidden units, dropout)
    {"emb": 0, "hidden": 128, "dropout": 0.1},
    {"emb": 16, "hidden": 128, "dropout": 0.1},
    {"emb": 32, "hidden": 256, "dropout": 0.15},
    {"emb": 64, "hidden": 256, "dropout": 0.25},
]


def history_inputs(h: np.ndarray) -> np.ndarray:
    """Counts of earlier answers (fold, call, raise overall, then in this context) as smoothed rates and log sizes."""
    h = h.astype(np.float32)
    overall, ctx = h[:, :3], h[:, 3:]
    n_all, n_ctx = overall.sum(1, keepdims=True), ctx.sum(1, keepdims=True)
    return np.concatenate([(overall + 1) / (n_all + 3), (ctx + 1) / (n_ctx + 3), np.log1p(n_all), np.log1p(n_ctx)], 1)


class Net(nn.Module):
    def __init__(self, n_in: int, n_players: int, emb: int, hidden: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(n_players, emb) if emb > 0 else None
        d = n_in + emb
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor, pid: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, self.emb(pid)], 1) if self.emb is not None else x
        logits = self.mlp(z)
        return logits.masked_fill(~legal, -1e9)


def train_one(cfg, seed, data, device, epochs, batch, lr, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, pid, legal, y, tr, va, n_players = data["x"], data["pid"], data["legal"], data["y"], data["train"], data["valid"], data["n_players"]
    net = Net(x.shape[1], n_players, cfg["emb"], cfg["hidden"], cfg["dropout"]).to(device)
    params = [{"params": [p for n, p in net.named_parameters() if not n.startswith("emb")], "weight_decay": 1e-4}]
    if net.emb is not None:
        params.append({"params": net.emb.parameters(), "weight_decay": 1e-3})
    opt = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, best_epoch = float("inf"), None, -1
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen).to(tr.device)]
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            p = pid[idx]
            if net.emb is not None:  # embedding dropout: also learn the "unknown player" vector
                p = torch.where(torch.rand(len(idx), device=device) < 0.2, torch.zeros_like(p), p)
            loss = loss_fn(net(x[idx], p, legal[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = evaluate(net, data, va, device)
        log(f"    {cfg} seed {seed} epoch {epoch + 1}: validation log loss {val:.5f}")
        if val < best - 1e-5:
            best, best_epoch = val, epoch + 1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif epoch + 1 - best_epoch >= 2:
            break
    net.load_state_dict(best_state)
    return net, best, best_epoch


@torch.no_grad()
def predict(net, data, rows, device, batch=65536):
    net.eval()
    out = []
    for start in range(0, len(rows), batch):
        idx = rows[start:start + batch]
        out.append(torch.softmax(net(data["x"][idx], data["pid"][idx], data["legal"][idx]), 1).float().cpu())
    return torch.cat(out).numpy()


def evaluate(net, data, rows, device) -> float:
    p = predict(net, data, rows, device)
    y = data["y"][rows].cpu().numpy()
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-6)).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true", help="30%% of the training rows, 2 configurations, 1 seed, 2 epochs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    t0 = time.time()
    device = torch.device(args.device)
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    log = lambda s: print(s, flush=True)  # noqa: E731
    log(f"device: {name}; torch {torch.__version__}")
    z = np.load(args.data)
    X = np.concatenate([z["X"].astype(np.float32), history_inputs(z["history"])], 1)
    split, main_flag, seq, y = z["split"], z["main"].astype(bool), z["seq"], z["y"].astype(np.int64)
    train_rows = np.flatnonzero(split == 0)
    train_rows = train_rows[np.argsort(seq[train_rows], kind="stable")]
    cut = int(0.9 * len(train_rows))
    fit_rows, valid_rows = train_rows[:cut], train_rows[cut:]
    valid_rows = valid_rows[main_flag[valid_rows]]
    if args.quick:
        rng = np.random.default_rng(0)
        fit_rows = np.sort(rng.choice(fit_rows, size=max(1000, int(0.3 * len(fit_rows))), replace=False))
    test_rows = z["test_rows"].astype(np.int64)
    to = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
    data = {"x": to(X, torch.float32), "pid": to(z["player_index"], torch.long), "legal": to(np.stack([np.ones(len(y), bool),
            np.ones(len(y), bool), z["raise_legal"].astype(bool)], 1), torch.bool), "y": to(y, torch.long),
            "train": to(fit_rows, torch.long), "valid": to(valid_rows, torch.long), "n_players": int(z["n_players"])}
    log(f"rows: fit {len(fit_rows)}, validation {len(valid_rows)} (main cohort, latest 10% of training), test {len(test_rows)}")
    configs = CONFIGS[:2] if args.quick else CONFIGS
    seeds = [1] if args.quick else list(range(1, args.seeds + 1))
    epochs = 2 if args.quick else args.epochs
    runs, nets = [], {}
    for ci, cfg in enumerate(configs):
        for seed in seeds:
            t1 = time.time()
            net, val, ep = train_one(cfg, seed, data, device, epochs, args.batch, args.lr, log)
            runs.append({"config": ci, **cfg, "seed": seed, "validation_log_loss": round(val, 5), "best_epoch": ep,
                         "seconds": round(time.time() - t1, 1)})
            nets[(ci, seed)] = net
            log(f"  config {ci} seed {seed}: {val:.5f} (epoch {ep}, {time.time() - t1:.0f} s)")
    mean_val = {ci: float(np.mean([r["validation_log_loss"] for r in runs if r["config"] == ci])) for ci in range(len(configs))}
    best = min(mean_val, key=mean_val.get)
    test_t = to(test_rows, torch.long)
    p_nn = np.mean([predict(nets[(best, s)], data, test_t, device) for s in seeds], axis=0)
    y_test, players = y[test_rows], z["player_raw"][test_rows]
    ll = lambda p: -np.log(np.maximum(p[np.arange(len(y_test)), y_test], 1e-6))  # noqa: E731
    base = {"population": z["base_population"], "situation": z["base_situation"],
            "personal_online": z["base_personal_online"], "situation_personal_online": z["base_combined_online"]}
    scores = {"neural": ll(p_nn), **{k: ll(v.astype(np.float64)) for k, v in base.items()}}
    lifts = {k: player_bootstrap(players, scores[k] - scores["neural"]) for k in base}
    names = [str(n) for n in z["feature_names"]]
    r = np.exp(z["X"][test_rows, names.index("log_ratio")].astype(np.float64))
    fold = y_test == 0
    pay = lambda p: np.where(p[:, 0] > r / (1 + r), np.where(fold, 100.0, -100.0 * r), 0.0)  # noqa: E731
    bluff = {"neural": pay(p_nn), "population": pay(base["population"]), "situation_personal_online": pay(base["situation_personal_online"])}
    result = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"), "device": name, "torch": torch.__version__,
        "quick": args.quick, "rows": {"fit": int(len(fit_rows)), "validation": int(len(valid_rows)), "test": int(len(test_rows))},
        "inputs": {"public_features": int(z["X"].shape[1]), "history_inputs": 8, "players_with_embedding": int(z["n_players"]) - 1},
        "runs": runs, "chosen": {"config": best, **configs[best], "seeds_averaged": len(seeds), "validation_log_loss": round(mean_val[best], 5)},
        "test_log_loss": {k: round(float(v.mean()), 5) for k, v in scores.items()},
        "neural_lift_vs": lifts,
        "bluff_chips_per_decision": {k: round(float(v.mean()), 3) for k, v in bluff.items()},
        "bluff_vs_population": player_bootstrap(players, bluff["neural"] - bluff["population"]),
        "bluff_vs_situation_personal": player_bootstrap(players, bluff["neural"] - bluff["situation_personal_online"]),
        "seconds": round(time.time() - t0, 1),
        "label": ("frozen test half of the main cohort; lifts are the reduction in log loss against each CPU forecaster on the "
                  "same decisions (positive means the network is better), 95% intervals from resampling players"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    log(json.dumps({k: result[k] for k in ("device", "chosen", "test_log_loss", "bluff_chips_per_decision", "seconds")}, indent=1))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
