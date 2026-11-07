"""PPO training of the 6-max RL advisor (GPU learner, CPU self-play workers).

    python -m pokersim rl-train --hours 9.5 --workers 24          # full overnight run on the GPU PC
    python -m pokersim rl-train --smoke                           # 1-2 minute end-to-end check on any PC
    python -m pokersim rl-train --resume data/rl/runs/<run>/latest.pt --hours 3

What is learned: one seat at a six-handed no-limit table (100bb deep on average, stacks reset each hand)
against opponents drawn from a wide range of styles: equity bots with random tightness/aggression,
calling stations, maniacs, random players, TAG/LAG presets, and frozen past versions of the policy
itself.  A table keeps the same opponents for a session of 50-300 hands, and the policy sees each
opponent's style estimate built up over that session (:mod:`pokersim.opponent_model`), so it can learn
to exploit how *these* opponents play.

Every ``--eval-every`` updates the policy plays fixed lineups (tight-passive, loose-passive,
loose-aggressive, tight-aggressive, mixed) against the same deal seeds as a baseline equity bot, and
once more with the style inputs blanked out; the gap between the two RL rows shows how much the
opponent model is used.  The best policy so far is exported to ``data/rl/policy.npz`` for the Play
tab, with ``data/rl/policy.json`` describing it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agents import make_agent, safe_act
from .engine import IllegalAction, Seat, Table
from .opponent_model import StyleTracker, estimate_style
from .rl_features import ACTIONS, N_ACTIONS, OBS_DIM, action_targets, encode, styles_for_hand
from .rl_policy import HIDDEN, POLICY_META, POLICY_NPZ, RL_DIR, NumpyPolicy

BB = 2.0
REWARD_SCALE = 10.0  # reward = net big blinds / 10

EVAL_LINEUPS: Dict[str, List[str]] = {
    "tight_passive": ["equity:tightness=0.9,aggression=0.1,iterations=40"] * 5,
    "loose_passive": ["station", "station", "equity:tightness=0.05,aggression=0.05,iterations=40",
                      "station", "equity:tightness=0.1,aggression=0.1,iterations=40"],
    "loose_aggressive": ["lag:iterations=40", "maniac", "lag:iterations=40", "maniac", "lag:iterations=40"],
    "tight_aggressive": ["tag:iterations=40"] * 5,
    "mixed": ["tag:iterations=40", "lag:iterations=40", "equity:iterations=40", "station", "random"],
}


# ============================================================ opponents
def sample_opponent(rng: random.Random) -> str:
    u = rng.random()
    if u < 0.62:
        return f"equity:tightness={rng.random():.2f},aggression={rng.random():.2f},iterations=40"
    if u < 0.72:
        return "station"
    if u < 0.79:
        return f"maniac:frequency={rng.uniform(0.3, 0.9):.2f}"
    if u < 0.84:
        return "random"
    if u < 0.92:
        return "tag:iterations=40"
    return "lag:iterations=40"


# ============================================================ one table session (worker side)
class Session:
    """Hero in seat 1 against five opponents for ``hands_left`` hands; stacks reset every hand."""

    def __init__(self, rng: random.Random, p_self: float, snapshot_ids: Sequence[int], lineup: Optional[List[str]] = None,
                 hands: Optional[int] = None):
        self.rng = rng
        self.hands_left = hands if hands is not None else rng.randint(50, 300)
        self.opp: Dict[int, Any] = {}
        seats = [Seat(seat_no=1, name="hero", stack=100 * BB)]
        for k in range(5):
            seat_no = k + 2
            if lineup is not None:
                spec = lineup[k]
            elif snapshot_ids and rng.random() < p_self:
                spec = f"snapshot:{rng.choice(list(snapshot_ids))}"
            else:
                spec = sample_opponent(rng)
            name = f"p{seat_no}"
            if spec.startswith("snapshot:"):
                self.opp[seat_no] = ("snapshot", int(spec.split(":")[1]))
            else:
                agent = make_agent(spec, seed=rng.randrange(1 << 30))
                agent.name = name
                self.opp[seat_no] = ("bot", agent)
            seats.append(Seat(seat_no=seat_no, name=name, stack=100 * BB))
        self.table = Table(seats, small_blind=BB / 2, big_blind=BB, rng=random.Random(rng.randrange(1 << 30)))
        self.table.button_seat = rng.randint(1, 6)
        self.tracker = StyleTracker()


def play_hand(sess: Session, hero_fn, snapshot_policies: Dict[int, NumpyPolicy], rng: random.Random,
              deep_random: bool = True) -> Tuple[float, List[Tuple[np.ndarray, np.ndarray, int, float, float]]]:
    """Play one hand; returns hero net big blinds and the hero's decisions (x, mask, action, logp, value)."""
    t = sess.table
    for s in t.seats:
        s.stack = (rng.uniform(50, 150) if deep_random else 100.0) * BB
    hand = t.new_hand()
    hero_idx = next(i for i, s in enumerate(t.current_order) if s.seat_no == 1)
    start = hand.players[hero_idx].stack
    steps: List[Tuple[np.ndarray, np.ndarray, int, float, float]] = []
    guard = 0
    while hand.status == "betting" and hand.actor is not None and guard < 400:
        guard += 1
        actor = hand.actor
        seat = t.seat_for_player(actor)
        obs = hand.observation(actor)
        if seat.seat_no == 1:
            styles = styles_for_hand(sess.tracker, hand.players)
            action, rec = hero_fn(obs, styles)
            if rec is not None:
                steps.append(rec)
        else:
            kind, who = sess.opp[seat.seat_no]
            if kind == "snapshot":
                pol = snapshot_policies.get(who)
                if pol is None:
                    action = {"action": "call"}
                else:
                    styles = styles_for_hand(sess.tracker, hand.players)
                    x = encode(obs, styles, rng=rng)
                    mask, targets = action_targets(obs)
                    pr, _ = pol.probs(x, mask)
                    action = targets[int(rng.choices(range(N_ACTIONS), weights=pr)[0])] or {"action": "call"}
            else:
                action = safe_act(who, obs)
        try:
            hand.act(actor, action["action"], action.get("amount"))
        except IllegalAction:
            hand.act(actor, "call")
    if hand.status not in ("finished", "incomplete"):
        hand.finalize()
    net_bb = (hand.players[hero_idx].stack - start) / BB
    t.finish_hand()
    sess.tracker.add_hand(hand)
    sess.hands_left -= 1
    return net_bb, steps


def _sampling_hero(policy: NumpyPolicy, rng: random.Random, blank_styles: bool = False, greedy: bool = False,
                   blank_ranges: bool = False):
    def fn(obs, styles):
        if blank_styles:
            styles = {}
        x = encode(obs, styles, rng=rng, use_ranges=not blank_ranges)
        mask, targets = action_targets(obs)
        logits, value = policy.forward(x[None, :])
        logits = np.where(mask, logits[0], -1e9)
        e = np.exp(logits - logits.max())
        pr = e / e.sum()
        a = int(np.argmax(pr)) if greedy else int(rng.choices(range(N_ACTIONS), weights=pr)[0])
        return targets[a] or {"action": "call"}, (x, mask, a, float(math.log(max(pr[a], 1e-12))), float(value[0]))
    return fn


def _bot_hero(spec: str, rng: random.Random):
    agent = make_agent(spec, seed=rng.randrange(1 << 30))

    def fn(obs, styles):
        return safe_act(agent, obs), None
    return fn


def worker_main(wid: int, seed: int, cmd_q, out_q) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rng = random.Random(seed * 7919 + wid)
    snapshots: Dict[int, NumpyPolicy] = {}
    sess: Optional[Session] = None
    while True:
        msg = cmd_q.get()
        if msg is None:
            return
        kind = msg["kind"]
        try:
            for sid, params in (msg.get("add_snapshots") or {}).items():
                snapshots[int(sid)] = NumpyPolicy(params)
            for sid in list(snapshots):
                if sid not in set(msg.get("keep_snapshots", snapshots.keys())):
                    snapshots.pop(sid, None)
            policy = NumpyPolicy(msg["params"]) if msg.get("params") is not None else None
            if kind == "rollout":
                hero = _sampling_hero(policy, rng)
                xs, masks, acts, logps, vals, advs, rets = [], [], [], [], [], [], []
                hands, net_total = 0, 0.0
                while len(xs) < msg["steps"]:
                    if sess is None or sess.hands_left <= 0:
                        sess = Session(rng, msg["p_self"], list(snapshots))
                    net_bb, steps = play_hand(sess, hero, snapshots, rng)
                    hands += 1
                    net_total += net_bb
                    if not steps:
                        continue
                    r_final = net_bb / REWARD_SCALE
                    gae, next_v = 0.0, 0.0
                    out = []
                    for k in range(len(steps) - 1, -1, -1):
                        x, m, a, lp, v = steps[k]
                        r = r_final if k == len(steps) - 1 else 0.0
                        delta = r + next_v - v
                        gae = delta + msg["lam"] * gae
                        out.append((x, m, a, lp, v, gae, gae + v))
                        next_v = v
                    for x, m, a, lp, v, ad, rt in reversed(out):
                        xs.append(x); masks.append(m); acts.append(a); logps.append(lp); vals.append(v); advs.append(ad); rets.append(rt)
                out_q.put({"kind": "rollout", "wid": wid, "x": np.stack(xs), "mask": np.stack(masks), "a": np.asarray(acts, np.int64),
                           "logp": np.asarray(logps, np.float32), "v": np.asarray(vals, np.float32), "adv": np.asarray(advs, np.float32),
                           "ret": np.asarray(rets, np.float32), "hands": hands, "net_bb": net_total})
            elif kind == "eval":
                erng = random.Random(msg["seed"])
                if msg["hero"] == "rl":
                    hero = _sampling_hero(policy, erng, greedy=True)
                elif msg["hero"] == "rl_nostyle":
                    hero = _sampling_hero(policy, erng, greedy=True, blank_styles=True)
                elif msg["hero"] == "rl_norange":
                    hero = _sampling_hero(policy, erng, greedy=True, blank_ranges=True)
                else:
                    hero = _bot_hero(msg["hero"], erng)
                s = Session(random.Random(msg["seed"]), 0.0, [], lineup=EVAL_LINEUPS[msg["lineup"]], hands=msg["hands"])
                nets = []
                drng = random.Random(msg["seed"] + 1)
                while s.hands_left > 0:
                    net_bb, _ = play_hand(s, hero, {}, drng, deep_random=False)
                    nets.append(net_bb)
                out_q.put({"kind": "eval", "wid": wid, "lineup": msg["lineup"], "hero": msg["hero"], "nets": nets})
        except Exception as exc:  # noqa: BLE001
            import traceback
            out_q.put({"kind": "error", "wid": wid, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})


# ============================================================ learner (torch)
def build_net(torch, hidden=HIDDEN):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.lin = nn.Linear(i, o)
            self.ln = nn.LayerNorm(o)

        def forward(self, x):
            return torch.relu(self.ln(self.lin(x)))

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [OBS_DIM] + list(hidden)
            self.trunk = nn.ModuleList([Block(dims[i], dims[i + 1]) for i in range(len(hidden))])
            self.pi = nn.Linear(hidden[-1], N_ACTIONS)
            self.v = nn.Linear(hidden[-1], 1)
            nn.init.orthogonal_(self.pi.weight, 0.01)
            nn.init.zeros_(self.pi.bias)

        def forward(self, x):
            for b in self.trunk:
                x = b(x)
            return self.pi(x), self.v(x).squeeze(-1)

    return PolicyNet()


def export_params(net) -> Dict[str, np.ndarray]:
    out = {}
    for i, b in enumerate(net.trunk):
        out[f"trunk.{i}.w"] = b.lin.weight.detach().cpu().numpy()
        out[f"trunk.{i}.b"] = b.lin.bias.detach().cpu().numpy()
        out[f"trunk.{i}.ln_w"] = b.ln.weight.detach().cpu().numpy()
        out[f"trunk.{i}.ln_b"] = b.ln.bias.detach().cpu().numpy()
    out["pi.w"] = net.pi.weight.detach().cpu().numpy()
    out["pi.b"] = net.pi.bias.detach().cpu().numpy()
    out["v.w"] = net.v.weight.detach().cpu().numpy()
    out["v.b"] = net.v.bias.detach().cpu().numpy()
    return out


def _ci(nets: Sequence[float]) -> Tuple[float, float]:
    n = len(nets)
    if n < 2:
        return (float("nan"), float("nan"))
    m = sum(nets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in nets) / (n - 1))
    return (100 * m, 100 * 1.96 * sd / math.sqrt(n))


class Trainer:
    def __init__(self, args):
        import torch

        self.torch = torch
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        torch.manual_seed(args.seed)
        self.net = build_net(torch).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, eps=1e-5)
        self.update = 0
        self.steps = 0
        self.hands = 0
        self.seconds_before = 0.0
        self.snapshots: Dict[int, Dict[str, np.ndarray]] = {}
        self.best_score = -1e9
        self.history: List[Dict[str, Any]] = []
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = args.run_dir or os.path.join(RL_DIR, "runs", stamp)
        if args.resume:
            self._load(args.resume)
            self.run_dir = args.run_dir or os.path.dirname(os.path.abspath(args.resume))
        os.makedirs(self.run_dir, exist_ok=True)
        # the smoke test never replaces the model the Play tab uses
        self.export_dir = self.run_dir if args.smoke else RL_DIR

    # ---------------------------------------------------------------- checkpoints
    def _save(self, name: str) -> str:
        path = os.path.join(self.run_dir, name)
        self.torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(), "update": self.update, "steps": self.steps,
                         "hands": self.hands, "seconds": self.seconds_before + (time.time() - self.t0),
                         "snapshots": self.snapshots, "best_score": self.best_score, "history": self.history,
                         "args": vars(self.args), "obs_dim": OBS_DIM, "hidden": list(HIDDEN)}, path)
        return path

    def _load(self, path: str) -> None:
        ck = self.torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("obs_dim") != OBS_DIM:
            raise SystemExit(f"checkpoint has {ck.get('obs_dim')} features, this code makes {OBS_DIM}")
        self.net.load_state_dict(ck["net"])
        self.opt.load_state_dict(ck["opt"])
        self.update, self.steps, self.hands = ck["update"], ck["steps"], ck["hands"]
        self.seconds_before = ck.get("seconds", 0.0)
        self.snapshots = ck.get("snapshots", {})
        self.best_score = ck.get("best_score", -1e9)
        self.history = ck.get("history", [])
        print(f"resumed from {path}: update {self.update}, {self.steps:,} steps, {self.hands:,} hands")

    def _export(self, eval_result: Dict[str, Any], out_dir: str = RL_DIR) -> None:
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "policy.npz"), **export_params(self.net))
        meta = {"trained_at": _dt.datetime.now().isoformat(timespec="seconds"), "updates": self.update, "steps": self.steps,
                "hands": self.hands, "seconds": round(self.seconds_before + time.time() - self.t0),
                "device": str(self.device) + (f" ({self.torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""),
                "obs_dim": OBS_DIM, "hidden": list(HIDDEN), "actions": ACTIONS, "eval": eval_result,
                "run_dir": os.path.relpath(self.run_dir, os.path.join(RL_DIR, "..", "..")).replace(os.sep, "/"),
                "algorithm": "PPO (clipped), GAE; self-play workers vs a style-diverse opponent pool + league snapshots",
                "reward": "net big blinds per hand / 10"}
        with open(os.path.join(out_dir, "policy.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)

    # ---------------------------------------------------------------- main loop
    def run(self) -> None:
        a = self.args
        torch = self.torch
        ctx = mp.get_context("spawn")
        cmd_qs = [ctx.Queue() for _ in range(a.workers)]
        out_q = ctx.Queue()
        procs = [ctx.Process(target=worker_main, args=(w, a.seed, cmd_qs[w], out_q), daemon=True) for w in range(a.workers)]
        for p in procs:
            p.start()
        self.t0 = time.time()
        deadline = self.t0 + a.hours * 3600
        log_path = os.path.join(self.run_dir, "log.csv")
        new_log = not os.path.exists(log_path)
        logf = open(log_path, "a", newline="", encoding="utf-8")
        log = csv.writer(logf)
        if new_log:
            log.writerow(["update", "steps", "hands", "elapsed_s", "steps_per_s", "rollout_bb100", "pi_loss", "v_loss", "entropy",
                          "approx_kl", "clip_frac", "lr", "ent_coef"])
        print(f"device {self.device}{' (' + torch.cuda.get_device_name(0) + ')' if self.device.type == 'cuda' else ''} · "
              f"{a.workers} workers · {a.steps_per_update:,} steps/update · obs {OBS_DIM} · run {self.run_dir}", flush=True)
        added: Dict[int, Dict[str, np.ndarray]] = dict(self.snapshots)  # resend snapshots to fresh workers
        last_ckpt = time.time()
        try:
            while time.time() < deadline and (a.max_updates <= 0 or self.update < a.max_updates):
                frac = min(1.0, (time.time() - self.t0) / max(1.0, deadline - self.t0))
                lr = a.lr * (1.0 - 0.9 * frac)
                ent_coef = a.ent_coef * (1.0 - 0.8 * frac)
                for g in self.opt.param_groups:
                    g["lr"] = lr
                t_up = time.time()
                params = export_params(self.net)
                per = max(64, a.steps_per_update // a.workers)
                for q in cmd_qs:
                    q.put({"kind": "rollout", "params": params, "steps": per, "p_self": a.p_self, "lam": a.lam,
                           "add_snapshots": added, "keep_snapshots": list(self.snapshots)})
                added = {}
                batch = self._gather(out_q, a.workers, "rollout")
                X = torch.as_tensor(np.concatenate([b["x"] for b in batch]), device=self.device)
                M = torch.as_tensor(np.concatenate([b["mask"] for b in batch]), device=self.device)
                A = torch.as_tensor(np.concatenate([b["a"] for b in batch]), device=self.device)
                LP = torch.as_tensor(np.concatenate([b["logp"] for b in batch]), device=self.device)
                ADV = torch.as_tensor(np.concatenate([b["adv"] for b in batch]), device=self.device)
                RET = torch.as_tensor(np.concatenate([b["ret"] for b in batch]), device=self.device)
                hands = sum(b["hands"] for b in batch)
                net_bb = sum(b["net_bb"] for b in batch)
                ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
                n = X.shape[0]
                stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0, "k": 0}
                for _epoch in range(a.epochs):
                    perm = torch.randperm(n, device=self.device)
                    for s in range(0, n, a.minibatch):
                        idx = perm[s:s + a.minibatch]
                        logits, v = self.net(X[idx])
                        logits = logits.masked_fill(~M[idx], -1e9)
                        logp_all = torch.log_softmax(logits, -1)
                        lp = logp_all.gather(1, A[idx][:, None]).squeeze(1)
                        ratio = torch.exp(lp - LP[idx])
                        adv = ADV[idx]
                        pi_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
                        v_loss = 0.5 * ((v - RET[idx]) ** 2).mean()
                        p_all = logp_all.exp()
                        ent = -(p_all * logp_all.masked_fill(~M[idx], 0.0)).sum(-1).mean()
                        loss = pi_loss + a.vf_coef * v_loss - ent_coef * ent
                        self.opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                        self.opt.step()
                        with torch.no_grad():
                            stats["pi"] += pi_loss.item(); stats["v"] += v_loss.item(); stats["ent"] += ent.item()
                            stats["kl"] += (LP[idx] - lp).mean().item(); stats["clip"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                            stats["k"] += 1
                k = max(1, stats["k"])
                self.update += 1
                self.steps += n
                self.hands += hands
                elapsed = self.seconds_before + time.time() - self.t0
                sps = n / max(1e-6, time.time() - t_up)
                row = [self.update, self.steps, self.hands, round(elapsed), round(sps), round(100 * net_bb / max(1, hands), 2),
                       round(stats["pi"] / k, 4), round(stats["v"] / k, 4), round(stats["ent"] / k, 4), round(stats["kl"] / k, 5),
                       round(stats["clip"] / k, 4), f"{lr:.2e}", round(ent_coef, 4)]
                log.writerow(row)
                logf.flush()
                if self.update % a.print_every == 0:
                    print(f"upd {self.update:5d} · {self.steps:,} steps · {self.hands:,} hands · {sps:,.0f} steps/s · "
                          f"rollout {row[5]:+.1f} bb/100 · ent {row[8]:.3f} · kl {row[9]:.4f} · {elapsed / 3600:.2f} h", flush=True)
                if self.update % a.snapshot_every == 0:
                    sid = self.update
                    self.snapshots[sid] = export_params(self.net)
                    added[sid] = self.snapshots[sid]
                    if len(self.snapshots) > a.max_snapshots:
                        self.snapshots.pop(min(self.snapshots))
                if self.update % a.eval_every == 0:
                    self._evaluate(cmd_qs, out_q)
                if time.time() - last_ckpt > a.ckpt_minutes * 60:
                    self._save("latest.pt")
                    last_ckpt = time.time()
            print("time budget reached" if time.time() >= deadline else "max updates reached", flush=True)
            self._evaluate(cmd_qs, out_q, final=True)
            self._save("latest.pt")
        finally:
            for q in cmd_qs:
                q.put(None)
            for p in procs:
                p.join(timeout=10)
            logf.close()

    def _gather(self, out_q, n: int, kind: str) -> List[Dict[str, Any]]:
        got = []
        while len(got) < n:
            try:
                m = out_q.get(timeout=1800)
            except queue.Empty:
                raise SystemExit("workers stopped responding for 30 minutes")
            if m["kind"] == "error":
                print(f"worker {m['wid']} error: {m['error']}\n{m['trace']}", flush=True)
                raise SystemExit(1)
            if m["kind"] == kind:
                got.append(m)
        return got

    def _evaluate(self, cmd_qs, out_q, final: bool = False) -> Dict[str, Any]:
        a = self.args
        params = export_params(self.net)
        heroes = ["rl", "rl_nostyle", "rl_norange", "equity:iterations=40"]
        jobs = []
        chunk = max(50, a.eval_hands // max(1, a.eval_chunks))
        for li, lineup in enumerate(EVAL_LINEUPS):
            for c in range(max(1, a.eval_hands // chunk)):
                for h in heroes:
                    jobs.append({"kind": "eval", "params": params if h.startswith("rl") else None, "lineup": lineup, "hero": h,
                                 "hands": chunk, "seed": 10_000 + 97 * c + 1009 * li})
        nets: Dict[Tuple[str, str], List[float]] = {}
        sent = 0
        pending = 0
        while sent < len(jobs) or pending:
            while sent < len(jobs) and pending < len(cmd_qs):
                cmd_qs[sent % len(cmd_qs)].put(jobs[sent])
                sent += 1
                pending += 1
            m = self._gather(out_q, 1, "eval")[0]
            pending -= 1
            nets.setdefault((m["lineup"], m["hero"]), []).extend(m["nets"])
        res: Dict[str, Any] = {"update": self.update, "hands_per_lineup": a.eval_hands, "lineups": {}}
        score = []
        for lineup in EVAL_LINEUPS:
            row = {}
            for h in heroes:
                m, ci = _ci(nets.get((lineup, h), []))
                row["baseline_equity_bot" if h.startswith("equity") else h] = {"bb100": round(m, 2), "ci95": round(ci, 2)}
            row["rl_minus_baseline"] = round(row["rl"]["bb100"] - row["baseline_equity_bot"]["bb100"], 2)
            row["style_inputs_gain"] = round(row["rl"]["bb100"] - row["rl_nostyle"]["bb100"], 2)
            row["range_inputs_gain"] = round(row["rl"]["bb100"] - row["rl_norange"]["bb100"], 2)
            res["lineups"][lineup] = row
            score.append(row["rl"]["bb100"])
        res["mean_rl_bb100"] = round(sum(score) / len(score), 2)
        res["generated"] = _dt.datetime.now().isoformat(timespec="seconds")
        self.history.append(res)
        with open(os.path.join(self.run_dir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=1)
        line = " · ".join(f"{k} {v['rl']['bb100']:+.1f} (base {v['baseline_equity_bot']['bb100']:+.1f}, no-style {v['rl_nostyle']['bb100']:+.1f}, no-range {v['rl_norange']['bb100']:+.1f})"
                          for k, v in res["lineups"].items())
        print(f"EVAL upd {self.update}: mean {res['mean_rl_bb100']:+.1f} bb/100 · {line}", flush=True)
        if res["mean_rl_bb100"] > self.best_score or final and self.best_score == -1e9:
            self.best_score = res["mean_rl_bb100"]
            self._save("best.pt")
            self._export(res, self.export_dir)
            print(f"  new best -> exported {os.path.relpath(os.path.join(self.export_dir, 'policy.npz'))}", flush=True)
        if final:
            self._export_dir_copy(res)
        return res

    def _export_dir_copy(self, res: Dict[str, Any]) -> None:
        """Keep the final (not necessarily best) policy in the run directory too."""
        np.savez(os.path.join(self.run_dir, "final_policy.npz"), **export_params(self.net))
        with open(os.path.join(self.run_dir, "final_eval.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


# ============================================================ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pokersim rl-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=9.5, help="wall-clock budget; training stops cleanly after it")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="self-play processes (CPU cores - 2)")
    ap.add_argument("--steps-per-update", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--p-self", type=float, default=0.25, help="share of opponent seats taken by past policy snapshots")
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--max-snapshots", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-hands", type=int, default=3000, help="hands per lineup per hero at each evaluation")
    ap.add_argument("--eval-chunks", type=int, default=6)
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true", help="train on the CPU even if CUDA is available")
    ap.add_argument("--resume", default=None, help="checkpoint (.pt) to continue from")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (a minute or two) to check the setup")
    args = ap.parse_args(argv)
    if args.smoke:
        args.workers = min(args.workers, 2)
        args.steps_per_update, args.minibatch, args.epochs = 512, 256, 2
        args.max_updates, args.eval_every, args.snapshot_every = 3, 3, 2
        args.eval_hands, args.eval_chunks, args.hours = 60, 1, 0.5
        args.run_dir = args.run_dir or os.path.join(RL_DIR, "runs", "smoke")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is not installed. See TRAINING.md (pip install torch --index-url https://download.pytorch.org/whl/cu128).")
        return 1
    Trainer(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
