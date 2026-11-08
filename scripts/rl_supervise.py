"""Run `pokersim rl-train` under supervision: watch it, keep every checkpoint, log to wandb, reset on failure.

    python scripts/rl_supervise.py --run-dir data/rl/runs/v3-ranges --hours 5 --name v3-ranges -- --seed 1 --workers 14
    touch data/rl/runs/v3-ranges/STOP      # graceful stop: the trainer saves latest.pt and the supervisor exits

Everything after ``--`` goes to rl-train.  One line per event goes to stdout and to <run>/supervise.log:

* ``EVAL``     every evaluation (held-out score, in-distribution mean, new best)
* ``ALERT``    a spike in value loss / KL / clip fraction, a sudden entropy drop, a slow-down (no automatic action)
* ``STALL``    no new best held-out score in 4 evaluations
* ``RESET``    NaN/inf in the metrics or entropy collapse below 0.2: the trainer is killed and resumed from the last kept
               checkpoint *before* the problem, with the learning rate halved (at most 2 resets)
* ``RESTART``  the trainer died without finishing: resumed from latest.pt for the remaining time (at most 3 restarts)
* ``DONE`` / ``DEAD``

Every new latest.pt is copied to <run>/keep/updNNNNN.pt; best held-out policies are kept by rl-train itself in
<run>/best_heldout_history/.  Set WANDB_MODE=offline on machines without internet (sync later with `wandb sync`).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--hours", type=float, required=True, help="total wall-clock budget for this run, restarts included")
ap.add_argument("--name", default=None, help="wandb run name (default: run dir name)")
ap.add_argument("--project", default="poker-mirror-rl")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--max-restarts", type=int, default=3)
ap.add_argument("--max-resets", type=int, default=2)
ap.add_argument("--poll", type=float, default=20.0)
ap.add_argument("--first-resume", default=None, help="checkpoint to resume from at the first launch")
args, rest = ap.parse_known_args()
extra = [x for x in rest if x != "--"]

R = os.path.abspath(args.run_dir)
os.makedirs(os.path.join(R, "keep"), exist_ok=True)
name = args.name or os.path.basename(R)
deadline = time.time() + args.hours * 3600
log_path = os.path.join(R, "supervise.log")


def say(kind: str, msg: str) -> None:
    line = f"{time.strftime('%H:%M')} [{name}] {kind} {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- wandb (optional)
try:
    import wandb

    run = wandb.init(project=args.project, id=name, name=name, resume="allow",
                     config={"run_dir": R, "hours": args.hours, "trainer_args": " ".join(extra)})
    wandb.define_metric("train/update")
    wandb.define_metric("train/*", step_metric="train/update")
    wandb.define_metric("eval/update")
    wandb.define_metric("eval/*", step_metric="eval/update")
except Exception as exc:  # noqa: BLE001 - supervision must work without wandb
    run = None
    say("INFO", f"wandb off ({exc.__class__.__name__})")


# ---------------------------------------------------------------- trainer process
proc = None
lr = args.lr
restarts = resets = 0


def launch(resume: str | None) -> None:
    global proc
    hours = max(0.05, (deadline - time.time()) / 3600 - 0.08)  # leave ~5 min for the final evaluation
    cmd = [sys.executable, "-u", "-m", "pokersim", "rl-train", "--run-dir", R, "--hours", f"{hours:.3f}", "--lr", f"{lr:.2e}"]
    if resume:
        cmd += ["--resume", resume]
    cmd += extra
    out = open(os.path.join(R, "train.log"), "a")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    say("INFO", f"trainer pid {proc.pid}: {' '.join(cmd[3:])}")


def kill_hard() -> None:
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def keep_before(update: int) -> str | None:
    """Newest kept checkpoint strictly before ``update``."""
    cands = []
    for f in os.listdir(os.path.join(R, "keep")):
        if f.startswith("upd") and f.endswith(".pt"):
            u = int(f[3:8])
            if u < update:
                cands.append((u, f))
    return os.path.join(R, "keep", max(cands)[1]) if cands else None


# ---------------------------------------------------------------- watching
rows_seen = evals_seen = 0
ckpt_mtime = 0.0
best_heldout, best_idx = -1e18, -1


def floats(r):
    out = {}
    for k, v in r.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def check_rows() -> int | None:
    """Returns the update at which a reset is needed, else None."""
    global rows_seen
    path = os.path.join(R, "log.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = [floats(r) for r in csv.DictReader(f) if r.get("ent_coef")]
    reset_at = None
    for i in range(rows_seen, len(rows)):
        r = rows[i]
        upd = int(r.get("update", 0))
        if run:
            run.log({f"train/{k}": v for k, v in r.items()})
        if any(math.isnan(v) or math.isinf(v) for v in r.values()):
            say("RESET?", f"upd {upd}: NaN/inf in the metrics")
            reset_at = reset_at or upd
            continue
        prev = rows[max(0, i - 20):i]
        if len(prev) < 8:
            continue
        med_v = statistics.median(p["v_loss"] for p in prev)
        ent5 = statistics.median(p["entropy"] for p in prev[-5:])
        med_sps = statistics.median(p["steps_per_s"] for p in prev)
        why = []
        if r["v_loss"] > 2.5 * med_v and r["v_loss"] > 30:
            why.append(f"value loss {r['v_loss']:.1f} (median {med_v:.1f})")
        if r["approx_kl"] > 0.06:
            why.append(f"KL {r['approx_kl']:.3f}")
        if r["clip_frac"] > 0.35:
            why.append(f"clip {r['clip_frac']:.2f}")
        if r["entropy"] < ent5 - 0.15:
            why.append(f"entropy {r['entropy']:.3f} (last-5 median {ent5:.3f})")
        if r["steps_per_s"] < 0.4 * med_sps:
            why.append(f"speed {r['steps_per_s']:.0f} steps/s (median {med_sps:.0f})")
        if why:
            say("ALERT", f"upd {upd}: " + "; ".join(why))
        if r["entropy"] < 0.2:
            say("RESET?", f"upd {upd}: entropy collapsed to {r['entropy']:.3f}")
            reset_at = reset_at or upd
    rows_seen = len(rows)
    return reset_at


def check_evals() -> None:
    global evals_seen, best_heldout, best_idx
    path = os.path.join(R, "eval.json")
    if not os.path.exists(path):
        return
    try:
        hist = json.load(open(path))
    except json.JSONDecodeError:
        return
    for j in range(evals_seen, len(hist)):
        e = hist[j]
        hs = e.get("heldout_score")
        if hs is None:  # evaluation from before the held-out tables existed (e.g. resumed v2)
            continue
        new = hs is not None and hs > best_heldout
        if new:
            best_heldout, best_idx = hs, j
        parts = []
        for k, v in e["lineups"].items():
            if v.get("heldout"):
                parts.append(f"{k} {v['rl_minus_baseline']:+.0f}±{v.get('rl_minus_baseline_ci95', 0):.0f}")
        rg = [v["range_inputs_gain"] for v in e["lineups"].values() if "range_inputs_gain" in v]
        say("EVAL", f"upd {e['update']}: held-out vs equity bot {hs:+.1f} [{', '.join(parts)}] · in-distribution "
                    f"{e['mean_rl_bb100']:+.1f}" + (f" · range inputs {statistics.mean(rg):+.1f}" if rg else "")
                    + (" · NEW BEST" if new else f" · best {best_heldout:+.1f}"))
        if run:
            d = {"eval/update": e["update"], "eval/heldout_score": hs, "eval/heldout_min": e.get("heldout_min"),
                 "eval/in_distribution_mean": e["mean_rl_bb100"]}
            for k, v in e["lineups"].items():
                for key in ("rl_minus_baseline", "style_inputs_gain", "range_inputs_gain"):
                    if key in v:
                        d[f"eval/{k}/{key}"] = v[key]
            run.log(d)
        if j - best_idx >= 4 and (j - best_idx) % 4 == 0:
            say("STALL", f"no new best held-out in {j - best_idx} evaluations (best {best_heldout:+.1f})")
    evals_seen = len(hist)


def keep_checkpoint() -> None:
    global ckpt_mtime
    src = os.path.join(R, "latest.pt")
    if not os.path.exists(src):
        return
    m = os.path.getmtime(src)
    if m <= ckpt_mtime:
        return
    time.sleep(3)  # let the trainer finish writing
    try:
        import torch

        upd = int(torch.load(src, map_location="cpu", weights_only=False)["update"])
    except Exception:  # noqa: BLE001
        return
    shutil.copy2(src, os.path.join(R, "keep", f"upd{upd:05d}.pt"))
    ckpt_mtime = m


# ---------------------------------------------------------------- main loop
launch(args.first_resume)
while True:
    time.sleep(args.poll)
    reset_at = None
    try:
        keep_checkpoint()
        reset_at = check_rows()
    except Exception as exc:  # noqa: BLE001
        say("INFO", f"watcher error (rows) {exc!r}")
    try:
        check_evals()
    except Exception as exc:  # noqa: BLE001
        say("INFO", f"watcher error (evals) {exc!r}")
    if os.path.exists(os.path.join(R, "STOP")):
        say("INFO", "STOP file: asking the trainer to save and stop")
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait()
        keep_checkpoint()
        say("DONE", "stopped on request")
        break
    if reset_at is not None:
        ck = keep_before(reset_at)
        if resets >= args.max_resets or ck is None:
            say("DEAD", f"needs a reset at upd {reset_at} but {'no earlier checkpoint' if ck is None else 'reset limit reached'}")
            kill_hard()
            break
        resets += 1
        lr /= 2
        kill_hard()
        say("RESET", f"rolled back to {os.path.basename(ck)}, learning rate {lr:.1e} (reset {resets}/{args.max_resets})")
        launch(ck)
        continue
    code = proc.poll()
    if code is None:
        continue
    keep_checkpoint()
    check_evals()
    if os.path.exists(os.path.join(R, "final_eval.json")):
        say("DONE", f"finished; best held-out {best_heldout:+.1f}")
        break
    latest = os.path.join(R, "latest.pt")
    if restarts < args.max_restarts and deadline - time.time() > 15 * 60 and os.path.exists(latest):
        restarts += 1
        say("RESTART", f"trainer exited with code {code}; resuming from latest.pt ({restarts}/{args.max_restarts})")
        launch(latest)
        continue
    say("DEAD", f"trainer exited with code {code}; not restarting")
    break
if run:
    run.finish()
