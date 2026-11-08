"""What your move cost against the coach: the Poker Mirror's "If you had followed the coach" box.

At your first move in a hand that differs from the RL coach's pick (fold / check-call / bet-raise, as its bar on
screen shows it), the hand is copied.  The deck is shuffled once per hand, so the copy holds the same cards still to
come: everyone's cards and the rest of the board are what really happened.  From that spot each of the three moves is
played out many times: the coach takes your seat for the rest of the hand (playing its own mix, as it was trained and
tested), and every other player is their own bot with their own cards, choosing afresh each time (the Shark reads the
copy, not the real hand).

That gives, for each move, what it wins on average in this spot with these cards: in reinforcement-learning terms a
Monte Carlo estimate of Q(spot, move) under the coach's policy.  The coach's own value of the spot is its mix of them,
V = sum of p(move) x Q(move), and what your move cost is V - Q(your move), the negative of its advantage.  So:

- only one move is scored, and every move is finished the same way, so errors do not pile up from move to move;
- your move counts as partly right: cost = (1 - p(yours)) x (Q of the coach's other moves, mixed) - same for yours,
  so a move the coach makes 40% of the time is never called simply wrong;
- it can come out either way, and the box says so when your move did better.

A guess, not a certainty: the others' bots stand in for the people, and the cards are fixed as they fell.
"""
from __future__ import annotations

import copy
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .agents import safe_act
from .engine import Hand, IllegalAction

GROUPS = ("fold", "check/call", "bet/raise")


def group_of(action: str) -> int:
    return 0 if action == "fold" else 1 if action in ("check", "call") else 2


def coach_mix(obs: Dict[str, Any], styles: Dict[int, Dict[str, Any]], policy: Any) -> Tuple[List[float], List[Optional[Dict[str, Any]]]]:
    """The RL coach's chance of fold / check-call / bet-raise (None where illegal) and the move for each: call or
    check, and for a raise its likeliest size.  Seeded like rl_policy.recommend, so it is the bar that was on screen."""
    from .rl_features import action_targets, encode

    rng = random.Random(f"{obs.get('hand_id')}|{obs['street']}|{len(obs.get('history') or [])}|0")
    x = encode(obs, styles, rng=rng, use_ranges=policy.meta.get("use_ranges", True))
    mask, targets = action_targets(obs)
    pr, _ = policy.probs(x, mask)
    p = [float(pr[0]) if mask[0] else None, float(pr[1]), float(pr[2:][mask[2:]].sum()) if mask[2:].any() else None]
    raise_to = targets[2 + int(np.argmax(np.where(mask[2:], pr[2:], -1.0)))] if p[2] is not None else None
    return p, [targets[0] if mask[0] else None, targets[1], raise_to]


def coach_pick(p: List[Optional[float]], moves: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """The coach's top choice, as its bar on screen lights it."""
    g = max((k for k in range(3) if p[k] is not None), key=lambda k: p[k])
    return dict(moves[g] or {"action": "call"})


def play_out(snap: Hand, viewer: int, first: Dict[str, Any], agents: Dict[int, Any], styles: Dict[int, Dict[str, Any]],
             policy: Any, reads: Optional[Callable[[Hand, int], Optional[Dict[str, Any]]]] = None, n: int = 150,
             budget_s: float = 3.0, seed: int = 0) -> List[float]:
    """Play the copied hand ``snap`` (``viewer`` to act) from the move ``first`` to the end, up to ``n`` times or
    ``budget_s`` seconds, the coach in the viewer's seat afterwards; returns the viewer's net chips in each.  ``agents``
    maps every other player's index to their bot, copied with a fresh random stream (the live bots are never touched);
    ``reads`` lets a Shark read the copy being played."""
    from .rl_policy import act as coach_act

    rng = random.Random(seed)
    bots = {}
    for j, a in agents.items():
        b = copy.copy(a)
        b.rng = random.Random(rng.randrange(1 << 30))
        if hasattr(b, "last_read"):
            b.last_read = None
        bots[j] = b
    cur: Dict[str, Hand] = {}
    for b in bots.values():
        if getattr(b, "reads_provider", None) is not None:
            b.reads_provider = (lambda obs: reads(cur["h"], obs["hero"]["index"])) if reads else None
    start = snap.players[viewer].starting_stack
    nets: List[float] = []
    t0 = time.perf_counter()
    while len(nets) < n and (len(nets) < 20 or time.perf_counter() - t0 < budget_s):
        h = copy.deepcopy(snap)
        cur["h"] = h
        try:
            h.act(viewer, first["action"], first.get("amount"))
        except IllegalAction:
            h.act(viewer, "call")
        guard = 0
        while h.status == "betting" and h.actor is not None and guard < 80:
            guard += 1
            j = h.actor
            obs = h.observation(j)
            if j == viewer:
                a = coach_act(obs, styles, policy, rng)
            else:
                a = safe_act(bots[j], obs)
            try:
                h.act(j, a["action"], a.get("amount"))
            except IllegalAction:
                h.act(j, "call")
        nets.append(round(h.players[viewer].stack - start, 2))
        if first["action"] == "fold":  # nothing left to chance: one replay is the answer
            break
    return nets


def score_spot(snap: Hand, viewer: int, mine: int, p: List[Optional[float]], moves: List[Optional[Dict[str, Any]]],
               agents: Dict[int, Any], styles: Dict[int, Dict[str, Any]], policy: Any,
               reads: Optional[Callable[[Hand, int], Optional[Dict[str, Any]]]] = None, seed: int = 0) -> Dict[str, Any]:
    """Every legal move from the spot played out (``mine``: the group of the move you made, played at your size), and
    what your move cost against the coach's mix, in chips, with its standard error over the replays."""
    opts = []
    for g in range(3):
        if p[g] is None or moves[g] is None:
            continue
        nets = np.asarray(play_out(snap, viewer, moves[g], agents, styles, policy, reads=reads, seed=seed * 3 + g), dtype=float)
        se = float(nets.std(ddof=1) / np.sqrt(len(nets))) if len(nets) > 1 else 0.0
        opts.append({"group": g, "coach_p": round(p[g], 3), "mine": g == mine, "replays": len(nets),
                     "mean": round(float(nets.mean()), 1), "se": round(se, 2), "certain": bool(nets.min() == nets.max()),
                     "lo": round(float(np.percentile(nets, 10)), 1), "hi": round(float(np.percentile(nets, 90)), 1)})
    tot = sum(o["coach_p"] for o in opts) or 1.0
    you = next(o for o in opts if o["mine"])
    mix = sum(o["coach_p"] / tot * o["mean"] for o in opts)
    # cost = sum_g w_g Q_g - Q_you; your own weight cancels, leaving (1 - w_you) on your Q
    var = sum(((o["coach_p"] / tot) - (1.0 if o["mine"] else 0.0)) ** 2 * o["se"] ** 2 for o in opts)
    top = max(opts, key=lambda o: o["coach_p"])
    return {"options": opts, "coach_mix": round(mix, 1), "cost": round(mix - you["mean"], 1), "cost_se": round(float(np.sqrt(var)), 1),
            "vs_pick": round(top["mean"] - you["mean"], 1)}
