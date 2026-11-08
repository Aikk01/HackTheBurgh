"""The Shark learns you: a simulated player with a human leak in your seat, and the Shark reading it (for the replay).

    python scripts/build_sim_shark.py [--sessions 100] [--workers 14]   # -> data/demo/sim_you_shark.json

The player in your seat has a common human leak, "fit or fold": normal before the flop, but after it gives up to
almost any bet (the pro model with a fold tilt after the flop when facing a bet).  Nobody is told.  The live Mirror
table plays on (lab_api.MIRROR_TABLE, warm reads for the bots, none for you), and the Shark (agents.ReaderAgent) reads
the models as they learn, hand by hand, exactly as at the live table.

One sitting is 60 hands: 50 for the Shark to learn you, then 10 where it can use what it learned.  A single sitting
has only a spot or two where the Shark is heads-up with you after the flop, so the numbers come from ``--sessions``
sittings with different cards, each played twice: with the Shark reading the table and with the same Shark not
reading it.  Compared: how often it bets when heads-up with you after the flop, how often you fold to its bets, the
chips you lose to it, in hands 1-10 and 51-60.  The replay shows one sitting: the first one (in seed order) where you
meet the Shark both before hand 50 and after it, picked for having those meetings, not for its result.
"""
from __future__ import annotations

import argparse
import copy
from multiprocessing import Pool
import json
import os
import random
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import build_replay as BR  # noqa: E402
from pokersim import lab_api  # noqa: E402
from pokersim import next_move as NM  # noqa: E402
from pokersim import you_model as Y  # noqa: E402
from pokersim.agents import ProAgent, make_agent, safe_act  # noqa: E402
from pokersim.engine import IllegalAction, Seat, Table  # noqa: E402
from pokersim.lab_api import MIRROR_TABLE  # noqa: E402

ME, SHARK = "You", "Shark"
YOU_TEXT = "gives up to almost any bet after the flop"
LEARN, SITTING = 50, 60  # the Shark learns you over the first 50 hands of a 60-hand sitting


class FitOrFold(ProAgent):
    """The leak in your seat: the pro model, with a strong fold tilt after the flop when facing a bet."""

    def act(self, obs):
        from pokersim.next_move import sample_action

        facing = float((obs.get("legal_actions") or {}).get("call") or 0) > 0
        tilt = (4.0, 0.0, -1.0) if obs["street"] != "preflop" and facing else (0.0, 0.0, 0.0)
        return sample_action(obs, self.rng, pluribus=self.pluribus, temp=self.temp, tilt=tilt)


def money(x):
    return f"{x:g}"


def simulate(n, seed, reads, record):
    """Play ``n`` hands; ``record(h)`` says which hands to keep frames for.  Returns the measures and the frames."""
    Y.PLAYERS_DIR = tempfile.mkdtemp(prefix="sim_shark_players_")
    rng = random.Random(seed)
    st = BR.Stand()
    st.agents = {}
    warm = json.load(open(os.path.join(ROOT, "data", "demo", "warm_reads.json")))
    seats = []
    for i, (spec, name) in enumerate([("fit-or-fold", ME)] + MIRROR_TABLE):
        ag = FitOrFold(seed=rng.randrange(1 << 30)) if i == 0 else make_agent(spec, seed=rng.randrange(1 << 30))
        ag.name = name
        if i and spec in warm["by_agent"]:  # the bots are known from earlier hands, as at the live table; you are not
            m = st.watch.model(name)
            m.counts = copy.deepcopy(warm["by_agent"][spec])
            m.warm_hands = int(warm["hands"])
            m.range_obs = [list(x) for x in warm["range_obs"].get(spec, [])]
        if ag.kind == "reader" and reads:
            ag.reads_provider = lambda obs: lab_api.table_reads(st, obs["hero"]["index"])
        st.agents[i + 1] = ag
        seats.append(Seat(seat_no=i + 1, name=name, stack=200.0, agent=ag, kind=ag.kind, buy_in=200.0))
    table = Table(seats, small_blind=1.0, big_blind=2.0, rng=random.Random(rng.randrange(1 << 30)))
    st._seat_of = lambda i: table.seat_for_player(i).seat_no

    spots, folds, lost, series, meets = [], [], [], [], {}
    seen = [0, 0, 0.0]  # your decisions after the flop facing a bet: n, folds, people's fold chance summed
    snapshot = None
    for h in range(1, n + 1):
        for s in table.seats:
            s.stack = 200.0
        hand = table.new_hand()
        st.hand = hand
        k = next(i for i, p in enumerate(hand.players) if p.name == ME)
        st.human_seats = [st._seat_of(k)]
        disp = BR.display([p.name for p in hand.players], k)
        keep = record(h)
        log, frames, met = [], [], False
        last_bettor, guard = None, 0
        while hand.status == "betting" and hand.actor is not None and guard < 400:
            guard += 1
            actor = hand.actor
            if keep:
                frames.append({"st": BR.state_of(hand, k, log, h, disp), "pred": BR.rename_pred(lab_api.predict(st), disp)})
            obs, pred, _ = st.watch.before(hand, actor, human=(actor == k))
            name = hand.players[actor].name
            action = safe_act(st.agents[st._seat_of(actor)], obs)
            la = obs.get("legal_actions") or {}
            to_call = float(la.get("call") or 0.0)
            post = obs["street"] != "preflop"
            if name == SHARK and post and not hand.players[k].folded and la.get("raise"):
                met = True
                if sum(not p.folded for p in hand.players) == 2:  # heads-up with you: its read is about you alone
                    spots.append([h, int(action["action"] not in ("fold", "check", "call"))])
            if actor == k and to_call > 0 and post:
                seen[0] += 1
                seen[1] += action["action"] == "fold"
                seen[2] += float(NM.predict(obs, k, with_cards=False)[0])
                if last_bettor == SHARK:
                    folds.append([h, int(action["action"] == "fold")])
            try:
                res = hand.act(actor, action["action"], action.get("amount"))
            except IllegalAction:
                res = hand.act(actor, "call")
                action = {"action": "call"}
            st.watch.after(hand, actor, actor == k, obs, pred, res)
            if action["action"] not in ("fold", "check", "call"):
                last_bettor = name
            txt = ("folds" if action["action"] == "fold" else f"calls {money(to_call)}" if action["action"] == "call" and to_call > 0
                   else "checks" if action["action"] in ("check", "call") else f"raises to {money(float(action.get('amount') or 0))}")
            log.append({"text": f"{disp[actor]} {txt}", "player": actor})
        if hand.status not in ("finished", "incomplete"):
            hand.finalize()
        pay = hand.to_dict(viewer=None, reveal=True).get("payouts") or []
        shark_won = any(p.name == SHARK and p.index < len(pay) and float(pay[p.index] or 0) > 0 for p in hand.players)
        mine = hand.players[k].stack - 200.0
        lost.append([h, round(-mine, 1) if shark_won and mine < 0 else 0.0])
        if keep:
            frames.append({"st": BR.state_of(hand, k, log, h, disp), "pred": BR.rename_pred(lab_api.predict(st), disp)})
            if met:
                meets[h] = frames
        series.append([h, round(seen[1] / max(seen[0], 1), 3), round(seen[2] / max(seen[0], 1), 3), round(sum(x[1] for x in lost), 1)])
        if h == LEARN:
            you = st.watch.model(ME, True)
            snapshot = {"counts": copy.deepcopy(you.counts), "habits": you.habits(top=2),
                        "opponents": {nm: {"counts": copy.deepcopy(st.watch.model(nm).counts), "habits": st.watch.model(nm).habits(top=1)}
                                      for _, nm in MIRROR_TABLE}}
        table.finish_hand()

    return {"spots": spots, "folds": folds, "lost": lost, "series": series, "meets": meets, "snapshot": snapshot}


def sitting(seed):
    """One 60-hand sitting, with and without the Shark's reads (no frames): the raw rows for the comparison."""
    BR.HIM, BR.MINE = ME, True
    on, off = simulate(SITTING, seed, True, lambda h: False), simulate(SITTING, seed, False, lambda h: False)
    met = sorted({r[0] for r in on["spots"]} | {r[0] for r in on["folds"]})
    return seed, {k: on[k] for k in ("spots", "folds", "lost")}, {k: off[k] for k in ("spots", "folds", "lost")}, met


def summarise(runs, lo, hi):
    """Pooled over sittings, hands lo..hi: the Shark's bet rate heads-up with you after the flop, your fold rate to
    its bets, and chips you lost to it per sitting (with a 95% interval over sittings)."""
    spots = [r[1] for run in runs for r in run["spots"] if lo <= r[0] <= hi]
    folds = [r[1] for run in runs for r in run["folds"] if lo <= r[0] <= hi]
    lost = np.array([sum(r[1] for r in run["lost"] if lo <= r[0] <= hi) for run in runs])
    rate = lambda x: {"rate": round(float(np.mean(x)), 3) if x else None, "n": len(x)}  # noqa: E731
    return {"shark_bets": rate(spots), "you_fold": rate(folds),
            "lost_per_sitting": [round(float(lost.mean()), 2), round(float(1.96 * lost.std(ddof=1) / np.sqrt(len(lost))), 2)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        res = sorted(pool.imap_unordered(sitting, [a.seed + i for i in range(a.sessions)]))
    on, off = [r[1] for r in res], [r[2] for r in res]
    pick = next((r[0] for r in res if any(h <= LEARN for h in r[3]) and any(h > LEARN for h in r[3])), res[0][0])
    BR.HIM, BR.MINE = ME, True
    show = simulate(SITTING, pick, True, lambda h: True)  # the same sitting again, now with frames
    early = [h for h in sorted(show["meets"]) if h <= LEARN][:2]
    late = [h for h in sorted(show["meets"]) if h > LEARN][:3]
    shark = {"you_style": YOU_TEXT, "learn_hands": LEARN, "sitting_hands": SITTING, "sessions": a.sessions,
             "with_reads": {"first10": summarise(on, 1, 10), "last10": summarise(on, LEARN + 1, SITTING)},
             "without_reads": {"first10": summarise(off, 1, 10), "last10": summarise(off, LEARN + 1, SITTING)},
             "shown_sitting": pick, "series": show["series"][:LEARN]}
    snap = show["snapshot"]
    out = {"player": {"label": "You, with a leak", "venue": "simulated", "hands": LEARN, "mine": True, "sim": True, "shark": True,
                      "style": "shark", "style_text": YOU_TEXT,
                      "picked": "one 60-hand sitting: the first (in seed order) where you meet the Shark before and after hand 50",
                      "note": f"simulated; the numbers come from {a.sessions} sittings, each with and without the Shark's reads"},
           "habits": snap["habits"], "curve": [], "summary": {},
           "hands": [{"hand_no": h, "phase": "early", "frames": show["meets"][h]} for h in early] +
                    [{"hand_no": h, "phase": "late", "frames": show["meets"][h]} for h in late],
           "shark": shark, "counts": snap["counts"], "opponents": snap["opponents"]}
    path = os.path.join(ROOT, "data", "demo", "sim_you_shark.json")
    s = json.dumps(out, separators=(",", ":"), default=lambda x: round(float(x), 4))
    with open(path + ".tmp", "w") as f:
        f.write(s)
    os.replace(path + ".tmp", path)
    print(f"{a.sessions} sittings of {SITTING} hands, x2; shown sitting {pick}: early {early}, late {late}; {len(s) / 1e6:.1f} MB")
    for arm, d in (("reading", shark["with_reads"]), ("not reading", shark["without_reads"])):
        print(f"  {arm:11s} hands 1-10: {d['first10']} · hands 51-60: {d['last10']}")
    s50 = show["series"][LEARN - 1]
    print(f"  shown sitting, by hand {LEARN}: you fold {s50[1]:.0%} after the flop facing a bet (people {s50[2]:.0%})")


if __name__ == "__main__":
    main()
