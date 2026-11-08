"""The Shark learns you: a simulated player with a human leak in your seat, and the Shark reading it (for the replay).

    python scripts/build_sim_shark.py [--hands 400] [--seed 41]   # -> data/demo/sim_you_shark.json

The player in your seat has a common human leak, "fit or fold": normal before the flop, but after it gives up to
almost any bet (the pro model with a fold tilt after the flop when facing a bet).  Nobody is told.  The live Mirror
table plays on (lab_api.MIRROR_TABLE, warm reads for the bots, none for you), and the Shark (agents.ReaderAgent) reads
the models as they learn, hand by hand, exactly as at the live table.

What reading you is worth is measured, not shown: the same seed is played twice for ``--hands`` hands, once with the
Shark reading the table and once with the same Shark not reading it, and the two are compared: how often it bets when
it is heads-up with you after the flop (where its read is about you alone), how often you fold to its bets, and the
chips you lose to it.  The replay itself shows the first times you meet
the Shark (it does not know you yet), the fast-forward to hand 50 (its read of you building up), then the next times
you meet it.  "Now you play" carries the models as they stood after hand 50.
"""
from __future__ import annotations

import argparse
import copy
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
LEARN = 50  # the Shark learns you over the first 50 hands


class FitOrFold(ProAgent):
    """The leak in your seat: the pro model, with a strong fold tilt after the flop when facing a bet."""

    def act(self, obs):
        from pokersim.next_move import sample_action

        facing = float((obs.get("legal_actions") or {}).get("call") or 0) > 0
        tilt = (2.5, 0.0, -0.5) if obs["street"] != "preflop" and facing else (0.0, 0.0, 0.0)
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

    def rate(rows, lo):
        x = [r[1] for r in rows if r[0] > lo]
        return {"rate": round(float(np.mean(x)), 3) if x else None, "n": len(x)}

    late = [r[1] for r in lost if r[0] > LEARN]
    return {"bets_into_you": rate(spots, LEARN), "you_fold_to_shark": rate(folds, LEARN),
            "lost_per_10": round(10.0 * sum(late) / max(len(late), 1), 2), "series": series, "meets": meets, "snapshot": snapshot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--show", type=int, default=2, help="meetings shown before and after the Shark learns you")
    a = ap.parse_args()
    BR.HIM, BR.MINE = ME, True
    on = simulate(a.hands, a.seed, True, lambda h: h <= LEARN or LEARN < h <= LEARN + 60)
    off = simulate(a.hands, a.seed, False, lambda h: False)
    early = [h for h in sorted(on["meets"]) if h <= LEARN][: a.show]
    late = [h for h in sorted(on["meets"]) if h > LEARN][: a.show + 1]
    shark = {"you_style": YOU_TEXT, "learn_hands": LEARN, "run_hands": a.hands,
             "with_reads": {k: on[k] for k in ("bets_into_you", "you_fold_to_shark", "lost_per_10")},
             "without_reads": {k: off[k] for k in ("bets_into_you", "you_fold_to_shark", "lost_per_10")},
             "series": on["series"][:LEARN]}
    snap = on["snapshot"]
    out = {"player": {"label": "You, with a leak", "venue": "simulated", "hands": LEARN, "mine": True, "sim": True, "shark": True,
                      "style": "shark", "style_text": YOU_TEXT,
                      "picked": "the first times you meet the Shark, then the next times after it has watched you for 50 hands",
                      "note": "simulated; the comparison comes from two runs of the same seed, with and without the Shark's reads"},
           "habits": snap["habits"], "curve": [], "summary": {},
           "hands": [{"hand_no": h, "phase": "early", "frames": on["meets"][h]} for h in early] +
                    [{"hand_no": h, "phase": "late", "frames": on["meets"][h]} for h in late],
           "shark": shark, "counts": snap["counts"], "opponents": snap["opponents"]}
    path = os.path.join(ROOT, "data", "demo", "sim_you_shark.json")
    s = json.dumps(out, separators=(",", ":"), default=lambda x: round(float(x), 4))
    with open(path + ".tmp", "w") as f:
        f.write(s)
    os.replace(path + ".tmp", path)
    print(f"{a.hands} hands x2; shown early {early}, late {late}; {len(s) / 1e6:.1f} MB")
    for key, label in (("bets_into_you", "Shark bets, heads-up with you after the flop"), ("you_fold_to_shark", "you fold to its bets")):
        print(f"  {label}: reading {on[key]} · not reading {off[key]}")
    print(f"  chips lost to the Shark per 10 hands (after hand {LEARN}): reading {on['lost_per_10']} · not reading {off['lost_per_10']}")
    s50 = on["series"][LEARN - 1]
    print(f"  by hand {LEARN}: you fold {s50[1]:.0%} after the flop facing a bet (people {s50[2]:.0%})")


if __name__ == "__main__":
    main()
