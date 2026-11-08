"""Precompute the Poker Mirror replay of a real unseen player (data/demo/replay.json, served at /api/lab/replay).

    python scripts/build_replay.py [--showcase 7] [--player data/demo/replay_player.json --hands data/demo/replay_hands.phhs]
    python scripts/build_replay.py --me You     # your own hands at the live table -> data/demo/replay_you.json

The default player is one of the 30 unseen players of scripts/eval_unseen_players.py (no model was trained on them),
picked for a strong personal style so the personal model's adaptation is visible; data/demo/replay_player.json says
so, and the page says it too.  ``--player data/demo/reserved_player.json --hands data/demo/reserved_hands.phhs`` replays
the reserved player instead.

All of the player's recorded hands are replayed in time order through the same engine and the same model code as the
live table (pokersim.lab_api.predict on a stand-in game).  The personal models ("This player") learn him, and any
opponent who recurs, as the hands go by, like a poker tracker following a player across many online tables.

Output:
* ``curve``: for every one of his decisions, whether each model's prediction (made before the move) was right:
  his personal model, people in general (next-move model), and "his usual move" — for the learning-curve chart.
* ``hands``: full frames (table state + every model value, exactly the /api/lab/predict shape) for the showcase
  hands, chosen by a fixed rule: hands from the last half (after he has been studied), where his cards were revealed
  at showdown and at least one of his moves went against what people in general would do (the next-move model's top
  guess), evenly spaced.  Whether his own model caught those moves is shown, right or wrong.
* ``habits``: what his model learned about him by the end, in plain words.  His cards
  are taken from that showdown and shown from the start, so the coach (RL) can advise; the personal model never sees
  cards (public information only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pokersim import cards as C  # noqa: E402
from pokersim import lab_api  # noqa: E402
from pokersim import next_move as NM  # noqa: E402
from pokersim import phh  # noqa: E402
from pokersim import you_model as Y  # noqa: E402
from pokersim.opponent_model import StyleTracker  # noqa: E402
from pokersim.phh import parse_action  # noqa: E402
from pokersim.replay import build_hand  # noqa: E402

OUT = os.path.join(ROOT, "data", "demo", "replay.json")
HIM = "Unseen player"
MINE = False  # replaying a player of our own live table: the bots keep their names


class Stand:
    """The parts of GameSession that lab_api.predict reads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.hand = None
        self.human_seats = []
        self.styles = StyleTracker()
        self.watch = Y.TableWatch()
        self.last_lab = None

    def _seat_of(self, i):
        return i + 1


def his_index(rec, name):
    return rec.players.index(name)


def shown_cards(rec, k):
    for a in rec.actions:
        if a.startswith(f"p{k + 1} sm ") and not a.endswith(" -"):
            cs = a.split()[2]
            if "?" not in cs and len(cs) == 4:
                return cs
    for a in rec.actions:  # live-table records deal every card face up
        if a.startswith(f"d dh p{k + 1} "):
            cs = a.split()[3] if len(a.split()) > 3 else ""
            if "?" not in cs and len(cs) == 4:
                return cs
    return None


def n_decisions(rec, k):
    return sum(1 for a in rec.actions if a.startswith(f"p{k + 1} ") and a.split()[1] in ("f", "cc", "cbr"))


def display(names, k_him):
    return {i: (HIM if i == k_him else names[i] if MINE else f"Seat {i + 1}") for i in range(len(names))}


def state_of(hand, viewer, log, hand_no, disp):
    st = hand.to_dict(viewer=viewer, reveal=False)
    for p in st["players"]:
        p["seat"] = p["index"] + 1
        p["name"] = disp[p["index"]]
        p["is_human"] = p["index"] == viewer
    return {"active": True, "mode": "replay", "hand_no": hand_no, "hand": st, "viewer": viewer, "log": log[-30:],
            "table": {"small_blind": hand.config.blinds[0] if hand.config.blinds else None,
                      "big_blind": hand.config.min_bet}}


def rename_pred(pred, disp):
    for p in pred.get("players") or []:
        p["name"] = disp.get(p["index"], p["name"])
    return pred


def money(x):
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def run_hand(rec, stand, name, hand_no, frames_on, against=None):
    k = his_index(rec, name)
    sc = shown_cards(rec, k)
    actions = list(rec.actions)
    if sc and frames_on:  # hindsight: his cards from the showdown, from the start (for the coach only)
        actions = [f"d dh p{k + 1} {sc}" if a.startswith(f"d dh p{k + 1} ") else a for a in actions]
    rec.actions = actions
    hand = build_hand(rec)
    stand.hand = hand
    stand.human_seats = [k + 1]
    disp = display(rec.players, k)
    hand.post_forced_bets()
    log, frames, curve = [], [], []

    def frame(check=None):
        f = {"st": state_of(hand, k, log, hand_no, disp), "pred": rename_pred(lab_api.predict(stand), disp)}
        if check:
            f["check"] = check
        frames.append(f)

    for raw in actions:
        try:
            a = parse_action(raw)
        except ValueError:
            continue
        if a is None:
            continue
        if a.is_dealer:
            if a.verb == "dh" and len(a.args) >= 2:
                hand.deal_hole(int(a.args[0][1:]) - 1, C.parse_cards(a.args[1]))
            elif a.verb == "db":
                hand.deal_board(C.parse_cards(a.args[0]) if a.args else [])
            continue
        p = a.player
        if p is None or p >= hand.n:
            continue
        if a.verb in ("f", "cc", "cbr"):
            if hand.status == "dealing":
                hand._begin_street()
            if hand.status != "betting":
                continue
            if frames_on:
                frame()
            obs, pred, _ = stand.watch.before(hand, p, human=(p == k))
            prior = NM.predict(obs, p, with_cards=False) if p == k else None
            usual = stand.watch.model(rec.players[p], p == k).usual_group(obs, p) if p == k else None
            to_call = float((obs.get("legal_actions") or {}).get("call") or 0.0)
            if a.verb == "f":
                hand.fold(p)
                res, txt = {"action": "fold"}, "folds"
            elif a.verb == "cc":
                hand.check_call(p)
                res, txt = ({"action": "call"}, f"calls {money(to_call)}") if to_call > 0 else ({"action": "check"}, "checks")
            else:
                if not a.args:
                    continue
                amt = float(a.args[0])
                hand.bet_raise_to(p, amt)
                res, txt = {"action": "raise"}, f"raises to {money(amt)}"
            stand.watch.after(hand, p, p == k, obs, pred, res)
            log.append({"text": f"{disp[p]} {txt}", "player": p})
            if p == k:
                g = Y.group_of_action(res)
                if against is not None and prior is not None and int(Y.GROUP_OF[int(np.argmax(prior))]) != g:
                    against.add(hand_no)
                hit = int(Y.GROUP_OF[int(np.argmax(pred))]) == g
                curve.append([int(hit), int(int(Y.GROUP_OF[int(np.argmax(prior))]) == g) if prior is not None else 0,
                              int(usual == g)])
                if frames_on:
                    gp = int(Y.GROUP_OF[int(np.argmax(pred))])
                    frames[-1]["check"] = {"pred": Y.GROUPS[gp], "actual": Y.GROUPS[g], "hit": hit}
        elif a.verb == "sm":
            try:
                hand.show_muck(p, None if not a.args else (list(hand.players[p].hole) if a.args[0] == "-" else C.parse_cards(a.args[0])))
            except Exception:  # noqa: BLE001
                pass
    try:
        hand.finalize()
    except Exception:  # noqa: BLE001
        pass
    stand.styles.add_hand(hand)
    if frames_on:
        frame()
    return frames, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--showcase", type=int, default=7)
    ap.add_argument("--player", default=os.path.join(ROOT, "data", "demo", "replay_player.json"))
    ap.add_argument("--hands", default=None)
    ap.add_argument("--me", default=None, help="a human player of the live table (their data/players/<name>/hands.phhs)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global HIM, MINE
    if a.me:
        info = {"venue": "pokersim live table", "player": a.me, "label": a.me, "mine": True,
                "picked": "your own hands at this table"}
        hands = a.hands or os.path.join(Y.PlayerModel(a.me)._dir(), "hands.phhs")
        out_path = a.out or os.path.join(ROOT, "data", "demo", "replay_you.json")
    else:
        info = json.load(open(a.player))
        hands = a.hands or os.path.join(ROOT, "data", "demo", "replay_hands.phhs")
        out_path = a.out or OUT
    a.hands = hands
    HIM, MINE = info.get("label", HIM), bool(info.get("mine"))
    Y.PLAYERS_DIR = tempfile.mkdtemp(prefix="replay_players_")  # never touch the real player logs
    name = info["player"]
    recs = phh.load_file(a.hands)
    n = len(recs)
    # pass 1 (no frames): which of his hands had a move against what people in general would do
    against: set = set()
    probe = Stand()
    for i, rec in enumerate(phh.load_file(a.hands)):
        try:
            run_hand(rec, probe, name, i + 1, False, against)
        except Exception:  # noqa: BLE001
            continue
    pool = [i for i in range(n // 2, n) if (i + 1) in against and shown_cards(recs[i], his_index(recs[i], name))]
    pick = set(pool[j] for j in np.linspace(0, len(pool) - 1, min(a.showcase, len(pool))).round().astype(int)) if pool else set()
    Y.PLAYERS_DIR = tempfile.mkdtemp(prefix="replay_players_")  # the real pass starts knowing nothing about him
    stand = Stand()
    curve, showcase = [], []
    for i, rec in enumerate(recs):
        try:
            frames, c = run_hand(rec, stand, name, i + 1, i in pick)
        except Exception as exc:  # noqa: BLE001 - a malformed record is skipped, not fatal
            print(f"hand {i + 1}: skipped ({exc})")
            continue
        for row in c:
            curve.append([i + 1] + row)
        if i in pick:
            showcase.append({"hand_no": i + 1, "frames": frames})
    C_ = np.array([r[1:] for r in curve], float)
    him = stand.watch.model(name, True)
    out = {"player": {"label": HIM, "venue": info["venue"], "hands": n, "decisions": len(curve),
                      "picked": info.get("picked", ""), "mine": bool(info.get("mine")),
                      "note": "real online hands from many different tables; no model was trained on this player"},
           "habits": him.habits(top=3),
           "curve": curve, "summary": {"personal": C_[:, 0].mean(), "people": C_[:, 1].mean(), "usual": C_[:, 2].mean()},
           "showcase_rule": "last half of his hands, his cards revealed at showdown, at least one move against what people in general would do, evenly spaced",
           "hands": showcase}
    s = json.dumps(out, separators=(",", ":"), default=lambda x: round(float(x), 4))
    with open(out_path, "w") as f:
        f.write(s)
    print(f"{n} hands, {len(curve)} decisions, {len(showcase)} showcase hands, {len(s) / 1e6:.1f} MB; "
          f"accuracy personal {C_[:, 0].mean():.1%} · people {C_[:, 1].mean():.1%} · usual {C_[:, 2].mean():.1%}")


if __name__ == "__main__":
    main()
