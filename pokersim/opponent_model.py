"""Opponent modelling for the Play table: who plays how, and what that means for the next decision.

Two layers, both learned from data rather than from game theory:

1. **Style estimate per opponent.**  Every finished hand adds to each player's counts of the five public
   style features used by Poker Mirror (VPIP, PFR, post-flop aggression, fold to a bet, went to showdown;
   see :func:`pokersim.mirror_data.style_counts_from_hand`).  Rates are shrunk towards the population of
   48k real online players (``data/mirror/styles.json``), and the four player types fitted on that
   population (tight/loose x aggressive/passive) give a Bayesian type posterior.  Bot labels are never
   read: the model only sees actions, like a human would.

2. **Exploitative advice.**  Each opponent's range for the current hand is narrowed by their estimated
   style and by what they did in this hand (a tight raiser has a strong range, a loose caller a wide one,
   an aggressive player bluffs more).  The advisor simulates hands from those ranges and compares the
   expected chips of fold / check / call / bet-raise, including the chance that everyone folds, which
   comes from each opponent's estimated fold-to-bet rate.  It aims to beat *these* opponents, not to play
   an unexploitable equilibrium.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import cards as C
from . import evaluator as E
from .agents import legal, round_to_unit
from .equity import estimate_equity, preflop_strength
from .mirror_data import STYLE_FEATURES, style_counts_from_hand

STYLES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mirror", "styles.json")
SHRINK = 20.0  # pseudo-observations pulling a rate towards the population mean (as in mirror_styles)

TYPE_JA = {"tight-aggressive": "タイト・アグレッシブ", "tight-passive": "タイト・パッシブ",
           "loose-aggressive": "ルース・アグレッシブ", "loose-passive": "ルース・パッシブ"}
TYPE_HINT_JA = {
    "tight-aggressive": "参加する手は少ないが、参加すると強気。ベット・レイズは強い手の可能性が高い。",
    "tight-passive": "参加する手が少なく受け身。ベットされると降りやすく、自分から攻めてきたら本物。",
    "loose-aggressive": "多くの手に参加し強気に攻める。ブラフも多いので、ベットされても降りすぎない。",
    "loose-passive": "多くの手に参加するが受け身でコールが多い。ブラフは効きにくく、強い手で価値を取る。",
}
FEATURE_JA = {"vpip": "参加率 VPIP", "pfr": "プリフロップレイズ率 PFR", "af": "攻撃率（フロップ・ターン）",
              "fold_to_bet": "ベットに降りる率", "wtsd": "ショーダウン到達率"}
ACTION_JA = {"fold": "フォールド", "check": "チェック", "call": "コール", "bet": "ベット", "raise": "レイズ", "all_in": "オールイン"}

# Used only if data/mirror/styles.json is missing: rough 2009 online cash-game averages.
_FALLBACK = {
    "names": ["tight-aggressive", "tight-passive", "loose-aggressive", "loose-passive"],
    "shares": [0.15, 0.29, 0.35, 0.21],
    "centroids": [{"vpip": 0.17, "pfr": 0.13, "af": 0.30, "fold_to_bet": 0.69, "wtsd": 0.30},
                  {"vpip": 0.22, "pfr": 0.11, "af": 0.26, "fold_to_bet": 0.64, "wtsd": 0.28},
                  {"vpip": 0.38, "pfr": 0.13, "af": 0.27, "fold_to_bet": 0.58, "wtsd": 0.30},
                  {"vpip": 0.41, "pfr": 0.07, "af": 0.24, "fold_to_bet": 0.56, "wtsd": 0.30}],
}


def _load_types() -> Dict[str, Any]:
    try:
        with open(STYLES_PATH, encoding="utf-8") as f:
            st = json.load(f)
        out = {k: st[k] for k in ("names", "shares", "centroids")}
        out["source"] = f"data/mirror/styles.json（実データ {st.get('players_used', '?'):,} 人から推定した {st.get('k')} タイプ）"
        return out
    except (OSError, KeyError, ValueError, TypeError):
        return dict(_FALLBACK, source="組み込みの既定値（styles.json なし）")


TYPES = _load_types()
POP_MEAN = {f: sum(s * c[f] for s, c in zip(TYPES["shares"], TYPES["centroids"])) for f in STYLE_FEATURES}


# ----------------------------------------------------------------- 1. style estimate per opponent
class StyleTracker:
    """Running per-player counts over the hands of one table session."""

    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, List[int]]] = {}
        self.hands: Dict[str, int] = {}

    def reset(self) -> None:
        self.counts.clear()
        self.hands.clear()

    def add_hand(self, hand: Any, only: Optional[set] = None) -> None:
        for name, d in style_counts_from_hand(hand).items():
            if only is not None and name not in only:
                continue
            c = self.counts.setdefault(name, {f: [0, 0] for f in STYLE_FEATURES})
            for f, (num, den) in d.items():
                c[f][0] += num
                c[f][1] += den
            self.hands[name] = self.hands.get(name, 0) + 1

    def estimate(self, name: str) -> Dict[str, Any]:
        return estimate_style(self.counts.get(name, {f: [0, 0] for f in STYLE_FEATURES}), self.hands.get(name, 0))


def estimate_style(counts: Dict[str, Sequence[int]], hands: int) -> Dict[str, Any]:
    rates, raw = {}, {}
    for f in STYLE_FEATURES:
        num, den = counts.get(f, (0, 0))
        rates[f] = (num + SHRINK * POP_MEAN[f]) / (den + SHRINK)
        raw[f] = {"num": num, "den": den, "rate": (num / den) if den else None}
    # Bayes: type prior = population share, likelihood = binomial counts under each type's centre
    logs = []
    for share, cen in zip(TYPES["shares"], TYPES["centroids"]):
        lp = math.log(max(share, 1e-9))
        for f in STYLE_FEATURES:
            num, den = counts.get(f, (0, 0))
            p = min(max(cen[f], 0.01), 0.99)
            lp += num * math.log(p) + (den - num) * math.log(1 - p)
        logs.append(lp)
    m = max(logs)
    w = [math.exp(x - m) for x in logs]
    tot = sum(w)
    post = [x / tot for x in w]
    best = max(range(len(post)), key=post.__getitem__)
    name = TYPES["names"][best]
    reliability = hands / (hands + 30.0)  # how much the estimate rests on this player rather than the population
    return {
        "hands": hands, "rates": {f: round(v, 4) for f, v in rates.items()}, "raw": raw,
        "type": name, "type_ja": TYPE_JA.get(name, name), "type_prob": round(post[best], 3),
        "type_posterior": {TYPES["names"][i]: round(p, 3) for i, p in enumerate(post)},
        "hint_ja": TYPE_HINT_JA.get(name, ""), "reliability": round(reliability, 2),
        "confidence_ja": "まだ推測（観察が少ない）" if hands < 10 else "やや信頼できる" if hands < 40 else "信頼できる",
    }


# ----------------------------------------------------------------- 2. ranges from style + this hand's actions
_PCT_CACHE: Dict[bool, Dict[Tuple[int, int], float]] = {}


def preflop_percentile(short_deck: bool = False) -> Dict[Tuple[int, int], float]:
    """Hand -> share of starting hands that are at least as strong (0 = best hand, 1 = worst)."""
    if short_deck not in _PCT_CACHE:
        combos = list(itertools.combinations(C.full_deck(short_deck), 2))
        scored = sorted(combos, key=lambda h: -preflop_strength(list(h)))
        n = len(scored)
        _PCT_CACHE[short_deck] = {tuple(sorted(h)): (i + 0.5) / n for i, h in enumerate(scored)}
    return _PCT_CACHE[short_deck]


def hand_actions(obs: Dict[str, Any], index: int) -> Dict[str, Any]:
    """What player ``index`` has done so far in this hand."""
    acts = [h for h in obs.get("history") or [] if h.get("player") == index]
    pre = [a["action"] for a in acts if a["street"] == "preflop"]
    post = [a for a in acts if a["street"] != "preflop"]
    return {
        "pre_raised": any(a in ("bet", "raise") for a in pre),
        "pre_called": any(a == "call" for a in pre),
        "pre_acted": bool(pre),
        "post_aggr": sum(1 for a in post if a["action"] in ("bet", "raise")),
        "post_calls": sum(1 for a in post if a["action"] == "call"),
    }


def range_spec(style: Dict[str, Any], acts: Dict[str, Any], position: str) -> Dict[str, Any]:
    r = style["rates"]
    if acts["pre_raised"]:
        top, label = r["pfr"], f"上位 {r['pfr']:.0%} の手（レイズした）"
    elif acts["pre_called"] or not acts["pre_acted"]:
        top, label = r["vpip"], f"上位 {r['vpip']:.0%} の手（{'コールした' if acts['pre_called'] else 'まだ行動前'}）"
    else:  # checked the big blind: any two cards
        top, label = 1.0, "どんな手もありうる（BBでチェック）"
    bluff = min(max(r["af"] * 0.6, 0.05), 0.45)  # chance an aggressive action comes without a made hand
    if acts["post_aggr"]:
        label += f"・フロップ以降に攻めた（ブラフの可能性 {bluff:.0%}）"
    elif acts["post_calls"]:
        label += "・フロップ以降にコールした"
    return {"top": max(min(top, 1.0), 0.03), "bluff": bluff, "post_aggr": acts["post_aggr"], "post_calls": acts["post_calls"],
            "label_ja": label}


def _accept(hole: Tuple[int, int], spec: Dict[str, Any], board: List[int], pct: Dict[Tuple[int, int], float],
            rng: random.Random, short: bool) -> bool:
    p = pct.get(tuple(sorted(hole)), 1.0)
    if p > spec["top"] and rng.random() > 0.03:  # outside the style range (small leak keeps it robust)
        return False
    if len(board) >= 3 and (spec["post_aggr"] or spec["post_calls"]):
        made = E.category_of(E.evaluate(list(hole) + board, short), short) >= 1  # a pair or better
        if not made:
            keep = spec["bluff"] if spec["post_aggr"] else 0.35  # aggressive with nothing = bluff; calls = draws
            if rng.random() > keep:
                return False
    return True


# ----------------------------------------------------------------- 3. the advice
def advise(obs: Dict[str, Any], styles: Dict[int, Dict[str, Any]], iterations: int = 700, seed: int = 0) -> Dict[str, Any]:
    la = legal(obs)
    hole = [C.parse_card(c) for c in obs["hero"]["cards"]]
    if len(hole) != 2 or any(c is None for c in hole) or obs.get("variant", "NT") == "PO":
        return {"available": False, "reason_ja": "この手札・ゲーム形式には対応していません。"}
    short = obs.get("variant") == "NS"
    board = [c for c in (C.parse_card(x) for x in obs["board"]) if c is not None]
    street = obs["street"]
    rng = random.Random(f"{obs.get('hand_id')}|{street}|{len(obs.get('history') or [])}|{seed}")
    pct = preflop_percentile(short)

    opps = [p for p in obs["players"] if not p["folded"] and not p["is_hero"]]
    specs, info = [], []
    for p in opps:
        st = styles.get(p["index"]) or estimate_style({}, 0)
        acts = hand_actions(obs, p["index"])
        spec = range_spec(st, acts, p["position"])
        # chance this opponent gives up if we bet now
        fold_p = (1.0 - st["rates"]["vpip"]) if (street == "preflop" and not acts["pre_called"] and not acts["pre_raised"]) else st["rates"]["fold_to_bet"]
        betting_now = float(p["bet"] or 0.0) > float(obs["hero"]["bet"] or 0.0) + 1e-9
        if acts["post_aggr"] or acts["pre_raised"] or betting_now:
            fold_p *= 0.4  # players who have just bet or raised give up far less often against a raise
        specs.append(spec)
        info.append({"index": p["index"], "name": p["name"], "position": p["position"], "bet": float(p["bet"] or 0.0),
                     "style": st, "range_ja": spec["label_ja"], "fold_to_our_bet": round(fold_p, 3)})

    hero_hole = [c for c in hole if c is not None]
    deck0 = [c for c in C.full_deck(short) if c not in set(hero_hole) | set(board)]
    need_board = 5 - len(board)
    samples = []  # (hero_score, [opp_final_scores], [opp_now_strength])
    for _ in range(iterations):
        deck = deck0[:]
        rng.shuffle(deck)
        used = set()
        opp_holes = []
        for spec in specs:
            got = None
            for _try in range(40):
                a, b = rng.sample(deck, 2)
                if a in used or b in used:
                    continue
                if _accept((a, b), spec, board, pct, rng, short):
                    got = (a, b)
                    break
            if got is None:  # range too narrow for the remaining deck: take any free cards
                free = [c for c in deck if c not in used]
                got = (free[0], free[1])
            used.update(got)
            opp_holes.append(got)
        rest = [c for c in deck if c not in used]
        full = board + rest[:need_board]
        hero_s = E.evaluate(hero_hole + full, short)
        finals = [E.evaluate(list(h) + full, short) for h in opp_holes]
        now = [(E.evaluate(list(h) + board, short) if len(board) >= 3 else -pct.get(tuple(sorted(h)), 1.0)) for h in opp_holes]
        samples.append((hero_s, finals, now))

    def share(hero_s: int, scores: Sequence[int]) -> float:
        if not scores:
            return 1.0
        best = max(scores)
        if hero_s > best:
            return 1.0
        if hero_s == best:
            return 1.0 / (sum(1 for s in scores if s == best) + 1)
        return 0.0

    eq = sum(share(h, f) for h, f, _ in samples) / len(samples)
    eq_random = float(estimate_equity(hero_hole, board, max(1, len(opps)), iterations=600, short_deck=short, rng=random.Random(seed))["equity"])

    # each opponent continues against a bet with the stronger (1 - fold_p) part of their sampled range
    thresholds = []
    for j, o in enumerate(info):
        xs = sorted(s[2][j] for s in samples)
        k = min(len(xs) - 1, max(0, int(o["fold_to_our_bet"] * len(xs))))
        thresholds.append(xs[k])

    pot = float(obs["total_pot"])
    to_call = float(la["call"])
    hero_bet = float(obs["hero"]["bet"])
    current = hero_bet + to_call  # the bet level we face
    options: List[Dict[str, Any]] = []
    if la["fold"] and to_call > 0:
        options.append({"action": "fold", "amount": None, "ev": 0.0})
    if to_call <= 0:
        options.append({"action": "check", "amount": None, "ev": eq * pot})
    else:
        options.append({"action": "call", "amount": to_call, "ev": eq * (pot + to_call) - to_call})
    if la["raise"] and info:
        sizes = []
        if street == "preflop":
            bb = float(obs["min_bet"])
            sizes.append(3 * bb if current <= bb + 1e-9 else 3 * current)
        else:
            for frac in (0.5, 0.75, 1.0):
                sizes.append(hero_bet + to_call + frac * (pot + to_call))
        seen = set()
        for target in sizes:
            target = round_to_unit(max(target, la["min_to"]), obs, la)
            if target in seen:
                continue
            seen.add(target)
            risk = target - hero_bet
            ev_sum, all_fold = 0.0, 0
            for hero_s, finals, now in samples:
                cont = [j for j in range(len(info)) if now[j] >= thresholds[j]]
                if not cont:
                    all_fold += 1
                    ev_sum += pot
                    continue
                final_pot = pot + risk + sum(max(target - info[j]["bet"], 0.0) for j in cont)
                ev_sum += share(hero_s, [finals[j] for j in cont]) * final_pot - risk
            options.append({"action": "all_in" if target >= la["max_to"] - 1e-9 else la["kind"], "amount": target,
                            "ev": ev_sum / len(samples), "all_fold": all_fold / len(samples)})
    best = max(options, key=lambda o: o["ev"])
    ranked = sorted(options, key=lambda o: -o["ev"])
    gap = ranked[0]["ev"] - ranked[1]["ev"] if len(ranked) > 1 else pot
    confidence = "high" if gap >= 0.10 * max(pot, 1) else "medium" if gap >= 0.03 * max(pot, 1) else "close"

    why = []
    if abs(eq - eq_random) >= 0.03:
        why.append(f"相手のスタイルと今回の行動から手札を絞ると、あなたの勝率は {eq:.0%}（相手の手をランダムと仮定した場合は {eq_random:.0%}）。")
    else:
        why.append(f"あなたの勝率は {eq:.0%}（相手のスタイルを考慮済み）。")
    if best["action"] in ("bet", "raise", "all_in"):
        folders = "、".join(f"{o['name']} {o['fold_to_our_bet']:.0%}" for o in info)
        why.append(f"{ACTION_JA[best['action']]} {best['amount']:g} すると全員が降りる確率は {best.get('all_fold', 0):.0%}（降りやすさ: {folders}）。")
    elif best["action"] == "fold":
        why.append(f"コールに必要な勝率は {to_call / (pot + to_call):.0%} で、足りません。")
    elif best["action"] == "call":
        why.append(f"コールに必要な勝率 {to_call / (pot + to_call):.0%} を上回ります。")
    else:
        why.append("ベットしても降りる相手が少なく、勝負するほど強くもないのでチェックが得です。")
    if confidence == "close":
        why.append("上位の選択肢の差は小さく、微妙な場面です。")

    for o in options:
        o["ev"] = round(o["ev"], 2)
        if "all_fold" in o:
            o["all_fold"] = round(o["all_fold"], 3)
    return {
        "available": True, "street": street, "cards": obs["hero"]["cards"], "opponents_in_hand": len(opps),
        "equity": round(eq, 4), "equity_vs_random": round(eq_random, 4),
        "pot": pot, "to_call": to_call, "pot_odds": round(to_call / (pot + to_call), 4) if to_call > 0 else 0.0,
        "call_ev": round(eq * (pot + to_call) - to_call, 2) if to_call > 0 else None,
        "options": options, "recommend": {"action": best["action"], "amount": best["amount"]},
        "recommend_ja": ACTION_JA.get(best["action"], best["action"]), "confidence": confidence,
        "reason_ja": " ".join(why),
        "opponents": [{k: v for k, v in o.items() if k != "style"} | {"style": {k: o["style"][k] for k in ("type", "type_ja", "type_prob", "hands", "rates", "confidence_ja")}}
                      for o in info],
        "method_ja": "相手ごとのプレイスタイル（この卓での行動を実データ4.8万人の分布で補正）から手札の範囲を絞り、"
                     "降りる確率も含めて各選択肢の期待チップをシミュレーションで比較。ゲーム理論の均衡ではなく、この相手に勝つための助言。",
        "types_source": TYPES["source"],
    }
