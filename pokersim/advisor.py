"""Live fold / call / raise advice for a human seat at the Play table.

The advisor is deterministic and offline: Monte-Carlo equity against the opponents still in the
hand, compared with the pot odds of calling.  It explains every recommendation with the numbers
behind it so a player can learn from it, not just follow it.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from . import cards as C
from .agents import legal, round_to_unit
from .equity import estimate_equity, preflop_strength

ACTION_JA = {"fold": "フォールド", "check": "チェック", "call": "コール", "bet": "ベット", "raise": "レイズ", "all_in": "オールイン"}
STREET_JA = {"preflop": "プリフロップ", "flop": "フロップ", "turn": "ターン", "river": "リバー"}


def _raise_target(obs: Dict[str, Any], la: Dict[str, Any], fraction_of_pot: float) -> float:
    pot_after_call = float(obs["total_pot"]) + la["call"]
    target = float(obs["hero"]["bet"]) + la["call"] + fraction_of_pot * pot_after_call
    return round_to_unit(target, obs, la)


def advise(obs: Dict[str, Any], opponents: Optional[List[Dict[str, Any]]] = None, iterations: int = 1500,
           seed: int = 0) -> Dict[str, Any]:
    """Recommendation for the player whose observation ``obs`` is (it must be their turn)."""
    la = legal(obs)
    hole = [C.parse_card(c) for c in obs["hero"]["cards"]]
    if not hole or any(c is None for c in hole):
        return {"available": False, "reason_ja": "自分の手札が分からないため助言できません。"}
    board = [C.parse_card(c) for c in obs["board"]]
    variant = obs.get("variant", "NT")
    street = obs["street"]
    live_opp = [p for p in obs["players"] if not p["folded"] and not p["is_hero"]]
    n_opp = max(1, len(live_opp))

    # deterministic per decision, so refreshing the page gives the same numbers
    rng = random.Random(f"{obs.get('hand_id')}|{street}|{len(obs.get('history') or [])}|{seed}")
    eq_res = estimate_equity([c for c in hole if c is not None], board, n_opp, iterations=iterations,
                             short_deck=variant == "NS", omaha=variant == "PO", rng=rng)
    eq = float(eq_res["equity"])
    fair = 1.0 / (n_opp + 1)
    pot = float(obs["total_pot"])
    to_call = float(la["call"])
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
    call_ev = eq * (pot + to_call) - to_call  # chips, ignoring later streets
    stack = float(obs["hero"]["stack"]) if obs["hero"]["stack"] is not None else float("inf")
    preflop = street == "preflop"
    margin = 0.05 if preflop else 0.03
    strong_bar = fair + (0.20 if preflop else 0.25)
    pf = preflop_strength(hole) if preflop and variant != "PO" else None  # type: ignore[arg-type]

    verb = la["kind"]  # "bet" when nobody has bet yet, otherwise "raise"
    rec: Dict[str, Any]
    why: List[str] = []
    if la["raise"] and eq >= strong_bar:
        if preflop:
            bb = float(obs["min_bet"])
            facing = float(obs["hero"]["bet"]) + to_call  # the bet we are facing (raise-to level)
            target = round_to_unit(max(la["min_to"], 3 * bb if facing <= bb + 1e-9 else 3 * facing), obs, la)
        else:
            target = _raise_target(obs, la, 0.66)
        all_in = target >= la["max_to"] - 1e-9
        rec = {"action": "all_in" if all_in else verb, "amount": target}
        why.append(f"勝率 {eq:.0%} は相手{n_opp}人に対する平均（{fair:.0%}）を大きく上回る強い手です。")
        why.append(f"{ACTION_JA[verb]}して相手からチップを引き出しましょう（目安 {target:g} まで）。")
    elif to_call <= 0:
        rec = {"action": "check", "amount": None}
        why.append(f"勝率 {eq:.0%}。強い手ではないので、タダで次のカードを見られるチェックが無難です。")
    elif eq >= pot_odds + margin:
        rec = {"action": "call", "amount": to_call}
        why.append(f"コールに必要な勝率は {pot_odds:.0%}（{to_call:g} を払って {pot + to_call:g} を取り合う）で、あなたの勝率 {eq:.0%} がそれを上回ります。")
        if to_call >= stack:
            why.append("コールするとオールインになります。")
    else:
        rec = {"action": "fold", "amount": None}
        why.append(f"コールに必要な勝率は {pot_odds:.0%} ですが、あなたの勝率は {eq:.0%} しかありません。長い目で見て損なのでフォールドが得です。")
    gap = eq - (pot_odds if to_call > 0 else fair)
    confidence = "high" if abs(gap) >= 0.12 else "medium" if abs(gap) >= 0.05 else "close"
    if confidence == "close":
        why.append("ただし差はわずかで、どちらを選んでも大きな損はない微妙な場面です。")
    if preflop and pf is not None:
        why.append(f"プリフロップの手の強さ指標 {pf:.2f}（0〜1）。")

    return {
        "available": True,
        "street": street, "street_ja": STREET_JA.get(street, street),
        "cards": obs["hero"]["cards"], "board": obs["board"], "position": obs["hero"]["position"],
        "opponents_in_hand": n_opp, "fair_share": round(fair, 4),
        "equity": round(eq, 4), "win": round(float(eq_res.get("win", eq)), 4), "tie": round(float(eq_res.get("tie", 0.0)), 4),
        "iterations": int(eq_res.get("iterations", iterations)),
        "pot": pot, "to_call": to_call, "pot_odds": round(pot_odds, 4), "call_ev": round(call_ev, 2),
        "recommend": rec, "recommend_ja": ACTION_JA.get(rec["action"], rec["action"]), "confidence": confidence,
        "reason_ja": " ".join(why),
        "opponents": opponents or [],
        "method": "モンテカルロ勝率（残っている相手の手札をランダムと仮定）とポットオッズの比較",
    }
