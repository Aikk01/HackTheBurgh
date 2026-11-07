"""Replay recorded hands through the engine.

``replay(record)`` produces a :class:`ReplayResult` with one :class:`Step` per
recorded action (plus a setup step and a final payout step).  Every step carries
a JSON-friendly snapshot of the table so a UI can scrub through the hand, and
every player decision is exposed as a *decision point* carrying the observation
the acting player had at that moment - the hook used to test models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import cards as C
from .engine import STREETS, GameConfig, Hand, UnsupportedVariant, num, q
from .phh import HandRecord, ParsedAction, parse_action

SUPPORTED_VARIANTS = ("NT", "NS", "PO", "FT")


@dataclass
class Step:
    index: int                      # index into record.actions (-1 setup, -2 final)
    raw: Optional[str]
    label: str
    state: Dict[str, Any]
    events: List[Dict[str, Any]] = field(default_factory=list)
    player: Optional[int] = None
    decision: bool = False
    decision_id: Optional[int] = None


@dataclass
class DecisionPoint:
    id: int
    step_index: int                 # index into ReplayResult.steps of the state *before* the action
    player: int
    street: str
    observation: Dict[str, Any]
    legal: Dict[str, Any]
    actual: Dict[str, Any]          # {"action": fold|check|call|bet|raise, "to": float|None, "amount": float}
    cards_at_time: List[str]
    cards_final: List[str] = field(default_factory=list)


@dataclass
class ReplayResult:
    record: HandRecord
    hand: Hand
    steps: List[Step]
    decisions: List[DecisionPoint]
    warnings: List[str]
    complete: bool
    stacks_match: Optional[bool]
    stack_diffs: Optional[List[float]]

    def to_dict(self, include_observations: bool = False) -> Dict[str, Any]:
        return {
            "record": self.record.summary(),
            "steps": [{
                "index": s.index, "raw": s.raw, "label": s.label, "state": s.state, "events": s.events,
                "player": s.player, "decision": s.decision, "decision_id": s.decision_id,
            } for s in self.steps],
            "decisions": [{
                "id": d.id, "step_index": d.step_index, "player": d.player, "street": d.street,
                "actual": d.actual, "legal": d.legal, "cards_at_time": d.cards_at_time,
                "cards_final": d.cards_final,
                **({"observation": d.observation} if include_observations else {}),
            } for d in self.decisions],
            "warnings": self.warnings,
            "complete": self.complete,
            "stacks_match": self.stacks_match,
            "stack_diffs": self.stack_diffs,
            "finishing_stacks": [num(p.stack) for p in self.hand.players],
            "recorded_finishing_stacks": self.record.finishing_stacks,
            "recorded_winnings": self.record.winnings,
            "payouts": self.hand.payouts,
            "pots": self.hand.pots,
        }


def config_from_record(rec: HandRecord) -> GameConfig:
    if rec.variant not in SUPPORTED_VARIANTS:
        raise UnsupportedVariant(f"variant {rec.variant!r} is not supported (hold'em family only)")
    return GameConfig(rec.variant, antes=rec.antes, blinds=rec.blinds, min_bet=rec.min_bet,
                      small_bet=rec.small_bet, big_bet=rec.big_bet, ante_trimming_status=rec.ante_trimming_status)


def build_hand(rec: HandRecord) -> Hand:
    cfg = config_from_record(rec)
    names = [rec.player_name(i) for i in range(rec.n)]
    hand = Hand(cfg, rec.starting_stacks, names, scripted=True, hand_id=rec.meta.get("hand", rec.label),
                seats=rec.seats)
    hand.winnings_hint = rec.winnings
    return hand


def _money(x: float) -> str:
    x = q(x)
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}"


def _label(hand: Hand, a: ParsedAction, events: List[Dict[str, Any]]) -> str:
    if a.is_dealer:
        if a.verb == "dh":
            cards = a.args[1] if len(a.args) > 1 else ""
            who = hand.players[int(a.args[0][1:]) - 1].name if a.args else "?"
            return f"Deal {who}: {_pretty_cards(cards)}"
        if a.verb == "db":
            street = STREETS[hand.street]
            return f"{street.capitalize()}: {_pretty_cards(a.args[0] if a.args else '')}"
        return a.raw
    name = hand.players[a.player].name if a.player is not None and a.player < hand.n else a.actor
    act = next((e for e in events if e["type"] == "action"), None)
    if a.verb == "f":
        return f"{name} folds"
    if a.verb == "cc":
        if act and act["action"] == "check":
            return f"{name} checks"
        amt = act["amount"] if act else 0
        return f"{name} calls {_money(amt)}" + (" (all-in)" if act and act.get("all_in") else "")
    if a.verb == "cbr":
        if act and act["action"] == "bet":
            return f"{name} bets {_money(act['to'])}" + (" (all-in)" if act.get("all_in") else "")
        if act and act["action"] == "raise":
            return f"{name} raises to {_money(act['to'])}" + (" (all-in)" if act.get("all_in") else "")
        if act and act["action"] == "call":
            return f"{name} calls {_money(act['amount'])} (all-in)"
        return f"{name} bets/raises to {a.args[0] if a.args else '?'}"
    if a.verb == "sm":
        if not a.args:
            return f"{name} mucks"
        return f"{name} shows {_pretty_cards(a.args[0])}"
    return f"{name} {a.verb} {' '.join(a.args)}"


def _pretty_cards(text: str) -> str:
    try:
        return " ".join(C.pretty(c) for c in C.parse_cards(text))
    except ValueError:
        return text


def _classify(hand: Hand, a: ParsedAction, legal: Dict[str, Any]) -> Dict[str, Any]:
    if a.verb == "f":
        return {"action": "fold", "to": None, "amount": 0.0}
    if a.verb == "cc":
        if legal.get("check"):
            return {"action": "check", "to": None, "amount": 0.0}
        return {"action": "call", "to": None, "amount": legal.get("call", 0.0)}
    if a.verb == "cbr":
        to = float(a.args[0]) if a.args else None
        kind = "bet" if hand.max_bet <= 1e-9 else "raise"
        return {"action": kind, "to": to, "amount": None}
    return {"action": a.verb, "to": None, "amount": None}


def replay(rec: HandRecord, *, snapshots: bool = True, observations: bool = True,
           reveal: bool = True, split_mode: Optional[str] = None) -> ReplayResult:
    """Replay a recorded hand through the engine.

    Split pots are divided exactly (half chips, as in the Pluribus logs); when a
    record's ``finishing_stacks`` disagree, the replay is retried with the
    odd-chip rule used by real poker rooms before reporting a mismatch.
    """
    res = _replay(rec, snapshots=snapshots, observations=observations, reveal=reveal,
                  split_mode=split_mode or "exact")
    if split_mode is None and res.stacks_match is False:
        alt = _replay(rec, snapshots=snapshots, observations=observations, reveal=reveal, split_mode="odd_chip")
        if alt.stacks_match:
            return alt
    return res


def _replay(rec: HandRecord, *, snapshots: bool, observations: bool, reveal: bool, split_mode: str) -> ReplayResult:
    hand = build_hand(rec)
    hand.split_mode = split_mode
    steps: List[Step] = []
    decisions: List[DecisionPoint] = []

    def snap() -> Dict[str, Any]:
        return hand.to_dict(reveal=reveal) if snapshots else {}

    hand.post_forced_bets()
    steps.append(Step(-1, None, "Antes and blinds posted", snap(), list(hand.events)))

    for i, raw in enumerate(rec.actions):
        try:
            a = parse_action(raw)
        except ValueError as exc:
            hand.warn(f"action {i}: {exc}")
            continue
        if a is None:
            continue
        before = len(hand.events)
        decision_id = None
        player = a.player
        if a.is_dealer:
            if a.verb == "dh":
                if len(a.args) < 2:
                    hand.warn(f"action {i}: malformed hole card deal {raw!r}")
                    continue
                target = int(a.args[0][1:]) - 1
                player = target
                hand.deal_hole(target, C.parse_cards(a.args[1]))
            elif a.verb == "db":
                hand.deal_board(C.parse_cards(a.args[0]) if a.args else [])
            else:
                raise UnsupportedVariant(f"dealer action {a.verb!r} is not supported")
        else:
            if player is None or player >= hand.n:
                hand.warn(f"action {i}: unknown player in {raw!r}")
                continue
            if a.verb in ("f", "cc", "cbr"):
                if hand.status == "dealing":
                    hand._begin_street()
                if hand.status == "betting":
                    legal = hand.legal_actions(player)
                    obs = hand.observation(player) if observations else {}
                    dp = DecisionPoint(len(decisions), len(steps) - 1, player, STREETS[hand.street], obs, legal,
                                       _classify(hand, a, legal), [C.card_str(c) for c in hand.players[player].hole])
                    decisions.append(dp)
                    decision_id = dp.id
                if a.verb == "f":
                    hand.fold(player)
                elif a.verb == "cc":
                    hand.check_call(player)
                else:
                    if not a.args:
                        hand.warn(f"action {i}: raise without an amount {raw!r}")
                        continue
                    hand.bet_raise_to(player, float(a.args[0]))
            elif a.verb == "sm":
                if not a.args:
                    hand.show_muck(player, None)
                elif a.args[0] == "-":
                    hand.show_muck(player, list(hand.players[player].hole))
                else:
                    hand.show_muck(player, C.parse_cards(a.args[0]))
            elif a.verb in ("sd", "pb"):
                raise UnsupportedVariant(f"player action {a.verb!r} belongs to a draw/stud variant")
            else:
                hand.warn(f"action {i}: unknown action {raw!r}")
                continue
        events = hand.events[before:]
        steps.append(Step(i, raw, _label(hand, a, events), snap(), events, player, decision_id is not None,
                          decision_id))

    before = len(hand.events)
    hand.finalize()
    final_events = hand.events[before:]
    if hand.status == "finished":
        winners = [p for p in hand.players if p.payout > 0]
        label = "Pot awarded: " + ", ".join(f"{p.name} wins {_money(p.payout)}" for p in winners) if winners else "Hand over"
    else:
        label = "Record ends before the hand is resolved"
    steps.append(Step(-2, None, label, snap(), final_events))

    for d in decisions:
        d.cards_final = [C.card_str(c) for c in hand.players[d.player].final_cards()]

    stacks_match: Optional[bool] = None
    diffs: Optional[List[float]] = None
    if rec.finishing_stacks and len(rec.finishing_stacks) == hand.n and hand.status == "finished":
        diffs = []
        for p, s in zip(hand.players, rec.finishing_stacks):
            if p.stack == float("inf") or s == float("inf"):
                diffs.append(0.0)
            else:
                diffs.append(q(p.stack - s))
        stacks_match = all(abs(d) <= 0.011 for d in diffs)
        if not stacks_match:
            hand.warn("computed finishing stacks differ from the record: " +
                      ", ".join(f"p{i + 1} {d:+}" for i, d in enumerate(diffs) if abs(d) > 0.011))
            steps[-1].state = snap()

    return ReplayResult(rec, hand, steps, decisions, list(hand.warnings),
                        hand.status == "finished" and not hand.unresolved, stacks_match, diffs)
