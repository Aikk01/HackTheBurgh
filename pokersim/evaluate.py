"""Shadow evaluation: how often does a model make the same decision as the humans
(or bots) in recorded hands?

For every decision point in a recorded hand the agent is shown the observation
the acting player had (hole cards included when they are known - either dealt
face up in the record or revealed at showdown) and its action is compared with
what really happened.
"""
from __future__ import annotations

import collections
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .agents import Agent, legal, safe_act
from .engine import UnsupportedVariant
from .phh import HandRecord
from .replay import DecisionPoint, replay

ACTIONS = ["fold", "check", "call", "bet", "raise"]
COARSE = {"fold": "fold", "check": "passive", "call": "passive", "bet": "aggressive", "raise": "aggressive"}


def normalise_prediction(action: Dict[str, Any], legal_actions: Dict[str, Any]) -> Dict[str, Any]:
    """Map a model's raw action onto what the engine would actually do."""
    la = legal(({"legal_actions": legal_actions}))
    raw = str(action.get("action", "")).lower().replace("-", "_").replace(" ", "_")
    amount = action.get("amount")
    note = action.get("error")
    if raw in ("fold", "f"):
        if la["check"]:
            return {"action": "check", "to": None, "note": "fold with nothing to call -> check"}
        return {"action": "fold", "to": None, "note": note}
    if raw in ("check", "call", "cc", "k", "c", "check_call"):
        return {"action": "check" if la["check"] else "call", "to": None, "note": note}
    if raw in ("all_in", "allin", "shove", "jam"):
        if la["raise"]:
            return {"action": la["kind"], "to": la["max_to"], "note": note}
        return {"action": "check" if la["check"] else "call", "to": None, "note": "all-in with no raise allowed -> call"}
    if raw in ("bet", "raise", "raise_to", "bet_to", "cbr", "b", "r"):
        if not la["raise"]:
            return {"action": "check" if la["check"] else "call", "to": None, "note": "raise not allowed -> call/check"}
        try:
            amt = float(amount) if amount is not None else la["min_to"]
        except (TypeError, ValueError):
            amt = la["min_to"]
        clamped = min(max(amt, la["min_to"]), la["max_to"])
        n = note
        if abs(clamped - amt) > 1e-9:
            n = f"size {amt} clamped to {clamped}"
        return {"action": la["kind"], "to": clamped, "note": n}
    return {"action": "check" if la["check"] else "call", "to": None, "note": f"unknown action {raw!r} -> call/check"}


@dataclass
class DecisionRow:
    hand: str
    hand_index: Optional[int]
    list_index: int
    decision_id: int
    player: int
    name: str
    street: str
    position: str
    pot: float
    to_call: float
    cards: List[str]
    actual: str
    actual_to: Optional[float]
    predicted: str
    predicted_to: Optional[float]
    agree: bool
    agree_coarse: bool
    latency_ms: float
    note: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class EvalReport:
    agent: Dict[str, Any]
    rows: List[DecisionRow] = field(default_factory=list)
    hands: int = 0
    hands_evaluated: int = 0
    skipped_unknown_cards: int = 0
    unsupported: int = 0
    seconds: float = 0.0

    def summary(self) -> Dict[str, Any]:
        n = len(self.rows)
        by_street: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
        by_action: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
        confusion = {a: {b: 0 for b in ACTIONS} for a in ACTIONS}
        coarse = {a: {b: 0 for b in ("fold", "passive", "aggressive")} for a in ("fold", "passive", "aggressive")}
        log_ratios = []
        latency = []
        for r in self.rows:
            by_street[r.street][0] += r.agree
            by_street[r.street][1] += 1
            by_action[r.actual][0] += r.agree
            by_action[r.actual][1] += 1
            if r.actual in confusion and r.predicted in confusion[r.actual]:
                confusion[r.actual][r.predicted] += 1
            ca, cp = COARSE.get(r.actual), COARSE.get(r.predicted)
            if ca and cp:
                coarse[ca][cp] += 1
            if r.actual in ("bet", "raise") and r.predicted in ("bet", "raise") and r.actual_to and r.predicted_to:
                log_ratios.append(abs(math.log(r.predicted_to / r.actual_to)))
            latency.append(r.latency_ms)
        agree = sum(r.agree for r in self.rows)
        agree_c = sum(r.agree_coarse for r in self.rows)
        return {
            "agent": self.agent,
            "decisions": n,
            "hands": self.hands,
            "hands_evaluated": self.hands_evaluated,
            "skipped_unknown_cards": self.skipped_unknown_cards,
            "unsupported": self.unsupported,
            "accuracy": agree / n if n else None,
            "accuracy_coarse": agree_c / n if n else None,
            "by_street": {k: {"agree": v[0], "n": v[1], "accuracy": v[0] / v[1]} for k, v in by_street.items()},
            "by_action": {k: {"agree": v[0], "n": v[1], "accuracy": v[0] / v[1]} for k, v in by_action.items()},
            "confusion": confusion,
            "coarse_confusion": coarse,
            "size_error_mean_abs_log_ratio": (sum(log_ratios) / len(log_ratios)) if log_ratios else None,
            "sized_comparisons": len(log_ratios),
            "latency_ms_mean": (sum(latency) / len(latency)) if latency else None,
            "seconds": round(self.seconds, 2),
        }

    def to_dict(self, include_rows: bool = True, max_rows: Optional[int] = None) -> Dict[str, Any]:
        d = self.summary()
        if include_rows:
            rows = self.rows if max_rows is None else self.rows[:max_rows]
            d["rows"] = [asdict(r) for r in rows]
        return d


def observation_for(dp: DecisionPoint, use_final_cards: bool) -> Optional[Dict[str, Any]]:
    """Observation with the hero's cards filled in when they became known later."""
    obs = dict(dp.observation)
    cards = list(dp.cards_at_time)
    if any(c == "??" for c in cards) and use_final_cards and dp.cards_final:
        cards = list(dp.cards_final)
    if not cards or any(c == "??" for c in cards):
        return None
    hero = dict(obs["hero"])
    hero["cards"] = cards
    obs["hero"] = hero
    return obs


def shadow_evaluate(records: Iterable[HandRecord], agent: Agent, *, use_final_cards: bool = True,
                    max_decisions: Optional[int] = None, progress: Optional[Callable[[int, int, EvalReport], None]] = None,
                    should_stop: Optional[Callable[[], bool]] = None, only_player: Optional[str] = None) -> EvalReport:
    records = list(records)
    report = EvalReport(agent=agent.describe())
    t0 = time.perf_counter()
    report.hands = len(records)
    for hi, rec in enumerate(records):
        if should_stop and should_stop():
            break
        try:
            res = replay(rec, snapshots=False, observations=True)
        except UnsupportedVariant:
            report.unsupported += 1
            continue
        evaluated_here = False
        for dp in res.decisions:
            if max_decisions is not None and len(report.rows) >= max_decisions:
                break
            if only_player and rec.player_name(dp.player) != only_player:
                continue
            obs = observation_for(dp, use_final_cards)
            if obs is None:
                report.skipped_unknown_cards += 1
                continue
            raw = safe_act(agent, obs)
            pred = normalise_prediction(raw, dp.legal)
            actual = dp.actual
            agree = pred["action"] == actual["action"]
            agree_c = COARSE.get(pred["action"]) == COARSE.get(actual["action"])
            report.rows.append(DecisionRow(
                hand=rec.label, hand_index=rec.index, list_index=hi, decision_id=dp.id, player=dp.player,
                name=rec.player_name(dp.player), street=dp.street, position=obs["hero"]["position"],
                pot=obs["total_pot"], to_call=obs["to_call"], cards=obs["hero"]["cards"],
                actual=actual["action"], actual_to=actual.get("to"), predicted=pred["action"],
                predicted_to=pred.get("to"), agree=agree, agree_coarse=agree_c,
                latency_ms=raw.get("latency_ms", 0.0), note=pred.get("note"), reasoning=raw.get("reasoning"),
            ))
            evaluated_here = True
        if evaluated_here:
            report.hands_evaluated += 1
        if progress:
            progress(hi + 1, len(records), report)
        if max_decisions is not None and len(report.rows) >= max_decisions:
            break
    report.seconds = time.perf_counter() - t0
    return report
