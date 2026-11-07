"""Self-play: agents against each other for many hands, with PHH export."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .agents import Agent, make_agent, safe_act
from .engine import Hand, Seat, Table, num, q
from .phh import HandRecord


def hand_to_record(hand: Hand, *, names: Optional[Sequence[str]] = None, extra_meta: Optional[Dict[str, Any]] = None,
                   seats: Optional[Sequence[int]] = None) -> HandRecord:
    """Convert a finished live hand into a PHH record (same format as the dataset)."""
    cfg = hand.config
    rec = HandRecord(
        variant=cfg.variant,
        antes=list(cfg.antes),
        blinds=list(cfg.blinds),
        starting_stacks=[p.starting_stack for p in hand.players],
        actions=hand.to_phh_actions(),
        min_bet=cfg.min_bet if cfg.limit != "fixed" else None,
        small_bet=cfg.small_bet if cfg.limit == "fixed" else None,
        big_bet=cfg.big_bet if cfg.limit == "fixed" else None,
        ante_trimming_status=cfg.ante_trimming_status,
        players=list(names) if names else [p.name for p in hand.players],
        finishing_stacks=[p.stack for p in hand.players],
        winnings=[p.payout for p in hand.players],
        seats=list(seats) if seats else (list(hand.seats) if hand.seats else None),
        meta=dict(extra_meta or {}),
    )
    if hand.hand_id is not None:
        rec.meta.setdefault("hand", hand.hand_id)
    return rec


def play_hand(table: Table, agents: Dict[int, Agent], *, on_action: Optional[Callable[[Hand, Dict[str, Any]], None]] = None,
              max_actions: int = 500) -> Hand:
    """Run one hand at ``table`` with ``agents`` keyed by seat number."""
    hand = table.new_hand()
    steps = 0
    while hand.status == "betting" and hand.actor is not None and steps < max_actions:
        steps += 1
        actor = hand.actor
        seat = table.seat_for_player(actor)
        agent = agents[seat.seat_no]
        obs = hand.observation(actor)
        action = safe_act(agent, obs)
        result = hand.act(actor, action["action"], action.get("amount"))
        result.update({"player": actor, "seat": seat.seat_no, "raw": action})
        if on_action:
            on_action(hand, result)
    if hand.status not in ("finished", "incomplete"):
        hand.finalize()
    table.finish_hand()
    return hand


@dataclass
class MatchResult:
    seats: List[Dict[str, Any]]
    hands: int
    records: List[HandRecord] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"seats": self.seats, "hands": self.hands, "seconds": round(self.seconds, 2)}


def run_match(agent_specs: Sequence[str], *, hands: int = 200, stack: float = 200.0, small_blind: float = 0.5,
              big_blind: float = 1.0, ante: float = 0.0, seed: Optional[int] = None, reset_stacks: bool = True,
              keep_records: bool = True, names: Optional[Sequence[str]] = None,
              progress: Optional[Callable[[int, int], None]] = None) -> MatchResult:
    """Play ``hands`` hands between the given agents and report bb/100 for each."""
    rng = random.Random(seed)
    agents: Dict[int, Agent] = {}
    seats = []
    for i, spec in enumerate(agent_specs):
        agent = make_agent(spec, seed=rng.randrange(1 << 30))
        name = names[i] if names and i < len(names) else f"{agent.label} {i + 1}"
        agent.name = name
        agents[i + 1] = agent
        seats.append(Seat(seat_no=i + 1, name=name, stack=float(stack), agent=agent, kind=agent.kind, buy_in=float(stack)))
    table = Table(seats, small_blind=small_blind, big_blind=big_blind, ante=ante, rng=rng)
    nets: Dict[int, List[float]] = {s.seat_no: [] for s in seats}
    records: List[HandRecord] = []
    t0 = time.perf_counter()
    played = 0
    for h in range(hands):
        if len(table.eligible()) < 2:
            break
        before = {s.seat_no: s.stack for s in seats}
        hand = play_hand(table, agents)
        played += 1
        for s in seats:
            nets[s.seat_no].append(q(s.stack - before[s.seat_no]))
        if keep_records:
            records.append(hand_to_record(hand, extra_meta={"venue": "pokersim self-play"}))
        if reset_stacks:
            for s in seats:
                s.stack = float(stack)
        if progress:
            progress(h + 1, hands)
    out_seats = []
    for s in seats:
        xs = nets[s.seat_no]
        n = len(xs)
        total = q(sum(xs))
        mean = total / n if n else 0.0
        var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n) if n else 0.0
        out_seats.append({
            "seat": s.seat_no, "name": s.name, "agent": agents[s.seat_no].describe(), "hands": n,
            "net": total, "bb_per_100": q(100 * mean / big_blind) if n else 0.0,
            "bb_per_100_stderr": q(100 * se / big_blind) if n else 0.0, "final_stack": num(s.stack),
        })
    return MatchResult(out_seats, played, records, time.perf_counter() - t0)
