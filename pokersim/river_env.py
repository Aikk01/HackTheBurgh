"""PokerKit-backed river exercise (PLAN.md section 4).

One episode starts just after the out-of-position defender checks the river
with 100 chips in the pot and 150 behind each seat.  The learner is the
in-position bettor and has at most two decisions:

* stage 1: ``check_back`` or ``bet_half`` (50 into 100);
* stage 2 (only after the defender raises to 100): ``fold`` or ``call``.

The defender is an injected policy with the interface
``respond(context, bucket, rng) -> 'fold' | 'call' | 'raise'``.  Rewards are
``(final bettor stack - bettor stack at episode start) / 100``.
"""
from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pokerkit import Automation, NoLimitTexasHoldem

from . import cards as C
from .engine import GameConfig, Hand
from .mirror_advisor import ranker_for
from .mirror_types import (BET_SIZE, CONTEXTS, CTX_BETTOR_AGGR, CTX_CHECKED_THROUGH, CTX_DEFENDER_AGGR,
                           OPENING_POT, RAISE_TO, RESPONSES)

ENV_VERSION = "river-v1"
START_STACK = 200
BLINDS = (1, 2)
RIVER_STACK = 150
STAGE_ACTIONS = {1: ("check_back", "bet_half"), 2: ("fold", "call")}

AUTOMATIONS = (
    Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING, Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
)

# Legal heads-up prefixes that leave 100 in the pot with 150 behind each seat at the river check.
# Player 0 is the big blind (out of position, the defender); player 1 is the button (the bettor).
PREFIXES: Dict[str, List[Tuple[str, Tuple[Any, ...]]]] = {
    CTX_CHECKED_THROUGH: [("raise", (10,)), ("call", ()), ("flop", ()), ("check", ()), ("bet", (40,)), ("call", ()),
                          ("turn", ()), ("check", ()), ("check", ()), ("river", ()), ("check", ())],
    CTX_BETTOR_AGGR: [("raise", (10,)), ("call", ()), ("flop", ()), ("check", ()), ("check", ()),
                      ("turn", ()), ("check", ()), ("bet", (40,)), ("call", ()), ("river", ()), ("check", ())],
    CTX_DEFENDER_AGGR: [("raise", (10,)), ("call", ()), ("flop", ()), ("check", ()), ("check", ()),
                        ("turn", ()), ("bet", (40,)), ("call", ()), ("river", ()), ("check", ())],
}


def _cards(cards: Sequence[str]) -> List[int]:
    out = []
    for c in cards:
        v = C.parse_card(c)
        if v is None:
            raise ValueError(f"unknown card {c!r}")
        out.append(v)
    return out


def build_state(board: Sequence[str], defender_cards: Sequence[str], bettor_cards: Sequence[str], context: str):
    """A PokerKit state positioned right after the river check."""
    if context not in PREFIXES:
        raise ValueError(f"unknown context {context!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PokerKit warns when cards are dealt without a shuffled deck
        st = NoLimitTexasHoldem.create_state(AUTOMATIONS, False, 0, BLINDS, 2, (START_STACK, START_STACK), 2)
        st.deal_hole("".join(defender_cards))
        st.deal_hole("".join(bettor_cards))
        board = list(board)
        for kind, args in PREFIXES[context]:
            if kind == "raise" or kind == "bet":
                st.complete_bet_or_raise_to(args[0])
            elif kind in ("call", "check"):
                st.check_or_call()
            elif kind == "flop":
                st.deal_board("".join(board[:3]))
            elif kind == "turn":
                st.deal_board(board[3])
            elif kind == "river":
                st.deal_board(board[4])
    return st


def build_pokersim_hand(board: Sequence[str], defender_cards: Sequence[str], bettor_cards: Sequence[str],
                        context: str) -> Hand:
    """The same prefix on the project's own engine (used for parity checks)."""
    hand = Hand(GameConfig("NT", antes=[0, 0], blinds=list(BLINDS), min_bet=2), [START_STACK, START_STACK],
                ["defender", "bettor"], scripted=True)
    hand.post_forced_bets()
    hand.deal_hole(0, _cards(defender_cards))
    hand.deal_hole(1, _cards(bettor_cards))
    board = list(board)
    actor = 1  # button acts first pre-flop; post-flop the big blind acts first
    for kind, args in PREFIXES[context]:
        if kind in ("raise", "bet"):
            hand.bet_raise_to(actor, args[0])
            actor = 1 - actor
        elif kind in ("call", "check"):
            hand.check_call(actor)
            actor = 1 - actor
        elif kind == "flop":
            hand.deal_board(_cards(board[:3]))
            actor = 0
        elif kind == "turn":
            hand.deal_board(_cards([board[3]]))
            actor = 0
        elif kind == "river":
            hand.deal_board(_cards([board[4]]))
            actor = 0
    return hand


def apply_path_pokersim(hand: Hand, path: Sequence[str]) -> List[float]:
    """Apply a bettor/defender action path (names as in PATHS) and return final stacks."""
    for step in path:
        if step == "check_back":
            hand.check_call(1)
        elif step == "bet":
            hand.bet_raise_to(1, BET_SIZE)
        elif step == "fold":
            hand.fold(0)
        elif step == "call":
            hand.check_call(0)
        elif step == "raise":
            hand.bet_raise_to(0, RAISE_TO)
        elif step == "bettor_fold":
            hand.fold(1)
        elif step == "bettor_call":
            hand.check_call(1)
    hand.finalize()
    return [p.stack for p in hand.players]


def apply_path_pokerkit(st, path: Sequence[str]) -> List[int]:
    for step in path:
        if step in ("check_back", "call", "bettor_call"):
            st.check_or_call()
        elif step == "bet":
            st.complete_bet_or_raise_to(BET_SIZE)
        elif step == "raise":
            st.complete_bet_or_raise_to(RAISE_TO)
        elif step in ("fold", "bettor_fold"):
            st.fold()
    return list(st.stacks)


PATHS = {
    "check_back": ["check_back"],
    "bet_fold": ["bet", "fold"],
    "bet_call": ["bet", "call"],
    "bet_raise_fold": ["bet", "raise", "bettor_fold"],
    "bet_raise_call": ["bet", "raise", "bettor_call"],
}


# ----------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    board: List[str]
    defender_cards: List[str]
    bettor_cards: List[str]
    context: str
    defender_bucket: str
    bettor_bucket: str
    defender_equity: float
    bettor_equity: float
    seed: int

    @classmethod
    def make(cls, board: Sequence[str], defender_cards: Sequence[str], bettor_cards: Sequence[str], context: str,
             seed: int = 0) -> "Scenario":
        r = ranker_for(_cards(board))
        de = r.equity(_cards(defender_cards))
        be = r.equity(_cards(bettor_cards))
        from .mirror_types import bucket_of
        return cls(list(board), list(defender_cards), list(bettor_cards), context, bucket_of(de), bucket_of(be),
                   round(de, 6), round(be, 6), seed)


def generate_scenarios(n: int, seed: int, contexts: Sequence[str] = CONTEXTS) -> List[Scenario]:
    """Uniform random deals: five board cards, two cards each, a uniform context."""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        deck = C.full_deck()
        rng.shuffle(deck)
        board = [C.card_str(c) for c in deck[:5]]
        dc = [C.card_str(c) for c in deck[5:7]]
        bc = [C.card_str(c) for c in deck[7:9]]
        out.append(Scenario.make(board, dc, bc, rng.choice(list(contexts)), seed=seed * 1_000_003 + k))
    return out


def scenario_pool(n: int, seed: int, cache_dir: Optional[str] = None) -> List[Scenario]:
    """Cached scenario pool (equities are the expensive part)."""
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cache", "mirror")
    path = os.path.join(cache_dir, f"scenarios_{seed}_{n}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [Scenario(**d) for d in json.load(f)]
    pool = generate_scenarios(n, seed)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"  # atomic: parallel trainers may build the same pool
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in pool], f)
    os.replace(tmp, path)
    return pool


# ----------------------------------------------------------------- defenders
class ScriptedDefender:
    """Fixed fold/call/raise probabilities regardless of context or strength (synthetic)."""

    kind = "scripted"

    def __init__(self, fold: float, call: float, raise_: float, name: str = "scripted"):
        total = fold + call + raise_
        self.probs = {"fold": fold / total, "call": call / total, "raise": raise_ / total}
        self.name = name

    def respond(self, context: str, bucket: str, rng: random.Random) -> str:
        u = rng.random()
        acc = 0.0
        for a in RESPONSES:
            acc += self.probs[a]
            if u < acc:
                return a
        return "raise"

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "probs": self.probs, "synthetic": True}


class StrengthDefender:
    """Responds from a context x bucket table (synthetic strength-dependent opponent)."""

    kind = "strength"

    def __init__(self, table: Dict[str, Dict[str, Dict[str, float]]], name: str = "strength"):
        self.table = table
        self.name = name

    def respond(self, context: str, bucket: str, rng: random.Random) -> str:
        probs = self.table[context][bucket]
        u = rng.random()
        acc = 0.0
        for a in RESPONSES:
            acc += probs.get(a, 0.0)
            if u < acc:
                return a
        return "raise"

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "synthetic": True}


# ----------------------------------------------------------------- environment
class RiverResponseEnv:
    """Gymnasium-style single-agent environment over a headless PokerKit state."""

    def __init__(self, defender: Any, pool: Sequence[Scenario], seed: Optional[int] = None,
                 opponent_seed: Optional[int] = None):
        if not pool:
            raise ValueError("empty scenario pool")
        self.defender = defender
        self.pool = list(pool)
        self.deal_rng = random.Random(seed)
        self.opp_rng = random.Random(opponent_seed if opponent_seed is not None else (seed + 1 if seed is not None else None))
        self.state = None
        self.scenario: Optional[Scenario] = None
        self.stage = 0
        self.start_stack = 0
        self.last_response: Optional[str] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.deal_rng.seed(seed)
        sc = (options or {}).get("scenario")
        if sc is None:
            sc = self.pool[self.deal_rng.randrange(len(self.pool))]
        self.scenario = sc
        self.state = build_state(sc.board, sc.defender_cards, sc.bettor_cards, sc.context)
        self.start_stack = self.state.stacks[1]
        self.stage = 1
        self.last_response = None
        return self._obs(), {"legal_mask": [True, True], "scenario": asdict(sc)}

    def _obs(self) -> Dict[str, Any]:
        sc = self.scenario
        st = self.state
        return {
            "stage": self.stage,
            "context": sc.context,
            "board": list(sc.board),
            "own_cards": list(sc.bettor_cards),
            "pot": int(st.total_pot_amount),
            "to_call": int(st.checking_or_calling_amount) if self.stage == 2 else 0,
            "legal": [True, True],
            "actions": list(STAGE_ACTIONS[self.stage]),
            "bucket": sc.bettor_bucket,
            "equity": sc.bettor_equity,
        }

    def state_key(self) -> Tuple[int, str, str]:
        return (self.stage, self.scenario.context, self.scenario.bettor_bucket)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.state is None or self.stage == 0:
            raise RuntimeError("call reset() first")
        st = self.state
        info: Dict[str, Any] = {}
        if self.stage == 1:
            if action == 0:
                st.check_or_call()
                return self._finish(info)
            st.complete_bet_or_raise_to(BET_SIZE)
            resp = self.defender.respond(self.scenario.context, self.scenario.defender_bucket, self.opp_rng)
            self.last_response = resp
            info["response"] = resp
            if resp == "fold":
                st.fold()
                return self._finish(info)
            if resp == "call":
                st.check_or_call()
                return self._finish(info)
            st.complete_bet_or_raise_to(RAISE_TO)
            self.stage = 2
            return self._obs(), 0.0, False, False, info
        if action == 0:
            st.fold()
        else:
            st.check_or_call()
        return self._finish(info)

    def _finish(self, info: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        st = self.state
        reward = (st.stacks[1] - self.start_stack) / OPENING_POT
        info.update({"final_stacks": list(st.stacks), "response": self.last_response, "scenario": asdict(self.scenario)})
        self.stage = 0
        obs = {"stage": 0, "terminal": True}
        return obs, reward, True, False, info
