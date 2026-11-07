"""Hold'em game engine.

The engine follows the conventions of the PHH format (and PokerKit):

* players are indexed ``0..n-1`` in the order they are listed in a hand history;
  ``p1`` (index 0) is the first player left of the button and the last player
  is the button.  In heads-up play the forced bets are reversed, so the button
  posts the small blind and acts first before the flop.
* ``cbr`` amounts are *raise to* amounts: the player's total bet for the street.

A :class:`Hand` can run in two modes:

* **live** (``scripted=False``): the engine shuffles, deals, enforces legality and
  automatically runs out the board / pays the pot.
* **scripted** (``scripted=True``): dealing and showdown information come from a
  recorded history (see :mod:`pokersim.replay`); the engine is permissive and
  records inconsistencies in ``hand.warnings`` instead of raising.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import cards as C
from . import evaluator as E

INF = float("inf")
STREETS = ("preflop", "flop", "turn", "river")
BOARD_CARDS = (0, 3, 1, 1)  # community cards dealt at the start of each street
EPS = 1e-9


class EngineError(Exception):
    pass


class IllegalAction(EngineError):
    pass


class UnsupportedVariant(EngineError):
    pass


VARIANTS: Dict[str, Dict[str, Any]] = {
    "NT": {"name": "No-limit Texas hold 'em", "limit": "no", "hole": 2, "short": False, "omaha": False},
    "NS": {"name": "No-limit short-deck hold 'em", "limit": "no", "hole": 2, "short": True, "omaha": False},
    "PO": {"name": "Pot-limit Omaha hold 'em", "limit": "pot", "hole": 4, "short": False, "omaha": True},
    "FT": {"name": "Fixed-limit Texas hold 'em", "limit": "fixed", "hole": 2, "short": False, "omaha": False},
}


def q(x: float) -> float:
    """Round a chip amount to cents (keeps infinities)."""
    if x == INF or x == -INF:
        return x
    return round(x + 0.0, 2)


def num(x: Optional[float]) -> Optional[float]:
    """JSON friendly number (JavaScript cannot parse Infinity)."""
    if x is None or x == INF:
        return None
    return q(x)


def position_names(n: int) -> List[str]:
    if n == 2:
        return ["BB", "BTN"]
    names: List[Optional[str]] = [None] * n
    names[0], names[1], names[n - 1] = "SB", "BB", "BTN"
    if n >= 4:
        names[n - 2] = "CO"
    if n >= 5:
        names[n - 3] = "HJ"
    if n >= 8:
        names[n - 4] = "LJ"
    k = 0
    for i in range(2, n):
        if names[i] is None:
            names[i] = "UTG" if k == 0 else f"UTG+{k}"
            k += 1
    return [str(x) for x in names]


@dataclass
class GameConfig:
    """Static parameters of a hand (mirrors the PHH required fields)."""

    variant: str = "NT"
    antes: List[float] = field(default_factory=list)
    blinds: List[float] = field(default_factory=list)  # blinds_or_straddles
    min_bet: Optional[float] = None
    small_bet: Optional[float] = None
    big_bet: Optional[float] = None
    ante_trimming_status: bool = False

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise UnsupportedVariant(f"variant {self.variant!r} is not supported by the engine")
        self.antes = [float(a) for a in self.antes]
        self.blinds = [float(b) for b in self.blinds]
        biggest_blind = max(self.blinds) if self.blinds else 0.0
        if self.limit == "fixed":
            if self.small_bet is None:
                self.small_bet = self.min_bet if self.min_bet is not None else (biggest_blind or 1.0)
            if self.big_bet is None:
                self.big_bet = self.small_bet * 2
        if self.min_bet is None:
            self.min_bet = self.small_bet if self.small_bet is not None else (biggest_blind or 1.0)

    @property
    def info(self) -> Dict[str, Any]:
        return VARIANTS[self.variant]

    @property
    def limit(self) -> str:
        return VARIANTS[self.variant]["limit"]

    @property
    def hole_count(self) -> int:
        return VARIANTS[self.variant]["hole"]

    @property
    def short_deck(self) -> bool:
        return VARIANTS[self.variant]["short"]

    @property
    def omaha(self) -> bool:
        return VARIANTS[self.variant]["omaha"]

    def bet_size(self, street: int) -> float:
        if self.limit == "fixed":
            return float(self.small_bet if street < 2 else self.big_bet)  # type: ignore[arg-type]
        return float(self.min_bet)  # type: ignore[arg-type]

    def for_players(self, n: int) -> "GameConfig":
        antes = (self.antes + [0.0] * n)[:n]
        blinds = (self.blinds + [0.0] * n)[:n]
        return GameConfig(self.variant, antes, blinds, self.min_bet, self.small_bet,
                          self.big_bet, self.ante_trimming_status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "variant_name": self.info["name"],
            "antes": self.antes,
            "blinds": self.blinds,
            "min_bet": self.min_bet,
            "small_bet": self.small_bet,
            "big_bet": self.big_bet,
            "limit": self.limit,
            "hole_count": self.hole_count,
        }


class Player:
    __slots__ = ("index", "name", "starting_stack", "stack", "bet", "contributed", "ante_paid", "folded",
                 "hole", "shown", "mucked", "acted", "raise_blocked", "payout", "hand_score")

    def __init__(self, index: int, name: str, stack: float):
        self.index = index
        self.name = name
        self.starting_stack = stack
        self.stack = stack
        self.bet = 0.0
        self.contributed = 0.0
        self.ante_paid = 0.0
        self.folded = False
        self.hole: List[Optional[int]] = []
        self.shown: Optional[List[Optional[int]]] = None
        self.mucked = False
        self.acted = False
        self.raise_blocked = False
        self.payout = 0.0
        self.hand_score: Optional[int] = None

    @property
    def all_in(self) -> bool:
        return not self.folded and self.stack <= EPS

    @property
    def active(self) -> bool:
        return not self.folded

    @property
    def can_act(self) -> bool:
        return not self.folded and self.stack > EPS

    def final_cards(self) -> List[Optional[int]]:
        """Hole cards, with cards revealed at showdown filled in."""
        if not self.shown:
            return list(self.hole)
        out = list(self.hole)
        if len(out) < len(self.shown):
            out += [None] * (len(self.shown) - len(out))
        for i, c in enumerate(self.shown):
            if c is not None:
                if c in out and out.index(c) != i:
                    j = out.index(c)
                    out[j], out[i] = out[i], c
                else:
                    out[i] = c
        return out

    def cards_known(self) -> bool:
        cards = self.final_cards()
        return bool(cards) and all(c is not None for c in cards)


class Hand:
    """One hand of poker.  See the module docstring for the two modes."""

    def __init__(self, config: GameConfig, stacks: Sequence[float], names: Optional[Sequence[str]] = None, *,
                 rng: Optional[random.Random] = None, scripted: bool = False, hand_id: Any = None,
                 seats: Optional[Sequence[int]] = None):
        n = len(stacks)
        if n < 2:
            raise EngineError("a hand needs at least two players")
        self.config = config.for_players(n)
        self.n = n
        self.players = [Player(i, (names[i] if names and i < len(names) and names[i] else f"p{i + 1}"),
                               float(stacks[i])) for i in range(n)]
        self.rng = rng or random.Random()
        self.scripted = scripted
        self.hand_id = hand_id
        self.seats = list(seats) if seats else None
        self.button = n - 1
        self.street = 0
        self.board: List[Optional[int]] = []
        self.pot = 0.0  # chips collected from completed streets
        self.status = "init"  # init, dealing, betting, awaiting_board, showdown, finished, incomplete
        self.actor: Optional[int] = None
        self.max_bet = 0.0
        self.last_raise_size = 0.0
        self.aggressor: Optional[int] = None
        self.events: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.deck = C.Deck(short=self.config.short_deck, rng=self.rng)
        self.payouts: Optional[List[float]] = None
        self.pots: Optional[List[Dict[str, Any]]] = None
        self.winnings_hint: Optional[Sequence[float]] = None
        self.unresolved = False
        self.quantum = self._compute_quantum()
        # "exact": split pots evenly (recorded payoffs such as the Pluribus logs use
        # half chips); "odd_chip": indivisible remainders go to the earliest position.
        self.split_mode = "exact" if scripted else "odd_chip"
        self.positions = position_names(n)

    # ------------------------------------------------------------------ helpers
    def _compute_quantum(self) -> float:
        values = [p.stack for p in self.players] + list(self.config.antes) + list(self.config.blinds)
        values.append(float(self.config.min_bet or 1))
        for v in values:
            if v != INF and not float(v).is_integer():
                return 0.01
        return 1.0

    def _emit(self, kind: str, **data: Any) -> None:
        ev = {"type": kind, "street": STREETS[self.street]}
        ev.update(data)
        self.events.append(ev)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self._emit("warning", message=message)

    def effective_antes(self) -> List[float]:
        a = list(self.config.antes)
        return a[::-1] if self.n == 2 else a

    def effective_blinds(self) -> List[float]:
        b = list(self.config.blinds)
        return b[::-1] if self.n == 2 else b

    def active_players(self) -> List[Player]:
        return [p for p in self.players if not p.folded]

    def actionable_players(self) -> List[Player]:
        return [p for p in self.players if p.can_act]

    def total_pot(self) -> float:
        return q(self.pot + sum(p.bet for p in self.players))

    def to_call(self, index: int) -> float:
        p = self.players[index]
        return q(min(max(self.max_bet - p.bet, 0.0), p.stack))

    # ------------------------------------------------------------- forced bets
    def post_forced_bets(self) -> None:
        if self.status != "init":
            raise EngineError("forced bets already posted")
        antes = self.effective_antes()
        blinds = self.effective_blinds()
        for i, p in enumerate(self.players):
            a = antes[i]
            if a < 0:
                self.warn(f"negative ante {a} for p{i + 1} treated as 0")
                a = 0.0
            a = min(q(a), p.stack)
            if a > 0:
                p.stack = q(p.stack - a)
                p.contributed = q(p.contributed + a)
                p.ante_paid = q(p.ante_paid + a)
                self.pot = q(self.pot + a)
                self._emit("ante", player=i, amount=a)
        for i, p in enumerate(self.players):
            b = blinds[i]
            if b < 0:
                self.warn(f"negative blind {b} for p{i + 1} treated as 0")
                b = 0.0
            b = min(q(b), p.stack)
            if b > 0:
                p.stack = q(p.stack - b)
                p.contributed = q(p.contributed + b)
                p.bet = b
                self._emit("blind", player=i, amount=b)
        self.max_bet = max((p.bet for p in self.players), default=0.0)
        self.last_raise_size = max(self.config.bet_size(0), self.max_bet)
        self.status = "dealing"

    # ----------------------------------------------------------------- dealing
    def deal_hole(self, index: int, cards: Optional[Sequence[Optional[int]]] = None) -> List[Optional[int]]:
        if self.status == "init":
            raise EngineError("post the forced bets before dealing")
        p = self.players[index]
        if cards is None:
            dealt: List[Optional[int]] = list(self.deck.draw(self.config.hole_count))
        else:
            dealt = list(cards)
            dup = self.deck.remove(dealt)
            if dup:
                self.warn(f"card(s) {C.cards_str(dup)} dealt to p{index + 1} were already in play")
        p.hole.extend(dealt)
        self._emit("deal_hole", player=index, cards=[C.card_str(c) for c in dealt])
        return dealt

    def deal_board(self, cards: Optional[Sequence[Optional[int]]] = None) -> List[Optional[int]]:
        if self.status == "dealing":
            if not self.scripted:
                raise EngineError("deal the hole cards and run the betting first")
            self.warn("board dealt before any betting action")
            self._begin_street()
        if self.status == "betting":
            if not self.scripted:
                raise EngineError("betting round in progress")
            self.warn(f"board dealt while the {STREETS[self.street]} betting round was incomplete")
            self._end_street()
        if self.status in ("finished", "incomplete"):
            if not self.scripted:
                raise EngineError("hand is over")
            self.warn("board dealt after the hand finished")
            return []
        if self.status == "showdown":
            if not self.scripted:
                raise EngineError("board complete")
            self.warn("board dealt after the board was complete")
        if self.street >= 3:
            self.warn("too many board cards")
            return []
        expected = BOARD_CARDS[self.street + 1]
        if cards is None:
            dealt: List[Optional[int]] = list(self.deck.draw(expected))
        else:
            dealt = list(cards)
            dup = self.deck.remove(dealt)
            if dup:
                self.warn(f"board card(s) {C.cards_str(dup)} were already in play")
            if len(dealt) != expected:
                self.warn(f"expected {expected} board card(s) for the {STREETS[self.street + 1]}, got {len(dealt)}")
        self.board.extend(dealt)
        self.street += 1
        self._emit("deal_board", cards=[C.card_str(c) for c in dealt])
        if len(self.actionable_players()) >= 2:
            self._begin_street()
        elif len(self.board) >= 5:
            self.status = "showdown"
            if not self.scripted:
                self.finalize()
        else:
            self.status = "awaiting_board"
        return dealt

    def start(self) -> None:
        """Live mode: post forced bets, deal everybody and open the betting."""
        self.post_forced_bets()
        for i in range(self.n):
            self.deal_hole(i)
        self._begin_street()

    # ------------------------------------------------------------- the streets
    def _begin_street(self) -> None:
        self.status = "betting"
        for p in self.players:
            p.acted = False
            p.raise_blocked = False
        if self.street == 0:
            blinds = self.effective_blinds()
            ref = max(range(self.n), key=lambda i: (blinds[i], i))
            start = (ref + 1) % self.n
        else:
            self.max_bet = 0.0
            self.last_raise_size = self.config.bet_size(self.street)
            self.aggressor = None
            start = 0
        if self._round_complete():
            self._end_street()
            return
        self.actor = self._next_actor(start)
        if self.actor is None:
            self._end_street()
            return
        self._emit("street", actor=self.actor)

    def _next_actor(self, start: int) -> Optional[int]:
        for k in range(self.n):
            i = (start + k) % self.n
            p = self.players[i]
            if p.can_act and (not p.acted or p.bet < self.max_bet - EPS):
                return i
        return None

    def _round_complete(self) -> bool:
        act = self.actionable_players()
        if not act:
            return True
        if len(act) == 1:
            return act[0].bet >= self.max_bet - EPS
        return all(p.acted and p.bet >= self.max_bet - EPS for p in act)

    def _after_action(self, index: int) -> None:
        if len(self.active_players()) <= 1 or self._round_complete():
            self._end_street()
            return
        self.actor = self._next_actor((index + 1) % self.n)
        if self.actor is None:
            self._end_street()

    def _end_street(self) -> None:
        self.actor = None
        bets = sorted((p.bet for p in self.players), reverse=True)
        if len(bets) >= 2 and bets[0] > bets[1] + EPS:
            top = next(p for p in self.players if abs(p.bet - bets[0]) < EPS)
            refund = q(bets[0] - bets[1])
            top.stack = q(top.stack + refund)
            top.bet = q(top.bet - refund)
            top.contributed = q(top.contributed - refund)
            self._emit("uncalled", player=top.index, amount=refund)
        collected = q(sum(p.bet for p in self.players))
        for p in self.players:
            p.bet = 0.0
        self.pot = q(self.pot + collected)
        self._emit("collect", amount=collected, pot=self.pot)
        active = self.active_players()
        if len(active) == 1:
            self._award_uncontested(active[0])
            return
        if self.street >= 3:
            self.status = "showdown"
            if not self.scripted:
                self.finalize()
            return
        self.status = "awaiting_board"
        if not self.scripted:
            while self.status == "awaiting_board":
                self.deal_board()

    # ----------------------------------------------------------------- actions
    def _check_turn(self, index: int) -> bool:
        if self.status == "dealing":
            if not self.scripted:
                raise EngineError("the hand has not started")
            self._begin_street()
        if self.status != "betting":
            if self.scripted:
                self.warn(f"p{index + 1} acted while the hand status was '{self.status}'")
                return False
            raise IllegalAction(f"no betting in progress (status: {self.status})")
        if self.actor != index:
            if self.scripted:
                exp = f"p{self.actor + 1}" if self.actor is not None else "nobody"
                self.warn(f"expected {exp} to act but the record has p{index + 1}")
                if not self.players[index].can_act:
                    return False
            else:
                raise IllegalAction(f"it is p{(self.actor or 0) + 1}'s turn, not p{index + 1}'s")
        return True

    def fold(self, index: int) -> None:
        if not self._check_turn(index):
            return
        p = self.players[index]
        if not self.scripted and self.max_bet - p.bet <= EPS:
            raise IllegalAction("cannot fold when checking is free")
        p.folded = True
        p.acted = True
        self._emit("action", player=index, action="fold", amount=0.0, to=p.bet)
        self._after_action(index)

    def check_call(self, index: int) -> None:
        if not self._check_turn(index):
            return
        p = self.players[index]
        owed = q(self.max_bet - p.bet)
        amt = q(min(max(owed, 0.0), p.stack))
        p.stack = q(p.stack - amt)
        p.bet = q(p.bet + amt)
        p.contributed = q(p.contributed + amt)
        p.acted = True
        self._emit("action", player=index, action="check" if owed <= EPS else "call", amount=amt, to=p.bet,
                   all_in=p.stack <= EPS)
        self._after_action(index)

    def bet_raise_to(self, index: int, amount: float) -> None:
        if not self._check_turn(index):
            return
        p = self.players[index]
        amount = q(float(amount))
        max_possible = q(p.bet + p.stack)
        if amount > max_possible + EPS:
            if not self.scripted:
                raise IllegalAction(f"p{index + 1} cannot bet {amount}; maximum is {max_possible}")
            self.warn(f"p{index + 1} raised to {amount} with only {max_possible} available; clamped")
            amount = max_possible
        if amount <= self.max_bet + EPS:
            if not self.scripted:
                raise IllegalAction("a bet or raise must exceed the current bet")
            self.warn(f"p{index + 1} 'raised' to {amount} which does not exceed the current bet "
                      f"{self.max_bet}; treated as a call")
            delta = q(max(amount - p.bet, 0.0))
            p.stack = q(p.stack - delta)
            p.bet = q(p.bet + delta)
            p.contributed = q(p.contributed + delta)
            p.acted = True
            self._emit("action", player=index, action="call", amount=delta, to=p.bet, all_in=p.stack <= EPS)
            self._after_action(index)
            return
        if not self.scripted:
            la = self.legal_actions(index)
            if not la["raise"]:
                raise IllegalAction("raising is not allowed here")
            if amount < la["min_raise_to"] - EPS and amount < max_possible - EPS:
                raise IllegalAction(f"minimum raise is to {la['min_raise_to']}")
            if amount > la["max_raise_to"] + EPS:
                raise IllegalAction(f"maximum raise is to {la['max_raise_to']}")
        delta = q(amount - p.bet)
        raise_size = q(amount - self.max_bet)
        full = raise_size >= self.last_raise_size - EPS
        kind = "bet" if self.max_bet <= EPS else "raise"
        p.stack = q(p.stack - delta)
        p.bet = amount
        p.contributed = q(p.contributed + delta)
        p.acted = True
        for o in self.players:
            if o is p or not o.can_act:
                continue
            if full:
                o.acted = False
                o.raise_blocked = False
            elif o.acted:
                o.acted = False
                o.raise_blocked = True
        if full:
            self.last_raise_size = raise_size
        self.max_bet = amount
        self.aggressor = index
        self._emit("action", player=index, action=kind, amount=delta, to=amount, raise_size=raise_size,
                   all_in=p.stack <= EPS, full_raise=full)
        self._after_action(index)

    def show_muck(self, index: int, cards: Optional[Sequence[Optional[int]]]) -> None:
        p = self.players[index]
        if cards is None:
            p.mucked = True
            self._emit("muck", player=index)
            return
        shown = list(cards)
        new_known = [c for c in shown if c is not None and c not in p.hole]
        if new_known:
            dup = self.deck.remove(new_known)
            if dup:
                self.warn(f"p{index + 1} showed card(s) {C.cards_str(dup)} that were already in play")
        p.shown = shown
        self._emit("show", player=index, cards=[C.card_str(c) for c in shown])

    def legal_actions(self, index: Optional[int] = None) -> Dict[str, Any]:
        """Legal options for ``index`` (default: the player whose turn it is)."""
        i = self.actor if index is None else index
        if i is None:
            return {"player": None, "fold": False, "check": False, "call": 0.0, "raise": False}
        p = self.players[i]
        owed = q(max(self.max_bet - p.bet, 0.0))
        to_call = q(min(owed, p.stack))
        res: Dict[str, Any] = {
            "player": i,
            "fold": owed > EPS,
            "check": owed <= EPS,
            "call": to_call if owed > EPS else 0.0,
            "call_is_all_in": owed > EPS and owed >= p.stack - EPS,
            "raise": False,
        }
        can_raise = p.stack > owed + EPS and not p.raise_blocked and self.status == "betting"
        if can_raise:
            all_in_to = q(p.bet + p.stack)
            min_to = q(self.max_bet + self.last_raise_size)
            if self.config.limit == "no":
                max_to = all_in_to
            elif self.config.limit == "pot":
                pot_after_call = q(self.pot + sum(pl.bet for pl in self.players) + owed)
                max_to = min(all_in_to, q(self.max_bet + pot_after_call))
            else:
                max_to = min_to
            if min_to > all_in_to:
                min_to = all_in_to
            if max_to < min_to:
                max_to = min_to
            res.update({"raise": True, "min_raise_to": min_to, "max_raise_to": max_to,
                        "raise_kind": "bet" if self.max_bet <= EPS else "raise", "all_in_to": all_in_to})
        return res

    def act(self, index: int, action: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """Apply a normalised action (``fold``/``check``/``call``/``bet``/``raise``/``all_in``).

        Amounts are *raise to* totals for the street.  Out-of-range raise sizes
        are clamped to the legal range and the adjustment is reported in the
        returned dict so that model mistakes stay visible.
        """
        if self.status != "betting" or self.actor != index:
            raise IllegalAction(f"p{index + 1} cannot act now")
        a = (action or "").strip().lower().replace("-", "_").replace(" ", "_")
        la = self.legal_actions(index)
        note = None
        if a in ("fold", "f"):
            if la["check"]:
                note = "fold converted to check (nothing to call)"
                self.check_call(index)
                return {"action": "check", "note": note}
            self.fold(index)
            return {"action": "fold"}
        if a in ("check", "call", "cc", "check_call", "k", "c"):
            if a == "check" and not la["check"]:
                note = "check converted to call (facing a bet)"
            self.check_call(index)
            return {"action": "check" if la["check"] else "call", "amount": la["call"], "note": note}
        if a in ("bet", "raise", "raise_to", "bet_to", "cbr", "b", "r"):
            if not la["raise"]:
                note = "raise not allowed; converted to call/check"
                self.check_call(index)
                return {"action": "check" if la["check"] else "call", "amount": la["call"], "note": note}
            if amount is None:
                amount = la["min_raise_to"]
            amount = q(float(amount))
            clamped = min(max(amount, la["min_raise_to"]), la["max_raise_to"])
            if abs(clamped - amount) > EPS:
                note = f"raise size {amount} clamped to {clamped}"
            self.bet_raise_to(index, clamped)
            return {"action": la["raise_kind"], "amount": clamped, "note": note}
        if a in ("all_in", "allin", "shove", "jam"):
            if la["raise"]:
                self.bet_raise_to(index, la["max_raise_to"])
                return {"action": la["raise_kind"], "amount": la["max_raise_to"]}
            self.check_call(index)
            return {"action": "check" if la["check"] else "call", "amount": la["call"]}
        raise IllegalAction(f"unknown action {action!r}")

    # ---------------------------------------------------------------- showdown
    def _award_uncontested(self, p: Player) -> None:
        amount = self.pot
        p.stack = q(p.stack + amount)
        p.payout = q(p.payout + amount)
        self.pot = 0.0
        self.payouts = [pl.payout for pl in self.players]
        self.pots = [{"amount": amount, "eligible": [p.index], "winners": [p.index], "hand": None}]
        self._emit("payout", player=p.index, amount=amount, uncontested=True)
        self.status = "finished"
        self._emit("finished")

    def finalize(self) -> None:
        """Resolve the showdown and pay the pot (idempotent)."""
        if self.status == "finished":
            return
        if self.status != "showdown":
            previous = self.status
            self.status = "incomplete"
            self._emit("incomplete", reason=f"the record ended while the hand status was '{previous}'")
            return
        short = self.config.short_deck
        contenders = [p for p in self.players if not p.folded and not p.mucked]
        board_complete = len(self.board) == 5 and all(c is not None for c in self.board)
        for p in self.players:
            p.hand_score = None
        for p in contenders:
            cards = p.final_cards()
            if board_complete and len(cards) >= self.config.hole_count and all(c is not None for c in cards):
                p.hand_score = E.best_hand(cards, self.board, self.config.omaha, short)
        if len(contenders) == 1:
            self._award_uncontested(contenders[0])
            return
        known = [p for p in contenders if p.hand_score is not None]
        if not known:
            self.unresolved = True
            if self.winnings_hint is not None and len(self.winnings_hint) == self.n and sum(self.winnings_hint) > 0:
                self.warn("showdown hands unknown; pot awarded from the recorded winnings (post-rake)")
                self._apply_hint()
            else:
                self.warn("showdown hands unknown and no recorded winnings; pot split between contenders")
                for p in contenders:
                    p.hand_score = 0
                self._distribute(contenders)
            return
        for p in contenders:
            if p.hand_score is None:
                self.warn(f"{p.name} reached showdown with unknown cards; treated as a muck")
        self._distribute(known)
        if self.winnings_hint is not None and len(self.winnings_hint) == self.n:
            hinted = {i for i, w in enumerate(self.winnings_hint) if w > 0}
            ours = {p.index for p in self.players if p.payout > 0}
            if hinted and hinted != ours:
                self.warn(f"recorded winners {sorted(i + 1 for i in hinted)} differ from the computed winners "
                          f"{sorted(i + 1 for i in ours)}")

    def _apply_hint(self) -> None:
        assert self.winnings_hint is not None
        for p, w in zip(self.players, self.winnings_hint):
            w = q(float(w))
            if w > 0:
                p.stack = q(p.stack + w)
                p.payout = q(p.payout + w)
                self._emit("payout", player=p.index, amount=w, from_record=True)
        self.payouts = [p.payout for p in self.players]
        self.pots = [{"amount": self.pot, "eligible": [p.index for p in self.players if not p.folded],
                      "winners": [p.index for p in self.players if p.payout > 0], "hand": None,
                      "from_record": True}]
        self.pot = 0.0
        self.status = "finished"
        self._emit("finished")

    def _split(self, amount: float, k: int) -> List[float]:
        if self.split_mode == "exact":
            share = q(amount / k)
            shares = [share] * k
            drift = q(amount - share * k)
            if abs(drift) > EPS:
                shares[0] = q(shares[0] + drift)
            return shares
        # odd chips: use the finest unit actually in play this hand
        unit = self.quantum
        if unit > 0.01 and any(not float(q(p.contributed)).is_integer() for p in self.players):
            unit = 0.01
        if unit > 0.01 and not float(q(amount)).is_integer():
            unit = 0.01
        units = int(round(amount / unit))
        base, rem = divmod(units, k)
        return [q((base + (1 if j < rem else 0)) * unit) for j in range(k)]

    def _distribute(self, contenders: List[Player]) -> None:
        payouts = [0.0] * self.n
        pots: List[Dict[str, Any]] = []
        total = self.pot
        # Side pots are built from bet contributions.  Unless antes are "trimmed"
        # (ante_trimming_status), antes form a pool any contender can win - this is
        # what makes a big-blind ante work.
        trimming = self.config.ante_trimming_status
        if trimming:
            contrib = {p.index: q(p.contributed) for p in self.players}
            ante_pool = 0.0
        else:
            contrib = {p.index: q(p.contributed - p.ante_paid) for p in self.players}
            ante_pool = q(sum(p.ante_paid for p in self.players))
        bands: List[List[Any]] = []
        prev = 0.0
        for level in sorted({c for c in contrib.values() if c > EPS}):
            portion = q(sum(max(0.0, min(contrib[p.index], level) - prev) for p in self.players))
            prev = level
            if portion <= EPS:
                continue
            eligible = [p for p in contenders if contrib[p.index] >= level - EPS] or list(contenders)
            bands.append([portion, eligible])
        if ante_pool > EPS:
            bands.insert(0, [ante_pool, list(contenders)])
        for portion, eligible in bands:
            best = max(p.hand_score for p in eligible)  # type: ignore[type-var]
            winners = sorted([p for p in eligible if p.hand_score == best], key=lambda pl: pl.index)
            for w, s in zip(winners, self._split(portion, len(winners))):
                payouts[w.index] = q(payouts[w.index] + s)
            key = sorted(p.index for p in eligible)
            if pots and pots[-1]["eligible"] == key:
                pots[-1]["amount"] = q(pots[-1]["amount"] + portion)
            else:
                pots.append({"amount": portion, "eligible": key, "winners": [w.index for w in winners],
                             "hand": E.describe(best, self.config.short_deck) if best else None})  # type: ignore[arg-type]
        if abs(sum(payouts) - total) > 0.011:
            self.warn(f"pot accounting mismatch: pot {total} vs payouts {q(sum(payouts))}")
        for p in self.players:
            p.payout = payouts[p.index]
            if p.payout > EPS:
                p.stack = q(p.stack + p.payout)
                self._emit("payout", player=p.index, amount=p.payout,
                           hand=E.describe(p.hand_score, self.config.short_deck) if p.hand_score else None)
            if not self.scripted and p in contenders and p.shown is None:
                p.shown = list(p.hole)
                self._emit("show", player=p.index, cards=[C.card_str(c) for c in p.hole])
        self.payouts = payouts
        self.pots = pots
        self.pot = 0.0
        self.status = "finished"
        self._emit("finished")

    # --------------------------------------------------------------- reporting
    def action_history(self) -> List[Dict[str, Any]]:
        out = []
        for ev in self.events:
            if ev["type"] == "action":
                out.append({"street": ev["street"], "player": ev["player"], "action": ev["action"],
                            "amount": num(ev.get("amount")), "to": num(ev.get("to"))})
            elif ev["type"] == "deal_board":
                out.append({"street": ev["street"], "player": None, "action": "board", "cards": ev["cards"]})
        return out

    def player_dict(self, p: Player, visible: bool) -> Dict[str, Any]:
        cards = p.final_cards()
        hand_name = None
        if visible and cards and all(c is not None for c in cards) and len(cards) >= self.config.hole_count:
            score = E.best_hand(cards, self.board, self.config.omaha, self.config.short_deck)
            if score is not None:
                hand_name = E.describe(score, self.config.short_deck)
        return {
            "index": p.index,
            "name": p.name,
            "seat": self.seats[p.index] if self.seats else p.index + 1,
            "position": self.positions[p.index],
            "stack": num(p.stack),
            "starting_stack": num(p.starting_stack),
            "bet": q(p.bet),
            "contributed": q(p.contributed),
            "folded": p.folded,
            "all_in": p.all_in,
            "acted": p.acted,
            "mucked": p.mucked,
            "shown": p.shown is not None,
            "has_cards": bool(cards),
            "cards": [(C.card_str(c) if visible else ("??" if c is None else "XX")) for c in cards],
            "cards_known": bool(cards) and all(c is not None for c in cards),
            "hand": hand_name,
            "payout": q(p.payout),
            "net": num(p.stack - p.starting_stack) if p.stack != INF else None,
            "is_button": p.index == self.button,
        }

    def to_dict(self, viewer: Optional[int] = None, reveal: bool = False) -> Dict[str, Any]:
        players = []
        for p in self.players:
            visible = reveal or (viewer is not None and p.index == viewer) or p.shown is not None
            players.append(self.player_dict(p, visible))
        legal = self.legal_actions() if self.status == "betting" and self.actor is not None else None
        return {
            "hand_id": self.hand_id,
            "variant": self.config.variant,
            "variant_name": self.config.info["name"],
            "limit": self.config.limit,
            "status": self.status,
            "street": STREETS[self.street],
            "street_index": self.street,
            "board": [C.card_str(c) for c in self.board],
            "pot": q(self.pot),
            "total_pot": self.total_pot(),
            "max_bet": q(self.max_bet),
            "min_raise_to": q(self.max_bet + self.last_raise_size),
            "actor": self.actor,
            "button": self.button,
            "blinds": self.effective_blinds(),
            "antes": self.effective_antes(),
            "min_bet": self.config.min_bet,
            "players": players,
            "legal": legal,
            "pots": self.pots,
            "payouts": self.payouts,
            "unresolved": self.unresolved,
            "warnings": list(self.warnings),
        }

    def observation(self, index: int) -> Dict[str, Any]:
        """What a model playing ``index`` is allowed to know."""
        p = self.players[index]
        legal = self.legal_actions(index) if (self.status == "betting" and self.actor == index) else None
        return {
            "hand_id": self.hand_id,
            "variant": self.config.variant,
            "street": STREETS[self.street],
            "board": [C.card_str(c) for c in self.board],
            "pot": q(self.pot),
            "total_pot": self.total_pot(),
            "to_call": self.to_call(index),
            "min_bet": self.config.min_bet,
            "chip_unit": self.quantum,
            "blinds": self.effective_blinds(),
            "antes": self.effective_antes(),
            "hero": {
                "index": index,
                "name": p.name,
                "position": self.positions[index],
                "cards": [C.card_str(c) for c in p.hole],
                "stack": num(p.stack),
                "bet": q(p.bet),
                "contributed": q(p.contributed),
            },
            "players": [{
                "index": o.index, "name": o.name, "position": self.positions[o.index],
                "stack": num(o.stack), "bet": q(o.bet), "contributed": q(o.contributed),
                "folded": o.folded, "all_in": o.all_in, "is_button": o.index == self.button,
                "is_hero": o.index == index,
            } for o in self.players],
            "num_active": len(self.active_players()),
            "legal_actions": legal,
            "history": self.action_history(),
        }

    def to_phh_actions(self) -> List[str]:
        out = []
        for ev in self.events:
            t = ev["type"]
            if t == "deal_hole":
                out.append(f"d dh p{ev['player'] + 1} {''.join(ev['cards'])}")
            elif t == "deal_board":
                out.append(f"d db {''.join(ev['cards'])}")
            elif t == "action":
                pn = f"p{ev['player'] + 1}"
                if ev["action"] == "fold":
                    out.append(f"{pn} f")
                elif ev["action"] in ("check", "call"):
                    out.append(f"{pn} cc")
                else:
                    to = ev["to"]
                    out.append(f"{pn} cbr {int(to) if float(to).is_integer() else to}")
            elif t == "show":
                out.append(f"p{ev['player'] + 1} sm {''.join(ev['cards'])}")
            elif t == "muck":
                out.append(f"p{ev['player'] + 1} sm")
        return out


# ----------------------------------------------------------------------- table
@dataclass
class Seat:
    seat_no: int
    name: str
    stack: float
    agent: Any = None
    kind: str = "bot"
    sitting_out: bool = False
    hands_played: int = 0
    buy_in: float = 0.0


class Table:
    """Keeps seats, stacks and the dealer button across hands."""

    def __init__(self, seats: Sequence[Seat], *, small_blind: float, big_blind: float, ante: float = 0.0,
                 ante_mode: str = "uniform", variant: str = "NT", min_bet: Optional[float] = None,
                 rng: Optional[random.Random] = None, straddle: float = 0.0):
        self.seats = sorted(seats, key=lambda s: s.seat_no)
        if len(self.seats) < 2:
            raise EngineError("a table needs at least two seats")
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        self.ante = float(ante)
        self.ante_mode = ante_mode  # uniform | bb | button
        self.variant = variant
        self.min_bet = float(min_bet) if min_bet is not None else float(big_blind)
        self.straddle = float(straddle)
        self.rng = rng or random.Random()
        self.hand_no = 0
        self.button_seat = self.seats[0].seat_no
        self.current: Optional[Hand] = None
        self.current_order: List[Seat] = []
        self.history: List[Dict[str, Any]] = []

    def eligible(self) -> List[Seat]:
        return [s for s in self.seats if s.stack > EPS and not s.sitting_out]

    def _advance_button(self) -> None:
        nos = [s.seat_no for s in self.seats]
        start = nos.index(self.button_seat) if self.button_seat in nos else 0
        eligible = {s.seat_no for s in self.eligible()}
        for k in range(1, len(nos) + 1):
            cand = nos[(start + k) % len(nos)]
            if cand in eligible:
                self.button_seat = cand
                return

    def new_hand(self) -> Hand:
        if self.current is not None and self.current.status not in ("finished", "incomplete"):
            raise EngineError("the current hand is still running")
        elig = self.eligible()
        if len(elig) < 2:
            raise EngineError("need at least two players with chips")
        if self.button_seat not in {s.seat_no for s in elig}:
            self._advance_button()
        nos = [s.seat_no for s in elig]
        b = nos.index(self.button_seat)
        order = [elig[(b + 1 + k) % len(elig)] for k in range(len(elig))]  # ends with the button
        n = len(order)
        if self.ante_mode == "bb":
            antes = [0.0] * n
            antes[1 if n > 2 else 0] = self.ante
        elif self.ante_mode == "button":
            antes = [0.0] * n
            antes[n - 1] = self.ante
        else:
            antes = [self.ante] * n
        blinds = [self.small_blind, self.big_blind] + [0.0] * (n - 2)
        if self.straddle > 0 and n >= 3:
            blinds[2] = self.straddle
        cfg = GameConfig(self.variant, antes=antes, blinds=blinds, min_bet=self.min_bet)
        self.hand_no += 1
        hand = Hand(cfg, [s.stack for s in order], [s.name for s in order], rng=self.rng,
                    hand_id=self.hand_no, seats=[s.seat_no for s in order])
        self.current = hand
        self.current_order = order
        hand.start()
        return hand

    def finish_hand(self) -> Dict[str, Any]:
        """Write the finished hand's stacks back to the seats and move the button."""
        hand = self.current
        if hand is None:
            raise EngineError("no hand to finish")
        if hand.status not in ("finished", "incomplete"):
            raise EngineError("hand still running")
        summary = {"hand_no": self.hand_no, "results": []}
        for seat, p in zip(self.current_order, hand.players):
            net = q(p.stack - seat.stack)
            seat.stack = p.stack
            seat.hands_played += 1
            summary["results"].append({"seat": seat.seat_no, "name": seat.name, "net": net, "stack": num(seat.stack)})
        self.history.append(summary)
        self._advance_button()
        self.current = None  # current_order is kept so the finished hand can still be displayed
        return summary

    def seat_for_player(self, index: int) -> Seat:
        return self.current_order[index]
