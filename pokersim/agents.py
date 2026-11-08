"""The model interface.

An *agent* receives an observation (see :meth:`pokersim.engine.Hand.observation`)
and returns an action dict::

    {"action": "fold" | "check" | "call" | "bet" | "raise" | "all_in",
     "amount": <raise-to total for the street, optional>,
     "reasoning": <optional free text shown in the UI>}

Built-in agents: ``random``, ``station`` (calling station), ``maniac``, ``equity``
(Monte-Carlo pot-odds player), ``tag`` and ``lag`` (tight/loose aggressive
presets of the equity bot), ``http`` (POSTs the observation to your own model
server) and ``claude`` (asks a Claude model for a structured decision).
"""
from __future__ import annotations

import json
import os
import math
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import cards as C
from .equity import estimate_equity, preflop_strength

Observation = Dict[str, Any]
Action = Dict[str, Any]


class AgentError(Exception):
    pass


class Agent:
    kind = "agent"
    label = "Agent"
    description = ""

    def __init__(self, **params: Any):
        self.params = params
        self.name = params.get("name") or self.label
        self.rng = random.Random(params.get("seed"))

    def act(self, obs: Observation) -> Action:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "name": self.name,
                "params": {k: v for k, v in self.params.items() if k not in ("api_key",)}}


# --------------------------------------------------------------------- helpers
def legal(obs: Observation) -> Dict[str, Any]:
    la = obs.get("legal_actions") or {}
    return {
        "check": bool(la.get("check")),
        "fold": bool(la.get("fold")),
        "call": float(la.get("call") or 0.0),
        "raise": bool(la.get("raise")),
        "min_to": float(la.get("min_raise_to") or 0.0),
        "max_to": float(la.get("max_raise_to") or 0.0),
        "kind": la.get("raise_kind", "bet"),
    }


def hero_cards(obs: Observation) -> List[Optional[int]]:
    return [C.parse_card(c) for c in obs["hero"]["cards"]]


def board_cards(obs: Observation) -> List[Optional[int]]:
    return [C.parse_card(c) for c in obs["board"]]


def passive(obs: Observation) -> Action:
    return {"action": "check" if legal(obs)["check"] else "call"}


def round_to_unit(amount: float, obs: Observation, la: Dict[str, Any]) -> float:
    """Round a raise-to amount to the table's chip unit and keep it legal."""
    unit = float(obs.get("chip_unit") or 0.01)
    if unit >= 1:
        amount = round(amount / unit) * unit
    amount = min(max(amount, la["min_to"]), la["max_to"])
    return round(amount, 2)


def sized_raise(obs: Observation, fraction_of_pot: float, la: Dict[str, Any]) -> Action:
    """A bet/raise of roughly ``fraction_of_pot`` of the pot after calling."""
    pot_after_call = float(obs["total_pot"]) + la["call"]
    hero_bet = float(obs["hero"]["bet"])
    target = hero_bet + la["call"] + fraction_of_pot * pot_after_call
    return {"action": la["kind"], "amount": round_to_unit(target, obs, la)}


# ------------------------------------------------------------------- built-ins
class RandomAgent(Agent):
    kind = "random"
    label = "Wildcard"
    description = "Plays at random: any hand, any bet."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        options = ["check" if la["check"] else "call"]
        if la["fold"]:
            options.append("fold")
        if la["raise"]:
            options.append("raise")
        choice = self.rng.choice(options)
        if choice == "raise":
            return sized_raise(obs, self.rng.choice([0.33, 0.5, 0.75, 1.0]), la)
        return {"action": choice}


class CallingStationAgent(Agent):
    kind = "station"
    label = "Caller"
    description = "Calls almost everything and rarely folds or raises."

    def act(self, obs: Observation) -> Action:
        return passive(obs)


class ManiacAgent(Agent):
    kind = "maniac"
    label = "Maniac"
    description = "Raises and bluffs constantly, whatever the cards."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        freq = float(self.params.get("frequency", 0.7))
        if la["raise"] and self.rng.random() < freq:
            return sized_raise(obs, self.rng.choice([0.66, 1.0, 1.5]), la)
        return passive(obs)


class EquityAgent(Agent):
    """Pot-odds player: Chen formula pre-flop, Monte-Carlo equity post-flop.

    ``aggression`` (0..1) scales how often it bets/raises and bluffs,
    ``tightness`` (0..1) raises the strength needed to continue, ``iterations``
    controls the Monte-Carlo sample size.
    """

    kind = "equity"
    label = "Calculator"
    description = "Plays the odds: calls when the price is right for its hand's chances."

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.aggression = float(params.get("aggression", 0.5))
        self.tightness = float(params.get("tightness", 0.5))
        self.iterations = int(params.get("iterations", 300))

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        hole = hero_cards(obs)
        if any(c is None for c in hole) or not hole:
            return passive(obs)
        board = board_cards(obs)
        active = [p for p in obs["players"] if not p["folded"] and not p["is_hero"]]
        n_opp = max(1, len(active))
        pot = float(obs["total_pot"])
        to_call = la["call"]
        stack = obs["hero"]["stack"] if obs["hero"]["stack"] is not None else 1e12
        variant = obs.get("variant", "NT")
        omaha = variant == "PO"
        short = variant == "NS"

        if obs["street"] == "preflop" and not omaha:
            strength = preflop_strength(hole)  # 0..1
            # position bonus: later positions play more hands
            n = len(obs["players"])
            pos_idx = obs["hero"]["index"]
            late = (pos_idx + 1) / n  # 1.0 for the button
            strength += 0.06 * late
            raise_bar = 0.46 + 0.25 * self.tightness - 0.10 * self.aggression
            call_bar = 0.26 + 0.25 * self.tightness
            facing_raise = to_call > float(obs["min_bet"]) * 1.01
            if facing_raise:
                raise_bar += 0.12
                call_bar += 0.10
            if la["raise"] and strength >= raise_bar and self.rng.random() < 0.55 + 0.45 * self.aggression:
                frac = 2.5 if not facing_raise else 1.0
                target = max(la["min_to"], float(obs["min_bet"]) * (2.2 + self.rng.random() * 0.8) if not facing_raise
                             else la["min_to"] * (1.0 + 0.2 * self.rng.random()))
                target = min(target, la["max_to"], max(la["min_to"], pot * frac))
                return {"action": la["kind"], "amount": round_to_unit(target, obs, la)}
            if strength >= call_bar or to_call <= 0:
                if to_call > stack * 0.5 and strength < raise_bar:
                    return {"action": "fold"} if la["fold"] else {"action": "check"}
                return passive(obs)
            if la["fold"] and self.rng.random() > 0.08 * self.aggression:
                return {"action": "fold"}
            return passive(obs)

        eq = estimate_equity(
            [c for c in hole if c is not None], board, n_opp, iterations=self.iterations,
            short_deck=short, omaha=omaha, rng=self.rng,
        )["equity"]
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
        value_bar = 0.55 + 0.2 * self.tightness - 0.1 * self.aggression
        if la["raise"] and eq >= value_bar and self.rng.random() < 0.6 + 0.4 * self.aggression:
            frac = 0.5 + 0.5 * self.rng.random() + 0.4 * (eq - value_bar)
            act = sized_raise(obs, frac, la)
            if eq > 0.9 and stack < pot * 2:
                act = {"action": "all_in"}
            return act
        if la["raise"] and to_call == 0 and self.rng.random() < 0.12 * self.aggression:
            return sized_raise(obs, 0.5, la)  # occasional bluff / probe
        if to_call <= 0:
            return {"action": "check"}
        margin = 0.03 + 0.08 * self.tightness
        if eq >= pot_odds + margin:
            return {"action": "call"}
        if la["fold"]:
            return {"action": "fold"}
        return passive(obs)


class TightAggressiveAgent(EquityAgent):
    kind = "tag"
    label = "Shark"
    description = "Tight and aggressive: plays few hands, and bets them hard."

    def __init__(self, **params: Any):
        params.setdefault("aggression", 0.75)
        params.setdefault("tightness", 0.75)
        super().__init__(**params)


class LooseAggressiveAgent(EquityAgent):
    kind = "lag"
    label = "Hustler"
    description = "Loose and aggressive: plays many hands and keeps the pressure on."

    def __init__(self, **params: Any):
        params.setdefault("aggression", 0.9)
        params.setdefault("tightness", 0.2)
        super().__init__(**params)


class ScriptedAgent(Agent):
    """Plays a predetermined list of actions (used in tests)."""

    kind = "scripted"
    label = "Scripted"

    def __init__(self, actions: Sequence[Action], **params: Any):
        super().__init__(**params)
        self.actions = list(actions)
        self.pos = 0

    def act(self, obs: Observation) -> Action:
        if self.pos >= len(self.actions):
            return passive(obs)
        a = self.actions[self.pos]
        self.pos += 1
        return a


class HTTPAgent(Agent):
    """POSTs the observation as JSON to ``url`` and expects an action JSON back.

    Request body: the observation dict.  Response: ``{"action": ..., "amount": ...}``.
    """

    kind = "http"
    label = "HTTP model"
    description = "Your own model behind an HTTP endpoint (POST observation JSON, reply with an action JSON)."

    def __init__(self, url: str = "http://127.0.0.1:5001/act", timeout: float = 30.0, **params: Any):
        super().__init__(url=url, timeout=timeout, **params)
        self.url = url
        self.timeout = float(timeout)

    def act(self, obs: Observation) -> Action:
        data = json.dumps(obs).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise AgentError(f"HTTP agent request failed: {exc}") from exc
        try:
            out = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AgentError(f"HTTP agent returned invalid JSON: {body[:200]!r}") from exc
        if not isinstance(out, dict) or "action" not in out:
            raise AgentError(f"HTTP agent response must be an object with an 'action': {body[:200]!r}")
        return out


def describe_observation(obs: Observation) -> str:
    """Render an observation as compact text for an LLM."""
    hero = obs["hero"]
    la = legal(obs)
    lines = [
        f"Game: {obs.get('variant', 'NT')} ({'no-limit hold em' if obs.get('variant') == 'NT' else obs.get('variant')}).",
        f"Blinds: {'/'.join(str(b) for b in obs['blinds'] if b)}; antes: {sum(obs['antes'])}.",
        f"Street: {obs['street']}. Board: {' '.join(C.pretty(C.parse_card(c)) for c in obs['board']) or '(none yet)'}.",
        f"You are {hero['name']} in position {hero['position']} with {' '.join(C.pretty(C.parse_card(c)) for c in hero['cards'])}.",
        f"Your stack: {hero['stack']}; your bet this street: {hero['bet']}; pot (including bets): {obs['total_pot']}.",
        "Players:",
    ]
    for p in obs["players"]:
        status = "folded" if p["folded"] else ("all-in" if p["all_in"] else "active")
        lines.append(f"  - {p['name']} ({p['position']}{', button' if p['is_button'] else ''}): stack {p['stack']}, "
                     f"bet {p['bet']}, {status}{' <- you' if p['is_hero'] else ''}")
    if obs.get("history"):
        lines.append("Action so far:")
        for h in obs["history"]:
            if h["action"] == "board":
                lines.append(f"  [{h['street']}] board {' '.join(h['cards'])}")
            else:
                name = obs["players"][h["player"]]["name"]
                amt = f" to {h['to']}" if h["action"] in ("bet", "raise") else (f" {h['amount']}" if h["action"] == "call" else "")
                lines.append(f"  [{h['street']}] {name} {h['action']}{amt}")
    opts = []
    if la["fold"]:
        opts.append("fold")
    if la["check"]:
        opts.append("check")
    else:
        opts.append(f"call {la['call']}")
    if la["raise"]:
        opts.append(f"{la['kind']} to any total between {la['min_to']} and {la['max_to']} (amount = your total bet this street)")
    lines.append("Legal actions: " + "; ".join(opts) + ".")
    return "\n".join(lines)


class ClaudeAgent(Agent):
    """Asks a Claude model for a decision using structured JSON output.

    Needs the ``anthropic`` package and credentials (``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile).  Refusal fallbacks are enabled by default.
    """

    kind = "claude"
    label = "Claude"
    description = "Claude decides each action (requires the anthropic package and an API key)."

    SCHEMA = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["fold", "check", "call", "bet", "raise", "all_in"]},
            "amount": {"type": "number", "description": "Total bet for the street when betting/raising; 0 otherwise."},
            "reasoning": {"type": "string", "description": "One or two sentences."},
        },
        "required": ["action", "amount", "reasoning"],
        "additionalProperties": False,
    }
    SYSTEM = (
        "You are a strong, disciplined no-limit hold'em player. You receive the full table state and must "
        "choose exactly one legal action. Think about position, pot odds, stack sizes and the betting history. "
        "Amounts are totals for the current street (raise *to*), and must respect the legal range given. "
        "Reply only with the JSON object."
    )

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", style: str = "", **params: Any):
        super().__init__(model=model, effort=effort, style=style, **params)
        self.model = model
        self.effort = effort
        self.style = style
        self._client = None
        self.last_usage: Optional[Dict[str, Any]] = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise AgentError("the 'anthropic' package is not installed (pip install anthropic)") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def act(self, obs: Observation) -> Action:
        client = self._get_client()
        prompt = describe_observation(obs)
        if self.style:
            prompt += f"\n\nStyle instructions: {self.style}"
        prompt += "\n\nChoose your action."
        try:
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=2048,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=self.SYSTEM,
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": self.SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a fallback
            raise AgentError(f"Claude request failed: {exc}") from exc
        if getattr(response, "stop_reason", None) == "refusal":
            raise AgentError("Claude declined to answer (refusal)")
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {"input_tokens": getattr(usage, "input_tokens", None),
                               "output_tokens": getattr(usage, "output_tokens", None)}
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
        if not text:
            raise AgentError("Claude returned no text block")
        try:
            out = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentError(f"Claude returned invalid JSON: {text[:200]!r}") from exc
        return out


# -------------------------------------------------------------------- registry
class RLAgent(Agent):
    """The PPO-trained 6-max policy (``data/rl/policy.npz``, see TRAINING.md).

    A table can hand it opponent style estimates through ``styles_provider(obs) -> {index: style}``;
    without one it assumes population-average opponents."""

    kind = "rl"
    label = "RL model"
    description = "PPO policy trained by self-play against many opponent styles; uses opponent style estimates."

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.greedy = bool(params.get("greedy", False))
        self.styles_provider: Optional[Callable[[Observation], Dict[int, Dict[str, Any]]]] = None

    def act(self, obs: Observation) -> Action:
        from .rl_policy import act as rl_act, load_policy

        policy = load_policy()
        if policy is None:
            return passive(obs)
        styles = self.styles_provider(obs) if self.styles_provider else {}
        return rl_act(obs, styles, policy, self.rng, greedy=self.greedy)


class ProAgent(Agent):
    """Plays like the players of the Pluribus logs: the next-move model (data/rl/next_move.npz) with its own cards.

    ``temp`` scales the randomness, ``tilt_f`` / ``tilt_c`` / ``tilt_r`` shift the logits of fold / call / raise."""

    kind = "pro"
    label = "Pro"
    description = "Plays like the professionals in the Pluribus games (learned from their decisions)."
    pluribus = False

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.temp = float(params.get("temp", 1.0))
        self.tilt = (float(params.get("tilt_f", 0.0)), float(params.get("tilt_c", 0.0)), float(params.get("tilt_r", 0.0)))

    def act(self, obs: Observation) -> Action:
        from .next_move import sample_action

        return sample_action(obs, self.rng, pluribus=self.pluribus, temp=self.temp, tilt=self.tilt)


TYPE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "demo", "type_tilts.json")
TYPE_CODES = {"tag": "tight aggressive", "lag": "loose aggressive", "lp": "loose passive", "tp": "tight passive"}
_TYPE_TABLE: Dict[str, Any] = {}


def type_table() -> Dict[str, Any]:
    if "t" not in _TYPE_TABLE:
        try:
            with open(TYPE_FILE, encoding="utf-8") as f:
                _TYPE_TABLE["t"] = json.load(f)
        except (OSError, ValueError):
            _TYPE_TABLE["t"] = {}
    return _TYPE_TABLE["t"]


class TypeAgent(ProAgent):
    """A player of one classic type, mined from real regulars (scripts/mine_types.py, scripts/fit_types.py): the
    card-aware model of the professionals, tilted situation by situation (street x facing a bet x position) so that
    its fold / call / raise rates match its type's, played at the temperature that won most for that type."""

    kind = "type"
    label = "Regular"
    description = "Plays like real regulars of one type (tight or loose, aggressive or passive), mined from online hands."

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.style = TYPE_CODES.get(str(params.get("style", "")).lower())
        fit = type_table().get("types", {}).get(self.style or "", {})
        self.table = fit.get("tilt")  # [street][facing][position] -> [fold, call, raise]
        if self.table is not None and "temp" not in params:
            self.temp = float(fit.get("temp", 1.0))

    def tilt_for(self, obs: Observation) -> Tuple[float, float, float]:
        if not self.table:
            return self.tilt
        n, i = len(obs["players"]), obs["hero"]["index"]
        street = ["preflop", "flop", "turn", "river"].index(obs["street"]) if obs["street"] in ("preflop", "flop", "turn", "river") else 3
        facing = 1 if float((obs.get("legal_actions") or {}).get("call") or 0) > 0 else 0
        pos = 0 if i < 2 else 2 if i >= n - 2 else 1  # blinds / early / late; the button is the last index
        t = self.table[street][facing][pos]
        return (t[0] + self.tilt[0], t[1] + self.tilt[1], t[2] + self.tilt[2])

    def act(self, obs: Observation) -> Action:
        from .next_move import sample_action

        return sample_action(obs, self.rng, pluribus=self.pluribus, temp=self.temp, tilt=self.tilt_for(obs))


class ReaderAgent(TypeAgent):
    """A Pro that reads the table (the Poker Mirror's Shark): before each move it asks the models of the players still
    in how likely they all are to fold if it bets about the pot, against how often people in general would, and raises
    more when they fold more (it bluffs them) and less when they call more.  One level only: it reads what players do,
    never what they think it will do, and only it adapts, so there is no loop.  The shift is capped at ±1 on the
    raise logit.  Without a ``reads_provider`` (outside a live table) it plays as a plain tilted Pro."""

    kind = "reader"
    label = "Shark"
    description = "Plays like the professionals and reads the table: bluffs players who fold too much, not those who call."
    CAP = 1.0

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.reads_provider = None
        self.last_read: Optional[Dict[str, Any]] = None

    def act(self, obs: Observation) -> Action:
        from .next_move import sample_action

        tilt = self.tilt_for(obs)
        if self.reads_provider is not None:
            try:
                r = self.reads_provider(obs)
            except Exception:  # noqa: BLE001 - reading must never stop the bot from playing
                r = None
            if r and r.get("all_fold") is not None:
                lo = lambda q: math.log(max(q, 1e-3) / max(1.0 - q, 1e-3))  # noqa: E731
                delta = max(-self.CAP, min(self.CAP, lo(r["all_fold"]) - lo(r["people"])))
                tilt = (tilt[0], tilt[1], tilt[2] + delta)
                self.last_read = {**r, "delta": round(delta, 2), "hand_id": obs.get("hand_id"),
                                  "action": "bluffs more" if delta > 0.15 else "bluffs less" if delta < -0.15 else "plays its normal game"}
        return sample_action(obs, self.rng, pluribus=self.pluribus, temp=self.temp, tilt=tilt)


class PluribusLikeAgent(ProAgent):
    kind = "plur"
    label = "Machine"
    description = "Plays like the Pluribus poker AI (learned from its decisions); never seen by the RL model in training."
    pluribus = True


class _RuleAgent(Agent):
    """Shared helpers for the held-out rule bots (never used as RL training opponents)."""

    def _strength(self, obs: Observation) -> float:
        hole = [c for c in hero_cards(obs) if c is not None]
        return preflop_strength(hole) if len(hole) == 2 else 0.0

    def _equity(self, obs: Observation) -> float:
        hole = [c for c in hero_cards(obs) if c is not None]
        n_opp = max(1, sum(1 for p in obs["players"] if not p["folded"] and not p["is_hero"]))
        return float(estimate_equity(hole, board_cards(obs), n_opp, iterations=60, rng=self.rng)["equity"])

    def _fold_or_check(self, obs: Observation) -> Action:
        la = legal(obs)
        return {"action": "check"} if la["check"] else {"action": "fold"} if la["fold"] else passive(obs)


class NitShoverAgent(_RuleAgent):
    kind = "nit"
    label = "Rock"
    description = "Waits for premium hands, then goes all-in."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        if obs["street"] == "preflop":
            if self._strength(obs) >= 0.62:
                return {"action": "all_in"} if la["raise"] else passive(obs)
            return self._fold_or_check(obs) if la["call"] > 0 else passive(obs)
        eq = self._equity(obs)
        if eq > 0.7 and la["raise"]:
            return {"action": "all_in"}
        return passive(obs) if (la["call"] <= 0 or eq > 0.5) else self._fold_or_check(obs)


class TrapperAgent(_RuleAgent):
    kind = "trapper"
    label = "Trapper"
    description = "Slow-plays strong hands, then springs a big raise on the river."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        if obs["street"] == "preflop":
            return passive(obs) if self._strength(obs) >= 0.38 or la["call"] <= 0 else self._fold_or_check(obs)
        eq = self._equity(obs)
        if eq > 0.75:
            if obs["street"] == "river" and la["raise"]:
                return sized_raise(obs, 1.0, la)
            return passive(obs)
        pot = float(obs["total_pot"])
        to_call = la["call"]
        if to_call <= 0:
            return {"action": "check"}
        return passive(obs) if eq >= to_call / (pot + to_call) + 0.05 else self._fold_or_check(obs)


class MinRaiserAgent(_RuleAgent):
    kind = "minraiser"
    label = "Nibbler"
    description = "Makes lots of small raises, backs off when someone fights back."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        s = self._strength(obs) if obs["street"] == "preflop" else self._equity(obs)
        facing = la["call"] > float(obs["min_bet"]) * 1.01
        if la["raise"] and s >= (0.45 if obs["street"] == "preflop" else 0.5) and not (facing and s < 0.7):
            return {"action": la["kind"], "amount": round_to_unit(la["min_to"], obs, la)}
        if facing and s < 0.6:
            return self._fold_or_check(obs)
        return passive(obs)


class PotOddsAgent(_RuleAgent):
    kind = "potodds"
    label = "Accountant"
    description = "Calls exactly when the odds say so, bets for value, never bluffs."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        eq = self._strength(obs) * 0.9 if obs["street"] == "preflop" else self._equity(obs)
        pot = float(obs["total_pot"])
        to_call = la["call"]
        if la["raise"] and eq > 0.68:
            return sized_raise(obs, 0.5, la)
        if to_call <= 0:
            return {"action": "check"}
        return passive(obs) if eq >= to_call / (pot + to_call) else self._fold_or_check(obs)


# The three bots below are for the sealed final test only (scripts/sealed_eval.py): written after training started,
# never RL training opponents and never used to pick a model.
class OverbettorAgent(_RuleAgent):
    kind = "overbettor"
    label = "Bully"
    description = "Bets twice the pot with strong hands and with some bluffs; folds the middle."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        s = self._strength(obs) if obs["street"] == "preflop" else self._equity(obs)
        strong, weak = (0.6, 0.25) if obs["street"] == "preflop" else (0.7, 0.3)
        if la["raise"] and (s >= strong or (s < weak and self.rng.random() < 0.2)):
            return sized_raise(obs, 2.0, la)
        to_call = la["call"]
        if to_call <= 0:
            return {"action": "check"}
        return passive(obs) if s >= to_call / (float(obs["total_pot"]) + to_call) + 0.1 else self._fold_or_check(obs)


class LimperAgent(_RuleAgent):
    kind = "limper"
    label = "Tourist"
    description = "Limps in with many hands, leads out small with any piece, rarely raises."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        to_call = la["call"]
        if obs["street"] == "preflop":
            s = self._strength(obs)
            if s >= 0.75 and la["raise"]:
                return {"action": la["kind"], "amount": round_to_unit(la["min_to"], obs, la)}
            cheap = to_call <= float(obs["min_bet"]) * 1.01
            return passive(obs) if (s >= 0.3 and cheap) or s >= 0.5 or to_call <= 0 else self._fold_or_check(obs)
        eq = self._equity(obs)
        if to_call <= 0:
            return sized_raise(obs, 0.5, la) if eq > 0.45 and la["raise"] else {"action": "check"}
        if eq > 0.85 and la["raise"]:
            return sized_raise(obs, 1.0, la)
        return passive(obs) if eq >= 0.35 else self._fold_or_check(obs)


class CheckRaiserAgent(_RuleAgent):
    kind = "checkraiser"
    label = "Ambusher"
    description = "Checks after the flop, then raises when bet into with a strong hand."

    def act(self, obs: Observation) -> Action:
        la = legal(obs)
        to_call = la["call"]
        if obs["street"] == "preflop":
            s = self._strength(obs)
            if s >= 0.6 and la["raise"]:
                return sized_raise(obs, 0.75, la)
            return passive(obs) if s >= 0.4 or to_call <= 0 else self._fold_or_check(obs)
        eq = self._equity(obs)
        if to_call <= 0:
            return sized_raise(obs, 1.0, la) if obs["street"] == "river" and eq > 0.8 and la["raise"] else {"action": "check"}
        if eq > 0.65 and la["raise"]:
            return sized_raise(obs, 1.0, la)
        return passive(obs) if eq >= to_call / (float(obs["total_pot"]) + to_call) else self._fold_or_check(obs)


AGENT_TYPES: Dict[str, type] = {
    "rl": RLAgent,
    "pro": ProAgent,
    "plur": PluribusLikeAgent,
    "reader": ReaderAgent,
    "type": TypeAgent,
    "nit": NitShoverAgent,
    "trapper": TrapperAgent,
    "minraiser": MinRaiserAgent,
    "potodds": PotOddsAgent,
    "overbettor": OverbettorAgent,
    "limper": LimperAgent,
    "checkraiser": CheckRaiserAgent,
    "random": RandomAgent,
    "station": CallingStationAgent,
    "maniac": ManiacAgent,
    "equity": EquityAgent,
    "tag": TightAggressiveAgent,
    "lag": LooseAggressiveAgent,
    "http": HTTPAgent,
    "claude": ClaudeAgent,
}


def available_agents() -> List[Dict[str, Any]]:
    out = []
    for key, cls in AGENT_TYPES.items():
        out.append({"id": key, "label": cls.label, "description": cls.description})
    return out


def _coerce(v: str) -> Any:
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def make_agent(spec: str, **extra: Any) -> Agent:
    """Build an agent from a spec string.

    Examples: ``"equity"``, ``"equity:aggression=0.8,iterations=200"``,
    ``"http:http://localhost:5001/act"``, ``"claude:claude-opus-5-5"``.
    """
    spec = (spec or "equity").strip()
    kind, _, rest = spec.partition(":")
    kind = kind.lower()
    if kind not in AGENT_TYPES:
        raise AgentError(f"unknown agent type {kind!r}; available: {', '.join(AGENT_TYPES)}")
    params: Dict[str, Any] = dict(extra)
    if kind == "http":
        if rest:
            params["url"] = rest
    elif kind == "claude":
        if rest and "=" not in rest.split(",")[0]:
            model, _, rest = rest.partition(",")
            params["model"] = model
        for part in filter(None, rest.split(",")):
            k, _, v = part.partition("=")
            params[k.strip()] = _coerce(v.strip())
    else:
        for part in filter(None, rest.split(",")):
            k, _, v = part.partition("=")
            if k.strip():
                params[k.strip()] = _coerce(v.strip())
    return AGENT_TYPES[kind](**params)


def safe_act(agent: Agent, obs: Observation, fallback: Callable[[Observation], Action] = passive) -> Action:
    """Call ``agent.act`` and fall back to check/call on any error (recorded in the result)."""
    t0 = time.perf_counter()
    try:
        action = agent.act(obs)
        if not isinstance(action, dict) or "action" not in action:
            raise AgentError(f"agent returned {action!r}")
    except Exception as exc:  # noqa: BLE001
        action = fallback(obs)
        action["error"] = f"{type(exc).__name__}: {exc}"
    action["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return action
