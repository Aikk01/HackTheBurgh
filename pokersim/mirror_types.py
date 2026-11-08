"""Shared vocabulary for Poker Mirror.

Every workstream (data, model, environment, demo) emits and consumes the same
:class:`DecisionRecord`, so a historical replay, a practice drill and a
simulated episode all look alike downstream.  Keep this module dependency-free.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Public turn context: what happened on the turn before the river check / bet.
CTX_CHECKED_THROUGH = "checked_through"   # both players checked the turn
CTX_BETTOR_AGGR = "bettor_aggr"           # the river bettor made the last bet/raise on the turn
CTX_DEFENDER_AGGR = "defender_aggr"       # the river defender made the last bet/raise on the turn
CONTEXTS = (CTX_CHECKED_THROUGH, CTX_BETTOR_AGGR, CTX_DEFENDER_AGGR)

# Defender responses to the half-pot bet.
RESPONSES = ("fold", "call", "raise")

# Bettor strength buckets on the river (exact equity vs. all legal holdings).
BUCKETS = ("weak", "medium", "strong")
BUCKET_CUTS = (0.33, 0.67)

# The exercise is fixed: opening pot 100, bet 50, raise to 100.
OPENING_POT = 100
BET_SIZE = 50
RAISE_TO = 100

# Model arms.
ARM_POPULATION = "population"
ARM_TYPE = "type"
ARM_PERSONAL = "personal"


def bucket_of(equity: float) -> str:
    if equity < BUCKET_CUTS[0]:
        return BUCKETS[0]
    if equity < BUCKET_CUTS[1]:
        return BUCKETS[1]
    return BUCKETS[2]


@dataclass
class Prediction:
    """A forecast committed *before* the defender acts."""

    probs: Dict[str, float]                  # fold / call / raise, sums to one over legal responses
    fold_mean: float
    fold_p05: float
    fold_p95: float
    arm: str                                 # population | type | personal
    model_version: str
    observations: int                        # how many responses of this player the arm has seen
    type_posterior: Optional[Dict[str, float]] = None


@dataclass
class DecisionRecord:
    """One river decision: the defender checks and faces a half-pot bet."""

    session_id: str
    decision_id: str
    context: str                             # one of CONTEXTS
    pot_before_bet: float
    bet: float
    legal: Dict[str, bool]                   # fold / call / raise
    board: List[str]
    own_cards: List[str] = field(default_factory=list)      # the viewer's cards (bettor seat) or []
    defender_strength: Optional[str] = None                  # bucket, known only inside the defender policy
    prediction: Optional[Prediction] = None
    actual: Optional[str] = None             # fold / call / raise once observed
    source: str = "practice"                 # practice | historical | simulated
    player_id: Optional[str] = None          # historical handle or session token
    range_readout: Optional[Dict[str, Any]] = None
    ev_table: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DecisionRecord":
        d = json.loads(text)
        pred = d.get("prediction")
        d["prediction"] = Prediction(**pred) if pred else None
        return cls(**d)


EXAMPLE = DecisionRecord(
    session_id="demo",
    decision_id="demo-1",
    context=CTX_CHECKED_THROUGH,
    pot_before_bet=100,
    bet=50,
    legal={"fold": True, "call": True, "raise": True},
    board=["Kc", "9d", "7h", "4s", "3c"],
    own_cards=["6h", "2s"],
    prediction=Prediction({"fold": 0.62, "call": 0.33, "raise": 0.05}, 0.62, 0.40, 0.82,
                          ARM_POPULATION, "population-v1", 0),
    actual=None,
)
