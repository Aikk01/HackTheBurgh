"""Opponent models for Poker Mirror (PLAN.md section 3).

* :class:`PopulationModel` - a categorical response distribution per public
  turn context, fitted on the training manifest.
* :class:`PersonalModel` - an online Dirichlet-categorical classifier per
  player with an optional mixture-of-types prior (K = 1 is the plain model).
* :class:`StrengthTable` - response probabilities by context and the
  defender's own strength bucket, with calibration backoff.
* :class:`FrozenProfile` / :class:`Twin` - an immutable snapshot that plays
  as the defender in practice and in RL training.

Everything here is standard library only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .mirror_types import ARM_PERSONAL, ARM_POPULATION, ARM_TYPE, BUCKETS, CONTEXTS, RESPONSES, Prediction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POPULATION_PATH = os.path.join(ROOT, "data", "mirror", "population.json")
DEFAULT_PRIOR_STRENGTH = 8.0

# Synthetic default strength prior: how much more a weak hand folds than a strong one.
# Labelled as a default everywhere it is shown; replaced by calibration sessions.
DEFAULT_STRENGTH_PRIOR: Dict[str, Dict[str, float]] = {
    "weak": {"fold": 0.80, "call": 0.17, "raise": 0.03},
    "medium": {"fold": 0.40, "call": 0.52, "raise": 0.08},
    "strong": {"fold": 0.05, "call": 0.60, "raise": 0.35},
}
# Bucket shares under the uniform deal distribution (equity is close to uniform over holdings).
DEFAULT_BUCKET_SHARES = {"weak": 0.33, "medium": 0.34, "strong": 0.33}


# ----------------------------------------------------------------- beta helpers (no SciPy at runtime)
def _betacf(a: float, b: float, x: float) -> float:
    max_it, eps, tiny = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    front = math.exp(ln_beta)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def beta_ppf(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def dirichlet_multinomial_log(counts: Dict[str, int], alpha: Dict[str, float]) -> float:
    """log DirMult(n; alpha) = lgamma(sum alpha) - lgamma(sum alpha + sum n) + sum_a [lgamma(alpha_a + n_a) - lgamma(alpha_a)]."""
    sa = sum(alpha.values())
    sn = sum(counts.values())
    out = math.lgamma(sa) - math.lgamma(sa + sn)
    for a in RESPONSES:
        out += math.lgamma(alpha.get(a, 0.0) + counts.get(a, 0)) - math.lgamma(alpha.get(a, 0.0))
    return out


def _normalise(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    return {k: (v / s if s > 0 else 1.0 / len(d)) for k, v in d.items()}


def _masked(probs: Dict[str, float], legal: Optional[Dict[str, bool]]) -> Dict[str, float]:
    if not legal:
        return dict(probs)
    out = {a: (probs[a] if legal.get(a, True) else 0.0) for a in RESPONSES}
    return _normalise(out)


# ----------------------------------------------------------------- population
class PopulationModel:
    """q[c][a] = (n[c][a] + 1) / (sum_a n[c][a] + 3), with the pooled distribution as fallback."""

    version = "population-v1"

    def __init__(self, by_context: Dict[str, Dict[str, int]], pooled: Dict[str, int], meta: Optional[Dict[str, Any]] = None):
        self.counts = {c: {a: int(by_context.get(c, {}).get(a, 0)) for a in RESPONSES} for c in CONTEXTS}
        self.pooled_counts = {a: int(pooled.get(a, 0)) for a in RESPONSES}
        self.meta = dict(meta or {})
        self.q = {c: self._smooth(self.counts[c]) for c in CONTEXTS}
        self.q_pooled = self._smooth(self.pooled_counts)

    @staticmethod
    def _smooth(n: Dict[str, int]) -> Dict[str, float]:
        total = sum(n.values()) + 3
        return {a: (n[a] + 1) / total for a in RESPONSES}

    @classmethod
    def load(cls, path: str = POPULATION_PATH) -> "PopulationModel":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        meta = {k: v for k, v in d.items() if k not in ("by_context", "pooled")}
        return cls(d["by_context"], d["pooled"], meta)

    @classmethod
    def from_spots(cls, spots: Iterable[Any]) -> "PopulationModel":
        by = {c: {a: 0 for a in RESPONSES} for c in CONTEXTS}
        pooled = {a: 0 for a in RESPONSES}
        for s in spots:
            if getattr(s, "exact_half", False) and getattr(s, "all_three_legal", False):
                by[s.context][s.response] += 1
                pooled[s.response] += 1
        return cls(by, pooled, {"source": "spots"})

    def probs(self, context: str) -> Dict[str, float]:
        if context in self.q and sum(self.counts[context].values()) > 0:
            return dict(self.q[context])
        return dict(self.q_pooled)

    def n(self, context: str) -> int:
        return sum(self.counts.get(context, {}).values())

    def predict(self, context: str, legal: Optional[Dict[str, bool]] = None) -> Prediction:
        probs = _masked(self.probs(context), legal)
        n = self.n(context)
        a = self.counts[context]["fold"] + 1 if n else self.pooled_counts["fold"] + 1
        b = (n - self.counts[context]["fold"] + 2) if n else (sum(self.pooled_counts.values()) - self.pooled_counts["fold"] + 2)
        return Prediction(probs, probs["fold"], beta_ppf(0.05, a, b), beta_ppf(0.95, a, b), ARM_POPULATION,
                          self.version, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {"by_context": self.counts, "pooled": self.pooled_counts, "q": self.q, "q_pooled": self.q_pooled,
                "version": self.version, **{k: v for k, v in self.meta.items() if k not in ("by_context", "pooled")}}


# ----------------------------------------------------------------- type prior
@dataclass
class TypePrior:
    """Mixture-of-Dirichlet prior: weights pi over K types, each with q_k[c][a]."""

    names: List[str]
    pi: List[float]
    q: List[Dict[str, Dict[str, float]]]   # per type: context -> response -> prob
    version: str = "types-v1"

    @property
    def k(self) -> int:
        return len(self.names)

    def to_dict(self) -> Dict[str, Any]:
        return {"names": self.names, "pi": self.pi, "q": self.q, "version": self.version}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TypePrior":
        return cls(list(d["names"]), [float(x) for x in d["pi"]], d["q"], d.get("version", "types-v1"))


# ----------------------------------------------------------------- personal
class PersonalModel:
    """Online Dirichlet-categorical classifier for one player.

    With a :class:`TypePrior` the prior is a mixture ``sum_k pi_k Dir(s q_k[c])``
    with weights shared across contexts and updated by Dirichlet-multinomial
    likelihoods (PLAN.md section 3).  With ``K = 1`` this is the plain model
    ``alpha[c][a] = s q[c][a]``.
    """

    def __init__(self, population: PopulationModel, prior_strength: float = DEFAULT_PRIOR_STRENGTH,
                 type_prior: Optional[TypePrior] = None, pi0: Optional[Sequence[float]] = None,
                 player_id: Optional[str] = None):
        self.population = population
        self.s = float(prior_strength)
        self.type_prior = type_prior
        self.player_id = player_id
        self.counts: Dict[str, Dict[str, int]] = {c: {a: 0 for a in RESPONSES} for c in CONTEXTS}
        if type_prior is not None:
            self.pi0 = list(pi0) if pi0 is not None else list(type_prior.pi)
        else:
            self.pi0 = [1.0]
        self.pi0 = _normalise_list(self.pi0)
        self.version = f"personal-v1/s={self.s:g}/K={self.k}"

    @property
    def k(self) -> int:
        return self.type_prior.k if self.type_prior is not None else 1

    @property
    def arm(self) -> str:
        return ARM_TYPE if self.type_prior is not None else ARM_PERSONAL

    def _base(self, k: int, context: str) -> Dict[str, float]:
        if self.type_prior is None:
            return self.population.probs(context)
        return self.type_prior.q[k].get(context) or self.population.probs(context)

    def observations(self, context: Optional[str] = None) -> int:
        if context is None:
            return sum(sum(c.values()) for c in self.counts.values())
        return sum(self.counts[context].values())

    def type_posterior(self) -> List[float]:
        if self.type_prior is None:
            return [1.0]
        logs = []
        for k in range(self.k):
            lp = math.log(self.pi0[k]) if self.pi0[k] > 0 else -1e9
            for c in CONTEXTS:
                alpha = {a: self.s * self._base(k, c)[a] for a in RESPONSES}
                lp += dirichlet_multinomial_log(self.counts[c], alpha)
            logs.append(lp)
        m = max(logs)
        w = [math.exp(x - m) for x in logs]
        return _normalise_list(w)

    def probs(self, context: str) -> Dict[str, float]:
        post = self.type_posterior()
        out = {a: 0.0 for a in RESPONSES}
        n = self.counts[context]
        total = self.s + sum(n.values())
        for k, w in enumerate(post):
            base = self._base(k, context)
            for a in RESPONSES:
                out[a] += w * (self.s * base[a] + n[a]) / total
        return out

    def fold_interval(self, context: str, draws: int = 4000, rng: Optional[random.Random] = None) -> Tuple[float, float]:
        n = self.counts[context]
        if self.k == 1:
            base = self._base(0, context)
            a = self.s * base["fold"] + n["fold"]
            b = self.s * (base["call"] + base["raise"]) + n["call"] + n["raise"]
            return beta_ppf(0.05, a, b), beta_ppf(0.95, a, b)
        rng = rng or random.Random(1234)
        post = self.type_posterior()
        xs = []
        for _ in range(draws):
            u = rng.random()
            acc = 0.0
            k = self.k - 1
            for j, w in enumerate(post):
                acc += w
                if u < acc:
                    k = j
                    break
            base = self._base(k, context)
            a = self.s * base["fold"] + n["fold"]
            b = self.s * (base["call"] + base["raise"]) + n["call"] + n["raise"]
            xs.append(rng.betavariate(a, b))
        xs.sort()
        return xs[int(0.05 * draws)], xs[min(draws - 1, int(0.95 * draws))]

    def predict(self, context: str, legal: Optional[Dict[str, bool]] = None) -> Prediction:
        probs = _masked(self.probs(context), legal)
        lo, hi = self.fold_interval(context)
        post = self.type_posterior()
        tp = {self.type_prior.names[k]: round(w, 4) for k, w in enumerate(post)} if self.type_prior else None
        return Prediction(probs, probs["fold"], lo, hi, self.arm, self.version, self.observations(), tp)

    def observe(self, context: str, action: str) -> None:
        if action not in RESPONSES:
            raise ValueError(action)
        self.counts[context][action] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {"player_id": self.player_id, "prior_strength": self.s, "counts": self.counts, "pi0": self.pi0,
                "version": self.version, "type_prior": self.type_prior.to_dict() if self.type_prior else None}

    def freeze(self, strength: Optional["StrengthTable"] = None) -> "FrozenProfile":
        public = {c: self.probs(c) for c in CONTEXTS}
        intervals = {c: list(self.fold_interval(c)) for c in CONTEXTS}
        st = strength if strength is not None else StrengthTable.default(self.population)
        return FrozenProfile(player_id=self.player_id or "anonymous", arm=self.arm, model_version=self.version,
                             public=public, fold_intervals=intervals, counts=copy.deepcopy(self.counts),
                             observations=self.observations(), type_posterior=self.type_posterior(),
                             type_names=list(self.type_prior.names) if self.type_prior else ["all"],
                             strength=st.table_for_profile(self), strength_sources=st.sources)


def _normalise_list(xs: Sequence[float]) -> List[float]:
    s = sum(xs)
    return [x / s for x in xs] if s > 0 else [1.0 / len(xs)] * len(xs)


# ----------------------------------------------------------------- strength table
class StrengthTable:
    """R[c][bucket][a]: how a defender of strength ``bucket`` responds in context ``c``.

    Sources, in order: the player's own practice cells, then ``prior_weight``
    pseudo-observations from calibration sessions, then the synthetic default
    prior fitted so that its bucket-weighted marginal matches the population
    context distribution.  ``sources`` records which was used.
    """

    def __init__(self, prior: Dict[str, Dict[str, Dict[str, float]]], prior_weight: float = DEFAULT_PRIOR_STRENGTH,
                 sources: Optional[Dict[str, Any]] = None):
        self.prior = prior  # context -> bucket -> response -> prob
        self.prior_weight = float(prior_weight)
        self.sources = dict(sources or {})
        self.cells: Dict[str, Dict[str, Dict[str, int]]] = {c: {b: {a: 0 for a in RESPONSES} for b in BUCKETS} for c in CONTEXTS}

    @classmethod
    def default(cls, population: PopulationModel, shares: Optional[Dict[str, float]] = None,
                iterations: int = 25) -> "StrengthTable":
        """Synthetic default prior reconciled with the population marginal per context."""
        shares = shares or DEFAULT_BUCKET_SHARES
        prior: Dict[str, Dict[str, Dict[str, float]]] = {}
        for c in CONTEXTS:
            target = population.probs(c)
            table = {b: dict(DEFAULT_STRENGTH_PRIOR[b]) for b in BUCKETS}
            for _ in range(iterations):
                marginal = {a: sum(shares[b] * table[b][a] for b in BUCKETS) for a in RESPONSES}
                for b in BUCKETS:
                    table[b] = _normalise({a: table[b][a] * (target[a] / max(marginal[a], 1e-9)) for a in RESPONSES})
            prior[c] = table
        return cls(prior, sources={"calibration_sessions": 0, "prior": "synthetic default reconciled with population marginal"})

    @classmethod
    def from_calibration(cls, population: PopulationModel, calibration_counts: Dict[str, Dict[str, Dict[str, int]]],
                         sessions: int, prior_weight: float = DEFAULT_PRIOR_STRENGTH) -> "StrengthTable":
        base = cls.default(population)
        prior: Dict[str, Dict[str, Dict[str, float]]] = {}
        for c in CONTEXTS:
            prior[c] = {}
            for b in BUCKETS:
                n = calibration_counts.get(c, {}).get(b, {})
                total = sum(n.values())
                if total > 0:
                    prior[c][b] = _normalise({a: (n.get(a, 0) + prior_weight * base.prior[c][b][a]) / (total + prior_weight)
                                              for a in RESPONSES})
                else:
                    prior[c][b] = dict(base.prior[c][b])
        return cls(prior, prior_weight, {"calibration_sessions": sessions, "prior": "calibration sessions + default"})

    def observe(self, context: str, bucket: str, action: str) -> None:
        self.cells[context][bucket][action] += 1

    def probs(self, context: str, bucket: str) -> Dict[str, float]:
        n = self.cells[context][bucket]
        total = sum(n.values())
        p = self.prior[context][bucket]
        return _normalise({a: (n[a] + self.prior_weight * p[a]) / (total + self.prior_weight) for a in RESPONSES})

    def table_for_profile(self, personal: Optional[PersonalModel] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Full R table; cells with no personal data in a context the player has river history for
        are tilted towards the player's public posterior (PLAN.md backoff rule)."""
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for c in CONTEXTS:
            out[c] = {}
            public = personal.probs(c) if personal is not None else None
            pop = personal.population.probs(c) if personal is not None else None
            for b in BUCKETS:
                base = self.probs(c, b)
                if public is not None and pop is not None and sum(self.cells[c][b].values()) == 0:
                    # shift the cell by the player's deviation from the population in this context
                    base = _normalise({a: base[a] * (public[a] / max(pop[a], 1e-9)) for a in RESPONSES})
                out[c][b] = base
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"prior": self.prior, "prior_weight": self.prior_weight, "cells": self.cells, "sources": self.sources}


# ----------------------------------------------------------------- frozen profile / twin
@dataclass
class FrozenProfile:
    player_id: str
    arm: str
    model_version: str
    public: Dict[str, Dict[str, float]]
    fold_intervals: Dict[str, List[float]]
    counts: Dict[str, Dict[str, int]]
    observations: int
    type_posterior: List[float]
    type_names: List[str]
    strength: Dict[str, Dict[str, Dict[str, float]]]
    strength_sources: Dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"player_id": self.player_id, "arm": self.arm, "model_version": self.model_version, "public": self.public,
                "fold_intervals": self.fold_intervals, "counts": self.counts, "observations": self.observations,
                "type_posterior": self.type_posterior, "type_names": self.type_names, "strength": self.strength,
                "strength_sources": self.strength_sources, "label": self.label, "hash": self.hash}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrozenProfile":
        d = {k: v for k, v in d.items() if k != "hash"}
        return cls(**d)

    @property
    def hash(self) -> str:
        payload = json.dumps({"public": self.public, "strength": self.strength, "counts": self.counts}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "FrozenProfile":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class Twin:
    """Plays the defender from a frozen profile (strength table when the bucket is known)."""

    kind = "twin"

    def __init__(self, profile: FrozenProfile):
        self.profile = profile
        self.name = f"twin:{profile.player_id}"

    def respond(self, context: str, bucket: Optional[str], rng: random.Random) -> str:
        probs = self.profile.strength[context][bucket] if bucket else self.profile.public[context]
        u = rng.random()
        acc = 0.0
        for a in RESPONSES:
            acc += probs.get(a, 0.0)
            if u < acc:
                return a
        return "raise"

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "profile_hash": self.profile.hash, "arm": self.profile.arm,
                "observations": self.profile.observations, "synthetic": False}


def population_twin(population: PopulationModel) -> Twin:
    """The context-only opponent: a fresh personal model frozen before any observation."""
    pm = PersonalModel(population, player_id="population")
    prof = pm.freeze()
    prof.label = "population (context-only)"
    prof.arm = ARM_POPULATION
    return Twin(prof)
