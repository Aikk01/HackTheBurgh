"""Poker Mirror sessions: one judge, one lock, one profile (PLAN.md sections 1, 3 and 5).

A session runs two flows on the same :class:`DecisionRecord` contract:

* **defender drills** - the judge checks and faces a half-pot bet; a forecast is
  committed before the click and the personal model is updated once after it;
* **bettor play** - after freezing, the judge takes the bettor seat against
  their own twin; the EV advisor and the range readout explain each spot.

Sessions never share state; the registry keys them by a browser token.
"""
from __future__ import annotations

import json
import os
import random
import secrets
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from . import mirror_advisor as ADV
from .mirror_model import (DEFAULT_PRIOR_STRENGTH, FrozenProfile, PersonalModel, PopulationModel, StrengthTable,
                           Twin, TypePrior, population_twin)
from .mirror_types import (BET_SIZE, BUCKETS, CONTEXTS, OPENING_POT, RAISE_TO, RESPONSES, DecisionRecord, Prediction)
from .river_env import RiverResponseEnv, Scenario, scenario_pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIRROR_DIR = os.path.join(ROOT, "data", "mirror")
POLICY_DIR = os.path.join(ROOT, "data", "cache", "mirror", "policies")
# Volunteer decisions stay on this machine (data/cache is git-ignored); only aggregates are committed.
SESSIONS_DIR = os.path.join(ROOT, "data", "cache", "mirror", "sessions")
SESSION_TAGS = ("practice", "calibration", "heldout")
DRILL_POOL_SEED = 42
DRILL_POOL_SIZE = 3000
PLAY_POOL_SEED = 43
PLAY_POOL_SIZE = 1500
CONTEXT_LABELS = {
    "checked_through": "both players checked the turn",
    "bettor_aggr": "the river bettor bet or raised the turn",
    "defender_aggr": "you (the defender) bet or raised the turn",
}
PREFIX_STORY = {
    "checked_through": ["pre-flop: button raises to 10, big blind calls", "flop: check, bet 40, call", "turn: check, check"],
    "bettor_aggr": ["pre-flop: button raises to 10, big blind calls", "flop: check, check", "turn: check, bet 40, call"],
    "defender_aggr": ["pre-flop: button raises to 10, big blind calls", "flop: check, check", "turn: bet 40, call"],
}


def _load_json(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(MIRROR_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SharedModels:
    """Artifacts loaded once per process (population model, type prior, pools, policies)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._population: Optional[PopulationModel] = None
        self._type_prior: Optional[TypePrior] = None
        self._type_prior_checked = False
        self._drill_pool: Optional[List[Scenario]] = None
        self._play_pool: Optional[List[Scenario]] = None
        self._population_policy: Optional[Any] = None
        self._real_players: Optional[Dict[str, Dict[str, Any]]] = None
        self._population_wide: Optional[PopulationModel] = None
        self.sessions_dir: Optional[str] = SESSIONS_DIR  # None disables session logging (tests)

    def make_strength(self) -> StrengthTable:
        """Strength table for a new session: calibration sessions when they exist, else the labelled default."""
        cal = _load_json("strength_calibration.json")
        if cal and cal.get("sessions", 0) > 0:
            return StrengthTable.from_calibration(self.population, cal["counts"], int(cal["sessions"]))
        from .mirror_real import strength_table
        return strength_table(self.population)  # Pluribus-calibrated when 6.7 passed, else the labelled default

    @property
    def real_players(self) -> Dict[str, Dict[str, Any]]:
        """Players from the dataset with enough recorded river answers to build a twin (Phase 6)."""
        with self.lock:
            if self._real_players is None:
                d = _load_json("real_players.json") or {}
                self._real_players = {p["id"]: p for p in d.get("players", [])}
            return self._real_players

    @property
    def population_wide(self) -> PopulationModel:
        """Population over every table size (Phase 6); falls back to the two-player-table population."""
        with self.lock:
            if self._population_wide is None:
                path = os.path.join(MIRROR_DIR, "population_wide.json")
                if os.path.exists(path):
                    self._population_wide = PopulationModel.load(path)
            pop = self._population_wide
        return pop if pop is not None else self.population

    @property
    def wide_prior_strength(self) -> float:
        ev = _load_json("eval_wide.json")
        return float(ev["prior_strength"]) if ev and ev.get("prior_strength") else float(DEFAULT_PRIOR_STRENGTH)

    @property
    def population(self) -> PopulationModel:
        with self.lock:
            if self._population is None:
                self._population = PopulationModel.load()
            return self._population

    @property
    def type_prior(self) -> Optional[TypePrior]:
        with self.lock:
            if not self._type_prior_checked:
                self._type_prior_checked = True
                d = _load_json("styles.json")
                if d and d.get("gate", {}).get("pass") and d.get("type_prior"):
                    self._type_prior = TypePrior.from_dict(d["type_prior"])
            return self._type_prior

    @property
    def drill_pool(self) -> List[Scenario]:
        with self.lock:
            if self._drill_pool is None:
                self._drill_pool = scenario_pool(DRILL_POOL_SIZE, DRILL_POOL_SEED)
            return self._drill_pool

    @property
    def play_pool(self) -> List[Scenario]:
        with self.lock:
            if self._play_pool is None:
                self._play_pool = scenario_pool(PLAY_POOL_SIZE, PLAY_POOL_SEED)
            return self._play_pool

    @property
    def population_policy(self) -> Optional[Any]:
        with self.lock:
            if self._population_policy is None:
                path = os.path.join(MIRROR_DIR, "policy_population.json")
                if os.path.exists(path):
                    from .mirror_rl import Policy
                    self._population_policy = Policy.load(path)
            return self._population_policy


SHARED = SharedModels()


class MirrorSession:
    def __init__(self, token: str, shared: SharedModels = SHARED, seed: Optional[int] = None):
        self.token = token
        self.shared = shared
        self.lock = threading.RLock()
        self.created = time.time()
        self.rng = random.Random(seed if seed is not None else secrets.randbits(32))
        self.population = shared.population
        self.personal = PersonalModel(self.population, DEFAULT_PRIOR_STRENGTH, shared.type_prior, player_id=token)
        self.strength = shared.make_strength()
        self.tag = "practice"  # practice | calibration | heldout (what this judge's decisions may be used for)
        self.decisions: Dict[str, DecisionRecord] = {}
        self.order: List[str] = []
        self.pending: Optional[DecisionRecord] = None
        self.pending_scenario: Optional[Scenario] = None
        self.drill_count = 0
        self.frozen: Optional[FrozenProfile] = None
        self.twin: Optional[Twin] = None
        self.play_env: Optional[RiverResponseEnv] = None
        self.play_scenario: Optional[Scenario] = None
        self.play_stage = 0
        self.play_log: List[Dict[str, Any]] = []
        self.play_current: Optional[Dict[str, Any]] = None
        self.training: Dict[str, Any] = {"status": "idle"}
        self.personal_policy: Optional[Any] = None
        self.opponent: Dict[str, Any] = {"kind": "judge"}
        self.version = 0

    # ------------------------------------------------------------ drills (judge defends)
    def _pick_drill_scenario(self) -> Scenario:
        """Balance contexts and the judge's own strength across drills."""
        ctx = CONTEXTS[self.drill_count % 3]
        bucket = ("strong", "weak", "medium", "weak", "strong", "medium")[self.drill_count % 6]
        pool = [s for s in self.shared.drill_pool if s.context == ctx and s.defender_bucket == bucket]
        if not pool:
            pool = [s for s in self.shared.drill_pool if s.context == ctx]
        return pool[self.rng.randrange(len(pool))]

    def new_drill(self) -> Dict[str, Any]:
        with self.lock:
            if self.pending is not None:
                return self.drill_view()
            sc = self._pick_drill_scenario()
            self.drill_count += 1
            legal = {"fold": True, "call": True, "raise": True}
            pred = self.personal.predict(sc.context, legal)
            rec = DecisionRecord(
                session_id=self.token, decision_id=f"{self.token[:6]}-d{self.drill_count}", context=sc.context,
                pot_before_bet=OPENING_POT, bet=BET_SIZE, legal=legal, board=list(sc.board), own_cards=list(sc.defender_cards),
                defender_strength=sc.defender_bucket, prediction=pred, source="practice", player_id=self.token,
                extra={"story": PREFIX_STORY[sc.context], "context_label": CONTEXT_LABELS[sc.context], "equity": sc.defender_equity,
                       "population_prediction": self.population.probs(sc.context), "tag": self.tag},
            )
            self.pending = rec
            self.pending_scenario = sc
            self.decisions[rec.decision_id] = rec
            self.order.append(rec.decision_id)
            self.version += 1
            return self.drill_view()

    def drill_view(self) -> Dict[str, Any]:
        """What the judge may see before acting (the bettor's cards stay hidden)."""
        with self.lock:
            rec = self.pending
            if rec is None:
                return {"pending": None, "drills": self.drill_count}
            d = asdict(rec)
            d["prediction"] = asdict(rec.prediction) if rec.prediction else None
            d.pop("defender_strength", None)
            return {"pending": d, "drills": self.drill_count, "pot": OPENING_POT, "bet": BET_SIZE,
                    "to_call": BET_SIZE, "raise_to": RAISE_TO}

    def respond(self, decision_id: str, action: str) -> Dict[str, Any]:
        with self.lock:
            rec = self.pending
            if rec is None or rec.decision_id != decision_id:
                raise ValueError("no such pending decision")
            action = action.lower()
            if action not in RESPONSES or not rec.legal.get(action, False):
                raise ValueError(f"illegal response {action!r}")
            sc = self.pending_scenario
            rec.actual = action
            # one update per observed decision, under the session lock
            self.personal.observe(rec.context, action)
            if sc is not None:
                self.strength.observe(rec.context, sc.defender_bucket, action)
            outcome = None
            if sc is not None:
                from .mirror_rl import showdown_value
                w = showdown_value(sc)  # bettor's showdown result
                # defender's result relative to the river start, in opening pots (the bettor calls a raise)
                if action == "fold":
                    net = 0.0
                    result = "you folded"
                elif action == "call":
                    net = {1.0: -0.5, 0.0: 1.5, 0.5: 0.5}[w]
                    result = "you won at showdown" if w == 0.0 else "you lost at showdown" if w == 1.0 else "split pot"
                else:
                    net = {1.0: -1.0, 0.0: 2.0, 0.5: 0.5}[w]
                    result = "the bettor called your raise: " + ("you won" if w == 0.0 else "you lost" if w == 1.0 else "split")
                outcome = {"bettor_cards": list(sc.bettor_cards), "result": result, "net_pots": net,
                           "your_bucket": sc.defender_bucket, "your_equity": sc.defender_equity}
                rec.extra["outcome"] = outcome
            rec.extra["tag"] = self.tag
            self._log({"type": "drill", "tag": self.tag, "session": self.token, "time": time.time(), "decision": asdict(rec)})
            self.pending = None
            self.pending_scenario = None
            self.version += 1
            after = self.personal.predict(rec.context, rec.legal)
            return {"decision": {**asdict(rec), "prediction": asdict(rec.prediction) if rec.prediction else None},
                    "outcome": outcome, "after": asdict(after), "profile": self.profile_summary()}

    def profile_summary(self) -> Dict[str, Any]:
        with self.lock:
            contexts = {}
            for c in CONTEXTS:
                p = self.personal.predict(c)
                contexts[c] = {"probs": p.probs, "fold_p05": p.fold_p05, "fold_p95": p.fold_p95,
                               "observations": self.personal.observations(c), "label": CONTEXT_LABELS[c]}
            pop = {c: self.population.probs(c) for c in CONTEXTS}
            tp = self.personal.type_prior
            return {
                "player_id": self.token[:8], "arm": self.personal.arm, "model_version": self.personal.version,
                "prior_strength": self.personal.s, "observations": self.personal.observations(),
                "contexts": contexts, "population": pop,
                "type_posterior": ({"names": tp.names, "weights": self.personal.type_posterior()} if tp else None),
                "strength_sources": self.strength.sources,
                "strength_cells": {c: {b: sum(self.strength.cells[c][b].values()) for b in BUCKETS} for c in CONTEXTS},
                "frozen": self.frozen.hash if self.frozen else None,
                "history": [self._decision_row(i) for i in self.order if self.decisions[i].actual],
            }

    def _decision_row(self, decision_id: str) -> Dict[str, Any]:
        r = self.decisions[decision_id]
        return {"decision_id": r.decision_id, "context": r.context, "predicted_fold": r.prediction.fold_mean if r.prediction else None,
                "arm": r.prediction.arm if r.prediction else None, "observations_before": r.prediction.observations if r.prediction else None,
                "actual": r.actual, "board": r.board, "own_cards": r.own_cards}

    # ------------------------------------------------------------ freeze and play (judge bets)
    def freeze(self) -> Dict[str, Any]:
        with self.lock:
            self.frozen = self.personal.freeze(self.strength)
            self.frozen.label = f"judge twin ({self.personal.observations()} observations)"
            self.twin = Twin(self.frozen)
            self._forget_other_bot()
            self.opponent = {"kind": "judge", "observations": self.frozen.observations}
            self.play_env = RiverResponseEnv(self.twin, self.shared.play_pool, seed=self.rng.randrange(1 << 30))
            self.version += 1
            return {"hash": self.frozen.hash, "observations": self.frozen.observations, "arm": self.frozen.arm,
                    "public": self.frozen.public, "strength_sources": self.frozen.strength_sources}

    def _forget_other_bot(self) -> None:
        """A bettor trained against another twin no longer applies."""
        if self.personal_policy is not None and self.personal_policy.meta.get("profile_hash") not in (None, self.frozen.hash):
            self.personal_policy = None
            self.training = {"status": "idle"}

    def load_real_player(self, player_id: str) -> Dict[str, Any]:
        """Bet against a twin built from a real player's recorded river answers, so no drills are needed."""
        from .mirror_real import decode, strength_table
        rec = self.shared.real_players.get(player_id)
        if rec is None:
            raise ValueError(f"unknown real player {player_id!r}")
        population = self.shared.population_wide
        pm = PersonalModel(population, self.shared.wide_prior_strength, player_id=player_id)
        for c, a in decode(rec["sequence"]):
            pm.observe(c, a)
        frozen = pm.freeze(strength_table(population))
        frozen.label = f"real player {player_id} ({rec['decisions']} recorded decisions, {rec['venue']})"
        with self.lock:
            self.frozen = frozen
            self.twin = Twin(frozen)
            self.play_env = RiverResponseEnv(self.twin, self.shared.play_pool, seed=self.rng.randrange(1 << 30))
            self._forget_other_bot()
            self.opponent = {"kind": "real", "id": player_id, "venue": rec["venue"], "decisions": rec["decisions"],
                             "fold_rate": rec["fold_rate"], "table_mix": rec.get("table_mix")}
            self.version += 1
            return {"hash": frozen.hash, "observations": frozen.observations, "arm": frozen.arm, "public": frozen.public,
                    "strength_sources": frozen.strength_sources, "opponent": dict(self.opponent)}

    def twin_for_training(self) -> Twin:
        with self.lock:
            if self.twin is None:
                self.freeze()
            return self.twin  # type: ignore[return-value]

    def new_play_spot(self, scenario_index: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            if self.play_env is None:
                self.freeze()
            env = self.play_env
            assert env is not None
            pool = self.shared.play_pool
            sc = pool[scenario_index % len(pool)] if scenario_index is not None else pool[self.rng.randrange(len(pool))]
            env.reset(options={"scenario": sc})
            self.play_scenario = sc
            self.play_stage = 1
            self.play_current = {"actions": [], "response": None, "reward": None}
            self.version += 1
            return self.play_view()

    def _advice(self, sc: Scenario) -> Dict[str, Any]:
        prof = self.frozen
        assert prof is not None
        probs = prof.public[sc.context]
        lo, hi = prof.fold_intervals[sc.context]
        adv = ADV.advise(sc.board, sc.bettor_cards, sc.context, probs, prof.strength, (lo, hi))
        adv["forecast"] = {"fold": probs["fold"], "interval": [lo, hi], "arm": prof.arm, "observations": prof.observations}
        return adv

    def play_view(self) -> Dict[str, Any]:
        with self.lock:
            sc = self.play_scenario
            if sc is None or self.play_env is None:
                return {"spot": None, "frozen": self.frozen.hash if self.frozen else None}
            cur = self.play_current or {}
            view: Dict[str, Any] = {
                "spot": {"board": list(sc.board), "own_cards": list(sc.bettor_cards), "context": sc.context,
                         "context_label": CONTEXT_LABELS[sc.context].replace(
                             "you (the defender)", "the player's twin" if self.opponent.get("kind") == "real" else "your twin"),
                         "story": PREFIX_STORY[sc.context], "pot": OPENING_POT, "bet": BET_SIZE, "raise_to": RAISE_TO,
                         "stage": self.play_stage, "bettor_bucket": sc.bettor_bucket, "bettor_equity": sc.bettor_equity},
                "advice": self._advice(sc) if self.play_stage == 1 else None,
                "actions": cur.get("actions", []), "response": cur.get("response"), "readout": cur.get("readout"),
                "result": cur.get("result"), "frozen": self.frozen.hash if self.frozen else None,
                "bots": self._bots_on_deal(sc) if self.play_stage == 1 else cur.get("bots"),
                "log": self.play_log[-20:],
            }
            return view

    def _bots_on_deal(self, sc: Scenario) -> Dict[str, Any]:
        from .mirror_rl import state_key
        out: Dict[str, Any] = {}
        k = state_key(1, sc.context, sc.bettor_bucket)
        for name, pol in (("population", self.shared.population_policy), ("personal", self.personal_policy)):
            if pol is None:
                out[name] = None
                continue
            a = pol.greedy(k)
            out[name] = {"action": pol.action_name(k, a), "q": [round(x, 4) for x in pol.q[k]], "visits": list(pol.visits[k]),
                         "q_chips": [round(x * OPENING_POT, 1) for x in pol.q[k]], "episodes": pol.meta.get("episodes"),
                         "profile_hash": pol.meta.get("profile_hash"), "oracle": pol.meta.get("oracle_check")}
        return out

    def play_act(self, action: str) -> Dict[str, Any]:
        with self.lock:
            env = self.play_env
            sc = self.play_scenario
            if env is None or sc is None or self.play_stage == 0:
                raise ValueError("no spot in play")
            action = action.lower()
            cur = self.play_current or {}
            if self.play_stage == 1:
                if action not in ("check", "bet"):
                    raise ValueError("choose check or bet")
                a = 1 if action == "bet" else 0
            else:
                if action not in ("fold", "call"):
                    raise ValueError("choose fold or call")
                a = 1 if action == "call" else 0
            cur["actions"] = cur.get("actions", []) + [action]
            obs, reward, done, _, info = env.step(a)
            if info.get("response") and cur.get("response") is None:
                resp = info["response"]
                cur["response"] = resp
                cur["readout"] = ADV.range_readout(sc.board, sc.bettor_cards, sc.context, resp, self.frozen.strength)  # type: ignore[union-attr]
            if done:
                self.play_stage = 0
                from .mirror_rl import showdown_value
                cur["result"] = {"reward_pots": reward, "reward_chips": reward * OPENING_POT, "final_stacks": info["final_stacks"],
                                 "defender_cards": list(sc.defender_cards), "defender_bucket": sc.defender_bucket,
                                 "showdown": showdown_value(sc), "bots": cur.get("bots")}
                self.play_log.append({"board": sc.board, "own": sc.bettor_cards, "context": sc.context, "actions": cur["actions"],
                                      "response": cur.get("response"), "reward_chips": reward * OPENING_POT})
            else:
                self.play_stage = 2
            self.play_current = cur
            self.version += 1
            return self.play_view()

    # ------------------------------------------------------------ personal counter-bot
    def start_training(self, episodes: int = 20000, seed: int = 7) -> Dict[str, Any]:
        with self.lock:
            twin = self.twin_for_training()
            if self.training.get("status") == "running":
                return dict(self.training)
            key = f"{twin.profile.hash}_{seed}_{episodes}"
            path = os.path.join(POLICY_DIR, key + ".json")
            self.training = {"status": "running", "progress": 0, "episodes": episodes, "profile_hash": twin.profile.hash,
                             "seed": seed, "path": path}

        def run() -> None:
            from .mirror_rl import Policy, compare_with_oracle, q_star, train
            try:
                if os.path.exists(path):
                    pol = Policy.load(path)
                else:
                    pool = scenario_pool(4000, 1)
                    val = scenario_pool(1000, 2)

                    def progress(d: int, n: int) -> None:
                        self.training["progress"] = d
                    pol = train(twin, pool, episodes=episodes, seed=seed, validation_pool=val, progress=progress,
                                meta={"arm": "personal", "session": self.token[:8]})
                    cmp = compare_with_oracle(pol, q_star(twin, val))
                    pol.meta["oracle_check"] = {"pass": cmp["pass"], "failures": cmp["failures"]}
                    pol.save(path)
                with self.lock:
                    self.personal_policy = pol
                    self.training = {"status": "done", "progress": episodes, "episodes": episodes, "profile_hash": twin.profile.hash,
                                     "seed": seed, "path": path, "oracle": pol.meta.get("oracle_check"),
                                     "validation": pol.checkpoints[-1].get("validation_mean_reward") if pol.checkpoints else None}
                    self.version += 1
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.training = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        threading.Thread(target=run, daemon=True).start()
        return dict(self.training)

    def compare_bots(self) -> Dict[str, Any]:
        """Population versus personalised bettor on the current (or last) deal, plus policy charts."""
        with self.lock:
            sc = self.play_scenario
            pop = self.shared.population_policy
            per = self.personal_policy
            return {
                "deal": asdict(sc) if sc else None,
                "on_deal": self._bots_on_deal(sc) if sc else None,
                "population_chart": pop.chart() if pop else None,
                "personal_chart": per.chart() if per else None,
                "population_meta": {k: pop.meta.get(k) for k in ("episodes", "seed", "profile_hash", "oracle_check", "train_seconds")} if pop else None,
                "personal_meta": {k: per.meta.get(k) for k in ("episodes", "seed", "profile_hash", "oracle_check", "train_seconds")} if per else None,
                "training": dict(self.training),
                "advice": self._advice(sc) if (sc and self.frozen) else None,
            }

    # ------------------------------------------------------------ session data
    def set_tag(self, tag: str) -> Dict[str, Any]:
        """Declare what this judge's decisions are for: practice, calibration (development) or heldout (evaluation)."""
        if tag not in SESSION_TAGS:
            raise ValueError(f"tag must be one of {', '.join(SESSION_TAGS)}")
        with self.lock:
            self.tag = tag
            self.version += 1
            return {"tag": self.tag}

    def _log(self, entry: Dict[str, Any]) -> None:
        if not self.shared.sessions_dir:
            return
        try:
            os.makedirs(self.shared.sessions_dir, exist_ok=True)
            with open(os.path.join(self.shared.sessions_dir, f"{self.token}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            pass  # logging must never break a live demo

    # ------------------------------------------------------------ state
    def state(self) -> Dict[str, Any]:
        with self.lock:
            return {"token": self.token, "version": self.version, "tag": self.tag, "drill": self.drill_view(),
                    "profile": self.profile_summary(), "frozen": self.frozen.hash if self.frozen else None,
                    "play": self.play_view(), "training": dict(self.training), "opponent": dict(self.opponent),
                    "has_population_policy": self.shared.population_policy is not None}


class MirrorRegistry:
    def __init__(self, shared: SharedModels = SHARED):
        self.shared = shared
        self.lock = threading.Lock()
        self.sessions: Dict[str, MirrorSession] = {}

    def get(self, token: Optional[str], create: bool = True) -> MirrorSession:
        with self.lock:
            if token and token in self.sessions:
                return self.sessions[token]
            if not create:
                raise KeyError("unknown session")
            token = token or secrets.token_hex(8)
            s = MirrorSession(token, self.shared)
            self.sessions[token] = s
            return s

    def reset(self, token: str) -> MirrorSession:
        with self.lock:
            s = MirrorSession(token, self.shared)
            self.sessions[token] = s
            return s


REGISTRY = MirrorRegistry()
