"""HTTP + Server-Sent-Events server for the web UI (standard library only).

Run with ``python -m pokersim serve`` and open http://127.0.0.1:8000/.
"""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import random
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from . import dataset, phh
from .agents import Agent, AgentError, available_agents, make_agent, safe_act
from .engine import EngineError, Hand, IllegalAction, Seat, Table, UnsupportedVariant, num
from .evaluate import normalise_prediction, shadow_evaluate
from .replay import build_hand, replay
from .simulate import hand_to_record, run_match

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
FAST_AGENTS = {"random", "station", "maniac", "equity", "tag", "lag"}


def sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


class EventBus:
    def __init__(self) -> None:
        self.subscribers: List[queue.Queue] = []
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, event: Dict[str, Any]) -> None:
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            q.put(event)


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ----------------------------------------------------------------------- game
class GameSession:
    """A live table: humans act through the API, bots act in a worker thread."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.lock = threading.RLock()
        self.table: Optional[Table] = None
        self.hand: Optional[Hand] = None
        self.agents: Dict[int, Optional[Agent]] = {}
        self.human_seats: List[int] = []
        self.reveal = False
        self.bot_delay = 0.7
        self.log: List[Dict[str, Any]] = []
        self.records: List[phh.HandRecord] = []
        self.worker: Optional[threading.Thread] = None
        self.thinking: Optional[int] = None  # player index of the bot currently deciding
        self.mode = "table"  # table | scenario
        self.scenario: Optional[Dict[str, Any]] = None
        self.hand_counter = 0
        self.settled: set = set()  # id() of hands already written back to the table / archived
        from .opponent_model import StyleTracker
        self.styles = StyleTracker()  # per-player style counts for the AI advisor

    # -- setup ----------------------------------------------------------------
    def new_game(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            seats_cfg = cfg.get("seats") or []
            if len(seats_cfg) < 2:
                raise ApiError("at least two seats are required")
            rng = random.Random(cfg.get("seed"))
            seats = []
            self.agents = {}
            self.human_seats = []
            for i, s in enumerate(seats_cfg):
                seat_no = i + 1
                spec = (s.get("agent") or "human").strip()
                name = (s.get("name") or "").strip()
                stack = float(s.get("stack") or cfg.get("stack") or 200)
                if spec.lower() == "human":
                    agent = None
                    self.human_seats.append(seat_no)
                    name = name or f"You ({seat_no})"
                else:
                    try:
                        agent = make_agent(spec, seed=rng.randrange(1 << 30))
                    except AgentError as exc:
                        raise ApiError(str(exc))
                    name = name or f"{agent.label} {seat_no}"
                    agent.name = name
                    if agent.kind == "rl":  # the RL bot reads the same opponent style estimates as the advisor
                        agent.styles_provider = lambda obs: {p["index"]: self.styles.estimate(p["name"]) for p in obs["players"]}
                self.agents[seat_no] = agent
                seats.append(Seat(seat_no=seat_no, name=name, stack=stack, agent=agent,
                                  kind="human" if agent is None else agent.kind, buy_in=stack))
            try:
                self.table = Table(seats, small_blind=float(cfg.get("small_blind", 1)),
                                   big_blind=float(cfg.get("big_blind", 2)), ante=float(cfg.get("ante", 0) or 0),
                                   ante_mode=cfg.get("ante_mode", "uniform"), variant=cfg.get("variant", "NT"),
                                   rng=rng, straddle=float(cfg.get("straddle", 0) or 0))
            except (EngineError, UnsupportedVariant) as exc:
                raise ApiError(str(exc))
            self.bot_delay = max(0.0, float(cfg.get("bot_delay_ms", 700)) / 1000.0)
            self.reveal = bool(cfg.get("spectate", False)) or not self.human_seats
            self.log = []
            self.records = []
            self.styles.reset()
            self.hand = None
            self.mode = "table"
            self.scenario = None
            self.hand_counter = 0
            return self.next_hand()

    def next_hand(self) -> Dict[str, Any]:
        with self.lock:
            if self.mode == "scenario" and self.scenario is not None:
                return self.start_scenario(self.scenario)
            if self.table is None:
                raise ApiError("no game; create one first", 409)
            if self.hand is not None and self.hand.status not in ("finished", "incomplete"):
                raise ApiError("the current hand is still running", 409)
            self._settle()
            try:
                self.hand = self.table.new_hand()
            except EngineError as exc:
                raise ApiError(str(exc), 409)
            self.hand_counter += 1
            self.log = [{"t": time.time(), "text": f"Hand #{self.table.hand_no} - dealing"}]
            self._kick_bots()
            return self.state()

    def _settle(self) -> None:
        """Once a hand is over: archive it as a PHH record and write stacks back to the seats."""
        hand = self.hand
        if hand is None or hand.status not in ("finished", "incomplete") or id(hand) in self.settled:
            return
        self.settled.add(id(hand))
        try:
            self.styles.add_hand(hand)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self.records.append(hand_to_record(hand, extra_meta={"venue": "pokersim live table"}))
        if self.table is not None and self.table.current is hand:
            self.table.finish_hand()

    # -- scenario (fork a recorded hand) --------------------------------------
    def start_scenario(self, sc: Dict[str, Any]) -> Dict[str, Any]:
        """Replay ``record`` up to ``step_index`` then hand control to agents/humans."""
        with self.lock:
            rec: phh.HandRecord = sc["record"]
            step_index = int(sc.get("step_index", -1))
            seat_specs: Dict[int, str] = {int(k): v for k, v in (sc.get("seats") or {}).items()}
            use_board = bool(sc.get("use_recorded_board", True))
            rng = random.Random(sc.get("seed"))
            try:
                hand = build_hand(rec)
            except UnsupportedVariant as exc:
                raise ApiError(str(exc))
            hand.rng = rng
            hand.deck.rng = rng
            hand.split_mode = "odd_chip"
            hand.winnings_hint = None
            hand.post_forced_bets()
            parsed = rec.parsed_actions()
            future_board: List[int] = []
            cut = len(parsed) if step_index < 0 else min(step_index, len(parsed))
            for a in parsed[:cut]:
                _apply_recorded(hand, a)
            if hand.status in ("finished", "incomplete"):
                raise ApiError("the hand is already over at that point; pick an earlier step")
            for a in parsed[cut:]:
                if a.is_dealer and a.verb == "db" and a.args:
                    future_board += [c for c in phh_cards(a.args[0]) if c is not None]
                if a.is_dealer and a.verb == "dh" and len(a.args) > 1:
                    # cards dealt later in the record (draw games) are not used
                    pass
            # fill unknown hole cards of players still in the hand
            hand.deck.remove(future_board)
            for p in hand.players:
                if p.folded:
                    continue
                hole = list(p.hole)
                while len(hole) < hand.config.hole_count:
                    hole.append(None)
                for i, c in enumerate(hole):
                    if c is None:
                        hole[i] = hand.deck.draw(1)[0]
                p.hole = hole
            if use_board:
                hand.deck.cards = future_board + hand.deck.cards
            else:
                rng.shuffle(hand.deck.cards)
            hand.scripted = False
            hand.hand_id = f"{rec.label} (fork @ {cut})"
            # agents for each player index
            self.agents = {}
            self.human_seats = []
            for p in hand.players:
                spec = seat_specs.get(p.index, "equity" if not p.folded else "station")
                if spec == "human":
                    self.agents[p.index + 1] = None
                    self.human_seats.append(p.index + 1)
                else:
                    try:
                        ag = make_agent(spec, seed=rng.randrange(1 << 30))
                    except AgentError as exc:
                        raise ApiError(str(exc))
                    ag.name = p.name
                    self.agents[p.index + 1] = ag
            self.table = None
            self.hand = hand
            self.mode = "scenario"
            self.scenario = dict(sc)
            self.reveal = bool(sc.get("spectate", False)) or not self.human_seats
            self.bot_delay = max(0.0, float(sc.get("bot_delay_ms", 700)) / 1000.0)
            self.log = [{"t": time.time(), "text": f"Scenario: {rec.label} forked after action {cut}"}]
            self.hand_counter += 1
            # resume the state machine
            if hand.status == "dealing":
                hand._begin_street()
            elif hand.status == "awaiting_board":
                while hand.status == "awaiting_board":
                    hand.deal_board()
            elif hand.status == "showdown":
                hand.finalize()
            self._kick_bots()
            return self.state()

    # -- play -----------------------------------------------------------------
    def _seat_of(self, player_index: int) -> int:
        if self.mode == "scenario" or self.table is None:
            return player_index + 1
        return self.table.seat_for_player(player_index).seat_no

    def human_act(self, seat: Optional[int], action: str, amount: Optional[float]) -> Dict[str, Any]:
        with self.lock:
            hand = self.hand
            if hand is None or hand.status != "betting" or hand.actor is None:
                raise ApiError("no decision pending", 409)
            actor_seat = self._seat_of(hand.actor)
            if self.agents.get(actor_seat) is not None:
                raise ApiError("it is a bot's turn", 409)
            if seat is not None and int(seat) != actor_seat:
                raise ApiError(f"it is seat {actor_seat}'s turn", 409)
            try:
                result = hand.act(hand.actor, action, amount)
            except IllegalAction as exc:
                raise ApiError(str(exc))
            self._log_action(hand, actor_seat, result, raw=None)
            self._settle()
            self._kick_bots()
            return self.state()

    def _log_action(self, hand: Hand, seat: int, result: Dict[str, Any], raw: Optional[Dict[str, Any]]) -> None:
        ev = next((e for e in reversed(hand.events) if e["type"] == "action"), None)
        name = hand.players[ev["player"]].name if ev else f"seat {seat}"
        verb = result["action"]
        text = f"{name} {verb}s" if verb in ("fold", "check") else (
            f"{name} calls {_money(result.get('amount', 0))}" if verb == "call" else
            f"{name} {verb}s to {_money(result.get('amount', 0))}")
        if ev and ev.get("all_in"):
            text += " (all-in)"
        entry: Dict[str, Any] = {"t": time.time(), "text": text, "player": ev["player"] if ev else None,
                                 "street": ev["street"] if ev else hand.street}
        if result.get("note"):
            entry["note"] = result["note"]
        if raw:
            if raw.get("reasoning"):
                entry["reasoning"] = str(raw["reasoning"])[:600]
            if raw.get("error"):
                entry["error"] = raw["error"]
            entry["latency_ms"] = raw.get("latency_ms")
        self.log.append(entry)

    def _kick_bots(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.worker = threading.Thread(target=self._run_bots, daemon=True)
        self.worker.start()

    def _run_bots(self) -> None:
        try:
            while True:
                with self.lock:
                    hand = self.hand
                    if hand is None or hand.status != "betting" or hand.actor is None:
                        self.thinking = None
                        self._settle()
                        break
                    actor = hand.actor
                    seat = self._seat_of(actor)
                    agent = self.agents.get(seat)
                    if agent is None:
                        self.thinking = None
                        break
                    self.thinking = actor
                    obs = hand.observation(actor)
                self.bus.publish({"type": "game", "state": self.state()})
                t0 = time.perf_counter()
                action = safe_act(agent, obs)
                elapsed = time.perf_counter() - t0
                if self.bot_delay > elapsed:
                    time.sleep(self.bot_delay - elapsed)
                with self.lock:
                    if self.hand is not hand or hand.status != "betting" or hand.actor != actor:
                        continue
                    try:
                        result = hand.act(actor, action["action"], action.get("amount"))
                    except IllegalAction as exc:
                        result = hand.act(actor, "call")
                        result["note"] = f"illegal action from model ({exc}); called instead"
                    self._log_action(hand, seat, result, raw=action)
                    self.thinking = None
                self.bus.publish({"type": "game", "state": self.state()})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self.thinking = None
            self.bus.publish({"type": "game", "state": self.state()})
        else:
            self.bus.publish({"type": "game", "state": self.state()})

    # -- live advice for a human seat ----------------------------------------
    def advice(self) -> Dict[str, Any]:
        from .opponent_model import advise

        with self.lock:
            hand = self.hand
            if hand is None or hand.status != "betting" or hand.actor is None:
                return {"available": False, "reason_ja": "今は判断する場面ではありません。"}
            actor = hand.actor
            if self.agents.get(self._seat_of(actor)) is not None:
                return {"available": False, "reason_ja": "ボットの手番です。"}
            obs = hand.observation(actor)
            styles = {p.index: self.styles.estimate(p.name) for p in hand.players}
            key = (hand.hand_id, actor, len(obs.get("history") or []))
        out = advise(obs, styles)  # Monte-Carlo runs outside the lock
        try:
            from .rl_policy import recommend
            out["rl"] = recommend(obs, styles)
        except Exception as exc:  # noqa: BLE001 - the explanation layer still works without the RL model
            out["rl"] = {"available": False, "reason_ja": f"RLモデルを読み込めませんでした: {exc}"}
        out["key"] = "|".join(str(k) for k in key)
        return out

    def observe(self, hands: int = 60) -> Dict[str, Any]:
        """Watch the bots play ``hands`` hands among themselves (a stand-in takes the human seat) and
        feed only the bots' actions to the style tracker, like watching a table before sitting down."""
        from .replay import replay
        from .simulate import run_match

        hands = max(1, min(int(hands), 300))
        with self.lock:
            if self.table is None:
                raise ApiError("start a table first")
            specs, names, tracked = [], [], set()
            for s in self.table.seats:
                agent = self.agents.get(s.seat_no)
                if agent is None or agent.kind in ("http", "claude"):
                    specs.append("equity:iterations=80")  # stand-in; its actions are not recorded
                    names.append(f"(代役 {s.seat_no})")
                    continue
                params = {k: v for k, v in agent.params.items() if k not in ("seed",)}
                if agent.kind in ("equity", "tag", "lag"):
                    params["iterations"] = min(int(params.get("iterations", 300)), 80)  # faster, same style
                specs.append(agent.kind + (":" + ",".join(f"{k}={v}" for k, v in params.items()) if params else ""))
                names.append(s.name)
                tracked.add(s.name)
            bb = float(self.table.big_blind or 1.0)
            sb = float(self.table.small_blind or bb / 2)
        t0 = time.perf_counter()
        res = run_match(specs, hands=hands, stack=100 * bb, small_blind=sb, big_blind=bb, seed=random.randrange(1 << 30),
                        keep_records=True, names=names)
        added = 0
        for rec in res.records:
            try:
                h = replay(rec, snapshots=False, observations=False).hand
            except Exception:  # noqa: BLE001
                continue
            with self.lock:
                self.styles.add_hand(h, only=tracked)  # the stand-in's actions are not recorded
            added += 1
        return {"observed_hands": added, "seconds": round(time.perf_counter() - t0, 1), "opponents": self.opponents()}

    def opponents(self) -> Dict[str, Any]:
        """Style estimate for every seat; the bot's configured type is returned separately, only to check the model."""
        from .opponent_model import TYPES

        with self.lock:
            hand = self.hand
            if hand is None:
                return {"players": [], "types_source": TYPES["source"]}
            out = []
            for p in hand.players:
                seat = self._seat_of(p.index)
                agent = self.agents.get(seat)
                est = self.styles.estimate(p.name)
                out.append({"seat": seat, "name": p.name, "is_human": agent is None,
                            "actual_label": agent.describe().get("label") if agent else "人間", **est})
            return {"players": out, "types_source": TYPES["source"], "hands_total": max(self.styles.hands.values(), default=0)}

    # -- reporting ------------------------------------------------------------
    def set_reveal(self, reveal: bool) -> Dict[str, Any]:
        with self.lock:
            self.reveal = bool(reveal)
            return self.state()

    def state(self) -> Dict[str, Any]:
        with self.lock:
            hand = self.hand
            if hand is None:
                return {"active": False}
            viewer = None
            if len(self.human_seats) == 1:
                hs = self.human_seats[0]
                viewer = next((p.index for p in hand.players if self._seat_of(p.index) == hs), None)
            st = hand.to_dict(viewer=viewer, reveal=self.reveal)
            for p in st["players"]:
                seat = self._seat_of(p["index"])
                agent = self.agents.get(seat)
                p["seat"] = seat
                p["agent"] = agent.describe() if agent else {"kind": "human", "label": "Human", "name": p["name"]}
                p["is_human"] = agent is None
            self._settle()
            return {
                "active": True,
                "mode": self.mode,
                "hand_no": self.table.hand_no if self.table else self.hand_counter,
                "hand": st,
                "viewer": viewer,
                "reveal": self.reveal,
                "thinking": self.thinking,
                "log": self.log[-80:],
                "table": {
                    "seats": [{"seat": s.seat_no, "name": s.name, "stack": num(s.stack), "kind": s.kind,
                               "buy_in": s.buy_in, "hands": s.hands_played} for s in self.table.seats] if self.table else [],
                    "button_seat": self.table.button_seat if self.table else None,
                    "small_blind": self.table.small_blind if self.table else None,
                    "big_blind": self.table.big_blind if self.table else None,
                    "ante": self.table.ante if self.table else None,
                    "results": self.table.history[-50:] if self.table else [],
                },
                "hands_recorded": len(self.records),
                "scenario": {"label": self.scenario["record"].label, "step_index": self.scenario.get("step_index")}
                if self.scenario else None,
            }


def _money(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return str(int(v)) if v.is_integer() else f"{v:.2f}"


def phh_cards(text: str) -> List[Optional[int]]:
    from . import cards as C
    return C.parse_cards(text)


def _apply_recorded(hand: Hand, a: phh.ParsedAction) -> None:
    from . import cards as C
    if a.is_dealer:
        if a.verb == "dh" and len(a.args) > 1:
            hand.deal_hole(int(a.args[0][1:]) - 1, C.parse_cards(a.args[1]))
        elif a.verb == "db" and a.args:
            hand.deal_board(C.parse_cards(a.args[0]))
        return
    p = a.player
    if p is None:
        return
    if a.verb == "f":
        hand.fold(p)
    elif a.verb == "cc":
        hand.check_call(p)
    elif a.verb == "cbr" and a.args:
        hand.bet_raise_to(p, float(a.args[0]))
    elif a.verb == "sm":
        if not a.args:
            hand.show_muck(p, None)
        elif a.args[0] == "-":
            hand.show_muck(p, list(hand.players[p].hole))
        else:
            hand.show_muck(p, C.parse_cards(a.args[0]))


# -------------------------------------------------------------------- jobs
class BackgroundJob:
    def __init__(self, kind: str, bus: EventBus):
        self.kind = kind
        self.bus = bus
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = False
        self.status = "idle"
        self.progress: Dict[str, Any] = {}
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.payload: Any = None

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, target) -> None:
        if self.running():
            raise ApiError(f"{self.kind} already running", 409)
        self.stop_flag = False
        self.status = "running"
        self.result = None
        self.error = None
        self.progress = {}

        def run() -> None:
            try:
                self.result = target()
                self.status = "done" if not self.stop_flag else "stopped"
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.error = f"{type(exc).__name__}: {exc}"
                self.status = "error"
            self.bus.publish({"type": f"{self.kind}_done", "status": self.status, "error": self.error})

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def state(self) -> Dict[str, Any]:
        return {"status": self.status, "progress": self.progress, "error": self.error,
                "result": self.result}


# ------------------------------------------------------------------- app
class App:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.game = GameSession(self.bus)
        self.records: List[phh.HandRecord] = []
        self.records_label = ""
        self.eval_job = BackgroundJob("eval", self.bus)
        self.sim_job = BackgroundJob("sim", self.bus)
        self.sim_records: List[phh.HandRecord] = []
        self.replay_cache: Dict[int, Any] = {}
        self.lock = threading.Lock()

    # -- replay / dataset -----------------------------------------------------
    def load_records(self, body: Dict[str, Any]) -> Dict[str, Any]:
        source = body.get("source", "sample")
        limit = body.get("limit")
        limit = int(limit) if limit else None
        if source == "upload":
            text = body.get("text") or ""
            if not text.strip():
                raise ApiError("empty upload")
            recs = phh.load_text(text, source=body.get("name") or "upload", limit=limit)
            label = body.get("name") or "uploaded file"
        elif source == "hf":
            path = body.get("path") or ""
            if not path or path.endswith("/"):
                raise ApiError("choose a .phh or .phhs file")
            try:
                recs = dataset.fetch_hands(path, limit=limit)
            except FileNotFoundError as exc:
                raise ApiError(str(exc), 404)
            label = path
        elif source in ("sample", "cache", "path"):
            path = body.get("path") or ""
            if source == "sample":
                path = os.path.join(dataset.SAMPLES_DIR, os.path.basename(path))
            if not os.path.isfile(path):
                raise ApiError(f"file not found: {path}", 404)
            recs = phh.load_file(path, limit=limit)
            label = os.path.basename(path) if source == "sample" else path
        else:
            raise ApiError(f"unknown source {source!r}")
        if not recs:
            raise ApiError("no hands found in that file")
        with self.lock:
            self.records = recs
            self.records_label = label
            self.replay_cache = {}
        return self.records_summary()

    def records_summary(self) -> Dict[str, Any]:
        summaries = []
        variants: Dict[str, int] = {}
        for i, r in enumerate(self.records):
            s = r.summary()
            s["i"] = i
            s["supported"] = r.variant in ("NT", "NS", "PO", "FT")
            summaries.append(s)
            variants[r.variant] = variants.get(r.variant, 0) + 1
        return {"label": self.records_label, "count": len(self.records), "variants": variants, "hands": summaries}

    def get_replay(self, i: int, agent_spec: Optional[str]) -> Dict[str, Any]:
        if i < 0 or i >= len(self.records):
            raise ApiError("hand index out of range", 404)
        rec = self.records[i]
        with self.lock:
            cached = self.replay_cache.get(i)
        if cached is None:
            try:
                res = replay(rec)
            except UnsupportedVariant as exc:
                raise ApiError(str(exc), 422)
            cached = (res, res.to_dict(include_observations=False))
            with self.lock:
                self.replay_cache[i] = cached
        res, data = cached
        out = dict(data)
        out["i"] = i
        out["phh"] = rec.to_phh()
        if agent_spec:
            kind = agent_spec.split(":")[0].lower()
            if kind in FAST_AGENTS:
                out["predictions"] = self._predict_all(res, agent_spec)
            else:
                out["predictions"] = None  # fetched lazily by the client
        return out

    def _predict_all(self, res, agent_spec: str) -> Dict[str, Any]:
        try:
            agent = make_agent(agent_spec, seed=1)
        except AgentError as exc:
            raise ApiError(str(exc))
        from .evaluate import observation_for
        preds: Dict[str, Any] = {}
        for dp in res.decisions:
            obs = observation_for(dp, use_final_cards=True)
            if obs is None:
                preds[str(dp.id)] = {"skipped": "hero cards unknown"}
                continue
            raw = safe_act(agent, obs)
            pred = normalise_prediction(raw, dp.legal)
            preds[str(dp.id)] = {"prediction": pred, "raw": raw, "agree": pred["action"] == dp.actual["action"],
                                 "actual": dp.actual}
        return {"agent": agent.describe(), "decisions": preds}

    def predict_one(self, body: Dict[str, Any]) -> Dict[str, Any]:
        i = int(body.get("i", -1))
        did = int(body.get("decision_id", -1))
        spec = body.get("agent") or "equity"
        self.get_replay(i, None)
        res, _ = self.replay_cache[i]
        dp = next((d for d in res.decisions if d.id == did), None)
        if dp is None:
            raise ApiError("unknown decision", 404)
        from .evaluate import observation_for
        obs = observation_for(dp, use_final_cards=True)
        if obs is None:
            return {"skipped": "hero cards unknown"}
        try:
            agent = make_agent(spec, seed=1)
        except AgentError as exc:
            raise ApiError(str(exc))
        raw = safe_act(agent, obs)
        pred = normalise_prediction(raw, dp.legal)
        return {"prediction": pred, "raw": raw, "agree": pred["action"] == dp.actual["action"], "actual": dp.actual,
                "agent": agent.describe()}

    def start_scenario(self, body: Dict[str, Any]) -> Dict[str, Any]:
        i = int(body.get("i", -1))
        if i < 0 or i >= len(self.records):
            raise ApiError("hand index out of range", 404)
        sc = dict(body)
        sc["record"] = self.records[i]
        return self.game.start_scenario(sc)

    # -- evaluation -----------------------------------------------------------
    def start_eval(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if not self.records:
            raise ApiError("load a hand history first")
        spec = body.get("agent") or "equity"
        try:
            agent = make_agent(spec, seed=int(body.get("seed", 1) or 1))
        except AgentError as exc:
            raise ApiError(str(exc))
        limit = body.get("limit_hands")
        recs = self.records[: int(limit)] if limit else list(self.records)
        max_decisions = int(body["max_decisions"]) if body.get("max_decisions") else None
        use_final = bool(body.get("use_final_cards", True))
        only_player = body.get("only_player") or None
        job = self.eval_job

        def progress(done: int, total: int, report) -> None:
            n = len(report.rows)
            agree = sum(r.agree for r in report.rows)
            job.progress = {"done": done, "total": total, "decisions": n,
                            "accuracy": (agree / n) if n else None}
            if done % 5 == 0 or done == total:
                self.bus.publish({"type": "eval_progress", **job.progress})

        def target() -> Dict[str, Any]:
            report = shadow_evaluate(recs, agent, use_final_cards=use_final, max_decisions=max_decisions,
                                     progress=progress, should_stop=lambda: job.stop_flag, only_player=only_player)
            job.payload = report
            return report.to_dict(include_rows=True, max_rows=2000)

        job.start(target)
        return job.state()

    # -- simulation -----------------------------------------------------------
    def start_sim(self, body: Dict[str, Any]) -> Dict[str, Any]:
        specs = body.get("agents") or ["equity", "tag", "lag", "random"]
        if len(specs) < 2:
            raise ApiError("need at least two agents")
        for s in specs:
            try:
                make_agent(s)
            except AgentError as exc:
                raise ApiError(str(exc))
        hands = int(body.get("hands", 200))
        job = self.sim_job

        def progress(done: int, total: int) -> None:
            job.progress = {"done": done, "total": total}
            if done % max(1, total // 50) == 0 or done == total:
                self.bus.publish({"type": "sim_progress", **job.progress})

        def target() -> Dict[str, Any]:
            res = run_match(specs, hands=hands, stack=float(body.get("stack", 200)),
                            small_blind=float(body.get("small_blind", 1)), big_blind=float(body.get("big_blind", 2)),
                            ante=float(body.get("ante", 0) or 0), seed=body.get("seed"),
                            reset_stacks=bool(body.get("reset_stacks", True)), progress=progress,
                            names=body.get("names"))
            self.sim_records = res.records
            d = res.to_dict()
            # per-hand cumulative net for the chart
            cum = {s["seat"]: 0.0 for s in res.seats}
            series = {str(s["seat"]): [] for s in res.seats}
            for rec in res.records:
                for seat_no, (start, end) in enumerate(zip(rec.starting_stacks, rec.finishing_stacks or rec.starting_stacks)):
                    pass
                for idx, p in enumerate(rec.players):
                    seat = rec.seats[idx] if rec.seats else idx + 1
                    net = (rec.finishing_stacks[idx] if rec.finishing_stacks else 0) - rec.starting_stacks[idx]
                    cum[seat] = round(cum[seat] + net, 2)
                for seat in cum:
                    series[str(seat)].append(cum[seat])
            d["cumulative"] = series
            return d

        job.start(target)
        return job.state()


# ------------------------------------------------------------------ handler
APP = App()


class Handler(BaseHTTPRequestHandler):
    server_version = "pokersim/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        if args and "/api/events" in str(args[0]):
            return
        super().log_message(fmt, *args)

    # -- utils ----------------------------------------------------------------
    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(sanitize(data)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text: str, content_type: str = "text/plain", filename: Optional[str] = None) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            raise ApiError("invalid JSON body")
        return data if isinstance(data, dict) else {"value": data}

    def _static(self, rel: str) -> None:
        rel = rel.lstrip("/") or "index.html"
        path = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not path.startswith(os.path.normpath(WEB_DIR)) or not os.path.isfile(path):
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = APP.bus.subscribe()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(sanitize(ev))
                self.wfile.write(f"event: {ev.get('type', 'message')}\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            APP.bus.unsubscribe(q)

    # -- routing --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        try:
            if path == "/api/events":
                self._sse()
                return
            if path.startswith("/api/"):
                self._json(self._get_api(path, qs))
                return
            if path == "/":
                self._static("index.html")
            else:
                self._static(path)
        except ApiError as exc:
            self._json({"error": str(exc)}, exc.status)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            body = self._body()
            self._json(self._post_api(parsed.path, body))
        except ApiError as exc:
            self._json({"error": str(exc)}, exc.status)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _get_api(self, path: str, qs: Dict[str, str]) -> Any:
        if path.startswith("/api/mirror/"):
            from . import mirror_api
            try:
                out = mirror_api.get(path, qs)
            except (ValueError, KeyError) as exc:
                raise ApiError(str(exc))
            if out is None:
                raise ApiError("not found", 404)
            return out
        if path == "/api/agents":
            try:
                import anthropic  # type: ignore # noqa: F401
                has_anthropic = True
            except ImportError:
                has_anthropic = False
            return {"agents": available_agents(), "anthropic_installed": has_anthropic,
                    "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))}
        if path == "/api/game/state":
            return APP.game.state()
        if path == "/api/game/advice":
            return APP.game.advice()
        if path == "/api/game/opponents":
            return APP.game.opponents()
        if path == "/api/game/export":
            recs = APP.game.records
            if not recs:
                raise ApiError("no hands played yet", 404)
            self._text(phh.dump_hands(recs), "text/plain", "pokersim-live.phhs")
            return None
        if path == "/api/replay/list":
            return APP.records_summary()
        if path == "/api/replay/hand":
            return APP.get_replay(int(qs.get("i", 0)), qs.get("agent"))
        if path == "/api/dataset/featured":
            return {"featured": dataset.FEATURED, "page": dataset.HF_PAGE}
        if path == "/api/dataset/local":
            return {"samples": dataset.list_samples(), "cached": dataset.list_cached()}
        if path == "/api/dataset/list":
            try:
                return {"path": qs.get("path", ""), "items": dataset.list_dir(qs.get("path", ""))}
            except FileNotFoundError as exc:
                raise ApiError(str(exc), 404)
            except Exception as exc:  # noqa: BLE001
                raise ApiError(f"could not reach Hugging Face: {exc}", 502)
        if path == "/api/evaluate/state":
            return APP.eval_job.state()
        if path == "/api/evaluate/export":
            rep = APP.eval_job.payload
            if rep is None:
                raise ApiError("no evaluation yet", 404)
            import csv
            import io
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["hand", "decision", "player", "name", "street", "position", "pot", "to_call", "cards",
                        "actual", "actual_to", "predicted", "predicted_to", "agree", "agree_coarse", "latency_ms", "note"])
            for r in rep.rows:
                w.writerow([r.hand, r.decision_id, r.player, r.name, r.street, r.position, r.pot, r.to_call,
                            " ".join(r.cards), r.actual, r.actual_to, r.predicted, r.predicted_to, r.agree,
                            r.agree_coarse, r.latency_ms, r.note or ""])
            self._text(buf.getvalue(), "text/csv", "pokersim-eval.csv")
            return None
        if path == "/api/simulate/state":
            return APP.sim_job.state()
        if path == "/api/simulate/export":
            if not APP.sim_records:
                raise ApiError("no simulated hands yet", 404)
            self._text(phh.dump_hands(APP.sim_records), "text/plain", "pokersim-selfplay.phhs")
            return None
        raise ApiError("not found", 404)

    def _post_api(self, path: str, body: Dict[str, Any]) -> Any:
        if path.startswith("/api/mirror/"):
            from . import mirror_api
            try:
                out = mirror_api.post(path, body)
            except (ValueError, KeyError) as exc:
                raise ApiError(str(exc))
            if out is None:
                raise ApiError("not found", 404)
            return out
        if path == "/api/game/new":
            return APP.game.new_game(body)
        if path == "/api/game/next":
            return APP.game.next_hand()
        if path == "/api/game/act":
            amt = body.get("amount")
            return APP.game.human_act(body.get("seat"), body.get("action", ""), float(amt) if amt not in (None, "") else None)
        if path == "/api/game/reveal":
            return APP.game.set_reveal(bool(body.get("reveal", True)))
        if path == "/api/game/observe":
            return APP.game.observe(int(body.get("hands", 60)))
        if path == "/api/replay/load":
            return APP.load_records(body)
        if path == "/api/replay/predict":
            return APP.predict_one(body)
        if path == "/api/scenario/start":
            return APP.start_scenario(body)
        if path == "/api/evaluate/start":
            return APP.start_eval(body)
        if path == "/api/evaluate/stop":
            APP.eval_job.stop_flag = True
            return APP.eval_job.state()
        if path == "/api/simulate/start":
            return APP.start_sim(body)
        raise ApiError("not found", 404)


class _Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already taken, and requests keep
    # reaching the stale process.  Refuse instead, so an old server cannot silently shadow new code.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> None:
    try:
        httpd = _Server((host, port), Handler)
    except OSError as exc:
        print(f"cannot listen on {host}:{port} ({exc}). Another pokersim server is probably running: stop it or pass --port.")
        raise SystemExit(1)
    url = f"http://{host}:{port}/"
    print(f"pokersim serving on {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
