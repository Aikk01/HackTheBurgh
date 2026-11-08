"""FastAPI websocket that streams the market one JSON event per action.

    .venv/bin/python -m market.server            # http://127.0.0.1:8001  (pokersim UI stays on :8000)

Endpoints
  GET  /api/health
  GET  /api/schema                      event schema (same as NOTES.md)
  GET  /api/hands?limit=50              replayable held-out hands (test tables only)
  WS   /ws/hand/{hand_id}?tracked=0&delay_ms=700&mode=model
  WS   /ws/stream?delay_ms=700&mode=model     endless replay of random held-out hands

Messages on a websocket: one "meta", then one "event" per action, then "end" per hand.
The opponent's cards are only revealed in "end" - never while the hand is streaming.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .cards import card_str
from .evaluate import split_by_table
from .likelihood import MODEL_PATH, LikelihoodModel
from .parse import load_hands
from .tracker import COMBO_LABELS, GRID_LABELS, MarketTracker

SCHEMA = {
    "meta": {
        "type": "'meta'", "hand_id": "int", "tracked": "{position, cards[2]}",
        "opponent": "{position, preflop_role}", "pre_raises": "int (1 = single-raised pot, 2 = 3-bet pot)",
        "big_blind": "float (chip amounts below are in these units)",
        "combo_labels": "list[1326] e.g. 'AsKh' - index order of every posterior vector",
        "grid_labels": "13x13 e.g. 'AKs' (row/col 0 = Ace; suited above diagonal, offsuit below)",
    },
    "event": {
        "type": "'event'", "seq": "int, 0.. within the hand", "mode": "'model' | 'prior' | 'uniform'",
        "hand_id": "int", "street": "'flop' | 'turn' | 'river'", "board": "list[str] e.g. ['Kh','7c','2d']",
        "tracked": "{position, cards}",
        "action": "null on a board deal, else {actor: 'tracked'|'opponent', kind: check|bet|call|raise|fold, "
                  "size_frac_pot: chips added / pot before the action, pot: pot before the action, to_call}",
        "price": "float in [0,1]: tracked player's win probability vs the opponent posterior (ties = 1/2)",
        "pot_odds_price": "float | null: to_call / (pot + to_call), only when the tracked player faces a bet",
        "decision": "'call' | 'fold' | null: call iff price > pot_odds_price; emitted BEFORE the tracked "
                    "player's actual action",
        "latency_ms": "float: compute time for this event (Bayes update + equity + price)",
        "posterior": "list[1326] floats summing to 1 (combo_labels order); 0 = impossible/blocked",
        "grid": "13x13 posterior mass per hand class (grid_labels order)",
        "top_combos": "list[{combo, p}] - 8 most likely opponent combos",
        "effective_combos": "float: exp(entropy) of the posterior (1326 = uniform, 1 = certain)",
    },
    "end": {
        "type": "'end'", "hand_id": "int", "opponent_cards": "list[str] | null (revealed at showdown)",
        "tracked_share": "float | null: share of the pot won by the tracked player (1 / 0.5 / 0)",
        "final_board": "list[str]",
    },
}

app = FastAPI(title="Poker as a market")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE: Dict = {}


def _state() -> Dict:
    if not STATE:
        hands = load_hands()
        train, test, _ = split_by_table(hands)
        model = LikelihoodModel.load() if os.path.exists(MODEL_PATH) else LikelihoodModel().fit(train)
        replayable = [h for h in test if any(h["hole"]) and h["actions"]]
        STATE.update(model=model, hands={h["id"]: h for h in replayable},
                     showcase=[h["id"] for h in replayable if h["showdown"] and all(h["hole"])])
    return STATE


def _meta(h: Dict, me: int) -> Dict:
    return {"type": "meta", "hand_id": h["id"], "big_blind": h["bb"],
            "tracked": {"position": h["positions"][me], "cards": [card_str(c) for c in h["hole"][me]]},
            "opponent": {"position": h["positions"][1 - me], "preflop_role": h["pre_role"][1 - me]},
            "pre_raises": h["pre_raises"], "combo_labels": COMBO_LABELS, "grid_labels": GRID_LABELS}


def _end(h: Dict, me: int) -> Dict:
    final = h["boards"].get("river") or h["boards"].get("turn") or h["boards"].get("flop")
    opp = h["hole"][1 - me]
    return {"type": "end", "hand_id": h["id"], "final_board": [card_str(c) for c in final],
            "opponent_cards": [card_str(c) for c in opp] if opp and h["showdown"] else None,
            "tracked_share": h["share"][me] if h["share"] and h["showdown"] else None}


async def _stream_hand(ws: WebSocket, h: Dict, me: int, delay_ms: float, mode: str) -> None:
    st = _state()
    await ws.send_text(json.dumps(_meta(h, me)))
    for ev in MarketTracker(h, me, st["model"], mode=mode).run(include_posterior=True):
        await ws.send_text(json.dumps(ev))
        if delay_ms:
            await asyncio.sleep(delay_ms / 1e3)
    await ws.send_text(json.dumps(_end(h, me)))


@app.get("/")
def index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/api/health")
def health():
    return {"ok": True, "hands": len(_state()["hands"])}


@app.get("/api/schema")
def schema():
    return SCHEMA


@app.get("/api/hands")
def hands(limit: int = 50, showdown_only: bool = True):
    st = _state()
    ids = st["showcase"] if showdown_only else list(st["hands"])
    out = []
    for i in ids[:limit]:
        h = st["hands"][i]
        out.append({"hand_id": i, "positions": h["positions"], "pre_role": h["pre_role"],
                    "known": [x is not None for x in h["hole"]], "n_actions": len(h["actions"]),
                    "showdown": h["showdown"]})
    return out


@app.websocket("/ws/hand/{hand_id}")
async def ws_hand(ws: WebSocket, hand_id: int, tracked: int = 0, delay_ms: float = 700, mode: str = "model"):
    await ws.accept()
    h = _state()["hands"].get(hand_id)
    if h is None or h["hole"][tracked] is None:
        await ws.send_text(json.dumps({"type": "error", "error": "unknown hand or tracked player's cards unknown"}))
        await ws.close()
        return
    try:
        await _stream_hand(ws, h, tracked, delay_ms, mode)
        await ws.close()
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket, delay_ms: float = 700, mode: str = "model", seed: int = 0):
    await ws.accept()
    st = _state()
    rng = random.Random(seed)
    try:
        while True:
            h = st["hands"][rng.choice(st["showcase"])]
            await _stream_hand(ws, h, rng.choice([i for i in (0, 1) if h["hole"][i] is not None]), delay_ms, mode)
            await asyncio.sleep(3 * delay_ms / 1e3)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    _state()
    uvicorn.run(app, host="127.0.0.1", port=8001)
