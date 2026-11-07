"""Minimal model server for the ``http`` agent adapter.

Run:  python examples/model_server.py --port 5001
Then use the agent spec  http:http://127.0.0.1:5001/act  in the UI, or:
      python -m pokersim evaluate data/samples/pluribus-session-30.phhs --agent http:http://127.0.0.1:5001/act

Replace ``decide`` with your own model.  The observation format is documented in the
README (``pokersim.engine.Hand.observation``).
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RANKS = "23456789TJQKA"


def decide(obs: dict) -> dict:
    """A tiny rule-based policy - swap in your model here."""
    legal = obs["legal_actions"] or {}
    cards = obs["hero"]["cards"]
    ranks = sorted((RANKS.index(c[0]) for c in cards if c != "??"), reverse=True)
    pair = len(ranks) == 2 and ranks[0] == ranks[1]
    strong = pair and ranks[0] >= RANKS.index("T") or (len(ranks) == 2 and ranks[0] == 12 and ranks[1] >= RANKS.index("Q"))
    if obs["street"] == "preflop":
        if strong and legal.get("raise"):
            return {"action": "raise", "amount": min(legal["max_raise_to"], max(legal["min_raise_to"], 3 * obs["min_bet"])),
                    "reasoning": "premium hand, open/3-bet"}
        if legal.get("check") or (legal.get("call", 0) <= 2 * obs["min_bet"] and ranks and ranks[0] >= RANKS.index("9")):
            return {"action": "call", "reasoning": "cheap to see a flop"}
        return {"action": "fold", "reasoning": "weak hand facing a bet"}
    # post-flop: bet pairs, otherwise check/fold
    board_ranks = [RANKS.index(c[0]) for c in obs["board"]]
    hit = any(r in board_ranks for r in ranks) or pair
    if hit and legal.get("raise"):
        pot = obs["total_pot"]
        return {"action": legal.get("raise_kind", "bet"), "amount": obs["hero"]["bet"] + legal.get("call", 0) + 0.66 * (pot + legal.get("call", 0)),
                "reasoning": "made a pair, bet two thirds pot"}
    if legal.get("check"):
        return {"action": "check", "reasoning": "nothing yet"}
    return {"action": "fold", "reasoning": "missed, facing a bet"}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        obs = json.loads(self.rfile.read(length) or b"{}")
        try:
            action = decide(obs)
        except Exception as exc:  # noqa: BLE001
            action = {"action": "call", "reasoning": f"model error: {exc}"}
        body = json.dumps(action).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()
    print(f"model server listening on http://127.0.0.1:{args.port}/act")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
