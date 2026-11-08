"""Start the API server on a free port, exercise the main endpoints, shut down."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import server as srv  # noqa: E402


def call(base: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode()
        return json.loads(raw) if r.headers.get("Content-Type", "").startswith("application/json") else raw


def main() -> int:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        print("agents:", [a["id"] for a in call(base, "/api/agents")["agents"]])
        html = call(base, "/")
        assert "<title>" in html, "index.html not served"
        # live game: human + 3 bots, instant bots
        st = call(base, "/api/game/new", {"seats": [{"name": "Me", "agent": "human"}, {"agent": "equity"},
                                                     {"agent": "tag"}, {"agent": "random"}],
                                           "small_blind": 1, "big_blind": 2, "stack": 200, "bot_delay_ms": 0})
        for _ in range(50):
            if st["hand"]["status"] != "betting" or st["hand"]["players"][st["hand"]["actor"]]["is_human"]:
                break
            time.sleep(0.05)
            st = call(base, "/api/game/state")
        hand = st["hand"]
        print("game status:", hand["status"], "actor:", hand["actor"], "legal:", hand["legal"])
        hands_done = 0
        for _round in range(400):
            st = call(base, "/api/game/state")
            hand = st["hand"]
            if hand["status"] in ("finished", "incomplete"):
                hands_done += 1
                if hands_done >= 3:
                    break
                st = call(base, "/api/game/next", {})
                continue
            if hand["status"] == "betting" and hand["actor"] is not None and hand["players"][hand["actor"]]["is_human"]:
                la = hand["legal"]
                if la["raise"] and _round % 3 == 0:
                    call(base, "/api/game/act", {"action": "raise", "amount": la["min_raise_to"]})
                else:
                    call(base, "/api/game/act", {"action": "call"})
            else:
                time.sleep(0.03)
        print("hands played:", hands_done, "log tail:", [e["text"] for e in st["log"][-3:]])
        seats = st["table"]["seats"]
        print("seat stacks after 3 hands:", [(s["name"], s["stack"], s["hands"]) for s in seats], "button seat:", st["table"]["button_seat"])
        assert sum(s["stack"] for s in seats) == 800, "chips must be conserved"
        assert any(s["stack"] != 200 for s in seats), "stacks must carry over between hands"
        assert st["table"]["button_seat"] != 1, "the button must move"
        # Poker Mirror page and its model endpoint: every slot present, values only where a model file exists
        assert "<title>Poker Mirror</title>" in call(base, "/lab.html"), "lab.html not served"
        lab = call(base, "/api/lab/predict")
        assert set(lab["models"]) == {"base", "rl", "personal"}, lab["models"]
        # between hands there is no decision of yours to advise on, so "you" may be empty
        assert lab["you"] is None or (set(lab["you"]["should"]) >= {"base", "rl"} and "personal" in lab["you"]["would"]), lab["you"]
        assert all(set(p) >= {"seat", "name", "base", "this_player"} for p in lab["players"]), lab["players"]
        print("lab models:", lab["models"])
        export = call(base, "/api/game/export")
        print("export lines:", len(export.splitlines()))
        # replay: load a sample, fetch a hand with predictions
        lst = call(base, "/api/replay/load", {"source": "sample", "path": "pluribus-session-30.phhs"})
        print("loaded:", lst["label"], lst["count"], "hands; variants", lst["variants"])
        rp = call(base, "/api/replay/hand?i=0&agent=tag")
        print("replay steps:", len(rp["steps"]), "decisions:", len(rp["decisions"]), "stacks_match:", rp["stacks_match"],
              "predictions:", len(rp["predictions"]["decisions"]))
        pred = call(base, "/api/replay/predict", {"i": 0, "decision_id": 0, "agent": "equity"})
        print("single prediction:", pred["prediction"], "actual", pred["actual"])
        # scenario fork at step 8 with a human in seat of player 0
        sc = call(base, "/api/scenario/start", {"i": 0, "step_index": 8, "seats": {"0": "human"}, "bot_delay_ms": 0})
        print("scenario:", sc["scenario"], "status", sc["hand"]["status"])
        # evaluation job
        ev = call(base, "/api/evaluate/start", {"agent": "tag:iterations=50", "limit_hands": 20})
        for _ in range(200):
            ev = call(base, "/api/evaluate/state")
            if ev["status"] != "running":
                break
            time.sleep(0.05)
        print("eval:", ev["status"], {k: ev["result"][k] for k in ("decisions", "accuracy", "accuracy_coarse")})
        csv_text = call(base, "/api/evaluate/export")
        print("csv rows:", len(csv_text.splitlines()))
        # simulation job
        sim = call(base, "/api/simulate/start", {"agents": ["equity", "tag", "random"], "hands": 30, "seed": 3})
        for _ in range(400):
            sim = call(base, "/api/simulate/state")
            if sim["status"] != "running":
                break
            time.sleep(0.05)
        print("sim:", sim["status"], [(s["name"], s["bb_per_100"]) for s in sim["result"]["seats"]],
              "series len", len(sim["result"]["cumulative"]["1"]))
        # dataset listing (network)
        try:
            ds = call(base, "/api/dataset/list?path=pluribus/100")
            print("hf listing items:", len(ds["items"]))
        except Exception as exc:  # noqa: BLE001
            print("hf listing failed (offline?):", exc)
        print("SERVER SMOKE OK")
        return 0
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    sys.exit(main())
