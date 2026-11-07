"""Exercise the Poker Mirror API end to end on a free port (no browser needed)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import mirror_session as ms  # noqa: E402
from pokersim import server as srv  # noqa: E402


def call(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode()
        return json.loads(raw) if r.headers.get("Content-Type", "").startswith("application/json") else raw


def main() -> int:
    logs = tempfile.TemporaryDirectory()  # keep smoke drills out of the real volunteer logs
    ms.SHARED.sessions_dir = logs.name
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        html = call(base, "/")
        assert "mirror.js" in html
        st = call(base, "/api/mirror/state?session=")
        tok = st["token"]
        print("session", tok, "| obs", st["profile"]["observations"], "| population policy:", st["has_population_policy"])
        ex = call(base, "/api/mirror/exhibit")
        print("exhibit present:", ex["present"], "| two-shard all:", ex["exhibit"]["two_shard"]["cohorts"]["all_eligible"] if ex["present"] else None)
        dh = call(base, "/api/mirror/demo_hands")
        print("demo hands:", [(h["hand"], h["context"], h["response"], h["decision_step"], len(h["replay"]["steps"])) for h in dh["hands"]])
        # drills: commit, refresh, respond
        d = call(base, "/api/mirror/drill", {"session": tok})
        pend = d["pending"]
        print("drill:", pend["decision_id"], pend["context"], "forecast fold", round(pend["prediction"]["fold_mean"], 3), "cards", pend["own_cards"])
        st2 = call(base, f"/api/mirror/state?session={tok}")
        assert st2["drill"]["pending"]["decision_id"] == pend["decision_id"], "refresh must recover the pending decision"
        r = call(base, "/api/mirror/respond", {"session": tok, "decision_id": pend["decision_id"], "action": "call"})
        print("responded:", r["decision"]["actual"], "| outcome:", r["outcome"]["result"], "| after obs", r["after"]["observations"])
        for a in ("fold", "raise", "fold"):
            d = call(base, "/api/mirror/drill", {"session": tok})
            call(base, "/api/mirror/respond", {"session": tok, "decision_id": d["pending"]["decision_id"], "action": a})
        prof = call(base, f"/api/mirror/profile?session={tok}")
        print("profile obs:", prof["observations"], "| history:", len(prof["history"]))
        fr = call(base, "/api/mirror/freeze", {"session": tok})
        print("frozen:", fr["frozen"]["hash"], fr["frozen"]["observations"])
        p = call(base, "/api/mirror/play/new", {"session": tok, "scenario_index": 3})
        print("play spot:", p["spot"]["board"], p["spot"]["own_cards"], "| advice:", p["advice"]["recommend"], p["advice"]["ev_check"], p["advice"]["ev_bet"])
        p = call(base, "/api/mirror/play/act", {"session": tok, "action": "bet"})
        print("twin response:", p["response"], "| readout buckets:", {k: round(v, 3) for k, v in p["readout"]["buckets"].items()})
        if p["spot"]["stage"] == 2:
            p = call(base, "/api/mirror/play/act", {"session": tok, "action": "call"})
        print("result:", p["result"]["reward_chips"], "twin cards", p["result"]["defender_cards"])
        c = call(base, f"/api/mirror/compare?session={tok}")
        print("compare on deal:", {k: (v["action"] if v else None) for k, v in (c["on_deal"] or {}).items()}, "| advice:", c["advice"]["recommend"] if c["advice"] else None)
        tr = call(base, "/api/mirror/train", {"session": tok, "episodes": 1500})
        for _ in range(600):
            st = call(base, f"/api/mirror/state?session={tok}")
            if st["training"]["status"] != "running":
                break
            time.sleep(0.5)
        print("training:", st["training"]["status"], st["training"].get("oracle"), "validation", st["training"].get("validation"))
        ev = call(base, "/api/mirror/evidence")
        print("evidence panels:", {k: v["present"] for k, v in ev["panels"].items()})
        # isolation + reset
        other = call(base, "/api/mirror/state?session=")
        assert other["profile"]["observations"] == 0 and other["token"] != tok
        rs = call(base, "/api/mirror/reset", {"session": tok})
        assert rs["profile"]["observations"] == 0 and rs["token"] == tok
        print("MIRROR SMOKE OK")
        return 0
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    sys.exit(main())
