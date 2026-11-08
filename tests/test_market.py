"""Tests for the market package (pytest).  Run: .venv/bin/python -m pytest tests/test_market.py"""
import os
import time

import numpy as np
import pytest

from market.cards import COMBOS, N_COMBOS, available, card_from_str as C, combo_index
from market.equity import equity_vector
from market.prior import prior, uniform

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "hu_flop_hands.pkl")
needs_data = pytest.mark.skipif(not os.path.exists(DATA), reason="run `python -m market.parse` first")


def test_1326_combos_unique():
    assert N_COMBOS == 1326 == len(COMBOS)
    assert len({tuple(c) for c in COMBOS.tolist()}) == 1326
    assert (COMBOS[:, 0] < COMBOS[:, 1]).all()


def test_card_removal():
    assert available([C("As")]).sum() == 1326 - 51
    dead = [C(x) for x in ("As", "Kd", "7c", "7h", "2s")]
    m = available(dead)
    assert m.sum() == 47 * 46 // 2
    assert not m[combo_index(C("As"), C("Ah"))]
    assert m[combo_index(C("Qs"), C("Qh"))]
    p = prior("BTN", "agg", 1, dead)
    assert p[~m].sum() == 0


def test_aa_vs_kk_preflop():
    e = equity_vector([C("As"), C("Ah")], [], n_runouts=3000, seed=1)
    assert abs(e[combo_index(C("Kc"), C("Kd"))] - 0.82) < 0.025


def test_river_equity_exact():
    board = [C(x) for x in ("Kh", "7c", "2d", "9s", "3h")]
    e = equity_vector([C("As"), C("Ad")], board)
    assert e[combo_index(C("Kc"), C("Kd"))] == 0.0          # set of kings beats aces
    assert e[combo_index(C("Qc"), C("Qd"))] == 1.0
    assert e[combo_index(C("Ac"), C("Ah"))] == 0.5          # chop


def test_turn_equity_exact():
    board = [C(x) for x in ("Kh", "7c", "2d", "9s")]
    e = equity_vector([C("As"), C("Ad")], board)
    assert e[combo_index(C("Kc"), C("Kd"))] == pytest.approx(2 / 44)   # two aces left


def test_priors_are_distributions():
    for pos in ("UTG", "HJ", "CO", "BTN", "SB", "BB"):
        for role, n in (("agg", 1), ("call", 1), ("agg", 2), ("call", 2), ("limp", 0)):
            p = prior(pos, role, n, [C("2c"), C("2d")])
            assert p.min() >= 0 and p.sum() == pytest.approx(1.0)
    assert uniform().sum() == pytest.approx(1.0)


@pytest.fixture(scope="module")
def hands_and_model():
    from market.likelihood import MODEL_PATH, LikelihoodModel
    from market.parse import load_hands
    hands = [h for h in load_hands() if h["showdown"] and all(h["hole"])]
    model = LikelihoodModel.load() if os.path.exists(MODEL_PATH) else LikelihoodModel().fit(hands[:1500])
    return hands, model


@needs_data
def test_posterior_is_distribution(hands_and_model):
    from market.tracker import MarketTracker
    hands, model = hands_and_model
    for h in hands[:40]:
        for ev in MarketTracker(h, 0, model).run(include_posterior=True):
            p = np.asarray(ev["posterior"])
            assert p.min() >= 0
            assert p.sum() == pytest.approx(1.0, abs=1e-4)   # posterior is rounded to 7 dp for JSON
            assert 0.0 <= ev["price"] <= 1.0


@needs_data
def test_latency_per_action_under_50ms(hands_and_model):
    """Full update (likelihood on 1,326 rows + Bayes) + equity/price, per event, incl. board deals."""
    from market.tracker import MarketTracker
    hands, model = hands_and_model
    lat = []
    for h in hands[100:160]:
        t_prev = time.perf_counter()
        for ev in MarketTracker(h, 1, model, seed=7).run():
            now = time.perf_counter()
            lat.append((now - t_prev) * 1e3)
            t_prev = now
    lat = np.array(lat)
    print(f"latency ms: median {np.median(lat):.2f}  p95 {np.percentile(lat, 95):.2f}  max {lat.max():.2f}")
    assert lat.max() < 50


def test_fast_evaluator_matches_eval7():
    from market.cards import board_ranks, board_ranks_reference
    rng = np.random.default_rng(3)
    boards = [[0, 4, 8, 12, 16], [3, 7, 11, 15, 51], [48, 49, 50, 44, 40]]
    boards += [list(rng.choice(52, n, replace=False)) for n in (3, 4, 5) for _ in range(20)]
    for b in boards:
        assert (board_ranks(b) == board_ranks_reference(b)).all()


@needs_data
def test_websocket_stream_schema():
    import json
    from fastapi.testclient import TestClient
    from market.server import app
    client = TestClient(app)
    hid = client.get("/api/hands?limit=1").json()[0]["hand_id"]
    with client.websocket_connect(f"/ws/hand/{hid}?tracked=0&delay_ms=0") as ws:
        meta = json.loads(ws.receive_text())
        assert meta["type"] == "meta" and len(meta["combo_labels"]) == 1326
        msgs = []
        while True:
            m = json.loads(ws.receive_text())
            msgs.append(m)
            if m["type"] == "end":
                break
    events = [m for m in msgs if m["type"] == "event"]
    assert events and all(len(e["posterior"]) == 1326 for e in events)
    for e in events:
        assert set(("street", "board", "action", "posterior", "price", "pot_odds_price", "decision")) <= set(e)
        assert abs(sum(e["posterior"]) - 1) < 1e-4 and abs(sum(map(sum, e["grid"])) - 1) < 1e-4
        assert "opponent_cards" not in e                       # no leakage while streaming
