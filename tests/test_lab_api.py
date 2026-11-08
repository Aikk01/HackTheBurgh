"""Poker Mirror endpoint (pokersim.lab_api): the response always has every slot, and nothing leaks hidden cards."""
import json
import unittest
from unittest import mock

from pokersim import lab_api
from pokersim.server import EventBus, GameSession

SEATS = [{"name": "Me", "agent": "human"}, {"agent": "equity"}, {"agent": "tag"}, {"agent": "random"}]


class LabApiTest(unittest.TestCase):
    def setUp(self):
        self.game = GameSession(EventBus())
        self.game.new_game({"seats": SEATS, "small_blind": 1, "big_blind": 2, "stack": 200, "bot_delay_ms": 0})

    def tearDown(self):
        with self.game.lock:
            self.game.hand = None

    def test_shape_without_models(self):
        none = {"base": {"actions": False, "hands": False}, "rl": {"actions": False}, "personal": {"actions": False}}
        with mock.patch.object(lab_api, "_status", return_value=none):
            out = lab_api.predict(self.game)
        self.assertEqual(out["models"], none)
        you = out["you"]
        self.assertIsNotNone(you)
        self.assertEqual(you["should"], {"base": None, "rl": None, "rl_value_bb": None})
        self.assertEqual(you["would"], {"personal": None})
        self.assertIsNone(you["accuracy"])
        self.assertEqual(len(out["players"]), 3)
        for p in out["players"]:
            self.assertEqual(p["base"], {"action": None, "hands": None})
            self.assertEqual(p["this_player"], {"action": None, "hands": None, "seen": 0, "accuracy": None,
                                                "habits": [], "width": 0.0, "width_seen": 0})
            self.assertNotEqual(p["index"], you["index"])

    def test_no_hidden_cards_in_response(self):
        out = json.dumps(lab_api.predict(self.game))
        with self.game.lock:
            hidden = [c for p in self.game.hand.players if p.index != lab_api.predict(self.game)["you"]["index"] for c in p.hole]
        from pokersim import cards as C
        for c in hidden:
            self.assertNotIn(f'"{C.card_str(c)}"', out)

    def test_no_table(self):
        g = GameSession(EventBus())
        out = lab_api.predict(g)
        self.assertIsNone(out["you"])
        self.assertEqual(out["players"], [])


if __name__ == "__main__":
    unittest.main()
