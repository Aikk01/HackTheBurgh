import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import phh  # noqa: E402
from pokersim.agents import Agent, AgentError, HTTPAgent, make_agent, safe_act  # noqa: E402
from pokersim.evaluate import normalise_prediction, shadow_evaluate  # noqa: E402

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "samples")

LEGAL = {"fold": True, "check": False, "call": 10.0, "raise": True, "min_raise_to": 20.0, "max_raise_to": 200.0,
         "raise_kind": "raise"}


class Specs(unittest.TestCase):
    def test_make_agent_specs(self):
        self.assertEqual(make_agent("equity").kind, "equity")
        a = make_agent("equity:aggression=0.8,iterations=50")
        self.assertEqual(a.aggression, 0.8)
        self.assertEqual(a.iterations, 50)
        h = make_agent("http:http://127.0.0.1:9/act")
        self.assertIsInstance(h, HTTPAgent)
        self.assertEqual(h.url, "http://127.0.0.1:9/act")
        c = make_agent("claude:claude-opus-5-5,effort=low")
        self.assertEqual((c.model, c.effort), ("claude-opus-5-5", "low"))
        with self.assertRaises(AgentError):
            make_agent("nope")

    def test_safe_act_falls_back_on_errors(self):
        class Broken(Agent):
            def act(self, obs):
                raise RuntimeError("boom")

        obs = {"legal_actions": LEGAL}
        out = safe_act(Broken(), obs)
        self.assertEqual(out["action"], "call")
        self.assertIn("boom", out["error"])
        self.assertIn("latency_ms", out)

    def test_http_agent_connection_error(self):
        obs = {"legal_actions": LEGAL}
        out = safe_act(make_agent("http:http://127.0.0.1:9/act", timeout=0.5), obs)
        self.assertEqual(out["action"], "call")
        self.assertIn("HTTP agent", out["error"])


class Normalisation(unittest.TestCase):
    def test_prediction_normalisation(self):
        self.assertEqual(normalise_prediction({"action": "raise", "amount": 5}, LEGAL)["to"], 20.0)
        self.assertEqual(normalise_prediction({"action": "raise", "amount": 900}, LEGAL)["to"], 200.0)
        self.assertEqual(normalise_prediction({"action": "all_in"}, LEGAL)["to"], 200.0)
        self.assertEqual(normalise_prediction({"action": "check"}, LEGAL)["action"], "call")
        free = dict(LEGAL, fold=False, check=True, call=0.0)
        self.assertEqual(normalise_prediction({"action": "fold"}, free)["action"], "check")
        self.assertEqual(normalise_prediction({"action": "bet", "amount": 30}, free)["action"], "raise")


class ShadowEvaluation(unittest.TestCase):
    def test_report_shape(self):
        recs = phh.load_file(os.path.join(SAMPLES, "pluribus-session-30.phhs"))[:10]
        rep = shadow_evaluate(recs, make_agent("station"))
        s = rep.summary()
        self.assertGreater(s["decisions"], 20)
        self.assertEqual(s["skipped_unknown_cards"], 0)
        self.assertTrue(0 <= s["accuracy"] <= 1)
        # a calling station never folds or raises, so every recorded fold is a disagreement
        self.assertEqual(s["by_action"]["fold"]["agree"], 0)
        self.assertEqual(len(rep.rows), s["decisions"])
        self.assertTrue(all(r.list_index >= 0 for r in rep.rows))

    def test_unknown_cards_are_skipped_unless_revealed(self):
        recs = phh.load_file(os.path.join(SAMPLES, "handhq-pokerstars-200nl-sample.phhs"))[:30]
        rep = shadow_evaluate(recs, make_agent("station"), use_final_cards=False)
        self.assertEqual(len(rep.rows), 0)
        rep2 = shadow_evaluate(recs, make_agent("station"), use_final_cards=True)
        self.assertGreaterEqual(len(rep2.rows), 0)
        self.assertGreater(rep2.skipped_unknown_cards, 0)


if __name__ == "__main__":
    unittest.main()
