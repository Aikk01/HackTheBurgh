import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json  # noqa: E402
import tempfile  # noqa: E402

from pokersim import mirror_eval as EV  # noqa: E402
from pokersim import mirror_session as MS  # noqa: E402
from pokersim.mirror_model import PopulationModel, StrengthTable  # noqa: E402
from pokersim.river_env import generate_scenarios  # noqa: E402

COUNTS = {"checked_through": {"fold": 266, "call": 110, "raise": 18}, "bettor_aggr": {"fold": 52, "call": 42, "raise": 6},
          "defender_aggr": {"fold": 39, "call": 40, "raise": 6}}
POOLED = {"fold": 357, "call": 192, "raise": 30}


class FakeShared(MS.SharedModels):
    """Small pools and an in-memory population so the tests need no artifacts."""

    def __init__(self):
        super().__init__()
        self._population = PopulationModel(COUNTS, POOLED)
        self._type_prior_checked = True
        self._drill_pool = generate_scenarios(60, seed=1)
        self._play_pool = generate_scenarios(20, seed=2)
        self.sessions_dir = None  # never write test decisions into the real session logs

    def make_strength(self):
        return StrengthTable.default(self.population)


class SessionLogs(unittest.TestCase):
    """Volunteer decisions are logged locally and scored only from aggregates."""

    def play(self, session, n, action="call"):
        for _ in range(n):
            v = session.new_drill()
            session.respond(v["pending"]["decision_id"], action)

    def test_tags_logging_calibration_and_heldout_scores(self):
        shared = FakeShared()
        with tempfile.TemporaryDirectory() as logs, tempfile.TemporaryDirectory() as out:
            shared.sessions_dir = logs
            cal = MS.MirrorSession("cal1", shared, seed=1)
            cal.set_tag("calibration")
            self.play(cal, 4, "fold")
            held = MS.MirrorSession("held1", shared, seed=2)
            held.set_tag("heldout")
            self.play(held, 3, "call")
            practice = MS.MirrorSession("prac1", shared, seed=3)
            self.play(practice, 2)
            with self.assertRaises(ValueError):
                practice.set_tag("everything")
            with open(os.path.join(logs, "held1.jsonl"), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 3)
            d = rows[0]["decision"]
            self.assertEqual(rows[0]["tag"], "heldout")
            self.assertIn(d["defender_strength"], ("weak", "medium", "strong"))
            self.assertIn("population_prediction", d["extra"])
            self.assertIn("bettor_cards", d["extra"]["outcome"])
            counts = EV.session_counts(logs)
            self.assertEqual(counts["calibration"], {"sessions": 1, "decisions": 4})
            self.assertEqual(counts["heldout"], {"sessions": 1, "decisions": 3})
            res = EV.evaluate_sessions(logs, out_dir=out, population=shared.population)
            self.assertEqual(res["calibration"]["sessions"], 1)
            self.assertEqual(res["calibration"]["decisions"], 4)
            self.assertEqual(res["heldout_forecast"]["decisions"], 3)
            self.assertEqual(res["heldout_readout"]["decisions"], 3)
            self.assertEqual(res["heldout_readout"]["strength_table"], "calibration sessions")
            with open(os.path.join(out, "strength_calibration.json"), encoding="utf-8") as f:
                calib = json.load(f)
            self.assertEqual(sum(v for c in calib["counts"].values() for b in c.values() for v in b.values()), 4)


class Drills(unittest.TestCase):
    def setUp(self):
        self.shared = FakeShared()
        self.s = MS.MirrorSession("tok1", self.shared, seed=5)

    def test_prediction_is_committed_before_the_click_and_updated_once(self):
        v = self.s.new_drill()
        pending = v["pending"]
        self.assertIsNotNone(pending["prediction"])
        self.assertEqual(pending["prediction"]["observations"], 0)
        self.assertNotIn("defender_strength", pending)       # the simulator keeps the bucket private
        self.assertEqual(self.s.drill_view()["pending"]["decision_id"], pending["decision_id"])  # refresh recovers it
        again = self.s.new_drill()                             # no second pending decision
        self.assertEqual(again["pending"]["decision_id"], pending["decision_id"])
        res = self.s.respond(pending["decision_id"], "call")
        self.assertEqual(res["decision"]["actual"], "call")
        self.assertEqual(res["after"]["observations"], 1)
        self.assertEqual(self.s.personal.observations(), 1)
        with self.assertRaises(ValueError):
            self.s.respond(pending["decision_id"], "call")   # cannot be applied twice
        self.assertIsNone(self.s.drill_view()["pending"])
        self.assertEqual(len(res["profile"]["history"]), 1)
        self.assertIn("outcome", res["decision"]["extra"])

    def test_illegal_response_rejected(self):
        v = self.s.new_drill()
        with self.assertRaises(ValueError):
            self.s.respond(v["pending"]["decision_id"], "check")

    def test_drills_balance_contexts(self):
        seen = []
        for _ in range(6):
            v = self.s.new_drill()
            seen.append(v["pending"]["context"])
            self.s.respond(v["pending"]["decision_id"], "fold")
        self.assertEqual(set(seen), {"checked_through", "bettor_aggr", "defender_aggr"})


class PlayAndFreeze(unittest.TestCase):
    def setUp(self):
        self.shared = FakeShared()
        self.s = MS.MirrorSession("tok2", self.shared, seed=9)

    def test_freeze_play_and_readout(self):
        for _ in range(3):
            v = self.s.new_drill()
            self.s.respond(v["pending"]["decision_id"], "fold")
        fr = self.s.freeze()
        self.assertEqual(fr["observations"], 3)
        view = self.s.new_play_spot(scenario_index=0)
        self.assertEqual(view["spot"]["stage"], 1)
        adv = view["advice"]
        self.assertIn(adv["recommend"], ("bet", "check"))
        self.assertAlmostEqual(adv["forecast"]["fold"], fr["public"][view["spot"]["context"]]["fold"])
        after = self.s.play_act("bet")
        self.assertIn(after["response"], ("fold", "call", "raise"))
        self.assertAlmostEqual(sum(after["readout"]["buckets"].values()), 1.0)
        if after["spot"]["stage"] == 2:
            after = self.s.play_act("fold")
            self.assertEqual(after["result"]["reward_chips"], -50.0)
        else:
            self.assertIsNotNone(after["result"])
        self.assertEqual(len(self.s.play_log), 1)
        # frozen twin does not change when the judge keeps practising
        h = self.s.frozen.hash
        v = self.s.new_drill()
        self.s.respond(v["pending"]["decision_id"], "raise")
        self.assertEqual(self.s.frozen.hash, h)

    def test_sessions_are_isolated(self):
        reg = MS.MirrorRegistry(self.shared)
        a = reg.get("a")
        b = reg.get("b")
        v = a.new_drill()
        a.respond(v["pending"]["decision_id"], "call")
        self.assertEqual(a.personal.observations(), 1)
        self.assertEqual(b.personal.observations(), 0)
        self.assertIsNone(b.drill_view()["pending"])
        reg.reset("a")
        self.assertEqual(reg.get("a").personal.observations(), 0)

    def test_hidden_cards_do_not_change_the_forecast(self):
        v = self.s.new_drill()
        p1 = v["pending"]["prediction"]["probs"]
        sc = self.s.pending_scenario
        # swap the defender's hidden hand for another; the public forecast must be identical
        other = self.shared.drill_pool[-1]
        self.s.pending_scenario = type(sc)(sc.board, other.defender_cards, sc.bettor_cards, sc.context, "weak", sc.bettor_bucket,
                                           0.1, sc.bettor_equity, sc.seed)
        p2 = self.s.personal.predict(v["pending"]["context"], v["pending"]["legal"]).probs
        self.assertEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
