import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import mirror_eval as EV  # noqa: E402
from pokersim.mirror_data import RiverSpot  # noqa: E402
from pokersim.mirror_model import PopulationModel, TypePrior  # noqa: E402
from pokersim.mirror_types import CONTEXTS, CTX_CHECKED_THROUGH  # noqa: E402


def spot(hand_id, defender, response, ts, context=CTX_CHECKED_THROUGH, venue="v", source="shard-00001"):
    return RiverSpot(venue=venue, hand_id=hand_id, source_file=source, hand_number_in_file=1, timestamp=ts, stake="100NLH",
                     defender=defender, bettor="b", board=["Ah", "Kd", "2c", "7s", "9h"], bet=50.0, pot_before_bet=100.0, ratio=0.5,
                     exact_half=True, context=context, response=response, raise_to=None, defender_stack=150.0, bettor_stack=150.0,
                     all_three_legal=True, warning_free=True, defender_cards_known=False, defender_cards=["??", "??"],
                     bettor_cards=["??", "??"], bettor_cards_known=False, fingerprint=hand_id, min_bet=2.0, big_blind=2.0)


class Ladder(unittest.TestCase):
    def test_bins_and_personal_history(self):
        train = [spot(f"t{i}", "alice", "call", f"2009-07-0{1 + i % 5}T00:00:00") for i in range(20)]
        train += [spot(f"u{i}", "bob", "fold", "2009-07-01T00:00:00") for i in range(2)]
        test = [spot("x1", "alice", "call", "2009-07-20T00:00:00"), spot("x2", "bob", "fold", "2009-07-20T00:00:00"),
                spot("x3", "carol", "fold", "2009-07-20T00:00:00")]
        pop = PopulationModel({c: {"fold": 60, "call": 35, "raise": 5} for c in CONTEXTS}, {"fold": 180, "call": 105, "raise": 15})
        res = EV.forecast_ladder(train, test, pop, bootstrap=50)
        bins = {r["bin"]: r for r in res["ladder"]}
        self.assertEqual(set(bins), {"0", "1-3", "16-63"})
        self.assertEqual(res["new_players"]["n"], 1)
        self.assertEqual(res["known_players"]["n"], 2)
        # alice called 20 times before: the personal model must beat the population model on her call
        alice = bins["16-63"]
        self.assertLess(alice["log_loss"]["personal"], alice["log_loss"]["population"])
        # carol is new: personal equals population
        carol = bins["0"]
        self.assertAlmostEqual(carol["log_loss"]["personal"], carol["log_loss"]["population"])
        self.assertAlmostEqual(res["overall"]["log_loss"]["population"], -math.log((60 + 1) / 103) * 2 / 3 - math.log((35 + 1) / 103) / 3, places=9)

    def test_type_arm_and_bin_labels(self):
        self.assertEqual(EV.bin_label(0), "0")
        self.assertEqual(EV.bin_label(2), "1-3")
        self.assertEqual(EV.bin_label(100), "64+")
        pop = PopulationModel({c: {"fold": 60, "call": 35, "raise": 5} for c in CONTEXTS}, {"fold": 180, "call": 105, "raise": 15})
        tp = TypePrior(["station", "rock"], [0.5, 0.5], [{c: {"fold": 0.1, "call": 0.85, "raise": 0.05} for c in CONTEXTS},
                                                       {c: {"fold": 0.9, "call": 0.08, "raise": 0.02} for c in CONTEXTS}])
        test = [spot("x1", "dave", "call", "2009-07-20T00:00:00")]
        resp = {("v", "dave"): [0.95, 0.05]}
        res = EV.forecast_ladder([], test, pop, type_prior=tp, responsibilities=resp, bootstrap=20)
        self.assertIn("type", res["overall"]["log_loss"])
        self.assertLess(res["overall"]["log_loss"]["type"], res["overall"]["log_loss"]["population"])
        self.assertEqual(res["style_known_decisions"], 1)

    def test_pluribus_check_shape(self):
        spots = [{"response": "fold", "ratio": 0.5, "defender_equity": 0.2, "bucket": "weak"}] * 5 + \
                [{"response": "call", "ratio": 0.5, "defender_equity": 0.6, "bucket": "medium"}] * 5 + \
                [{"response": "raise", "ratio": 0.5, "defender_equity": 0.9, "bucket": "strong"}] * 3
        chk = EV.pluribus_check(spots, bootstrap=50)
        self.assertTrue(chk["monotone"])
        self.assertEqual(chk["spots"], 13)
        self.assertEqual(chk["half_pot_band"], 13)


if __name__ == "__main__":
    unittest.main()
