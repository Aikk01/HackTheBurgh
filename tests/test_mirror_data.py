import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import mirror_data as M  # noqa: E402
from pokersim.mirror_types import CTX_BETTOR_AGGR, CTX_CHECKED_THROUGH, DecisionRecord, EXAMPLE  # noqa: E402

HAVE_SHARDS = all(os.path.exists(M.shard_path(i)) for i in (35, 36))
try:
    import pyarrow  # noqa: F401
    HAVE_ARROW = True
except ImportError:
    HAVE_ARROW = False


class DecisionRecordRoundTrip(unittest.TestCase):
    def test_json_round_trip(self):
        again = DecisionRecord.from_json(EXAMPLE.to_json())
        self.assertEqual(again, EXAMPLE)
        self.assertEqual(again.prediction.probs["fold"], 0.62)


class RowMapping(unittest.TestCase):
    def test_takara_row_maps_to_record_with_integer_casts(self):
        row = {
            "variant": "NT", "venue": "Full Tilt Poker", "table": "t", "time": "08:27:34", "hand": "27015246844",
            "day": 20.0, "month": 7.0, "year": 2009.0, "seat_count": 6.0, "hand_number_in_file": 971.0, "min_bet": 6.0,
            "small_bet": None, "big_bet": None, "bring_in": None, "ante_trimming_status": False,
            "actions": ["d dh p1 ????", "d dh p2 ????", "p2 cbr 18", "p1 f"], "players": ["a", "b"],
            "antes": [0.0, 0.0], "blinds_or_straddles": [3.0, 6.0], "starting_stacks": [600.0, 120.0],
            "finishing_stacks": [], "winnings": [0.0, 0.0], "seats": [5, 6],
            "source_file": "handhq/FTP-2009-07-01_2009-07-23_600NLH_OBFU/6/ftp NLH handhq_286-OBFUSCATED.phhs",
        }
        rec = M.row_to_record(row)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.meta["day"], 20)
        self.assertEqual(rec.seat_count, 6)
        self.assertEqual(rec.index, 971)
        self.assertIsNone(rec.finishing_stacks)
        self.assertEqual(M.timestamp_of(rec), "2009-07-20T08:27:34")
        self.assertEqual(M.stake_of(rec.source), "600NLH")

    def test_malformed_rows_are_rejected(self):
        self.assertIsNone(M.row_to_record({"variant": "FB", "actions": ["["], "antes": [], "starting_stacks": []}))
        ok, why = M.prefilter({"variant": "NT", "starting_stacks": [100.0, 100.0], "venue": "x", "hand": "1",
                               "actions": ["d dh p1 ??", "p2 f"]})
        self.assertEqual((ok, why), (False, "no_river"))
        ok, why = M.prefilter({"variant": "NT", "starting_stacks": [float("inf"), 100.0], "venue": "x", "hand": "1",
                               "actions": ["d db a", "d db b", "d db c"]})
        self.assertEqual((ok, why), (False, "stacks"))


@unittest.skipUnless(HAVE_SHARDS and HAVE_ARROW, "needs shards 35 and 36 in data/cache/takara")
class ReferenceCounts(unittest.TestCase):
    """The ten numbers from PLAN.md section 2 must reproduce exactly."""

    def test_reference_counts(self):
        for shard in (36, 35):
            scan = M.scan_shard(shard)
            cohorts = M.exhibit_counts(scan.spots)["cohorts"]
            for name, (folds, opps) in M.REFERENCE_COUNTS[shard].items():
                got = (cohorts[name]["folds"], cohorts[name]["opportunities"])
                self.assertEqual(got, (folds, opps), f"shard {shard} cohort {name}")

    def test_split_is_chronological_and_disjoint(self):
        spots = M.scan_shard(36).spots
        kept, stats = M.dedupe(spots)
        self.assertEqual(stats["kept"], len(kept))
        split = M.assign_split(kept)
        eligible = [s for s in kept if s.exact_half and s.all_three_legal and s.timestamp]
        n_train = sum(1 for v in split.values() if v == "train")
        self.assertEqual(n_train, int(0.8 * len(eligible)))
        self.assertEqual(len(split), len(eligible))
        by_venue = {}
        for s in eligible:
            by_venue.setdefault(s.venue, []).append(s)
        for venue, ss in by_venue.items():
            train_ts = [s.timestamp for s in ss if split[s.key] == "train"]
            test_ts = [s.timestamp for s in ss if split[s.key] == "test"]
            if train_ts and test_ts:
                self.assertLessEqual(max(train_ts), min(test_ts), venue)


class ExhibitShape(unittest.TestCase):
    def test_contexts_partition(self):
        base = dict(venue="v", hand_id="1", source_file="f", hand_number_in_file=1, timestamp="2009-07-01T00:00:00",
                    stake="100NLH", defender="d", bettor="b", board=["Ah", "Kd", "2c", "7s", "9h"], bet=5.0,
                    pot_before_bet=10.0, ratio=0.5, exact_half=True, response="fold", raise_to=None,
                    defender_stack=100.0, bettor_stack=100.0, all_three_legal=True, warning_free=True,
                    defender_cards_known=False, defender_cards=["??", "??"], bettor_cards=["??", "??"],
                    bettor_cards_known=False, fingerprint="x", min_bet=2.0, big_blind=2.0)
        spots = [M.RiverSpot(**dict(base, hand_id="1", context=CTX_CHECKED_THROUGH)),
                 M.RiverSpot(**dict(base, hand_id="2", context=CTX_BETTOR_AGGR, response="call"))]
        ex = M.exhibit_counts(spots)
        self.assertEqual(ex["cohorts"]["all_eligible"], {"folds": 1, "opportunities": 2})
        self.assertEqual(ex["cohorts"]["half_checked_through"], {"folds": 1, "opportunities": 1})
        self.assertEqual(ex["cohorts"]["half_bettor_aggr"], {"folds": 0, "opportunities": 1})


SIX_MAX_ROW = {
    "variant": "NT", "venue": "PokerStars", "table": "t", "time": "10:00:00", "hand": "424242",
    "day": 10.0, "month": 7.0, "year": 2009.0, "seat_count": 6.0, "hand_number_in_file": 1.0, "min_bet": 2.0,
    "small_bet": None, "big_bet": None, "bring_in": None, "ante_trimming_status": False,
    "actions": ["d dh p1 ????", "d dh p2 ????", "d dh p3 ????", "d dh p4 ????", "d dh p5 ????", "d dh p6 ????",
                "p3 f", "p4 cbr 6", "p5 f", "p6 cc", "p1 f", "p2 cc",
                "d db AhKd2c", "p2 cc", "p4 cbr 10", "p6 f", "p2 cc",
                "d db 7s", "p2 cc", "p4 cc",
                "d db 3h", "p2 cc", "p4 cbr 19.5", "p2 f"],
    "players": ["a", "b", "c", "d", "e", "f"], "antes": [0.0] * 6, "blinds_or_straddles": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    "starting_stacks": [200.0] * 6, "finishing_stacks": [], "winnings": [0.0] * 6, "seats": [1, 2, 3, 4, 5, 6],
    "source_file": "handhq/PS-2009-07-01_2009-07-23_200NLH_OBFU/2/x.phhs",
}


class WideExtraction(unittest.TestCase):
    """Phase 6: heads-up river spots inside bigger games."""

    def test_six_handed_hand_that_ends_heads_up(self):
        ok, why = M.prefilter_wide(SIX_MAX_ROW)
        self.assertTrue(ok, why)
        self.assertEqual(M.prefilter(SIX_MAX_ROW), (False, "players"))  # the original cohort skips it
        spot, why = M.extract_spot(M.row_to_record(SIX_MAX_ROW), wide=True)
        self.assertIsNotNone(spot, why)
        self.assertEqual((spot.table_size, spot.defender_index, spot.bettor_index), (6, 1, 3))
        self.assertEqual((spot.defender, spot.bettor), ("b", "d"))
        self.assertEqual(spot.context, CTX_CHECKED_THROUGH)
        self.assertEqual(spot.flop_context, CTX_BETTOR_AGGR)
        self.assertEqual(spot.preflop_aggressor, "bettor")
        self.assertEqual(spot.flop_players, 3)
        self.assertAlmostEqual(spot.pot_before_bet, 39.0)
        self.assertAlmostEqual(spot.ratio, 0.5)
        self.assertTrue(spot.exact_half and spot.in_band and spot.all_three_legal)
        self.assertEqual(spot.response, "fold")
        self.assertEqual(spot.timestamp, "2009-07-10T10:00:00")

    def test_three_players_at_the_river_are_not_heads_up(self):
        row = dict(SIX_MAX_ROW, actions=["d dh p1 ????", "d dh p2 ????", "d dh p3 ????", "p3 cc", "p1 cc", "p2 cc",
                                         "d db AhKd2c", "d db 7s", "d db 3h"], starting_stacks=[100.0] * 3,
                   players=["a", "b", "c"], blinds_or_straddles=[1.0, 2.0, 0.0], antes=[0.0] * 3)
        self.assertEqual(M.prefilter_wide(row), (False, "not_two_at_river"))

    def test_split_follows_the_main_cohort_cutoff(self):
        info = [("A", t, f"h{t:02d}", True) for t in range(1, 11)]
        info += [("A", 5, "y", False), ("A", 9, "x", False), ("A", None, "z", False)]
        info += [("B", t, f"b{t}", False) for t in range(1, 6)]
        split, cutoffs = M._wide_split(info)
        self.assertEqual(split[:10], [0] * 8 + [1] * 2)       # 80/20 of the main cohort
        self.assertEqual(split[10:13], [0, 1, None])          # others follow the cutoff; no timestamp, no split
        self.assertEqual(split[13:], [0, 0, 0, 0, 1])          # a venue without main spots gets its own 80%
        self.assertEqual(set(cutoffs), {"A", "B"})


@unittest.skipUnless(HAVE_SHARDS and HAVE_ARROW, "needs shard 36 in data/cache/takara")
class WideMatchesOriginal(unittest.TestCase):
    def test_two_player_hands_extract_identically(self):
        compared = 0
        for k, row in enumerate(M.iter_rows(M.shard_path(36))):
            if k >= 30000:
                break
            if not M.prefilter(row)[0]:
                continue
            self.assertTrue(M.prefilter_wide(row)[0])
            rec = M.row_to_record(row)
            if rec is None:
                continue
            a, why_a = M.extract_spot(rec)
            b, why_b = M.extract_spot(rec, wide=True)
            if a is None:
                if why_a == "ratio":  # out-of-band bets are kept by the wide scan unless a later check rejects them
                    self.assertTrue((b is not None and not b.in_band) or why_b in ("defender_short", "response", "warnings"))
                else:
                    self.assertEqual(why_a, why_b)
                continue
            compared += 1
            self.assertEqual(M.asdict(a), M.asdict(b))
        self.assertGreater(compared, 20)


if __name__ == "__main__":
    unittest.main()
