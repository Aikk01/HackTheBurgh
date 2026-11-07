import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import phh  # noqa: E402
from pokersim.engine import UnsupportedVariant  # noqa: E402
from pokersim.replay import replay  # noqa: E402
from pokersim.simulate import run_match  # noqa: E402

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


class ParsePHH(unittest.TestCase):
    def test_single_hand_fields(self):
        rec = phh.load_file(os.path.join(SAMPLES, "dwan-ivey-2009.phh"))[0]
        self.assertEqual(rec.variant, "NT")
        self.assertEqual(rec.antes, [500, 500, 500])
        self.assertEqual(rec.blinds, [1000, 2000, 0])
        self.assertEqual(rec.players, ["Phil Ivey", "Patrik Antonius", "Tom Dwan"])
        self.assertEqual(rec.meta["year"], 2009)
        self.assertEqual(rec.actions[0], "d dh p1 Ac2d")

    def test_action_grammar(self):
        a = phh.parse_action("p3 cbr 47.50  # a comment")
        self.assertEqual((a.actor, a.player, a.verb, a.args, a.commentary), ("p3", 2, "cbr", ["47.50"], "a comment"))
        d = phh.parse_action("d dh p2 ????")
        self.assertTrue(d.is_dealer)
        self.assertEqual(d.args, ["p2", "????"])
        self.assertIsNone(phh.parse_action("   # only commentary"))

    def test_multi_hand_file_and_round_trip(self):
        recs = phh.load_file(os.path.join(SAMPLES, "pluribus-session-30.phhs"))
        self.assertEqual(len(recs), 80)
        text = phh.dump_hands(recs[:5])
        again = phh.load_text(text)
        self.assertEqual(len(again), 5)
        for a, b in zip(recs[:5], again):
            self.assertEqual(a.actions, b.actions)
            self.assertEqual(a.starting_stacks, b.starting_stacks)
            self.assertEqual(a.finishing_stacks, b.finishing_stacks)
            self.assertEqual(a.players, b.players)

    def test_lenient_parser_handles_odd_values(self):
        text = "variant = 'NT'\nantes = [0, 0]\nblinds_or_straddles = [1, 2]\nmin_bet = 2\n" \
               "starting_stacks = [inf, 100]\nactions = ['d dh p1 ????', 'd dh p2 ????', 'p2 f']\ntime = 00:00:26\n"
        rec = phh.parse_hand_text(text)
        self.assertEqual(rec.starting_stacks[0], float("inf"))
        d = phh._lenient_parse(text)
        self.assertEqual(d["time"], "00:00:26")
        self.assertEqual(d["starting_stacks"][0], float("inf"))


class ReplaySamples(unittest.TestCase):
    def test_pluribus_finishing_stacks_match_exactly(self):
        recs = phh.load_file(os.path.join(SAMPLES, "pluribus-session-30.phhs"))
        for rec in recs:
            res = replay(rec, snapshots=False)
            self.assertTrue(res.complete, rec.label)
            self.assertTrue(res.stacks_match, (rec.label, res.stack_diffs, res.warnings))

    def test_wsop_holdem_hands_match(self):
        recs = phh.load_file(os.path.join(SAMPLES, "wsop-2023-ppc-final-table-holdem.phhs"))
        self.assertEqual(len(recs), 25)
        for rec in recs:
            res = replay(rec, snapshots=False)
            self.assertTrue(res.stacks_match, (rec.label, res.stack_diffs, res.warnings))

    def test_historic_hands_complete(self):
        for name in ("dwan-ivey-2009.phh", "antonius-blom-2009.phh", "phua-xuan-2019.phh"):
            rec = phh.load_file(os.path.join(SAMPLES, name))[0]
            res = replay(rec)
            self.assertTrue(res.complete, name)
            self.assertEqual(res.warnings, [], name)
            # snapshots cover setup + every action + payout
            self.assertEqual(len(res.steps), len(rec.parsed_actions()) + 2)
        # Dwan (7h6h, 3-4-5-6-7 straight) beats Ivey (Ac2d, wheel).  Ivey's uncalled 572,100
        # comes back to him, Antonius loses only his ante and big blind.
        dwan = replay(phh.load_file(os.path.join(SAMPLES, "dwan-ivey-2009.phh"))[0])
        self.assertEqual([p.stack for p in dwan.hand.players], [572100, 1997500, 1109500])
        self.assertEqual(dwan.hand.players[2].payout, 1109500)
        self.assertEqual(dwan.hand.players[0].payout, 0)

    def test_cash_game_samples_replay(self):
        for name in ("handhq-pokerstars-200nl-sample.phhs", "handhq-ipoker-100nl-hero-cards.phhs"):
            recs = phh.load_file(os.path.join(SAMPLES, name))
            finished = 0
            for rec in recs:
                res = replay(rec, snapshots=False)
                if res.hand.status == "incomplete":
                    # some iPoker logs stop right after the river card: the record carries no
                    # winnings either, so "incomplete" is the right verdict
                    self.assertFalse(any(rec.winnings or []), (name, rec.label))
                    continue
                self.assertEqual(res.hand.status, "finished", (name, rec.label, res.warnings))
                finished += 1
                if rec.winnings and sum(rec.winnings) > 0 and not res.hand.unresolved:
                    hinted = {i for i, w in enumerate(rec.winnings) if w > 0}
                    ours = {p.index for p in res.hand.players if p.payout > 0}
                    self.assertTrue(hinted & ours, (name, rec.label))
            self.assertGreater(finished, len(recs) * 0.6)

    def test_unsupported_variant_is_reported(self):
        rec = phh.parse_hand_text("variant = 'FB'\nantes = [0, 0]\nblinds_or_straddles = [1, 2]\nsmall_bet = 2\n"
                                  "big_bet = 4\nstarting_stacks = [100, 100]\nactions = ['d dh p1 ????????']\n")
        with self.assertRaises(UnsupportedVariant):
            replay(rec)

    def test_decision_points_carry_observations(self):
        rec = phh.load_file(os.path.join(SAMPLES, "pluribus-session-30.phhs"))[0]
        res = replay(rec)
        self.assertTrue(res.decisions)
        d = res.decisions[0]
        self.assertEqual(d.street, "preflop")
        self.assertIn("legal_actions", d.observation)
        self.assertEqual(d.observation["hero"]["index"], d.player)
        self.assertIn(d.actual["action"], ("fold", "check", "call", "bet", "raise"))


class SelfPlayRoundTrip(unittest.TestCase):
    def test_exported_hands_replay_to_the_same_stacks(self):
        res = run_match(["equity", "tag", "random", "station"], hands=40, stack=200, small_blind=1, big_blind=2, seed=11)
        self.assertEqual(res.hands, 40)
        recs = phh.load_text(phh.dump_hands(res.records))
        self.assertEqual(len(recs), 40)
        for rec in recs:
            r = replay(rec, snapshots=False)
            self.assertTrue(r.stacks_match, (rec.actions, r.stack_diffs, r.warnings))
            self.assertEqual(r.warnings, [])


if __name__ == "__main__":
    unittest.main()
