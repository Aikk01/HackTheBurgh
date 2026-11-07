import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim.cards import parse_cards  # noqa: E402
from pokersim.engine import GameConfig, Hand, IllegalAction, Seat, Table  # noqa: E402


def cards(s):
    return parse_cards(s)


class BlindsAndOrder(unittest.TestCase):
    def test_six_max_opener_is_utg(self):
        h = Hand(GameConfig("NT", antes=[0] * 6, blinds=[1, 2, 0, 0, 0, 0], min_bet=2), [100] * 6, rng=random.Random(1))
        h.start()
        self.assertEqual(h.actor, 2)
        self.assertEqual(h.players[0].bet, 1)
        self.assertEqual(h.players[1].bet, 2)
        self.assertEqual(h.positions, ["SB", "BB", "UTG", "HJ", "CO", "BTN"])

    def test_heads_up_blinds_are_reversed(self):
        h = Hand(GameConfig("NT", antes=[0, 0], blinds=[1, 2], min_bet=2), [100, 100], rng=random.Random(1))
        h.start()
        self.assertEqual(h.players[0].bet, 2)  # p1 is the big blind
        self.assertEqual(h.players[1].bet, 1)  # p2 (button) is the small blind
        self.assertEqual(h.actor, 1)           # button acts first pre-flop
        h.act(1, "call")
        h.act(0, "check")
        self.assertEqual(h.street, 1)
        self.assertEqual(h.actor, 0)           # big blind acts first post-flop

    def test_big_blind_has_the_option(self):
        h = Hand(GameConfig("NT", antes=[0] * 3, blinds=[1, 2, 0], min_bet=2), [100] * 3, rng=random.Random(1))
        h.start()
        h.act(2, "call")
        h.act(0, "call")
        self.assertEqual(h.actor, 1)
        self.assertTrue(h.legal_actions()["check"])
        self.assertTrue(h.legal_actions()["raise"])
        h.act(1, "raise", 8)
        self.assertEqual(h.actor, 2)


class BettingRules(unittest.TestCase):
    def make3(self, stacks=(100, 100, 100)):
        h = Hand(GameConfig("NT", antes=[0] * 3, blinds=[1, 2, 0], min_bet=2), list(stacks), rng=random.Random(3))
        h.start()
        return h

    def test_uncalled_bet_is_returned(self):
        h = self.make3()
        h.act(2, "raise", 10)
        h.act(0, "fold")
        h.act(1, "fold")
        self.assertEqual(h.status, "finished")
        self.assertEqual(h.players[2].stack, 103)   # 8 of the 10 came back uncalled, +3 net
        self.assertEqual(h.players[2].payout, 5)    # the collected pot: 1 + 2 + the matched 2
        self.assertEqual(h.pot, 0)
        self.assertEqual([e for e in h.events if e["type"] == "uncalled"][0]["amount"], 8)

    def test_min_raise_tracks_last_raise_size(self):
        h = self.make3()
        la = h.legal_actions(2)
        self.assertEqual(la["min_raise_to"], 4)
        h.act(2, "raise", 6)              # raise of 4 over the big blind
        self.assertEqual(h.legal_actions(0)["min_raise_to"], 10)
        with self.assertRaises(IllegalAction):
            h.bet_raise_to(0, 8)
        h.act(0, "raise", 20)             # raise of 14
        self.assertEqual(h.legal_actions(1)["min_raise_to"], 34)

    def test_cannot_fold_when_check_is_free_in_live_mode(self):
        h = self.make3()
        h.act(2, "call")
        h.act(0, "call")
        with self.assertRaises(IllegalAction):
            h.fold(1)
        res = h.act(1, "fold")            # the tolerant API converts it to a check
        self.assertEqual(res["action"], "check")
        self.assertEqual(h.street, 1)

    def test_incomplete_all_in_raise_does_not_reopen_betting(self):
        h = self.make3(stacks=(100, 100, 7))
        h.act(2, "call")                  # button calls 2
        h.act(0, "raise", 6)              # small blind raises to 6 (raise of 4)
        h.act(1, "call")
        h.act(2, "all_in")                # 7 total: a raise of only 1, not a full raise
        self.assertEqual(h.actor, 0)
        self.assertFalse(h.legal_actions(0)["raise"])   # already acted -> may only call or fold
        self.assertFalse(h.legal_actions(1)["raise"])
        h.act(0, "call")
        h.act(1, "call")
        self.assertEqual(h.street, 1)

    def test_raise_sizes_are_clamped_and_reported(self):
        h = self.make3()
        res = h.act(2, "raise", 1000)
        self.assertEqual(res["amount"], 100)
        self.assertIn("clamped", res["note"])
        self.assertTrue(h.players[2].all_in)


class SidePots(unittest.TestCase):
    def test_three_way_all_in_with_side_pot(self):
        cfg = GameConfig("NT", antes=[0, 0, 0], blinds=[1, 2, 0], min_bet=2)
        h = Hand(cfg, [10, 50, 100], scripted=True)
        h.post_forced_bets()
        h.deal_hole(0, cards("AhAd"))
        h.deal_hole(1, cards("KhKd"))
        h.deal_hole(2, cards("2h2d"))
        h.bet_raise_to(2, 100)
        h.check_call(0)                   # all-in for 10
        h.check_call(1)                   # all-in for 50
        self.assertEqual(h.status, "awaiting_board")
        self.assertEqual(h.players[2].stack, 50)  # 50 uncalled, returned
        h.deal_board(cards("3c7c9s"))
        h.deal_board(cards("Jd"))
        h.deal_board(cards("Qs"))
        h.finalize()
        self.assertEqual(h.status, "finished")
        self.assertEqual([p.stack for p in h.players], [30, 80, 50])
        self.assertEqual(len(h.pots), 2)
        self.assertEqual(h.pots[0]["amount"], 30)
        self.assertEqual(h.pots[1]["amount"], 80)

    def test_split_pot_exact_and_odd_chip(self):
        for mode, expected in (("exact", [101.5, 101.5, 97]), ("odd_chip", [102, 101, 97])):
            cfg = GameConfig("NT", antes=[0, 0, 0], blinds=[1, 2, 0], min_bet=2)
            h = Hand(cfg, [100, 100, 100], scripted=True)
            h.split_mode = mode
            h.post_forced_bets()
            h.deal_hole(0, cards("AhKd"))
            h.deal_hole(1, cards("AdKh"))
            h.deal_hole(2, cards("3h4d"))
            h.bet_raise_to(2, 3)
            h.check_call(0)
            h.check_call(1)
            for b in ("5c7c9s", "Jd", "Qs"):
                h.deal_board(cards(b))
                for i in (0, 1, 2):
                    if h.status == "betting":
                        h.check_call(i)
            h.finalize()
            self.assertEqual([p.stack for p in h.players], expected, mode)

    def test_big_blind_ante_belongs_to_the_main_pot(self):
        cfg = GameConfig("NT", antes=[0, 120, 0], blinds=[40, 80, 0], min_bet=80)
        h = Hand(cfg, [1000, 1000, 1000], scripted=True)
        h.post_forced_bets()
        h.deal_hole(0, cards("7s4s"))
        h.deal_hole(1, cards("Js8h"))
        h.deal_hole(2, cards("AcKc"))
        h.bet_raise_to(2, 200)
        h.fold(0)
        h.check_call(1)
        h.deal_board(cards("2c8cTh"))
        h.check_call(1)
        h.check_call(2)
        h.deal_board(cards("Ah"))
        h.check_call(1)
        h.check_call(2)
        h.deal_board(cards("6d"))
        h.check_call(1)
        h.check_call(2)
        h.finalize()
        # p3 wins everything: both bets of 200, the folded small blind and the big blind ante
        self.assertEqual([p.stack for p in h.players], [960, 680, 1360])


class PotLimit(unittest.TestCase):
    def test_pot_size_raise_heads_up(self):
        h = Hand(GameConfig("PO", antes=[0, 0], blinds=[1, 2], min_bet=2), [100, 100], rng=random.Random(2))
        h.start()
        la = h.legal_actions(1)           # small blind / button
        self.assertEqual(la["min_raise_to"], 4)
        self.assertEqual(la["max_raise_to"], 6)


class TableRotation(unittest.TestCase):
    def test_button_moves_and_stacks_persist(self):
        seats = [Seat(1, "a", 100), Seat(2, "b", 100), Seat(3, "c", 100)]
        t = Table(seats, small_blind=1, big_blind=2, rng=random.Random(5))
        h = t.new_hand()
        self.assertEqual(t.button_seat, 1)
        self.assertEqual([s.seat_no for s in t.current_order], [2, 3, 1])
        # everybody folds to the big blind
        h.act(2, "fold")
        h.act(0, "fold")
        t.finish_hand()
        self.assertEqual(t.button_seat, 2)
        self.assertEqual(sum(s.stack for s in seats), 300)
        h2 = t.new_hand()
        self.assertEqual([s.seat_no for s in t.current_order], [3, 1, 2])
        self.assertEqual(h2.players[2].name, "b")


if __name__ == "__main__":
    unittest.main()
