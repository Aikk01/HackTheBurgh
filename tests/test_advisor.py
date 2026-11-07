"""Live fold / call / raise advice for the human seat (pokersim.advisor)."""
import unittest

from pokersim.advisor import advise


def obs(cards, board, street, to_call, pot, raise_ok=True, n_opp=1, hero_bet=0.0, stack=1000.0):
    players = [{"index": 0, "name": "You", "position": "BTN", "stack": stack, "bet": hero_bet, "folded": False, "is_hero": True}]
    players += [{"index": i, "name": f"P{i}", "position": "BB", "stack": 1000.0, "bet": 0.0, "folded": False, "is_hero": False}
                for i in range(1, n_opp + 1)]
    return {
        "hand_id": 1, "variant": "NT", "street": street, "board": board, "total_pot": pot, "min_bet": 2.0, "chip_unit": 1.0,
        "hero": {"index": 0, "name": "You", "position": "BTN", "cards": cards, "stack": stack, "bet": hero_bet, "contributed": 0.0},
        "players": players, "history": [],
        "legal_actions": {"fold": to_call > 0, "check": to_call == 0, "call": to_call, "raise": raise_ok,
                          "min_raise_to": max(4.0, 2 * to_call), "max_raise_to": stack, "raise_kind": "raise" if to_call > 0 else "bet"},
    }


class Advisor(unittest.TestCase):
    def test_nuts_on_the_river_raises(self):
        a = advise(obs(["As", "Ks"], ["Qs", "Js", "Ts", "2d", "3c"], "river", 50.0, 150.0))
        self.assertTrue(a["available"])
        self.assertIn(a["recommend"]["action"], ("raise", "all_in"))
        self.assertGreater(a["equity"], 0.95)
        self.assertIn("勝率", a["reason_ja"])

    def test_air_facing_a_big_bet_folds(self):
        a = advise(obs(["7c", "2d"], ["As", "Kh", "Qd", "Jc", "9s"], "river", 200.0, 300.0))
        self.assertEqual(a["recommend"]["action"], "fold")
        self.assertLess(a["call_ev"], 0)

    def test_good_price_with_a_draw_calls(self):
        # flush draw (9 outs, about 35% to the river) getting 10-to-1
        a = advise(obs(["Ah", "5h"], ["Kh", "9h", "2c"], "flop", 10.0, 100.0, raise_ok=False))
        self.assertEqual(a["recommend"]["action"], "call")
        self.assertGreater(a["equity"], a["pot_odds"])

    def test_no_bet_to_call_never_folds(self):
        a = advise(obs(["7c", "2d"], ["As", "Kh", "Qd"], "flop", 0.0, 30.0, n_opp=5))
        self.assertEqual(a["recommend"]["action"], "check")

    def test_same_spot_same_numbers(self):
        o = obs(["Td", "9d"], ["8d", "7s", "2c"], "flop", 20.0, 60.0)
        self.assertEqual(advise(o)["equity"], advise(o)["equity"])

    def test_unknown_cards_unavailable(self):
        self.assertFalse(advise(obs(["??", "??"], [], "preflop", 2.0, 3.0))["available"])


if __name__ == "__main__":
    unittest.main()
