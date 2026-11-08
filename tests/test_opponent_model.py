"""Opponent style estimates and exploitative advice for the Play table (pokersim.opponent_model)."""
import unittest

from pokersim.opponent_model import POP_MEAN, StyleTracker, advise, estimate_style


def counts(vpip, pfr, af, ftb, wtsd, n):
    return {"vpip": [round(vpip * n), n], "pfr": [round(pfr * n), n], "af": [round(af * n), n],
            "fold_to_bet": [round(ftb * n), n], "wtsd": [round(wtsd * n), n]}


def obs(cards, board, street, to_call, pot, n_opp=1, history=None):
    players = [{"index": 0, "name": "You", "position": "BTN", "stack": 1000.0, "bet": 0.0, "folded": False, "is_hero": True}]
    players += [{"index": i, "name": f"P{i}", "position": "BB", "stack": 1000.0, "bet": to_call, "folded": False, "is_hero": False}
                for i in range(1, n_opp + 1)]
    return {
        "hand_id": 7, "variant": "NT", "street": street, "board": board, "total_pot": pot, "min_bet": 2.0, "chip_unit": 1.0,
        "hero": {"index": 0, "name": "You", "position": "BTN", "cards": cards, "stack": 1000.0, "bet": 0.0, "contributed": 0.0},
        "players": players, "history": history or [],
        "legal_actions": {"fold": to_call > 0, "check": to_call == 0, "call": to_call, "raise": True,
                          "min_raise_to": max(4.0, 2 * to_call), "max_raise_to": 1000.0, "raise_kind": "raise" if to_call > 0 else "bet"},
    }


class Styles(unittest.TestCase):
    def test_no_data_means_population(self):
        e = estimate_style({}, 0)
        for f, m in POP_MEAN.items():
            self.assertAlmostEqual(e["rates"][f], m, places=4)
        self.assertEqual(e["hands"], 0)

    def test_types_follow_the_counts(self):
        tight = estimate_style(counts(0.12, 0.10, 0.35, 0.75, 0.25, 200), 200)
        loose_passive = estimate_style(counts(0.60, 0.05, 0.15, 0.40, 0.40, 200), 200)
        self.assertTrue(tight["type"].startswith("tight"))
        self.assertEqual(loose_passive["type"], "loose-passive")
        self.assertGreater(loose_passive["type_prob"], 0.9)

    def test_tracker_counts_real_hands(self):
        from pokersim import phh
        from pokersim.replay import replay
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "pluribus-session-30.phhs")
        t = StyleTracker()
        for rec in phh.load_file(path)[:40]:
            t.add_hand(replay(rec, snapshots=False, observations=False).hand)
        self.assertTrue(t.hands)
        name = max(t.hands, key=t.hands.get)
        self.assertEqual(t.counts[name]["vpip"][1], t.hands[name])


class Advice(unittest.TestCase):
    def test_nuts_bets_air_folds(self):
        a = advise(obs(["As", "Ks"], ["Qs", "Js", "Ts", "2d", "3c"], "river", 50.0, 150.0), {})
        self.assertIn(a["recommend"]["action"], ("raise", "all_in"))
        b = advise(obs(["7c", "2d"], ["As", "Kh", "Qd", "Jc", "9s"], "river", 200.0, 300.0), {})
        self.assertEqual(b["recommend"]["action"], "fold")

    def test_folding_opponents_make_bluffs_pay(self):
        spot = obs(["7c", "2d"], ["As", "Kh", "9d"], "flop", 0.0, 40.0)
        folder = estimate_style(counts(0.15, 0.05, 0.10, 0.95, 0.20, 300), 300)
        caller = estimate_style(counts(0.60, 0.05, 0.10, 0.10, 0.50, 300), 300)
        bet_ev = lambda a: max(o["ev"] for o in a["options"] if o["action"] in ("bet", "raise", "all_in"))
        self.assertGreater(bet_ev(advise(spot, {1: folder})), bet_ev(advise(spot, {1: caller})))
        self.assertIn(advise(spot, {1: folder})["recommend"]["action"], ("bet", "all_in"))
        self.assertEqual(advise(spot, {1: caller})["recommend"]["action"], "check")

    def test_tight_raiser_lowers_our_equity(self):
        hist = [{"street": "preflop", "player": 1, "action": "raise", "amount": 6, "to": 6}]
        spot = obs(["Qc", "Jd"], [], "preflop", 6.0, 9.0, history=hist)
        tight = estimate_style(counts(0.10, 0.06, 0.3, 0.7, 0.25, 300), 300)
        a = advise(spot, {1: tight})
        self.assertLess(a["equity"], a["equity_vs_random"] - 0.05)


if __name__ == "__main__":
    unittest.main()
