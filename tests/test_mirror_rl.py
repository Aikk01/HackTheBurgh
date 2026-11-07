import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import cards as C  # noqa: E402
from pokersim import evaluator as E  # noqa: E402
from pokersim import mirror_rl as RL  # noqa: E402
from pokersim.mirror_types import CTX_CHECKED_THROUGH  # noqa: E402
from pokersim.river_env import Scenario, ScriptedDefender  # noqa: E402

BOARD = ["Kc", "9d", "7h", "4s", "3c"]
BETTOR = ["6h", "2s"]


def one_pair_scenarios():
    """Every defender holding that makes exactly one pair on the oracle board."""
    board = [C.parse_card(c) for c in BOARD]
    used = set(board) | {C.parse_card(c) for c in BETTOR}
    out = []
    for h in itertools.combinations([c for c in C.full_deck() if c not in used], 2):
        if E.category_of(E.evaluate(list(h) + board)) == E.PAIR:
            out.append(Scenario.make(BOARD, [C.card_str(h[0]), C.card_str(h[1])], BETTOR, CTX_CHECKED_THROUGH))
    return out


class PureBluffOracle(unittest.TestCase):
    """Betting 50 into 100 with a hopeless hand is worth 1.5 p_fold - 0.5 (PLAN.md section 4)."""

    @classmethod
    def setUpClass(cls):
        cls.pool = one_pair_scenarios()

    def test_pool_is_a_pure_bluff(self):
        self.assertGreater(len(self.pool), 100)
        self.assertTrue(all(RL.showdown_value(s) == 0.0 for s in self.pool))
        self.assertTrue(all(s.bettor_bucket == "weak" for s in self.pool))

    def test_q_star_and_learning(self):
        for p_fold, target, expected_action in ((0.2, -0.20, "check_back"), (0.6, 0.40, "bet_half")):
            defender = ScriptedDefender(p_fold, 1 - p_fold, 0.0)
            oracle = RL.q_star(defender, self.pool)
            key = RL.state_key(1, CTX_CHECKED_THROUGH, "weak")
            self.assertAlmostEqual(oracle[key][0], 0.0)
            self.assertAlmostEqual(oracle[key][1], target, places=9)
            policy = RL.train(defender, self.pool, episodes=3000, seed=3, checkpoints=(3000,))
            learned = policy.q[key][1]
            self.assertLessEqual(abs(learned - target), 0.10, (p_fold, learned))
            self.assertEqual(policy.action_name(key, policy.greedy(key)), expected_action)
            cmp = RL.compare_with_oracle(policy, oracle)
            self.assertTrue(cmp["pass"], [c for c in cmp["cells"] if not c["ok"]])

    def test_raise_fixture_second_decision(self):
        raiser = ScriptedDefender(0.0, 0.0, 1.0)
        oracle = RL.q_star(raiser, self.pool)
        k1 = RL.state_key(1, CTX_CHECKED_THROUGH, "weak")
        k2 = RL.state_key(2, CTX_CHECKED_THROUGH, "weak")
        self.assertEqual(oracle[k2], [-0.5, -1.0])  # fold -0.5, call and lose -1.0
        self.assertAlmostEqual(oracle[k1][1], -0.5)   # bet, get raised, fold
        policy = RL.train(raiser, self.pool, episodes=1500, seed=5, checkpoints=(1500,))
        self.assertEqual(policy.action_name(k2, policy.greedy(k2)), "fold")
        self.assertEqual(policy.action_name(k1, policy.greedy(k1)), "check_back")


class Benchmark(unittest.TestCase):
    def test_matched_benchmark_identical_policies_differ_by_zero(self):
        pool = one_pair_scenarios()[:50]
        p = RL.Policy.blank()
        res = RL.matched_benchmark(p, p, ScriptedDefender(0.5, 0.4, 0.1), pool, pairs=100, seed=1, bootstrap=100)
        self.assertEqual(res["mean_diff_chips"], 0.0)
        self.assertEqual(res["fraction_changed"], 0.0)
        q = RL.Policy.blank()
        q.q[RL.state_key(1, CTX_CHECKED_THROUGH, "weak")] = [0.0, 1.0]  # always bet
        res = RL.matched_benchmark(p, q, ScriptedDefender(1.0, 0.0, 0.0), pool, pairs=100, seed=1, bootstrap=100)
        self.assertEqual(res["mean_diff_chips"], 100.0)
        self.assertEqual(res["fraction_changed"], 1.0)

    def test_matched_benchmark_zero_probes_identical(self):
        """With the baseline trained on the candidates' seed, 0 probes must give the very same bettor."""
        res = RL.run_benchmark(None, opponents=["caller"], probes=[0], pairs=40, episodes=300, seed=3, workers=1,
                               pool_sizes=(60, 30, 40))
        self.assertTrue(res["matched_seed"])
        self.assertTrue(res["zero_probe_identical"])
        self.assertEqual(res["rows"][0]["mean_diff_chips"], 0.0)
        self.assertIn("control", res)

    def test_policy_round_trip_and_version_check(self):
        p = RL.Policy.blank({"x": 1})
        path = os.path.join(os.path.dirname(__file__), "_tmp_policy.json")
        try:
            p.save(path)
            again = RL.Policy.load(path)
            self.assertEqual(again.q, p.q)
            self.assertEqual(again.meta["x"], 1)
            import json
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            d["env_version"] = "other"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f)
            with self.assertRaises(ValueError):
                RL.Policy.load(path)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
