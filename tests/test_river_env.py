import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import river_env as R  # noqa: E402
from pokersim.mirror_types import CTX_BETTOR_AGGR, CTX_CHECKED_THROUGH, CTX_DEFENDER_AGGR  # noqa: E402

BOARD = ["Kc", "9d", "7h", "4s", "3c"]
DEF = ["Kd", "8d"]
BET = ["6h", "2s"]
EXPECTED_DELTA = {"check_back": 0, "bet_fold": 100, "bet_call": -50, "bet_raise_fold": -50, "bet_raise_call": -100}


class ParityFixtures(unittest.TestCase):
    def test_plan_fixture_on_both_engines(self):
        for ctx in (CTX_CHECKED_THROUGH, CTX_BETTOR_AGGR, CTX_DEFENDER_AGGR):
            st = R.build_state(BOARD, DEF, BET, ctx)
            self.assertEqual(list(st.stacks), [150, 150], ctx)
            self.assertEqual(st.total_pot_amount, 100, ctx)
            self.assertEqual(st.actor_index, 1, ctx)
            hand = R.build_pokersim_hand(BOARD, DEF, BET, ctx)
            self.assertEqual([p.stack for p in hand.players], [150, 150], ctx)
            self.assertEqual(hand.total_pot(), 100, ctx)
            self.assertEqual(hand.actor, 1, ctx)
            for name, path in R.PATHS.items():
                kit = R.apply_path_pokerkit(R.build_state(BOARD, DEF, BET, ctx), path)
                sim = R.apply_path_pokersim(R.build_pokersim_hand(BOARD, DEF, BET, ctx), path)
                self.assertEqual(kit, [int(x) for x in sim], (ctx, name))
                self.assertEqual(kit[1] - 150, EXPECTED_DELTA[name], (ctx, name))
                self.assertEqual(sum(kit), 400, (ctx, name))

    def test_winning_hand_and_split_pot(self):
        # bettor holds the nuts: bet/call wins 150, raise/call wins 200
        kit = R.apply_path_pokerkit(R.build_state(BOARD, DEF, ["Kh", "Ks"], CTX_CHECKED_THROUGH), R.PATHS["bet_call"])
        sim = R.apply_path_pokersim(R.build_pokersim_hand(BOARD, DEF, ["Kh", "Ks"], CTX_CHECKED_THROUGH), R.PATHS["bet_call"])
        self.assertEqual(kit, [100, 300])
        self.assertEqual([int(x) for x in sim], kit)
        kit = R.apply_path_pokerkit(R.build_state(BOARD, DEF, ["Kh", "Ks"], CTX_CHECKED_THROUGH), R.PATHS["bet_raise_call"])
        self.assertEqual(kit, [50, 350])
        # split pot: identical straights on the board
        board = ["Th", "Jd", "Qs", "Kc", "Ah"]
        for path in ("check_back", "bet_call"):
            kit = R.apply_path_pokerkit(R.build_state(board, ["2c", "3d"], ["4h", "5s"], CTX_CHECKED_THROUGH), R.PATHS[path])
            sim = R.apply_path_pokersim(R.build_pokersim_hand(board, ["2c", "3d"], ["4h", "5s"], CTX_CHECKED_THROUGH), R.PATHS[path])
            self.assertEqual(kit, [200, 200], path)
            self.assertEqual([int(x) for x in sim], kit, path)


class Environment(unittest.TestCase):
    def scenario(self, ctx=CTX_CHECKED_THROUGH):
        return R.Scenario.make(BOARD, DEF, BET, ctx)

    def test_buckets_and_equity(self):
        sc = self.scenario()
        self.assertEqual(sc.bettor_bucket, "weak")
        self.assertLess(sc.bettor_equity, 0.05)
        self.assertEqual(sc.defender_bucket, "strong")

    def test_rewards_follow_the_payoff_table(self):
        sc = self.scenario()
        folder = R.ScriptedDefender(1, 0, 0)
        env = R.RiverResponseEnv(folder, [sc], seed=1)
        env.reset(options={"scenario": sc})
        _, reward, done, _, info = env.step(1)
        self.assertTrue(done)
        self.assertEqual(reward, 1.0)
        caller = R.ScriptedDefender(0, 1, 0)
        env = R.RiverResponseEnv(caller, [sc], seed=1)
        env.reset(options={"scenario": sc})
        _, reward, done, _, info = env.step(1)
        self.assertEqual(reward, -0.5)
        raiser = R.ScriptedDefender(0, 0, 1)
        env = R.RiverResponseEnv(raiser, [sc], seed=1)
        obs, _ = env.reset(options={"scenario": sc})
        obs, reward, done, _, info = env.step(1)
        self.assertFalse(done)
        self.assertEqual(obs["stage"], 2)
        self.assertEqual(obs["to_call"], 50)
        _, reward, done, _, info = env.step(0)
        self.assertTrue(done)
        self.assertEqual(reward, -0.5)
        env.reset(options={"scenario": sc})
        env.step(1)
        _, reward, done, _, info = env.step(1)
        self.assertEqual(reward, -1.0)
        env.reset(options={"scenario": sc})
        _, reward, done, _, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertEqual(sum(info["final_stacks"]), 400)

    def test_seeded_runs_are_reproducible(self):
        pool = R.generate_scenarios(20, seed=5)
        self.assertEqual(len({s.board[0] for s in pool}) > 1, True)
        d = R.ScriptedDefender(0.5, 0.4, 0.1)
        runs = []
        for _ in range(2):
            env = R.RiverResponseEnv(d, pool, seed=11, opponent_seed=12)
            trace = []
            for k in range(30):
                env.reset()
                _, r, done, _, info = env.step(1)
                if not done:
                    _, r, done, _, info = env.step(1)
                trace.append((info["scenario"]["board"][0], info["response"], r))
            runs.append(trace)
        self.assertEqual(runs[0], runs[1])

    def test_state_key_and_versions(self):
        sc = self.scenario(CTX_BETTOR_AGGR)
        env = R.RiverResponseEnv(R.ScriptedDefender(1, 0, 0), [sc], seed=0)
        env.reset(options={"scenario": sc})
        self.assertEqual(env.state_key(), (1, CTX_BETTOR_AGGR, "weak"))
        self.assertEqual(R.ENV_VERSION, "river-v1")


if __name__ == "__main__":
    unittest.main()
