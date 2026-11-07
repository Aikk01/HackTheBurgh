"""RL advisor: feature encoding, action mapping, NumPy policy and its match with the PyTorch network."""
import random
import unittest

import numpy as np

from pokersim.engine import IllegalAction
from pokersim.rl_features import N_ACTIONS, OBS_DIM, action_targets, encode
from pokersim.rl_policy import HIDDEN, NumpyPolicy, recommend
from pokersim.rl_train import Session, play_hand


def random_params(seed=0):
    rng = np.random.default_rng(seed)
    dims = [OBS_DIM] + list(HIDDEN)
    p = {}
    for i in range(len(HIDDEN)):
        p[f"trunk.{i}.w"] = rng.normal(0, 0.05, (dims[i + 1], dims[i]))
        p[f"trunk.{i}.b"] = np.zeros(dims[i + 1])
        p[f"trunk.{i}.ln_w"] = np.ones(dims[i + 1])
        p[f"trunk.{i}.ln_b"] = np.zeros(dims[i + 1])
    p["pi.w"] = rng.normal(0, 0.05, (N_ACTIONS, HIDDEN[-1]))
    p["pi.b"] = np.zeros(N_ACTIONS)
    p["v.w"] = rng.normal(0, 0.05, (1, HIDDEN[-1]))
    p["v.b"] = np.zeros(1)
    return p


class Features(unittest.TestCase):
    def test_every_masked_action_is_legal_in_the_engine(self):
        rng = random.Random(3)
        s = Session(rng, 0.0, [], hands=60)
        tried = [0]

        def hero(obs, styles):
            x = encode(obs, styles, rng=rng)
            self.assertEqual(x.shape, (OBS_DIM,))
            self.assertTrue(np.isfinite(x).all())
            mask, targets = action_targets(obs)
            self.assertTrue(mask[1])  # check/call is always possible
            for i in np.flatnonzero(mask):
                self.assertIsNotNone(targets[i])
            i = rng.choice(list(np.flatnonzero(mask)))
            tried[0] += 1
            return targets[i], None

        for _ in range(60):
            play_hand(s, hero, {}, rng)  # raises if the engine rejects a masked action
        self.assertGreater(tried[0], 20)
        self.assertEqual(s.tracker.hands["p2"], 60)


class Policy(unittest.TestCase):
    def test_numpy_policy_and_recommendation(self):
        pol = NumpyPolicy(random_params())
        rng = random.Random(5)
        s = Session(rng, 0.0, [], hands=5)
        seen = []

        def hero(obs, styles):
            r = recommend(obs, styles, policy=pol)
            self.assertTrue(r["available"])
            self.assertAlmostEqual(sum(p["p"] for p in r["probs"]), 1.0, places=3)
            seen.append(r["action"])
            return r["engine_action"], None

        for _ in range(5):
            play_hand(s, hero, {}, rng)
        self.assertTrue(seen)

    def test_torch_export_matches_numpy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from pokersim.rl_train import build_net, export_params
        net = build_net(torch)
        x = np.random.default_rng(1).normal(size=(4, OBS_DIM)).astype(np.float32)
        with torch.no_grad():
            lt, vt = net(torch.as_tensor(x))
        ln, vn = NumpyPolicy(export_params(net)).forward(x)
        np.testing.assert_allclose(lt.numpy(), ln, atol=1e-4)
        np.testing.assert_allclose(vt.numpy(), vn, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
