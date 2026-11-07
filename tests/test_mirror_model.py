import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim import mirror_model as MM  # noqa: E402
from pokersim.mirror_types import BUCKETS, CONTEXTS, CTX_CHECKED_THROUGH, RESPONSES  # noqa: E402

COUNTS = {"checked_through": {"fold": 266, "call": 110, "raise": 18}, "bettor_aggr": {"fold": 52, "call": 42, "raise": 6},
          "defender_aggr": {"fold": 39, "call": 40, "raise": 6}}
POOLED = {"fold": 357, "call": 192, "raise": 30}


class Population(unittest.TestCase):
    def test_smoothing_and_fallback(self):
        pop = MM.PopulationModel(COUNTS, POOLED)
        q = pop.probs(CTX_CHECKED_THROUGH)
        self.assertAlmostEqual(sum(q.values()), 1.0)
        self.assertAlmostEqual(q["fold"], (266 + 1) / (394 + 3))
        empty = MM.PopulationModel({}, POOLED)
        self.assertEqual(empty.probs(CTX_CHECKED_THROUGH), empty.q_pooled)
        pred = pop.predict(CTX_CHECKED_THROUGH, legal={"fold": True, "call": True, "raise": False})
        self.assertEqual(pred.probs["raise"], 0.0)
        self.assertAlmostEqual(sum(pred.probs.values()), 1.0)
        self.assertLess(pred.fold_p05, pred.fold_mean)
        self.assertGreater(pred.fold_p95, pred.fold_mean)


class BetaFunctions(unittest.TestCase):
    def test_incomplete_beta_against_known_values(self):
        self.assertAlmostEqual(MM.betainc(1, 1, 0.3), 0.3, places=12)
        self.assertAlmostEqual(MM.betainc(2, 3, 0.5), 0.6875, places=10)  # 1 - (1-x)^3 (1+3x) at x=.5
        self.assertAlmostEqual(MM.beta_ppf(0.5, 2, 2), 0.5, places=8)
        self.assertAlmostEqual(MM.betainc(5, 7, MM.beta_ppf(0.9, 5, 7)), 0.9, places=8)


class Personal(unittest.TestCase):
    def setUp(self):
        self.pop = MM.PopulationModel(COUNTS, POOLED)

    def test_prior_then_updates(self):
        pm = MM.PersonalModel(self.pop, prior_strength=8)
        self.assertEqual(pm.probs(CTX_CHECKED_THROUGH), self.pop.probs(CTX_CHECKED_THROUGH))
        p0 = pm.predict(CTX_CHECKED_THROUGH)
        for _ in range(10):
            pm.observe(CTX_CHECKED_THROUGH, "call")
        p1 = pm.predict(CTX_CHECKED_THROUGH)
        self.assertLess(p1.fold_mean, p0.fold_mean)
        self.assertEqual(p1.observations, 10)
        self.assertLess(p1.fold_p95 - p1.fold_p05, p0.fold_p95 - p0.fold_p05)
        q = self.pop.probs(CTX_CHECKED_THROUGH)
        self.assertAlmostEqual(p1.probs["call"], (8 * q["call"] + 10) / (8 + 10))

    def test_k1_mixture_equals_plain_model(self):
        tp = MM.TypePrior(["all"], [1.0], [{c: self.pop.probs(c) for c in CONTEXTS}])
        plain = MM.PersonalModel(self.pop, prior_strength=8)
        mix = MM.PersonalModel(self.pop, prior_strength=8, type_prior=tp)
        for a in ("fold", "fold", "call", "raise"):
            plain.observe(CTX_CHECKED_THROUGH, a)
            mix.observe(CTX_CHECKED_THROUGH, a)
        for c in CONTEXTS:
            for a in RESPONSES:
                self.assertAlmostEqual(plain.probs(c)[a], mix.probs(c)[a], places=12)
        self.assertEqual(mix.type_posterior(), [1.0])

    def test_mixture_moves_towards_the_matching_type(self):
        tight = {c: {"fold": 0.8, "call": 0.15, "raise": 0.05} for c in CONTEXTS}
        station = {c: {"fold": 0.1, "call": 0.85, "raise": 0.05} for c in CONTEXTS}
        tp = MM.TypePrior(["tight", "station"], [0.5, 0.5], [tight, station])
        pm = MM.PersonalModel(self.pop, prior_strength=8, type_prior=tp)
        self.assertAlmostEqual(sum(pm.type_posterior()), 1.0)
        for _ in range(6):
            pm.observe(CTX_CHECKED_THROUGH, "call")
        post = pm.type_posterior()
        self.assertGreater(post[1], 0.9)
        pred = pm.predict(CTX_CHECKED_THROUGH)
        self.assertEqual(set(pred.type_posterior), {"tight", "station"})
        self.assertLess(pred.fold_mean, 0.3)

    def test_freeze_is_immutable_and_hashable(self):
        pm = MM.PersonalModel(self.pop, player_id="judge")
        pm.observe(CTX_CHECKED_THROUGH, "fold")
        frozen = pm.freeze()
        h = frozen.hash
        pm.observe(CTX_CHECKED_THROUGH, "call")
        pm.observe(CTX_CHECKED_THROUGH, "call")
        self.assertEqual(frozen.hash, h)
        self.assertEqual(frozen.observations, 1)
        again = MM.FrozenProfile.from_dict(frozen.to_dict())
        self.assertEqual(again.hash, h)


class Strength(unittest.TestCase):
    def test_default_prior_matches_population_marginal(self):
        pop = MM.PopulationModel(COUNTS, POOLED)
        st = MM.StrengthTable.default(pop)
        for c in CONTEXTS:
            marginal = {a: sum(MM.DEFAULT_BUCKET_SHARES[b] * st.prior[c][b][a] for b in BUCKETS) for a in RESPONSES}
            for a in RESPONSES:
                self.assertAlmostEqual(marginal[a], pop.probs(c)[a], places=4, msg=(c, a))
            self.assertGreater(st.prior[c]["weak"]["fold"], st.prior[c]["strong"]["fold"])
        st.observe(CTX_CHECKED_THROUGH, "strong", "raise")
        st.observe(CTX_CHECKED_THROUGH, "strong", "raise")
        self.assertGreater(st.probs(CTX_CHECKED_THROUGH, "strong")["raise"], st.prior[CTX_CHECKED_THROUGH]["strong"]["raise"])

    def test_twin_is_seeded_and_uses_strength(self):
        pop = MM.PopulationModel(COUNTS, POOLED)
        pm = MM.PersonalModel(pop, player_id="x")
        twin = MM.Twin(pm.freeze())
        a = [twin.respond(CTX_CHECKED_THROUGH, "weak", random.Random(3)) for _ in range(5)]
        b = [twin.respond(CTX_CHECKED_THROUGH, "weak", random.Random(3)) for _ in range(5)]
        self.assertEqual(a, b)
        rng = random.Random(7)
        folds_weak = sum(twin.respond(CTX_CHECKED_THROUGH, "weak", rng) == "fold" for _ in range(2000)) / 2000
        folds_strong = sum(twin.respond(CTX_CHECKED_THROUGH, "strong", rng) == "fold" for _ in range(2000)) / 2000
        self.assertGreater(folds_weak, folds_strong + 0.3)


if __name__ == "__main__":
    unittest.main()
