import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # silence joblib's wmic probe on Windows

try:
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

from pokersim import mirror_styles as ST  # noqa: E402


def synthetic_rows(n=600, seed=3):
    """Two clear player types: loose-aggressive and tight-passive."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        loose = i % 2 == 0
        hands = rng.randint(80, 400)
        rates = {"vpip": 0.55 if loose else 0.18, "pfr": 0.30 if loose else 0.08, "af": 0.6 if loose else 0.25,
                 "fold_to_bet": 0.25 if loose else 0.6, "wtsd": 0.4 if loose else 0.25}
        r = {"venue": "v", "handle": f"p{i}", "hands": hands}
        for f, p in rates.items():
            den = hands if f in ("vpip", "pfr") else max(5, hands // 3)
            num = sum(1 for _ in range(den) if rng.random() < p)
            r[f + "_num"], r[f + "_den"] = num, den
        rows.append(r)
    return rows


@unittest.skipUnless(HAVE_SK, "needs numpy + scikit-learn")
class Types(unittest.TestCase):
    def test_reliability_and_two_types(self):
        rows = ST._rates(synthetic_rows())
        rel = ST.choose_n_min(rows)
        self.assertLessEqual(rel["n_min"], 100)
        self.assertEqual(rel["dropped"], [])
        kept, X, scaler = ST.feature_matrix(rows, rel["features"], rel["n_min"])
        self.assertEqual(X.shape[1], len(rel["features"]))
        fit = ST.fit_types(X, k_range=range(2, 5), bootstrap=5, seed=1)
        self.assertEqual(fit["chosen"]["k"], 2)
        self.assertTrue(fit["stable"])
        labels = fit["chosen"]["model"].predict(X)
        # the two synthetic groups should be separated almost perfectly
        agree = sum(1 for i, r in enumerate(kept) if (int(r["handle"][1:]) % 2 == 0) == (labels[i] == labels[0]))
        self.assertGreater(max(agree, len(kept) - agree) / len(kept), 0.95)
        ref = {f: 0.3 for f in rel["features"]}
        self.assertEqual(ST.name_type({"vpip": 0.55, "pfr": 0.3, "af": 0.6}, ref), "loose-aggressive")
        self.assertEqual(ST.name_type({"vpip": 0.18, "pfr": 0.08, "af": 0.25}, ref), "tight-passive")


if __name__ == "__main__":
    unittest.main()
