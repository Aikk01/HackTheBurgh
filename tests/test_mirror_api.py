"""Read-only Mirror API payloads built from the committed artifacts in data/mirror/."""
import os
import unittest

from pokersim import mirror_api
from pokersim.mirror_session import MIRROR_DIR


@unittest.skipUnless(os.path.exists(os.path.join(MIRROR_DIR, "styles.json")), "styles.json not generated")
class StyleMap(unittest.TestCase):
    def test_style_map_payload(self):
        s = mirror_api.get("/api/mirror/styles", {})
        self.assertTrue(s["present"])
        k = s["k"]
        self.assertEqual(len(s["names"]), k)
        self.assertEqual(len(s["centroid_xy"]), k)
        self.assertAlmostEqual(sum(s["shares"]), 1.0, places=2)
        self.assertTrue(s["map_points"])
        self.assertTrue(all(0 <= p["type"] < k for p in s["map_points"]))
        self.assertIn("pass", s["gate"])

    def test_evidence_keeps_map_points_out_of_type_gate(self):
        ev = mirror_api.get("/api/mirror/evidence", {})
        for name in ("filter_exhibit", "population", "forecast_ladder", "type_gate", "decision_benchmark", "range_checks", "population_bot"):
            self.assertIn(name, ev["panels"])
        self.assertNotIn("map_points", ev["panels"]["type_gate"]["data"] or {})


if __name__ == "__main__":
    unittest.main()
