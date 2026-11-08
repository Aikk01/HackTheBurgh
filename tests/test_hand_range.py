import random
import unittest

import numpy as np

from pokersim import cards as C
from pokersim import hand_range as H


def cards(text):
    return [C.parse_card(text[i:i + 2]) for i in range(0, len(text), 2)]


class HandRangeTest(unittest.TestCase):
    def test_grid_layout(self):
        # pairs on the diagonal, suited above it, offsuit below it (row/column A..2)
        self.assertEqual(H.CLASS_NAMES[H.class_of(*cards("AhAd"))], "AA")
        self.assertEqual(divmod(H.class_of(*cards("AsKs")), 13), (0, 1))
        self.assertEqual(divmod(H.class_of(*cards("AsKd")), 13), (1, 0))
        self.assertEqual(H.CLASS_NAMES[168], "22")
        self.assertEqual(len(set(H.CLASS_NAMES)), 169)

    def test_live_combos(self):
        lc = H.live_combos([])
        self.assertEqual(lc.sum(), 1326)
        self.assertEqual(lc[H.CLASS_NAMES.index("AA")], 6)
        self.assertEqual(lc[H.CLASS_NAMES.index("AKs")], 4)
        self.assertEqual(lc[H.CLASS_NAMES.index("AKo")], 12)
        lc = H.live_combos(cards("AsKsQd"))
        self.assertEqual(lc.sum(), 49 * 48 / 2)
        self.assertEqual(lc[H.CLASS_NAMES.index("AA")], 3)

    def test_board_features(self):
        g = H.class_board_features(cards("Ah9h2c"))
        aa, aks, x43 = (H.CLASS_NAMES.index(n) for n in ("AA", "AKs", "43o"))
        self.assertEqual(g[aa, 13], 1)   # set of aces
        self.assertEqual(g[aks, 10], 1)  # top pair
        self.assertEqual(g[aks, 19], 1)  # flush draw (suited, two of a suit on the board)
        self.assertEqual(g[x43, 17], 1)  # wheel draw
        self.assertEqual(H.class_board_features([]).shape, (169, H.G_DIM))

    def test_sample_hole_avoids_used_cards(self):
        p = np.zeros(169)
        p[H.CLASS_NAMES.index("AA")] = 1.0
        used = set(cards("AhAd"))
        a, b = H.sample_hole(p, used, random.Random(0))
        self.assertTrue(a not in used and b not in used)
        self.assertEqual(H.CLASS_NAMES[H.class_of(a, b)], "AA")


if __name__ == "__main__":
    unittest.main()
