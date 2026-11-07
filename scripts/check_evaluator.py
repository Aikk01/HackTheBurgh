"""Quick self-check of the hand evaluator (run: python scripts/check_evaluator.py)."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokersim.cards import full_deck, parse_cards  # noqa: E402
from pokersim.evaluator import describe, evaluate, evaluate_omaha  # noqa: E402

TESTS = {
    "AhKhQhJhTh": "Royal Flush",
    "9h8h7h6h5h2c3d": "Straight Flush, Nine high",
    "AhAdAcAs2d": "Four of a Kind, Aces",
    "KhKdKc9s9d": "Full House, Kings full of Nines",
    "AhKh9h5h2h": "Flush, Ace high",
    "Ah2d3c4s5h": "Straight, Five high",
    "ThJdQcKsAh": "Straight, Ace high",
    "7h7d7c2s9d": "Three of a Kind, Sevens",
    "AhAd8c8s3d": "Two Pair, Aces and Eights",
    "JhJd2c5s9d": "Pair of Jacks",
    "Ah2d5c9sJd": "Ace high",
    "KhKdKcKsAhAdAc": "Four of a Kind, Kings",
    "AhAdAcKsKdKc2h": "Full House, Aces full of Kings",
    "AhAdKcKsQdQc2h": "Two Pair, Aces and Kings",
    "AhAd9c9s5d5c2h": "Two Pair, Aces and Nines",
}

ok = True
for hand, expected in TESTS.items():
    got = describe(evaluate([c for c in parse_cards(hand) if c is not None]))
    if got != expected:
        ok = False
        print("FAIL", hand, got, "!=", expected)

cards = lambda s: [c for c in parse_cards(s) if c is not None]  # noqa: E731
assert evaluate(cards("AhAdAcKsKd")) > evaluate(cards("AhKhQhJh9h"))
assert evaluate(cards("AhKhQhJh9h"), True) > evaluate(cards("AhAdAcKsKd"), True)
assert describe(evaluate(cards("Ah6d7c8s9h"), True), True) == "Straight, Nine high"
assert evaluate(cards("Th6d7c8s9h"), True) > evaluate(cards("Ah6d7c8s9h"), True)
assert evaluate(cards("AhAdKc5s2d")) > evaluate(cards("AsAcQc5h2c"))
assert describe(evaluate_omaha(cards("AhAdKcQd"), cards("AcAsKd5h2c"))) == "Four of a Kind, Aces"
# Omaha must use exactly two hole cards: a single hole heart cannot make a flush
assert describe(evaluate_omaha(cards("2h3d9cTd"), cards("AhKhQhJh4c"))).startswith("Straight")
print("evaluator OK" if ok else "evaluator FAILED")

rng = random.Random(1)
deck = full_deck()
t = time.perf_counter()
for _ in range(20000):
    rng.shuffle(deck)
    evaluate(deck[:7])
print("20k 7-card evaluations: %.2fs" % (time.perf_counter() - t))
