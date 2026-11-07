"""Card representation and PHH card notation.

A card is an ``int`` in ``0..51``: ``rank * 4 + suit`` where rank ``0`` is the
deuce and ``12`` the ace, and suit follows the order clubs, diamonds, hearts,
spades.  Unknown cards (``??`` in PHH) are represented by ``None``.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Optional

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
RANK_NAMES = ["Deuce", "Trey", "Four", "Five", "Six", "Seven", "Eight", "Nine",
              "Ten", "Jack", "Queen", "King", "Ace"]
RANK_NAMES_PLURAL = ["Deuces", "Treys", "Fours", "Fives", "Sixes", "Sevens",
                     "Eights", "Nines", "Tens", "Jacks", "Queens", "Kings", "Aces"]

Card = int


def make_card(rank: int, suit: int) -> Card:
    return rank * 4 + suit


def rank_of(card: Card) -> int:
    return card >> 2


def suit_of(card: Card) -> int:
    return card & 3


def parse_card(text: str) -> Optional[Card]:
    """Parse a two character card such as ``Ah``.  ``??`` gives ``None``."""
    if len(text) != 2:
        raise ValueError(f"bad card {text!r}")
    if text == "??":
        return None
    r, s = text[0].upper(), text[1].lower()
    if r not in RANK_CHARS or s not in SUIT_CHARS:
        if "?" in text:
            # partially unknown card (e.g. 'A?') - treat as unknown
            return None
        raise ValueError(f"bad card {text!r}")
    return make_card(RANK_CHARS.index(r), SUIT_CHARS.index(s))


def parse_cards(text: str) -> List[Optional[Card]]:
    """Parse concatenated PHH card notation, e.g. ``AhKs??``."""
    text = text.strip()
    if len(text) % 2:
        raise ValueError(f"bad card string {text!r}")
    return [parse_card(text[i:i + 2]) for i in range(0, len(text), 2)]


def card_str(card: Optional[Card]) -> str:
    if card is None:
        return "??"
    return RANK_CHARS[card >> 2] + SUIT_CHARS[card & 3]


def cards_str(cards: Iterable[Optional[Card]]) -> str:
    return "".join(card_str(c) for c in cards)


def pretty(card: Optional[Card]) -> str:
    if card is None:
        return "??"
    return RANK_CHARS[card >> 2] + SUIT_SYMBOLS[SUIT_CHARS[card & 3]]


def full_deck(short: bool = False) -> List[Card]:
    """52-card deck, or the 36-card short deck (sixes and up)."""
    first_rank = 4 if short else 0
    return [make_card(r, s) for r in range(first_rank, 13) for s in range(4)]


class Deck:
    """A deck that can be drawn from and that tracks cards already in play."""

    def __init__(self, short: bool = False, rng: Optional[random.Random] = None):
        self.short = short
        self.rng = rng or random.Random()
        self.cards: List[Card] = full_deck(short)
        self.rng.shuffle(self.cards)

    def remove(self, cards: Iterable[Optional[Card]]) -> List[Card]:
        """Remove known cards (e.g. dealt from a hand history).  Returns the
        cards that were *not* in the deck (duplicates / impossible deals)."""
        missing = []
        for c in cards:
            if c is None:
                continue
            try:
                self.cards.remove(c)
            except ValueError:
                missing.append(c)
        return missing

    def draw(self, n: int = 1) -> List[Card]:
        if n > len(self.cards):
            raise ValueError("deck exhausted")
        out = self.cards[:n]
        del self.cards[:n]
        return out

    def sample(self, n: int) -> List[Card]:
        """Sample ``n`` cards without removing them (for Monte Carlo)."""
        return self.rng.sample(self.cards, n)

    def __len__(self) -> int:
        return len(self.cards)
