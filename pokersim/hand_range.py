"""Hand-range prediction learned from recorded hands: P(opponent holds each of the 169 starting-hand classes).

The 169 classes are the usual 13 x 13 grid (row/column rank A, K, ..., 2): the diagonal holds the pairs, above it the
suited hands (AKs at row 0, column 1), below it the offsuit hands (AKo at row 1, column 0).

What the model sees is what the player asking could see at the table (:func:`context_features`): the street, the
board texture, positions, the pot, and everything the opponent did in this hand.  Each class also gets features of
how it connects with the board (:func:`class_board_features`: pairs, two pair, sets, straights, flush chances, from
ranks only), so a bet on a flush board can move the probability onto suited hands.  The output is multiplied by the
number of combinations of each class still possible given the dead cards (board + the asking player's own cards).

Trained by ``scripts/train_hand_range.py`` on the Takara corpus: every Pluribus hand (all hole cards known) and the
showdown hands of the online games.  Inference is NumPy only (``data/rl/hand_range.npz``).
"""
from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cards as C

N_CLASSES = 169
RANKS_DESC = "AKQJT98765432"
STREETS = ["preflop", "flop", "turn", "river"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rl", "hand_range.npz")


# ----------------------------------------------------------------- classes
def _g(rank: int) -> int:
    """Grid index of a rank (0 = ace, 12 = deuce)."""
    return 12 - rank


def class_of(c1: int, c2: int) -> int:
    a, b = _g(C.rank_of(c1)), _g(C.rank_of(c2))
    if a == b:
        return a * 13 + a
    hi, lo = min(a, b), max(a, b)
    if C.suit_of(c1) == C.suit_of(c2):
        return hi * 13 + lo  # suited: above the diagonal
    return lo * 13 + hi      # offsuit: below the diagonal


def class_name(k: int) -> str:
    r, c = divmod(k, 13)
    if r == c:
        return RANKS_DESC[r] * 2
    if r < c:
        return RANKS_DESC[r] + RANKS_DESC[c] + "s"
    return RANKS_DESC[c] + RANKS_DESC[r] + "o"


CLASS_NAMES = [class_name(k) for k in range(N_CLASSES)]
COMBOS: List[Tuple[int, int]] = [(a, b) for a in range(52) for b in range(a + 1, 52)]
COMBO_CLASS = np.array([class_of(a, b) for a, b in COMBOS], dtype=np.int16)
_COMBO_CARDS = np.array(COMBOS, dtype=np.int16)
# per class: high/low rank (2..A as 0..12), pair, suited
_CLS_HI = np.zeros(N_CLASSES, np.int16)
_CLS_LO = np.zeros(N_CLASSES, np.int16)
_CLS_PAIR = np.zeros(N_CLASSES, np.float32)
_CLS_SUITED = np.zeros(N_CLASSES, np.float32)
for _k in range(N_CLASSES):
    _r, _c = divmod(_k, 13)
    _CLS_HI[_k], _CLS_LO[_k] = 12 - min(_r, _c), 12 - max(_r, _c)
    _CLS_PAIR[_k] = float(_r == _c)
    _CLS_SUITED[_k] = float(_r < _c)


def live_combos(dead: Sequence[int]) -> np.ndarray:
    """Number of combinations of each class that avoid the dead cards."""
    mask = np.ones(len(COMBOS), bool)
    if dead:
        d = np.zeros(52, bool)
        d[list(dead)] = True
        mask = ~(d[_COMBO_CARDS[:, 0]] | d[_COMBO_CARDS[:, 1]])
    return np.bincount(COMBO_CLASS[mask], minlength=N_CLASSES).astype(np.float32)


def counting(dead: Sequence[int]) -> np.ndarray:
    """Plain card counting: each hand type in proportion to its combinations that avoid the known cards."""
    lc = live_combos(dead)
    return lc / max(float(lc.sum()), 1.0)


def widen(probs: np.ndarray, width: float, dead: Sequence[int]) -> np.ndarray:
    """A guess blended towards card counting by ``width`` (a player's learned range width, 0 = unchanged)."""
    return probs if width <= 0 else (1.0 - width) * probs + width * counting(dead)


# ----------------------------------------------------------------- features
def _board_texture(board: Sequence[int]) -> List[float]:
    if not board:
        return [0.0] * 8
    ranks = [C.rank_of(c) for c in board]
    suits = [C.suit_of(c) for c in board]
    rc = [ranks.count(r) for r in range(13)]
    sc = max(suits.count(s) for s in range(4))
    rs = set(ranks) | ({-1} if 12 in ranks else set())
    window = max(sum(1 for r in range(lo, lo + 5) if r in rs) for lo in range(-1, 9))
    return [len(board) / 5.0, float(max(rc) >= 2), float(max(rc) >= 3), sc / 5.0, float(sc >= 3), float(sc >= 4),
            window / 5.0, max(ranks) / 12.0]


def _lg(x: float) -> float:
    return math.log1p(max(x, 0.0)) / 7.0


CTX_DIM = 4 + 2 + 6 + 2 + 12 + 6 + 4 + 3 + 1 + 2 + 1 + 8


def context_features(obs: Dict[str, Any], j: int) -> np.ndarray:
    """Public information about opponent ``j`` as seen in the observation ``obs`` (of any other player)."""
    bb = float(obs.get("min_bet") or 1.0) or 1.0
    players = obs["players"]
    n = len(players)
    p = players[j]
    hist = [h for h in (obs.get("history") or []) if h.get("player") is not None]
    street = obs["street"]
    board = [c for c in (C.parse_card(x) for x in obs.get("board") or []) if c is not None]
    f: List[float] = [float(street == s) for s in STREETS]
    f += [n / 6.0, sum(1 for q in players if not q["folded"]) / 6.0]
    pos = [0.0] * 6
    pos[min(5, n - 1 - j)] = 1.0
    f += pos + [float(p.get("position") == "SB"), float(p.get("position") == "BB")]
    for s in STREETS:
        mine = [h for h in hist if h["player"] == j and h["street"] == s]
        f += [min(sum(1 for h in mine if h["action"] in ("bet", "raise")), 3) / 3.0,
              min(sum(1 for h in mine if h["action"] == "call"), 3) / 3.0,
              min(sum(1 for h in mine if h["action"] == "check"), 3) / 3.0]
    raises = 0
    pre = {"open": 0.0, "threebet": 0.0, "limp": 0.0, "call_raise": 0.0, "call_3bet": 0.0}
    last_aggr_pre = None
    for h in hist:
        if h["street"] != "preflop":
            continue
        if h["player"] == j:
            if h["action"] in ("bet", "raise"):
                pre["open" if raises == 0 else "threebet"] = 1.0
            elif h["action"] == "call":
                pre["limp" if raises == 0 else "call_raise" if raises == 1 else "call_3bet"] = 1.0
        if h["action"] in ("bet", "raise"):
            raises += 1
            last_aggr_pre = h["player"]
    f += list(pre.values()) + [float(last_aggr_pre == j)]
    for s in STREETS:
        tos = [float(h.get("to") or 0.0) for h in hist if h["player"] == j and h["street"] == s and h["action"] in ("bet", "raise")]
        f.append(_lg(max(tos) / bb) if tos else 0.0)
    stack = p.get("stack")
    f += [_lg(float(p.get("contributed") or 0.0) / bb), _lg((float(stack) if stack is not None else 100 * bb) / bb),
          float(bool(p.get("all_in")))]
    f.append(_lg(float(obs.get("total_pot") or 0.0) / bb))
    f += [min(raises, 4) / 4.0, min(sum(1 for h in hist if h["street"] == street and h["action"] in ("bet", "raise")), 4) / 4.0]
    si = STREETS.index(street)
    prev_aggr = None
    if si > 0:
        for h in hist:
            if h["street"] == STREETS[si - 1] and h["action"] in ("bet", "raise"):
                prev_aggr = h["player"]
    f.append(float(prev_aggr == j))
    f += _board_texture(board)
    v = np.asarray(f, dtype=np.float32)
    assert v.shape[0] == CTX_DIM, v.shape
    return v


G_DIM = 4 + 4 + 2 + 1 + 5 + 2 + 4


def class_board_features(board: Sequence[int]) -> np.ndarray:
    """(169, G_DIM): how each class connects with the board, from ranks (and suit counts for flushes)."""
    g = np.zeros((N_CLASSES, G_DIM), np.float32)
    hi, lo = _CLS_HI.astype(np.int64), _CLS_LO.astype(np.int64)
    g[:, 0], g[:, 1], g[:, 2], g[:, 3] = hi / 12.0, lo / 12.0, _CLS_PAIR, _CLS_SUITED
    if not board:
        return g
    ranks = np.array([C.rank_of(c) for c in board])
    suits = [C.suit_of(c) for c in board]
    bc = np.bincount(ranks, minlength=13)
    h1 = bc[hi]
    h2 = np.where(_CLS_PAIR > 0, 0, bc[lo])
    g[:, 4], g[:, 5], g[:, 6], g[:, 7] = h1 >= 1, h1 >= 2, h2 >= 1, h2 >= 2
    top = ranks.max()
    g[:, 8] = (_CLS_PAIR > 0) & (hi > top)                       # overpair
    g[:, 9] = (_CLS_PAIR > 0) & (hi < ranks.min())               # underpair
    g[:, 10] = (hi == top) | ((lo == top) & (_CLS_PAIR == 0))     # top pair
    # made hand from ranks: counts of hole + board ranks
    cnt = np.tile(bc, (N_CLASSES, 1))
    np.add.at(cnt, (np.arange(N_CLASSES), hi), 1)
    np.add.at(cnt, (np.arange(N_CLASSES), lo), 1)
    srt = -np.sort(-cnt, axis=1)
    base = -np.sort(-bc)
    two_pair = (srt[:, 0] >= 2) & (srt[:, 1] >= 2)
    trips = srt[:, 0] >= 3
    full = trips & (srt[:, 1] >= 2) | (srt[:, 0] >= 4)
    pair_plus = srt[:, 0] >= 2
    improved = (srt[:, 0] > base[0]) | (two_pair & ~((base[0] >= 2) & (base[1] >= 2)))
    present = cnt > 0
    ace_low = present[:, 12:13]
    pres = np.concatenate([ace_low, present], axis=1)            # index 0 = ace as one
    win = np.stack([pres[:, i:i + 5].sum(axis=1) for i in range(10)], axis=1)
    straight = (win >= 5).any(axis=1)
    bpres = np.concatenate([[bc[12] > 0], bc > 0])
    bwin = np.array([bpres[i:i + 5].sum() for i in range(10)])
    draw = ((win >= 4) & (win > bwin[None, :])).any(axis=1) & ~straight
    g[:, 11:16] = np.stack([pair_plus & improved, two_pair & improved, trips & improved, straight, full & improved], axis=1)
    g[:, 16], g[:, 17] = straight & ~(bwin >= 5).any(), draw
    sc = max(suits.count(s) for s in range(4))
    suited = _CLS_SUITED > 0
    g[:, 18] = suited & (sc >= 3)
    g[:, 19] = suited & (sc == 2) & (len(board) < 5)
    g[:, 20] = sc >= 4
    g[:, 21] = ~suited & (sc == 3)
    return g


# ----------------------------------------------------------------- inference (NumPy)
class RangeModel:
    """logit[c] = b(ctx)[c] + u(ctx) . (g[c] W + E[c]) + log(live combos of c)."""

    def __init__(self, path: str = MODEL_PATH):
        z = np.load(path)
        self.w = {k: z[k] for k in z.files}
        self.n_layers = sum(1 for k in self.w if k.startswith("trunk") and k.endswith(".w"))

    def _trunk(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.w["ctx_mean"]) / self.w["ctx_std"]
        for i in range(self.n_layers):
            h = np.maximum(h @ self.w[f"trunk{i}.w"] + self.w[f"trunk{i}.b"], 0.0)
        return h

    def logits(self, ctx: np.ndarray, g: np.ndarray) -> np.ndarray:
        """ctx (B, CTX_DIM), g (169, G_DIM) shared by the batch -> (B, 169), without the combo term."""
        h = self._trunk(ctx)
        b = h @ self.w["head_b.w"] + self.w["head_b.b"]
        u = h @ self.w["head_u.w"] + self.w["head_u.b"]
        cls = g @ self.w["g_w"] + self.w["emb"]
        return b + u @ cls.T

    def predict(self, ctx: np.ndarray, board: Sequence[int], dead: Sequence[int]) -> np.ndarray:
        ctx = np.atleast_2d(ctx)
        lc = live_combos(dead)
        lg = self.logits(ctx, class_board_features(board)) + np.log(np.maximum(lc, 1e-9))[None, :]
        lg -= lg.max(axis=1, keepdims=True)
        p = np.exp(lg) * (lc > 0)[None, :]
        return p / p.sum(axis=1, keepdims=True)


_MODEL: Optional[RangeModel] = None


def model() -> Optional[RangeModel]:
    global _MODEL
    if _MODEL is None and os.path.exists(MODEL_PATH):
        _MODEL = RangeModel(MODEL_PATH)
    return _MODEL


def predict_ranges(obs: Dict[str, Any], opponents: Sequence[int]) -> Dict[int, np.ndarray]:
    """169-class probabilities for each opponent index, from the point of view of ``obs['hero']``."""
    m = model()
    if m is None or not opponents:
        return {}
    board = [c for c in (C.parse_card(x) for x in obs.get("board") or []) if c is not None]
    hole = [c for c in (C.parse_card(x) for x in obs["hero"].get("cards") or []) if c is not None]
    ctx = np.stack([context_features(obs, j) for j in opponents])
    probs = m.predict(ctx, board, board + hole)
    return {j: probs[i] for i, j in enumerate(opponents)}


def sample_hole(probs: np.ndarray, used: set, rng: random.Random) -> Tuple[int, int]:
    """Draw a class from ``probs`` then a free combination of it."""
    for _ in range(20):
        k = rng.choices(range(N_CLASSES), weights=probs)[0]
        opts = [cmb for cmb in CLASS_COMBOS[k] if cmb[0] not in used and cmb[1] not in used]
        if opts:
            return rng.choice(opts)
    free = [c for c in range(52) if c not in used]
    a, b = rng.sample(free, 2)
    return a, b


CLASS_COMBOS: List[List[Tuple[int, int]]] = [[] for _ in range(N_CLASSES)]
for _i, (_a, _b) in enumerate(COMBOS):
    CLASS_COMBOS[int(COMBO_CLASS[_i])].append((_a, _b))


def grid(probs: np.ndarray) -> List[List[float]]:
    """13 x 13 rows for display (row/column A..2)."""
    return [[round(float(probs[r * 13 + c]), 5) for c in range(13)] for r in range(13)]
