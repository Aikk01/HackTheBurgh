"""Historical evidence for Poker Mirror (PLAN.md section 2).

Reads the Takara ``poker_hands`` Parquet shards, replays heads-up no-limit
hands through the engine and extracts every river decision where the
out-of-position defender checked and then faced a bet.  Also builds the
filter exhibit, the chronological train/test manifests and the per-player
style table.

Offline only: needs ``pyarrow``.  The demo reads the small JSON artifacts.
"""
from __future__ import annotations

import collections
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import cards as C
from .engine import UnsupportedVariant
from .mirror_types import CONTEXTS, CTX_BETTOR_AGGR, CTX_CHECKED_THROUGH, CTX_DEFENDER_AGGR
from .phh import HandRecord, record_from_dict
from .replay import replay

TAKARA_REVISION = "6acb5afb6f43082e6a468fde578890d9188be393"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARD_DIR = os.path.join(ROOT, "data", "cache", "takara")
LARGE_OUT = os.path.join(ROOT, "data", "cache", "mirror")
SMALL_OUT = os.path.join(ROOT, "data", "mirror")
N_SHARDS = 37
RATIO_LOW, RATIO_HIGH = 0.40, 0.60
HALF_TOL = 1e-8
ALLOWED_WARNING = "reached showdown with unknown cards"
CTX_OTHER_AGGR = "other_aggr"   # a third player made the last bet (flop only)
_FOLD_ACTION = re.compile(r"^p\d+ f$")

STYLE_FEATURES = ("vpip", "pfr", "af", "fold_to_bet", "wtsd")


def shard_path(i: int) -> str:
    return os.path.join(SHARD_DIR, f"train-{i:05d}-of-00037.parquet")


# ----------------------------------------------------------------- rows -> records
def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def row_to_record(row: Dict[str, Any]) -> Optional[HandRecord]:
    """Map one Takara row onto a :class:`HandRecord`; ``None`` if malformed."""
    actions = row.get("actions") or []
    if len(actions) <= 1:
        return None  # the five hand-annotated rows carry a single '[' element
    d: Dict[str, Any] = {
        "variant": row.get("variant"),
        "antes": row.get("antes") or [],
        "blinds_or_straddles": row.get("blinds_or_straddles") or [],
        "starting_stacks": row.get("starting_stacks") or [],
        "actions": actions,
        "players": row.get("players") or [],
        "ante_trimming_status": bool(row.get("ante_trimming_status")),
    }
    for k in ("min_bet", "small_bet", "big_bet", "bring_in"):
        if row.get(k) is not None and not (isinstance(row[k], float) and math.isnan(row[k])):
            d[k] = row[k]
    for k in ("finishing_stacks", "winnings"):
        if row.get(k):
            d[k] = row[k]
    if row.get("seats"):
        d["seats"] = row["seats"]
    sc = _int_or_none(row.get("seat_count"))
    if sc:
        d["seat_count"] = sc
    for k in ("venue", "table", "time", "time_zone_abbreviation", "currency_symbol", "currency", "event",
              "source_file", "source_type"):
        if row.get(k) not in (None, "", "None"):
            d[k] = row[k]
    if row.get("hand") not in (None, "", "None"):
        d["hand"] = row["hand"]
    for k in ("day", "month", "year", "hand_number_in_file"):
        v = _int_or_none(row.get(k))
        if v is not None:
            d[k] = v
    try:
        return record_from_dict(d, source=row.get("source_file"), index=_int_or_none(row.get("hand_number_in_file")))
    except (ValueError, TypeError):
        return None


def timestamp_of(rec: HandRecord) -> Optional[str]:
    """ISO timestamp from day/month/year/time, or ``None`` when chronology is unreliable."""
    m = rec.meta
    y, mo, d, t = m.get("year"), m.get("month"), m.get("day"), m.get("time")
    if not (y and mo and d and t):
        return None
    try:
        tt = _dt.time.fromisoformat(str(t)) if not isinstance(t, _dt.time) else t
        return _dt.datetime(int(y), int(mo), int(d), tt.hour, tt.minute, tt.second).isoformat()
    except (ValueError, TypeError):
        return None


def stake_of(source_file: Optional[str]) -> Optional[str]:
    """e.g. 'handhq/FTP-2009-07-01_2009-07-23_600NLH_OBFU/6/ftp ...' -> '600NLH'."""
    if not source_file:
        return None
    for part in source_file.split("/"):
        for token in part.split("_"):
            if token.endswith("NLH"):
                return token
    return None


# ----------------------------------------------------------------- river spots
@dataclass
class RiverSpot:
    venue: str
    hand_id: str
    source_file: str
    hand_number_in_file: Optional[int]
    timestamp: Optional[str]
    stake: Optional[str]
    defender: str                       # handle of the out-of-position player (p1)
    bettor: str                         # handle of the in-position player (p2)
    board: List[str]
    bet: float
    pot_before_bet: float
    ratio: float
    exact_half: bool
    context: str
    response: str                       # fold / call / raise
    raise_to: Optional[float]
    defender_stack: float               # before responding
    bettor_stack: float                 # after betting
    all_three_legal: bool
    warning_free: bool
    defender_cards_known: bool
    defender_cards: List[str]
    bettor_cards: List[str]
    bettor_cards_known: bool
    fingerprint: str
    min_bet: float
    big_blind: float
    # Phase 6 (all table sizes); defaults keep older spot files loadable
    table_size: int = 2
    defender_index: int = 0
    bettor_index: int = 1
    raise_legal: bool = True
    in_band: bool = True
    flop_players: int = 2
    preflop_aggressor: str = ""
    flop_context: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.venue, self.hand_id)


def prefilter(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Cheap checks before replay.  Returns (ok, rejection reason)."""
    if row.get("variant") != "NT":
        return False, "variant"
    stacks = row.get("starting_stacks") or []
    if len(stacks) != 2:
        return False, "players"
    if any(s is None or not math.isfinite(s) or s <= 0 for s in stacks):
        return False, "stacks"
    if row.get("venue") in (None, "", "None") or row.get("hand") in (None, "", "None"):
        return False, "venue_or_hand"
    actions = row.get("actions") or []
    if len(actions) <= 1:
        return False, "malformed_actions"
    boards = sum(1 for a in actions if a.startswith("d db"))
    if boards != 3:
        return False, "no_river"
    return True, ""


def _fingerprint(rec: HandRecord) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([rec.starting_stacks, rec.blinds, rec.antes, rec.actions], sort_keys=True).encode())
    return h.hexdigest()[:16]


def _last_aggressor(history: Sequence[Dict[str, Any]], street: str) -> Optional[int]:
    agg = [h for h in history if h["street"] == street and h["action"] in ("bet", "raise")]
    return agg[-1]["player"] if agg else None


def _role(player: Optional[int], defender: int, bettor: int) -> str:
    if player is None:
        return CTX_CHECKED_THROUGH
    if player == bettor:
        return CTX_BETTOR_AGGR
    if player == defender:
        return CTX_DEFENDER_AGGR
    return CTX_OTHER_AGGR


def extract_spot(rec: HandRecord, wide: bool = False) -> Tuple[Optional[RiverSpot], str]:
    """Replay a hand and return its river check-bet-response spot, or (None, reason).

    By default this is the original cohort: a heads-up table, player 0 checks, player 1 bets 40-60% of the
    pot.  ``wide=True`` (Phase 6) accepts any table size where exactly two players reach the river and any
    bet size; the ratio, band and table size are recorded instead of filtering."""
    try:
        res = replay(rec, snapshots=False, observations=True)
    except UnsupportedVariant:
        return None, "unsupported"
    except Exception:  # noqa: BLE001 - a corrupt record must not stop the scan
        return None, "replay_error"
    river = [d for d in res.decisions if d.street == "river"]
    if len(river) < 3:
        return None, "no_river_response"
    d0, d1, d2 = river[0], river[1], river[2]
    if wide:
        ok = (d0.actual["action"] == "check" and d1.actual["action"] == "bet" and d1.player != d0.player
              and d2.player == d0.player and d0.observation.get("num_active", 2) == 2)
    else:
        ok = (d0.player == 0 and d0.actual["action"] == "check" and d1.player == 1
              and d1.actual["action"] == "bet" and d2.player == 0)
    if not ok:
        return None, "sequence"
    dfn, btr = d0.player, d1.player
    bet = float(d1.actual["to"] or 0)
    obs = d2.observation
    pot_before = float(obs["total_pot"]) - bet
    if pot_before <= 0 or bet <= 0:
        return None, "pot"
    ratio = bet / pot_before
    in_band = RATIO_LOW <= ratio <= RATIO_HIGH
    if not wide and not in_band:
        return None, "ratio"
    defender_stack = float(obs["hero"]["stack"])
    if defender_stack < bet:
        return None, "defender_short"
    response = d2.actual["action"]
    if response not in ("fold", "call", "raise"):
        return None, "response"
    other_warnings = [w for w in res.warnings if ALLOWED_WARNING not in w]
    if other_warnings:
        return None, "warnings"
    history = obs["history"]
    context = _role(_last_aggressor(history, "turn"), dfn, btr)
    if context == CTX_OTHER_AGGR:
        return None, "context"
    pf = _last_aggressor(history, "preflop")
    preflop_aggressor = "none" if pf is None else "bettor" if pf == btr else "defender" if pf == dfn else "other"
    preflop_folds = sum(1 for h in history if h["street"] == "preflop" and h["action"] == "fold")
    bettor_stack = float(next(p["stack"] for p in obs["players"] if p["index"] == btr) or 0)
    hand = res.hand
    def_cards = [C.card_str(c) for c in hand.players[dfn].final_cards()]
    bet_cards = [C.card_str(c) for c in hand.players[btr].final_cards()]
    raise_legal = bool(d2.legal.get("raise"))
    spot = RiverSpot(
        venue=str(rec.meta.get("venue")), hand_id=str(rec.meta.get("hand")), source_file=str(rec.source or ""),
        hand_number_in_file=rec.index, timestamp=timestamp_of(rec), stake=stake_of(rec.source),
        defender=rec.player_name(dfn), bettor=rec.player_name(btr),
        board=[C.card_str(c) for c in hand.board], bet=bet, pot_before_bet=round(pot_before, 2), ratio=ratio,
        exact_half=abs(bet - 0.5 * pot_before) <= HALF_TOL, context=context, response=response,
        raise_to=d2.actual.get("to"), defender_stack=defender_stack, bettor_stack=bettor_stack,
        all_three_legal=raise_legal and defender_stack >= 2 * bet,
        warning_free=not res.warnings,
        defender_cards_known=bool(def_cards) and all(c != "??" for c in def_cards),
        defender_cards=def_cards, bettor_cards=bet_cards,
        bettor_cards_known=bool(bet_cards) and all(c != "??" for c in bet_cards),
        fingerprint=_fingerprint(rec), min_bet=float(rec.min_bet or 0), big_blind=float(max(rec.blinds) if rec.blinds else 0),
        table_size=rec.n, defender_index=dfn, bettor_index=btr, raise_legal=raise_legal, in_band=in_band,
        flop_players=rec.n - preflop_folds, preflop_aggressor=preflop_aggressor,
        flop_context=_role(_last_aggressor(history, "flop"), dfn, btr),
    )
    return spot, ""


# ----------------------------------------------------------------- shard scan
def iter_rows(path: str, columns: Optional[Sequence[str]] = None, batch_size: int = 20000) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None):
        for row in batch.to_pylist():
            yield row


@dataclass
class ShardScan:
    shard: int
    rows: int = 0
    prefiltered: int = 0
    rejections: Dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    spots: List[RiverSpot] = field(default_factory=list)
    seconds: float = 0.0

    def counts(self) -> Dict[str, Any]:
        return {"shard": self.shard, "rows": self.rows, "prefiltered": self.prefiltered,
                "eligible": len(self.spots), "duplicates": self.duplicates, "rejections": self.rejections,
                "seconds": round(self.seconds, 1)}


def scan_shard(i: int, path: Optional[str] = None, limit: Optional[int] = None) -> ShardScan:
    """Replay every heads-up NLHE hand of one shard that reached the river."""
    path = path or shard_path(i)
    t0 = time.perf_counter()
    scan = ShardScan(shard=i)
    rej: collections.Counter = collections.Counter()
    seen: set = set()
    for row in iter_rows(path):
        scan.rows += 1
        if limit and scan.rows > limit:
            break
        ok, why = prefilter(row)
        if not ok:
            rej[why] += 1
            continue
        key = (row["venue"], str(row["hand"]))
        if key in seen:
            scan.duplicates += 1
            continue
        seen.add(key)
        scan.prefiltered += 1
        rec = row_to_record(row)
        if rec is None:
            rej["malformed_record"] += 1
            continue
        spot, why = extract_spot(rec)
        if spot is None:
            rej[why] += 1
            continue
        scan.spots.append(spot)
    scan.rejections = dict(rej)
    scan.seconds = time.perf_counter() - t0
    return scan


def spots_to_jsonl(spots: Iterable[RiverSpot], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in spots:
            f.write(json.dumps(asdict(s), sort_keys=True) + "\n")
            n += 1
    return n


def load_spots(path: str) -> List[RiverSpot]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(RiverSpot(**json.loads(line)))
    return out


# ----------------------------------------------------------------- exhibit
def folds_over(spots: Iterable[RiverSpot]) -> Tuple[int, int]:
    spots = list(spots)
    return sum(1 for s in spots if s.response == "fold"), len(spots)


def exhibit_counts(spots: Sequence[RiverSpot]) -> Dict[str, Any]:
    """The five cohorts of the judge's evidence screen, as folds / opportunities."""
    half = [s for s in spots if s.exact_half]
    rows = {
        "all_eligible": folds_over(spots),
        "half_checked_through": folds_over(s for s in half if s.context == CTX_CHECKED_THROUGH),
        "half_bettor_aggr": folds_over(s for s in half if s.context == CTX_BETTOR_AGGR),
        "half_defender_aggr": folds_over(s for s in half if s.context == CTX_DEFENDER_AGGR),
        "defender_cards_known": folds_over(s for s in spots if s.defender_cards_known),
        "warning_free": folds_over(s for s in spots if s.warning_free),
    }
    by_response = collections.Counter(s.response for s in spots)
    return {"cohorts": {k: {"folds": v[0], "opportunities": v[1]} for k, v in rows.items()},
            "responses": dict(by_response), "n": len(spots)}


REFERENCE_COUNTS = {
    36: {"all_eligible": (592, 1057), "half_checked_through": (266, 394), "half_bettor_aggr": (52, 100),
         "defender_cards_known": (0, 205), "warning_free": (592, 815)},
    35: {"all_eligible": (688, 1220), "half_checked_through": (299, 455), "half_bettor_aggr": (71, 141),
         "defender_cards_known": (0, 225), "warning_free": (688, 925)},
}


# ----------------------------------------------------------------- manifests
def dedupe(spots: Iterable[RiverSpot]) -> Tuple[List[RiverSpot], Dict[str, int]]:
    """Global deduplication on (venue, hand id) and on content fingerprint."""
    by_key: Dict[Tuple[str, str], RiverSpot] = {}
    by_fp: Dict[str, Tuple[str, str]] = {}
    stats = collections.Counter()
    quarantine = 0
    for s in spots:
        if s.key in by_key:
            stats["duplicate_key"] += 1
            if by_key[s.key].fingerprint != s.fingerprint:
                quarantine += 1
                del by_key[s.key]  # conflicting records for one key: trust neither
                stats["quarantined_conflicts"] += 1
            continue
        if s.fingerprint in by_fp and by_fp[s.fingerprint] != s.key:
            stats["duplicate_content"] += 1
            continue
        by_key[s.key] = s
        by_fp[s.fingerprint] = s.key
    stats["kept"] = len(by_key)
    return list(by_key.values()), dict(stats)


def assign_split(spots: Sequence[RiverSpot], train_fraction: float = 0.8) -> Dict[Tuple[str, str], str]:
    """Chronological 80/20 split per venue with largest-remainder allocation (PLAN.md section 2)."""
    eligible = [s for s in spots if s.exact_half and s.all_three_legal and s.timestamp]
    n = len(eligible)
    target = int(math.floor(train_fraction * n))
    by_venue: Dict[str, List[RiverSpot]] = collections.defaultdict(list)
    for s in eligible:
        by_venue[s.venue].append(s)
    venues = sorted(by_venue)
    quotas = {v: train_fraction * len(by_venue[v]) for v in venues}
    alloc = {v: int(math.floor(quotas[v])) for v in venues}
    remainder = target - sum(alloc.values())
    for v in sorted(venues, key=lambda v: (-(quotas[v] - alloc[v]), v))[:max(0, remainder)]:
        alloc[v] += 1
    split: Dict[Tuple[str, str], str] = {}
    for v in venues:
        ordered = sorted(by_venue[v], key=lambda s: (s.timestamp or "", s.hand_id))
        for k, s in enumerate(ordered):
            split[s.key] = "train" if k < alloc[v] else "test"
    return split


def write_manifest(spots: Sequence[RiverSpot], split: Dict[Tuple[str, str], str], path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cutoffs: Dict[str, str] = {}
    n_train = n_test = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["venue", "hand_id", "timestamp", "fingerprint", "split", "context", "response", "defender", "bettor",
                    "source_file", "hand_number_in_file", "stake"])
        for s in sorted(spots, key=lambda s: (s.venue, s.timestamp or "", s.hand_id)):
            sp = split.get(s.key)
            if sp is None:
                continue
            w.writerow([s.venue, s.hand_id, s.timestamp, s.fingerprint, sp, s.context, s.response, s.defender, s.bettor,
                        s.source_file, s.hand_number_in_file, s.stake])
            if sp == "train":
                n_train += 1
                if s.timestamp and s.timestamp > cutoffs.get(s.venue, ""):
                    cutoffs[s.venue] = s.timestamp
            else:
                n_test += 1
    return {"path": os.path.relpath(path, ROOT), "train": n_train, "test": n_test, "cutoffs": cutoffs,
            "revision": TAKARA_REVISION}


# ----------------------------------------------------------------- style scan
def style_counts_for_hand(rec: HandRecord) -> Dict[str, Dict[str, List[int]]]:
    """Per-handle numerators/denominators of the five public-action features for one hand.

    River decisions are excluded so that river responses stay the outcome being predicted.
    """
    try:
        res = replay(rec, snapshots=False, observations=False)
    except Exception:  # noqa: BLE001
        return {}
    return style_counts_from_hand(res.hand)


def style_counts_from_hand(hand: Any) -> Dict[str, Dict[str, List[int]]]:
    """The same five features counted on an engine ``Hand`` (also used live by the Play-table opponent model)."""
    n = hand.n
    out: Dict[str, Dict[str, List[int]]] = {}
    for p in hand.players:
        out[p.name] = {f: [0, 0] for f in STYLE_FEATURES}
    vpip = [False] * n
    pfr = [False] * n
    saw_flop = [False] * n
    postflop_actions = [0] * n
    postflop_aggr = [0] * n
    faced_bet = [0] * n
    folded_to_bet = [0] * n
    folded_pre = [False] * n
    for ev in hand.events:
        if ev["type"] != "action":
            continue
        i, act, street = ev["player"], ev["action"], ev["street"]
        if street == "preflop":
            if act in ("call", "bet", "raise"):
                vpip[i] = True
            if act in ("bet", "raise"):
                pfr[i] = True
            if act == "fold":
                folded_pre[i] = True
        elif street in ("flop", "turn"):
            postflop_actions[i] += 1
            if act in ("bet", "raise"):
                postflop_aggr[i] += 1
            if act == "fold":
                faced_bet[i] += 1
                folded_to_bet[i] += 1
            elif act == "call":
                faced_bet[i] += 1
    reached_flop = hand.street >= 1
    for p in hand.players:
        i = p.index
        d = out[p.name]
        d["vpip"] = [int(vpip[i]), 1]
        d["pfr"] = [int(pfr[i]), 1]
        if reached_flop and not folded_pre[i]:
            saw_flop[i] = True
        d["af"] = [postflop_aggr[i], postflop_actions[i]]
        d["fold_to_bet"] = [folded_to_bet[i], faced_bet[i]]
        went_sd = saw_flop[i] and not p.folded and hand.status == "finished" and len(hand.active_players()) > 1
        d["wtsd"] = [int(went_sd), int(saw_flop[i])]
    return out


def style_scan_shard(i: int, cutoffs: Dict[str, str], path: Optional[str] = None,
                     limit: Optional[int] = None) -> Dict[Tuple[str, str], Dict[str, List[int]]]:
    """Accumulate style counts per (venue, handle) for hands at or before the venue cutoff."""
    path = path or shard_path(i)
    table: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    rows = 0
    for row in iter_rows(path):
        rows += 1
        if limit and rows > limit:
            break
        if row.get("variant") != "NT" or row.get("venue") in (None, "", "None"):
            continue
        cutoff = cutoffs.get(row["venue"])
        if cutoff is None:
            continue
        rec = row_to_record(row)
        if rec is None:
            continue
        ts = timestamp_of(rec)
        if ts is None or ts > cutoff:
            continue
        for handle, feats in style_counts_for_hand(rec).items():
            key = (row["venue"], handle)
            acc = table.setdefault(key, {f: [0, 0] for f in STYLE_FEATURES})
            acc["hands"] = acc.get("hands", [0, 0])
            acc["hands"][0] += 1
            for f in STYLE_FEATURES:
                acc[f][0] += feats[f][0]
                acc[f][1] += feats[f][1]
    return table


def merge_style_tables(tables: Iterable[Dict[Tuple[str, str], Dict[str, List[int]]]]) -> Dict[Tuple[str, str], Dict[str, List[int]]]:
    out: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    for t in tables:
        for key, feats in t.items():
            acc = out.setdefault(key, {f: [0, 0] for f in STYLE_FEATURES})
            acc["hands"] = acc.get("hands", [0, 0])
            acc["hands"][0] += feats.get("hands", [0, 0])[0]
            for f in STYLE_FEATURES:
                acc[f][0] += feats[f][0]
                acc[f][1] += feats[f][1]
    return out


def write_style_table(table: Dict[Tuple[str, str], Dict[str, List[int]]], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["venue", "handle", "hands"] + [f"{feat}_{k}" for feat in STYLE_FEATURES for k in ("num", "den")])
        for (venue, handle), feats in sorted(table.items()):
            w.writerow([venue, handle, feats.get("hands", [0, 0])[0]] + [feats[feat][k] for feat in STYLE_FEATURES for k in (0, 1)])
            n += 1
    return n


def read_style_table(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [{k: (v if k in ("venue", "handle") else int(v)) for k, v in row.items()} for row in csv.DictReader(f)]


# ----------------------------------------------------------------- preparation
DEMO_HANDS = {36: ["3039372300", "3039405280"]}


def find_hands(shard: int, hand_ids: Sequence[str]) -> List[HandRecord]:
    wanted = set(str(h) for h in hand_ids)
    out = []
    for row in iter_rows(shard_path(shard)):
        if str(row.get("hand")) in wanted:
            rec = row_to_record(row)
            if rec is not None:
                out.append(rec)
    return out


def _scan_worker(args: Tuple[int, Optional[int]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    shard, limit = args
    scan = scan_shard(shard, limit=limit)
    return scan.counts(), [asdict(s) for s in scan.spots]


def population_counts(spots: Iterable[RiverSpot]) -> Dict[str, Any]:
    """Counts by public context for exact-half, all-three-legal offers."""
    counts: Dict[str, Dict[str, int]] = {c: {"fold": 0, "call": 0, "raise": 0} for c in CONTEXTS}
    pooled = {"fold": 0, "call": 0, "raise": 0}
    for s in spots:
        if s.exact_half and s.all_three_legal:
            counts[s.context][s.response] += 1
            pooled[s.response] += 1
    return {"by_context": counts, "pooled": pooled}


def prepare(shards: Sequence[int] = tuple(range(N_SHARDS)), workers: Optional[int] = None,
            limit: Optional[int] = None, out_large: str = LARGE_OUT, out_small: str = SMALL_OUT) -> Dict[str, Any]:
    """Run the first scan over the shards, deduplicate, split, and write every artifact."""
    import multiprocessing as mp

    os.makedirs(out_large, exist_ok=True)
    os.makedirs(out_small, exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(i, limit) for i in shards if os.path.exists(shard_path(i))]
    missing = [i for i in shards if not os.path.exists(shard_path(i))]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(min(workers, len(jobs))) as pool:
        results = pool.map(_scan_worker, jobs)
    counts = [c for c, _ in results]
    per_shard_spots = {c["shard"]: [RiverSpot(**d) for d in spots] for c, spots in zip(counts, (s for _, s in results))}
    all_spots = [s for spots in per_shard_spots.values() for s in spots]
    kept, dedupe_stats = dedupe(all_spots)
    spots_to_jsonl(kept, os.path.join(out_large, "spots.jsonl"))
    split = assign_split(kept)
    manifest = write_manifest(kept, split, os.path.join(out_large, "manifest.csv"))
    train = [s for s in kept if split.get(s.key) == "train"]
    test = [s for s in kept if split.get(s.key) == "test"]
    eligible_modelling = [s for s in kept if s.exact_half and s.all_three_legal and s.timestamp]
    exhibit = {
        "revision": TAKARA_REVISION,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "per_shard": {str(i): exhibit_counts(per_shard_spots[i]) for i in per_shard_spots},
        "two_shard": exhibit_counts(per_shard_spots.get(35, []) + per_shard_spots.get(36, [])) if 35 in per_shard_spots and 36 in per_shard_spots else None,
        "full_corpus": exhibit_counts(kept),
        "reference": {str(k): {c: list(v) for c, v in d.items()} for k, d in REFERENCE_COUNTS.items()},
    }
    with open(os.path.join(out_small, "exhibit.json"), "w", encoding="utf-8") as f:
        json.dump(exhibit, f, indent=1)
    pop = population_counts(train)
    pop.update({"revision": TAKARA_REVISION, "split": "train", "n_train": len(train), "n_test": len(test),
                "generated": exhibit["generated"], "prior_pseudo_count": 1})
    with open(os.path.join(out_small, "population.json"), "w", encoding="utf-8") as f:
        json.dump(pop, f, indent=1)
    meta = {
        "revision": TAKARA_REVISION, "generated": exhibit["generated"], "shards_scanned": [c["shard"] for c in counts],
        "shards_missing": missing, "rows": sum(c["rows"] for c in counts), "prefiltered": sum(c["prefiltered"] for c in counts),
        "eligible_before_dedupe": len(all_spots), "dedupe": dedupe_stats, "eligible_unique": len(kept),
        "eligible_modelling": len(eligible_modelling), "manifest": manifest, "per_shard": counts,
        "seconds": round(time.perf_counter() - t0, 1),
        "responses_unique": dict(collections.Counter(s.response for s in kept)),
        "contexts_modelling": dict(collections.Counter(s.context for s in eligible_modelling)),
        "venues": dict(collections.Counter(s.venue for s in kept)),
    }
    for shard, ids in DEMO_HANDS.items():
        if shard in per_shard_spots:
            from .phh import dump_hands
            recs = find_hands(shard, ids)
            with open(os.path.join(out_small, "demo_hands.phhs"), "w", encoding="utf-8") as f:
                f.write(dump_hands(recs))
            meta["demo_hands"] = [str(r.meta.get("hand")) for r in recs]
    with open(os.path.join(out_small, "scan_counts.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def _style_worker(args: Tuple[int, Dict[str, str], Optional[int]]) -> Dict[Tuple[str, str], Dict[str, List[int]]]:
    shard, cutoffs, limit = args
    return style_scan_shard(shard, cutoffs, limit=limit)


def prepare_styles(cutoffs: Optional[Dict[str, str]] = None, shards: Sequence[int] = tuple(range(N_SHARDS)),
                   workers: Optional[int] = None, limit: Optional[int] = None, out_large: str = LARGE_OUT,
                   out_small: str = SMALL_OUT, tag: str = "") -> Dict[str, Any]:
    """The second scan: per-player style counts up to each venue's training cutoff (type layer)."""
    import multiprocessing as mp

    t0 = time.perf_counter()
    suffix = f"_{tag}" if tag else ""
    if cutoffs is None:
        with open(os.path.join(out_small, f"scan_counts{suffix}.json"), encoding="utf-8") as f:
            sc = json.load(f)
            cutoffs = sc["cutoffs"] if tag else sc["manifest"]["cutoffs"]
    jobs = [(i, cutoffs, limit) for i in shards if os.path.exists(shard_path(i))]
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    with mp.Pool(min(workers, len(jobs))) as pool:
        # merged as the shards finish, so only one shard's table waits in memory at a time
        merged = merge_style_tables(pool.imap_unordered(_style_worker, jobs))
    path = os.path.join(out_large, f"style_table{suffix}.csv")
    n = write_style_table(merged, path)
    thresholds = [10, 25, 50, 100, 250, 500]
    per_venue: Dict[str, Dict[str, int]] = {}
    for (venue, _handle), feats in merged.items():
        h = feats.get("hands", [0, 0])[0]
        d = per_venue.setdefault(venue, {"handles": 0, **{f">={t}": 0 for t in thresholds}})
        d["handles"] += 1
        for t in thresholds:
            if h >= t:
                d[f">={t}"] += 1
    meta = {"revision": TAKARA_REVISION, "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "cutoffs": cutoffs, "handles": n, "per_venue": per_venue, "features": list(STYLE_FEATURES),
            "path": os.path.relpath(path, ROOT), "seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(out_small, f"style_counts{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


# ----------------------------------------------------------------- Phase 6: every table size
RESP_CODE = {"fold": 0, "call": 1, "raise": 2}
CTX_CODE = {CTX_CHECKED_THROUGH: 0, CTX_BETTOR_AGGR: 1, CTX_DEFENDER_AGGR: 2, CTX_OTHER_AGGR: 3}
PF_CODE = {"none": 0, "bettor": 1, "defender": 2, "other": 3}
RANKS, SUITS = "23456789TJQKA", "cdhs"
SIZE_CLASSES = ("2", "3-6", "7-10")
WIDE_TABLE = os.path.join(LARGE_OUT, "decisions_wide.npz")
WIDE_META = os.path.join(LARGE_OUT, "decisions_wide_meta.json")
WIDE_SPOTS = os.path.join(LARGE_OUT, "spots_wide.jsonl")
WIDE_MANIFEST = os.path.join(LARGE_OUT, "manifest_wide.csv")


def prefilter_wide(row: Dict[str, Any]) -> Tuple[bool, str]:
    """Cheap checks before replay: any table size, exactly two players left when the river is dealt."""
    if row.get("variant") != "NT":
        return False, "variant"
    stacks = row.get("starting_stacks") or []
    if len(stacks) < 2:
        return False, "players"
    if any(s is None or not math.isfinite(s) or s <= 0 for s in stacks):
        return False, "stacks"
    if row.get("venue") in (None, "", "None") or row.get("hand") in (None, "", "None"):
        return False, "venue_or_hand"
    actions = row.get("actions") or []
    if len(actions) <= 1:
        return False, "malformed_actions"
    boards = [k for k, a in enumerate(actions) if a.startswith("d db")]
    if len(boards) != 3:
        return False, "no_river"
    if len(stacks) - sum(1 for a in actions[:boards[2]] if _FOLD_ACTION.match(a)) != 2:
        return False, "not_two_at_river"
    return True, ""


def size_class(n: int) -> str:
    return "2" if n <= 2 else ("3-6" if n <= 6 else "7-10")


def card_code(c: str) -> int:
    if not c or len(c) < 2 or c[0] not in RANKS or c[1] not in SUITS:
        return -1
    return RANKS.index(c[0]) * 4 + SUITS.index(c[1])


def _epoch(ts: str) -> int:
    return int(_dt.datetime.fromisoformat(ts).replace(tzinfo=_dt.timezone.utc).timestamp())


def _h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def scan_shard_wide(i: int, out_path: str, limit: Optional[int] = None, path: Optional[str] = None) -> Dict[str, Any]:
    """Every heads-up river check-bet-response spot of one shard at any table size, streamed to ``out_path``."""
    path = path or shard_path(i)
    t0 = time.perf_counter()
    rows = pre = dup = n = 0
    rej: collections.Counter = collections.Counter()
    seen: set = set()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        for row in iter_rows(path, batch_size=5000):
            rows += 1
            if limit and rows > limit:
                break
            ok, why = prefilter_wide(row)
            if not ok:
                rej[why] += 1
                continue
            key = (row["venue"], str(row["hand"]))
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            pre += 1
            rec = row_to_record(row)
            if rec is None:
                rej["malformed_record"] += 1
                continue
            spot, why = extract_spot(rec, wide=True)
            if spot is None:
                rej[why] += 1
                continue
            f.write(json.dumps(asdict(spot), sort_keys=True) + "\n")
            n += 1
    os.replace(out_path + ".tmp", out_path)
    return {"shard": i, "rows": rows, "prefiltered": pre, "eligible": n, "duplicates": dup, "rejections": dict(rej),
            "seconds": round(time.perf_counter() - t0, 1)}


def _wide_worker(args: Tuple[int, str, Optional[int]]) -> Dict[str, Any]:
    shard, out_path, limit = args
    return scan_shard_wide(shard, out_path, limit=limit)


def _wide_split(info: Sequence[Tuple[str, Optional[int], str, bool]], train_fraction: float = 0.8):
    """Main cohort (40-60% bets, all three responses legal): chronological 80/20 per venue with
    largest-remainder allocation, as :func:`assign_split`.  Every other spot follows its venue's cutoff,
    so no bet size or table size leaks across the boundary.  Returns (split list, cutoffs)."""
    main_by_venue: Dict[str, List[int]] = collections.defaultdict(list)
    all_by_venue: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, (venue, ts, _hand, main) in enumerate(info):
        if ts is None:
            continue
        all_by_venue[venue].append(idx)
        if main:
            main_by_venue[venue].append(idx)
    n = sum(len(v) for v in main_by_venue.values())
    target = int(math.floor(train_fraction * n))
    venues = sorted(main_by_venue)
    quotas = {v: train_fraction * len(main_by_venue[v]) for v in venues}
    alloc = {v: int(math.floor(quotas[v])) for v in venues}
    for v in sorted(venues, key=lambda v: (-(quotas[v] - alloc[v]), v))[:max(0, target - sum(alloc.values()))]:
        alloc[v] += 1
    order = lambda i: (info[i][1], info[i][2])  # noqa: E731
    split: List[Optional[int]] = [None] * len(info)
    cut: Dict[str, Tuple[int, str]] = {}
    for v in venues:
        ordered = sorted(main_by_venue[v], key=order)
        for r, i in enumerate(ordered):
            split[i] = 0 if r < alloc[v] else 1
        if alloc[v] > 0:
            cut[v] = order(ordered[alloc[v] - 1])
    for v, idxs in all_by_venue.items():
        if v not in cut:
            ordered = sorted(idxs, key=order)
            m = int(math.floor(train_fraction * len(ordered)))
            if m > 0:
                cut[v] = order(ordered[m - 1])
        for i in idxs:
            if split[i] is None:
                split[i] = 0 if v in cut and order(i) <= cut[v] else 1
    cutoffs = {v: _dt.datetime.fromtimestamp(c[0], _dt.timezone.utc).replace(tzinfo=None).isoformat() for v, c in cut.items()}
    return split, cutoffs


WIDE_COLUMNS = (("player", "int32"), ("bettor", "int32"), ("venue", "int8"), ("ts", "int64"), ("seq", "int32"),
                ("split", "int8"), ("table_size", "int8"), ("def_idx", "int8"), ("bet_idx", "int8"), ("flop_players", "int8"),
                ("context", "int8"), ("flop_ctx", "int8"), ("pf_aggr", "int8"), ("response", "int8"),
                ("ratio", "float32"), ("spr", "float32"), ("pot_bb", "float32"), ("bb", "float32"),
                ("in_band", "bool"), ("exact_half", "bool"), ("legal3", "bool"), ("raise_legal", "bool"), ("def_known", "bool"))


def prepare_wide(shards: Sequence[int] = tuple(range(N_SHARDS)), workers: Optional[int] = None, limit: Optional[int] = None,
                 out_large: str = LARGE_OUT, out_small: str = SMALL_OUT, train_fraction: float = 0.8) -> Dict[str, Any]:
    """Phase 6 scan: all table sizes and bet sizes, deduplicated, one frozen chronological split, compact table."""
    import multiprocessing as mp

    import numpy as np

    t0 = time.perf_counter()
    wide_dir = os.path.join(out_large, "wide")
    os.makedirs(wide_dir, exist_ok=True)
    os.makedirs(out_small, exist_ok=True)
    jobs = [(i, os.path.join(wide_dir, f"spots_{i:02d}.jsonl"), limit) for i in shards if os.path.exists(shard_path(i))]
    missing = [i for i in shards if not os.path.exists(shard_path(i))]
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    counts: List[Dict[str, Any]] = []
    with mp.Pool(min(workers, len(jobs))) as pool:
        for c in pool.imap_unordered(_wide_worker, jobs):
            counts.append(c)
            print(f"  shard {c['shard']:2d}: {c['eligible']:6d} spots, {c['seconds']:6.1f} s", flush=True)
    counts.sort(key=lambda c: c["shard"])
    files = [j[1] for j in sorted(jobs)]

    def records():
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line, json.loads(line)

    stats: collections.Counter = collections.Counter()
    first_fp: Dict[int, int] = {}
    conflict: set = set()
    for _line, d in records():
        k = _h64(f"{d['venue']}|{d['hand_id']}")
        fp = int(d["fingerprint"], 16)
        if k in first_fp:
            stats["duplicate_key"] += 1
            if first_fp[k] != fp:
                conflict.add(k)
        else:
            first_fp[k] = fp
    del first_fp
    stats["quarantined_conflicts"] = len(conflict)
    kept: set = set()
    fp_owner: Dict[int, int] = {}
    info: List[Tuple[str, Optional[int], str, bool]] = []
    spots_path = os.path.join(out_large, os.path.basename(WIDE_SPOTS))
    with open(spots_path + ".tmp", "w", encoding="utf-8") as out:
        for line, d in records():
            k = _h64(f"{d['venue']}|{d['hand_id']}")
            if k in conflict or k in kept:
                continue
            fp = int(d["fingerprint"], 16)
            if fp_owner.get(fp, k) != k:
                stats["duplicate_content"] += 1
                continue
            kept.add(k)
            fp_owner[fp] = k
            out.write(line if line.endswith("\n") else line + "\n")
            info.append((d["venue"], _epoch(d["timestamp"]) if d.get("timestamp") else None, d["hand_id"],
                         bool(d.get("in_band") and d.get("all_three_legal"))))
    os.replace(spots_path + ".tmp", spots_path)
    del kept, fp_owner, conflict
    stats["kept"] = len(info)
    split, cutoffs = _wide_split(info, train_fraction)
    rows_idx = [i for i, s in enumerate(split) if s is not None]
    n = len(rows_idx)
    seq_of = np.full(len(info), -1, dtype=np.int64)
    seq_of[sorted(rows_idx, key=lambda i: (info[i][1], info[i][0], info[i][2]))] = np.arange(n)
    cols = {name: np.zeros(n, dtype=dt) for name, dt in WIDE_COLUMNS}
    board = np.full((n, 5), -1, dtype=np.int8)
    players: Dict[Tuple[str, str], int] = {}
    venues = sorted({v for v, *_ in info})
    vcode = {v: k for k, v in enumerate(venues)}
    by_size = {sc: collections.Counter() for sc in SIZE_CLASSES}
    pop = {c: {a: 0 for a in RESP_CODE} for c in CONTEXTS}
    pop_size = {c: {sc: {a: 0 for a in RESP_CODE} for sc in SIZE_CLASSES} for c in CONTEXTS}
    manifest_path = os.path.join(out_large, os.path.basename(WIDE_MANIFEST))
    r = 0
    with open(spots_path, encoding="utf-8") as f, open(manifest_path + ".tmp", "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["venue", "hand_id", "timestamp", "split", "table_size", "ratio", "in_band", "all_three_legal", "exact_half",
                    "context", "response", "defender", "bettor", "source_file", "stake"])
        for idx, line in enumerate(f):
            sp = split[idx]
            if sp is None:
                continue
            d = json.loads(line)
            sc = size_class(d["table_size"])
            main = bool(d["in_band"] and d["all_three_legal"])
            by_size[sc]["any"] += 1
            by_size[sc]["band"] += bool(d["in_band"])
            by_size[sc]["exact_half"] += bool(d["exact_half"] and d["all_three_legal"])
            if main:
                by_size[sc]["main"] += 1
                by_size[sc]["main_" + d["response"]] += 1
                by_size[sc]["main_" + ("train" if sp == 0 else "test")] += 1
                if sp == 0:
                    pop[d["context"]][d["response"]] += 1
                    pop_size[d["context"]][sc][d["response"]] += 1
            w.writerow([d["venue"], d["hand_id"], d["timestamp"], "train" if sp == 0 else "test", d["table_size"],
                        round(d["ratio"], 4), int(d["in_band"]), int(d["all_three_legal"]), int(d["exact_half"]), d["context"],
                        d["response"], d["defender"], d["bettor"], d["source_file"], d["stake"]])
            pot, bet, bb = float(d["pot_before_bet"]), float(d["bet"]), float(d.get("big_blind") or 0)
            eff = min(float(d["defender_stack"]), float(d["bettor_stack"]) + bet)
            vals = {"player": players.setdefault((d["venue"], d["defender"]), len(players)),
                    "bettor": players.setdefault((d["venue"], d["bettor"]), len(players)),
                    "venue": vcode[d["venue"]], "ts": info[idx][1], "seq": seq_of[idx], "split": sp,
                    "table_size": d["table_size"], "def_idx": d["defender_index"], "bet_idx": d["bettor_index"],
                    "flop_players": d.get("flop_players") or d["table_size"], "context": CTX_CODE[d["context"]],
                    "flop_ctx": CTX_CODE.get(d.get("flop_context") or CTX_CHECKED_THROUGH, 0),
                    "pf_aggr": PF_CODE.get(d.get("preflop_aggressor") or "none", 0), "response": RESP_CODE[d["response"]],
                    "ratio": d["ratio"], "spr": eff / pot if pot > 0 else 0.0, "pot_bb": pot / bb if bb > 0 else 0.0, "bb": bb,
                    "in_band": d["in_band"], "exact_half": d["exact_half"], "legal3": d["all_three_legal"],
                    "raise_legal": d.get("raise_legal", True), "def_known": d["defender_cards_known"]}
            for name, v in vals.items():
                cols[name][r] = v
            board[r, :] = [card_code(c) for c in (list(d["board"]) + [""] * 5)[:5]]
            r += 1
    os.replace(manifest_path + ".tmp", manifest_path)
    np.savez_compressed(os.path.join(out_large, os.path.basename(WIDE_TABLE)), board=board, **cols)
    gen = _dt.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(out_large, os.path.basename(WIDE_META)), "w", encoding="utf-8") as f:
        json.dump({"generated": gen, "n": n, "venues": venues, "players": [list(k) for k in players], "cutoffs": cutoffs,
                   "columns": [c for c, _ in WIDE_COLUMNS] + ["board"]}, f)
    main_mask = cols["in_band"] & cols["legal3"]
    summary = {
        "revision": TAKARA_REVISION, "generated": gen, "shards_scanned": [c["shard"] for c in counts], "shards_missing": missing,
        "rows": sum(c["rows"] for c in counts), "prefiltered": sum(c["prefiltered"] for c in counts),
        "eligible_before_dedupe": sum(c["eligible"] for c in counts), "dedupe": dict(stats), "kept": len(info),
        "no_timestamp": len(info) - n, "with_timestamp": n, "by_table_size": {sc: dict(by_size[sc]) for sc in SIZE_CLASSES},
        "main_cohort": {"definition": "bet 40-60% of the pot, all three responses legal, any table size",
                        "decisions": int(main_mask.sum()), "train": int((main_mask & (cols["split"] == 0)).sum()),
                        "test": int((main_mask & (cols["split"] == 1)).sum()),
                        "players": int(np.unique(cols["player"][main_mask]).size)},
        "any_size": {"train": int((cols["split"] == 0).sum()), "test": int((cols["split"] == 1).sum())},
        "exact_half_all_tables": int((cols["exact_half"] & cols["legal3"]).sum()), "cutoffs": cutoffs,
        "venues": dict(collections.Counter(venues[v] for v in cols["venue"].tolist())), "workers": workers,
        "per_shard": counts, "seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(out_small, "scan_counts_wide.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    population = {"by_context": pop, "pooled": {a: sum(pop[c][a] for c in CONTEXTS) for a in RESP_CODE},
                  "by_context_size": pop_size, "revision": TAKARA_REVISION, "split": "train",
                  "cohort": summary["main_cohort"]["definition"], "n_train": summary["main_cohort"]["train"],
                  "n_test": summary["main_cohort"]["test"], "generated": gen, "prior_pseudo_count": 1}
    with open(os.path.join(out_small, "population_wide.json"), "w", encoding="utf-8") as f:
        json.dump(population, f, indent=1)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pokersim mirror-prepare", description="Scan the Takara shards for Poker Mirror")
    ap.add_argument("--shards", default=None, help="comma separated shard ids (default: all 37)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="rows per shard (debugging)")
    ap.add_argument("--styles", action="store_true", help="also run the style scan (needs the manifests)")
    ap.add_argument("--styles-only", action="store_true", help="run only the style scan")
    ap.add_argument("--wide", action="store_true", help="Phase 6: every table size and bet size (scan_counts_wide.json)")
    ap.add_argument("--styles-wide", action="store_true", help="style scan with the wide split's cutoffs")
    args = ap.parse_args(argv)
    shards = [int(s) for s in args.shards.split(",")] if args.shards else list(range(N_SHARDS))
    if args.wide or args.styles_wide:
        if args.wide:
            meta = prepare_wide(shards, workers=args.workers, limit=args.limit)
            print(json.dumps({k: meta[k] for k in ("rows", "prefiltered", "kept", "with_timestamp", "by_table_size",
                                                    "main_cohort", "cutoffs", "seconds")}, indent=1))
        if args.styles_wide:
            meta = prepare_styles(shards=shards, workers=args.workers, limit=args.limit, tag="wide")
            print(json.dumps({k: meta[k] for k in ("handles", "per_venue", "seconds")}, indent=1))
        return 0
    if not args.styles_only:
        meta = prepare(shards, workers=args.workers, limit=args.limit)
        print(json.dumps({k: meta[k] for k in ("rows", "prefiltered", "eligible_unique", "eligible_modelling", "dedupe", "manifest",
                                                "seconds", "shards_missing", "demo_hands")}, indent=1))
    if args.styles or args.styles_only:
        meta = prepare_styles(shards=shards, workers=args.workers, limit=args.limit)
        print(json.dumps({k: meta[k] for k in ("handles", "per_venue", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
