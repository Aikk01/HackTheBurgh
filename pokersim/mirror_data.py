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


def extract_spot(rec: HandRecord) -> Tuple[Optional[RiverSpot], str]:
    """Replay a heads-up hand and return its river spot, or (None, reason)."""
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
    if not (d0.player == 0 and d0.actual["action"] == "check" and d1.player == 1
            and d1.actual["action"] == "bet" and d2.player == 0):
        return None, "sequence"
    bet = float(d1.actual["to"] or 0)
    obs = d2.observation
    pot_before = float(obs["total_pot"]) - bet
    if pot_before <= 0 or bet <= 0:
        return None, "pot"
    ratio = bet / pot_before
    if ratio < RATIO_LOW or ratio > RATIO_HIGH:
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
    turn_actions = [h for h in obs["history"] if h["street"] == "turn" and h["action"] != "board"]
    aggressive = [h for h in turn_actions if h["action"] in ("bet", "raise")]
    if not aggressive:
        context = CTX_CHECKED_THROUGH
    elif aggressive[-1]["player"] == 1:
        context = CTX_BETTOR_AGGR
    else:
        context = CTX_DEFENDER_AGGR
    bettor_stack = float(next(p["stack"] for p in obs["players"] if p["index"] == 1) or 0)
    hand = res.hand
    def_cards = [C.card_str(c) for c in hand.players[0].final_cards()]
    bet_cards = [C.card_str(c) for c in hand.players[1].final_cards()]
    spot = RiverSpot(
        venue=str(rec.meta.get("venue")), hand_id=str(rec.meta.get("hand")), source_file=str(rec.source or ""),
        hand_number_in_file=rec.index, timestamp=timestamp_of(rec), stake=stake_of(rec.source),
        defender=rec.player_name(0), bettor=rec.player_name(1),
        board=[C.card_str(c) for c in hand.board], bet=bet, pot_before_bet=round(pot_before, 2), ratio=ratio,
        exact_half=abs(bet - 0.5 * pot_before) <= HALF_TOL, context=context, response=response,
        raise_to=d2.actual.get("to"), defender_stack=defender_stack, bettor_stack=bettor_stack,
        all_three_legal=bool(d2.legal.get("raise")) and defender_stack >= 2 * bet,
        warning_free=not res.warnings,
        defender_cards_known=bool(def_cards) and all(c != "??" for c in def_cards),
        defender_cards=def_cards, bettor_cards=bet_cards,
        bettor_cards_known=bool(bet_cards) and all(c != "??" for c in bet_cards),
        fingerprint=_fingerprint(rec), min_bet=float(rec.min_bet or 0), big_blind=float(max(rec.blinds) if rec.blinds else 0),
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
    hand = res.hand
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
                   out_small: str = SMALL_OUT) -> Dict[str, Any]:
    """The second scan: per-player style counts up to each venue's training cutoff (type layer)."""
    import multiprocessing as mp

    t0 = time.perf_counter()
    if cutoffs is None:
        with open(os.path.join(out_small, "scan_counts.json"), encoding="utf-8") as f:
            cutoffs = json.load(f)["manifest"]["cutoffs"]
    jobs = [(i, cutoffs, limit) for i in shards if os.path.exists(shard_path(i))]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(min(workers, len(jobs))) as pool:
        tables = pool.map(_style_worker, jobs)
    merged = merge_style_tables(tables)
    path = os.path.join(out_large, "style_table.csv")
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
    with open(os.path.join(out_small, "style_counts.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="pokersim mirror-prepare", description="Scan the Takara shards for Poker Mirror")
    ap.add_argument("--shards", default=None, help="comma separated shard ids (default: all 37)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="rows per shard (debugging)")
    ap.add_argument("--styles", action="store_true", help="also run the style scan (needs the manifests)")
    ap.add_argument("--styles-only", action="store_true", help="run only the style scan")
    args = ap.parse_args(argv)
    shards = [int(s) for s in args.shards.split(",")] if args.shards else list(range(N_SHARDS))
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
