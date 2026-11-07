"""Poker Hand History (PHH) reader and writer.

PHH is a TOML based format (https://phh.readthedocs.io).  A ``.phh`` file holds
one hand; a ``.phhs`` file holds many, each under a ``[N]`` table header.  This
module parses both into :class:`HandRecord` objects and can write them back.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

_HEADER_RE = re.compile(r"^\[\s*([^\]\n]+?)\s*\]\s*$", re.M)

REQUIRED_ORDER = ["variant", "ante_trimming_status", "antes", "blinds_or_straddles", "bring_in",
                  "small_bet", "big_bet", "min_bet", "starting_stacks", "actions"]


@dataclass
class ParsedAction:
    actor: str            # 'd' for the dealer or 'pN'
    player: Optional[int]  # 0-based player index, None for the dealer
    verb: str             # dh, db, f, cc, cbr, sm, sd, pb
    args: List[str]
    commentary: Optional[str]
    raw: str

    @property
    def is_dealer(self) -> bool:
        return self.actor == "d"


def parse_action(text: str) -> Optional[ParsedAction]:
    """Parse one PHH action string.  Returns ``None`` for no-ops/commentary."""
    raw = text
    commentary = None
    if "#" in text:
        text, commentary = text.split("#", 1)
        commentary = commentary.strip() or None
    parts = text.split()
    if not parts:
        return None
    actor = parts[0]
    if len(parts) < 2:
        raise ValueError(f"malformed action {raw!r}")
    verb = parts[1]
    args = parts[2:]
    player = None
    if actor != "d":
        m = re.fullmatch(r"p(\d+)", actor)
        if not m:
            raise ValueError(f"unknown actor in action {raw!r}")
        player = int(m.group(1)) - 1
    return ParsedAction(actor, player, verb, args, commentary, raw)


@dataclass
class HandRecord:
    variant: str
    antes: List[float]
    blinds: List[float]
    starting_stacks: List[float]
    actions: List[str]
    min_bet: Optional[float] = None
    small_bet: Optional[float] = None
    big_bet: Optional[float] = None
    bring_in: Optional[float] = None
    ante_trimming_status: bool = False
    players: List[str] = field(default_factory=list)
    finishing_stacks: Optional[List[float]] = None
    winnings: Optional[List[float]] = None
    seats: Optional[List[int]] = None
    seat_count: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    index: Optional[int] = None

    @property
    def n(self) -> int:
        return len(self.starting_stacks)

    @property
    def label(self) -> str:
        h = self.meta.get("hand")
        base = os.path.basename(self.source) if self.source else "hand"
        if h is not None:
            return f"{base} #{h}"
        if self.index is not None:
            return f"{base} [{self.index}]"
        return base

    def player_name(self, i: int) -> str:
        if i < len(self.players) and self.players[i]:
            return str(self.players[i])
        return f"p{i + 1}"

    def parsed_actions(self) -> List[ParsedAction]:
        out = []
        for a in self.actions:
            pa = parse_action(a)
            if pa is not None:
                out.append(pa)
        return out

    def summary(self) -> Dict[str, Any]:
        """Light-weight description used by the UI's hand list."""
        n_board = 0
        went_to_showdown = False
        for a in self.parsed_actions():
            if a.is_dealer and a.verb == "db":
                n_board += len(a.args[0]) // 2 if a.args else 0
            if not a.is_dealer and a.verb == "sm":
                went_to_showdown = True
        known_holes = sum(1 for a in self.parsed_actions() if a.is_dealer and a.verb == "dh"
                          and a.args and "?" not in a.args[-1])
        return {
            "index": self.index,
            "label": self.label,
            "variant": self.variant,
            "players": [self.player_name(i) for i in range(self.n)],
            "n": self.n,
            "blinds": self.blinds,
            "antes": self.antes,
            "starting_stacks": [None if s == float("inf") else s for s in self.starting_stacks],
            "n_actions": len(self.actions),
            "board_cards": n_board,
            "showdown": went_to_showdown,
            "known_hole_cards": known_holes,
            "winnings": self.winnings,
            "finishing_stacks": self.finishing_stacks,
            "meta": _json_safe(self.meta),
        }

    def to_phh(self) -> str:
        return dump_hand(self)


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, float):
        if v == float("inf") or v == float("-inf") or v != v:
            return None
        return v
    if isinstance(v, (_dt.time, _dt.date, _dt.datetime)):
        return v.isoformat()
    return v


def _floats(v: Any) -> List[float]:
    return [float(x) if x is not None else float("inf") for x in (v or [])]


def record_from_dict(d: Dict[str, Any], source: Optional[str] = None, index: Optional[int] = None) -> HandRecord:
    known = {"variant", "antes", "blinds_or_straddles", "starting_stacks", "actions", "min_bet", "small_bet",
             "big_bet", "bring_in", "ante_trimming_status", "players", "finishing_stacks", "winnings", "seats",
             "seat_count"}
    for req in ("variant", "antes", "starting_stacks", "actions"):
        if req not in d:
            raise ValueError(f"missing required field {req!r}")
    meta = {k: v for k, v in d.items() if k not in known}
    rec = HandRecord(
        variant=str(d["variant"]),
        antes=_floats(d.get("antes")),
        blinds=_floats(d.get("blinds_or_straddles")),
        starting_stacks=_floats(d.get("starting_stacks")),
        actions=[str(a) for a in d["actions"]],
        min_bet=float(d["min_bet"]) if d.get("min_bet") is not None else None,
        small_bet=float(d["small_bet"]) if d.get("small_bet") is not None else None,
        big_bet=float(d["big_bet"]) if d.get("big_bet") is not None else None,
        bring_in=float(d["bring_in"]) if d.get("bring_in") is not None else None,
        ante_trimming_status=bool(d.get("ante_trimming_status", False)),
        players=[str(p) for p in d.get("players", [])],
        finishing_stacks=_floats(d["finishing_stacks"]) if d.get("finishing_stacks") is not None else None,
        winnings=_floats(d["winnings"]) if d.get("winnings") is not None else None,
        seats=[int(s) for s in d["seats"]] if d.get("seats") is not None else None,
        seat_count=int(d["seat_count"]) if d.get("seat_count") is not None else None,
        meta=meta,
        source=source,
        index=index,
    )
    if not rec.blinds:
        rec.blinds = [0.0] * rec.n
    return rec


# ----------------------------------------------------------------- lenient fallback
def _strip_comment(line: str) -> str:
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _lenient_parse(text: str) -> Dict[str, Any]:
    """Best-effort parser for PHH files that tomllib rejects."""
    import ast

    joined = "\n".join(_strip_comment(line) for line in text.splitlines())
    data: Dict[str, Any] = {}
    key_re = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*=\s*(.*)$", re.S)
    buf = ""
    key = None
    for line in joined.splitlines():
        if key is None:
            m = key_re.match(line)
            if not m:
                continue
            key, buf = m.group(1), m.group(2).strip()
        else:
            buf += "\n" + line.strip()
        if buf.count("[") > buf.count("]"):
            continue
        value = buf.strip()
        if re.fullmatch(r"\d{2}:\d{2}:\d{2}(\.\d+)?", value):
            data[key] = value
        else:
            py = re.sub(r"\btrue\b", "True", value)
            py = re.sub(r"\bfalse\b", "False", py)
            # literal_eval accepts no calls: 1e999 overflows to +inf, which is what we want
            py = re.sub(r"(?<![\w.])inf\b", "1e999", py)
            py = re.sub(r"(?<![\w.])nan\b", "None", py)
            try:
                data[key] = ast.literal_eval(py) if py else None
            except Exception:
                data[key] = value.strip("'\"")
        key, buf = None, ""
    return data


def parse_hand_text(text: str, source: Optional[str] = None, index: Optional[int] = None) -> HandRecord:
    try:
        d = tomllib.loads(text)
    except Exception:
        d = _lenient_parse(text)
    return record_from_dict(d, source, index)


def iter_hands_text(text: str, source: Optional[str] = None) -> Iterator[HandRecord]:
    """Yield hands from a ``.phh`` (single) or ``.phhs`` (multi) document."""
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        yield parse_hand_text(text, source, None)
        return
    for k, m in enumerate(headers):
        start = m.end()
        end = headers[k + 1].start() if k + 1 < len(headers) else len(text)
        body = text[start:end]
        if not body.strip():
            continue
        name = m.group(1).strip().strip("'\"")
        try:
            idx: Optional[int] = int(name)
        except ValueError:
            idx = k + 1
        yield parse_hand_text(body, source, idx)


def load_file(path: str, limit: Optional[int] = None) -> List[HandRecord]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    out: List[HandRecord] = []
    for rec in iter_hands_text(text, source=path):
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_text(text: str, source: Optional[str] = None, limit: Optional[int] = None) -> List[HandRecord]:
    out: List[HandRecord] = []
    for rec in iter_hands_text(text, source=source):
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------------ writing
def _fmt_num(x: Any) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        if x == float("inf"):
            return "inf"
        if x.is_integer():
            return str(int(x))
        return f"{x:.2f}".rstrip("0").rstrip(".") if abs(x) < 1e15 else repr(x)
    return str(x)


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _fmt_num(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "\\'") + "'" if "'" not in v else '"' + v.replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    if isinstance(v, (_dt.time, _dt.date, _dt.datetime)):
        return v.isoformat()
    return _fmt_value(str(v))


def dump_hand(rec: HandRecord) -> str:
    lines = [f"variant = {_fmt_value(rec.variant)}",
             f"ante_trimming_status = {_fmt_value(rec.ante_trimming_status)}",
             f"antes = {_fmt_value(rec.antes)}"]
    if rec.bring_in is not None:
        lines.append(f"bring_in = {_fmt_num(rec.bring_in)}")
    else:
        lines.append(f"blinds_or_straddles = {_fmt_value(rec.blinds)}")
    if rec.small_bet is not None:
        lines.append(f"small_bet = {_fmt_num(rec.small_bet)}")
    if rec.big_bet is not None:
        lines.append(f"big_bet = {_fmt_num(rec.big_bet)}")
    if rec.min_bet is not None and rec.small_bet is None:
        lines.append(f"min_bet = {_fmt_num(rec.min_bet)}")
    lines.append(f"starting_stacks = {_fmt_value(rec.starting_stacks)}")
    lines.append("actions = [")
    for a in rec.actions:
        lines.append(f"  {_fmt_value(a)},")
    lines.append("]")
    for k, v in rec.meta.items():
        if k == "time" and isinstance(v, str):
            lines.append(f"{k} = {v}")
        else:
            lines.append(f"{k} = {_fmt_value(v)}")
    if rec.seats:
        lines.append(f"seats = {_fmt_value(rec.seats)}")
    if rec.seat_count:
        lines.append(f"seat_count = {rec.seat_count}")
    if rec.players:
        lines.append(f"players = {_fmt_value(rec.players)}")
    if rec.finishing_stacks is not None:
        lines.append(f"finishing_stacks = {_fmt_value(rec.finishing_stacks)}")
    if rec.winnings is not None:
        lines.append(f"winnings = {_fmt_value(rec.winnings)}")
    return "\n".join(lines) + "\n"


def dump_hands(records: Sequence[HandRecord]) -> str:
    parts = []
    for i, rec in enumerate(records, start=1):
        parts.append(f"[{i}]\n{dump_hand(rec)}")
    return "\n".join(parts)
