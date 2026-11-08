"""The reserved demo player: one real human from the corpus that no model is trained or evaluated on.

Chosen by ``scripts/reserve_demo_player.py`` (written to ``data/demo/reserved_player.json``).  Every extraction script
calls :func:`row_has_reserved` and skips hands that contain this player, so the player stays unseen.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demo", "reserved_player.json")
_INFO: Optional[Dict[str, Any]] = None


def info() -> Optional[Dict[str, Any]]:
    global _INFO
    if _INFO is None and os.path.exists(PATH):
        with open(PATH, encoding="utf-8") as f:
            _INFO = json.load(f)
    return _INFO


def row_has_reserved(row: Dict[str, Any]) -> bool:
    """True if this corpus row (a Takara Parquet row) contains the reserved player."""
    d = info()
    if not d:
        return False
    return (row.get("venue") or "") == d["venue"] and d["player"] in (row.get("players") or [])
