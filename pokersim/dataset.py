"""Access to the ``gabbler/PokerData`` Hugging Face dataset.

The dataset is ~16 GB of ``.phh``/``.phhs`` files, so everything is fetched on
demand and cached under ``data/cache``.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from . import phh

DATASET = "gabbler/PokerData"
HF_API = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/"
HF_RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"
HF_PAGE = f"https://huggingface.co/datasets/{DATASET}"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT, "data", "cache")
SAMPLES_DIR = os.path.join(ROOT, "data", "samples")

# A few curated entry points into the dataset.
FEATURED: List[Dict[str, str]] = [
    {"path": "dwan-ivey-2009.phh", "title": "Dwan vs Ivey, the first televised $1M pot (NLHE)"},
    {"path": "antonius-blom-2009.phh", "title": "Antonius vs Blom, largest online pot (PLO)"},
    {"path": "phua-xuan-2019.phh", "title": "Phua vs Xuan, Triton London 2019 (short deck)"},
    {"path": "pluribus/30/0.phh", "title": "Pluribus vs pros (Brown & Sandholm 2019), session 30 hand 0"},
    {"path": "wsop/2023/43/5/1.phh", "title": "2023 WSOP $50k PPC final table, hand 1 (NLHE)"},
    {"path": "handhq/PS-2009-07-01_2009-07-23_200NLH_OBFU/2/ps NLH handhq_1-OBFUSCATED.phhs",
     "title": "PokerStars 200NL, 1000 anonymised cash-game hands (2009)"},
    {"path": "handhq/PTY-2009-07-01_2009-07-23_400NLH_OBFU/4/pty NLH handhq_1-OBFUSCATED.phhs",
     "title": "PartyPoker 400NL, 995 anonymised cash-game hands (2009)"},
    {"path": "handhq/ABS-2009-07-01_2009-07-23_1000NLH_OBFU/10/abs NLH handhq_1-OBFUSCATED.phhs",
     "title": "Absolute Poker 1000NL (with antes), 999 hands (2009)"},
]


def _get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pokersim/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_dir(path: str = "") -> List[Dict[str, Any]]:
    """List a folder of the dataset repository."""
    path = path.strip("/")
    url = HF_API + urllib.parse.quote(path)
    try:
        data = json.loads(_get(url).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"{path!r}: HTTP {exc.code}") from exc
    out = []
    for item in data:
        p = item.get("path", "")
        name = p.rsplit("/", 1)[-1]
        if name.startswith("."):
            continue
        out.append({"path": p, "name": name, "type": item.get("type"), "size": item.get("size")})
    out.sort(key=lambda x: (x["type"] != "directory", _natural_key(x["name"])))
    return out


def _natural_key(s: str) -> List[Any]:
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def cache_path(path: str) -> str:
    safe = path.strip("/").replace("..", "_")
    return os.path.join(CACHE_DIR, *safe.split("/"))


def download(path: str, force: bool = False) -> str:
    """Download one dataset file (cached) and return the local path."""
    local = cache_path(path)
    if os.path.exists(local) and not force and os.path.getsize(local) > 0:
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    url = HF_RESOLVE + urllib.parse.quote(path.strip("/"))
    try:
        data = _get(url, timeout=300)
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"{path!r}: HTTP {exc.code}") from exc
    with open(local, "wb") as f:
        f.write(data)
    return local


def fetch_hands(path: str, limit: Optional[int] = None) -> List[phh.HandRecord]:
    local = download(path)
    recs = phh.load_file(local, limit=limit)
    for r in recs:
        r.source = path
    return recs


def list_samples() -> List[Dict[str, Any]]:
    out = []
    if os.path.isdir(SAMPLES_DIR):
        for name in sorted(os.listdir(SAMPLES_DIR)):
            if name.endswith((".phh", ".phhs")):
                full = os.path.join(SAMPLES_DIR, name)
                out.append({"name": name, "path": full, "size": os.path.getsize(full)})
    return out


def list_cached() -> List[Dict[str, Any]]:
    out = []
    if os.path.isdir(CACHE_DIR):
        for root, _d, files in os.walk(CACHE_DIR):
            for f in files:
                if f.endswith((".phh", ".phhs")):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, CACHE_DIR).replace(os.sep, "/")
                    out.append({"name": rel, "path": full, "size": os.path.getsize(full)})
    out.sort(key=lambda x: x["name"])
    return out
