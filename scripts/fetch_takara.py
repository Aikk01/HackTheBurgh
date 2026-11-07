"""Download the 37 Takara poker_hands shards (pinned revision) into data/cache/takara/.

Resumable: shards whose size matches the server are skipped.  Verifies the two
SHA-256 values recorded in PLAN.md.

    python scripts/fetch_takara.py            # all shards
    python scripts/fetch_takara.py 35 36      # just these
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request

REVISION = "6acb5afb6f43082e6a468fde578890d9188be393"
BASE = f"https://huggingface.co/datasets/takara-ai/poker_hands/resolve/{REVISION}/data/"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST = os.path.join(ROOT, "data", "cache", "takara")
KNOWN_SHA = {
    36: "4f880233eb1a03d6a255986475d2be47739bf753b66dbaf0eb9516adca05401b",
    35: "0b5a566bc75c10a132480fe14d1a0804ba9c9224f0655339a8752b9aed7b9516",
}


def shard_name(i: int) -> str:
    return f"train-{i:05d}-of-00037.parquet"


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pokersim/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(i: int) -> str:
    name = shard_name(i)
    url = BASE + name
    dest = os.path.join(DEST, name)
    size = remote_size(url)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return f"{name}: cached ({size >> 20} MB)"
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "pokersim/0.1"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    msg = f"{name}: {os.path.getsize(dest) >> 20} MB in {time.time() - t0:.0f}s"
    if i in KNOWN_SHA:
        msg += " sha256 " + ("OK" if sha256(dest) == KNOWN_SHA[i] else "MISMATCH")
    return msg


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    ids = [int(a) for a in sys.argv[1:]] or list(range(37))
    failures = 0
    for i in ids:
        for attempt in range(3):
            try:
                print(fetch(i), flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{shard_name(i)}: attempt {attempt + 1} failed: {exc}", flush=True)
                time.sleep(5)
        else:
            failures += 1
    done = sum(1 for i in range(37) if os.path.exists(os.path.join(DEST, shard_name(i))))
    print(f"shards present: {done}/37, failures this run: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
