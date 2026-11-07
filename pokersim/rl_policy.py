"""The RL advisor's policy at play time: a small MLP evaluated with NumPy (no PyTorch needed to serve).

Training (``pokersim.rl_train``, PyTorch on a GPU) exports ``data/rl/policy.npz`` + ``data/rl/policy.json``;
this module loads them and turns an observation into action probabilities and a value estimate.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np

from . import hand_range as HR
from .rl_features import ACTIONS, ACTIONS_JA, OBS_DIM, action_targets, encode

RL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rl")
POLICY_NPZ = os.path.join(RL_DIR, "policy.npz")
POLICY_META = os.path.join(RL_DIR, "policy.json")
HIDDEN = (512, 512, 256)


class NumpyPolicy:
    """Same layers as ``rl_train.PolicyNet``: Linear+LayerNorm+ReLU trunk, policy and value heads."""

    def __init__(self, params: Dict[str, np.ndarray], meta: Optional[Dict[str, Any]] = None):
        self.p = {k: v.astype(np.float32) for k, v in params.items()}
        self.meta = meta or {}
        self.n_layers = sum(1 for k in self.p if k.startswith("trunk.") and k.endswith(".w"))
        if self.p["trunk.0.w"].shape[1] != OBS_DIM:
            raise ValueError(f"policy expects {self.p['trunk.0.w'].shape[1]} features, encoder makes {OBS_DIM}")

    @classmethod
    def load(cls, npz: str = POLICY_NPZ, meta: str = POLICY_META) -> "NumpyPolicy":
        with np.load(npz) as z:
            params = {k: z[k] for k in z.files}
        m = {}
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                m = json.load(f)
        return cls(params, m)

    def forward(self, x: np.ndarray):
        h = x.astype(np.float32)
        for i in range(self.n_layers):
            h = h @ self.p[f"trunk.{i}.w"].T + self.p[f"trunk.{i}.b"]
            mu = h.mean(axis=-1, keepdims=True)
            var = h.var(axis=-1, keepdims=True)
            h = (h - mu) / np.sqrt(var + 1e-5) * self.p[f"trunk.{i}.ln_w"] + self.p[f"trunk.{i}.ln_b"]
            h = np.maximum(h, 0.0)
        logits = h @ self.p["pi.w"].T + self.p["pi.b"]
        value = h @ self.p["v.w"].T + self.p["v.b"]
        return logits, value[..., 0]

    def probs(self, x: np.ndarray, mask: np.ndarray):
        logits, value = self.forward(x[None, :])
        logits = np.where(mask, logits[0], -1e9)
        e = np.exp(logits - logits.max())
        pr = e / e.sum()
        return pr, float(value[0])


_cached: Optional[NumpyPolicy] = None
_cached_mtime: float = -1.0


def load_policy() -> Optional[NumpyPolicy]:
    """The exported policy, reloaded when the file changes; None if no model has been trained yet."""
    global _cached, _cached_mtime
    if not os.path.exists(POLICY_NPZ):
        return None
    mt = os.path.getmtime(POLICY_NPZ)
    if _cached is None or mt != _cached_mtime:
        _cached = NumpyPolicy.load()
        _cached_mtime = mt
    return _cached


def recommend(obs: Dict[str, Any], styles: Dict[int, Dict[str, Any]], policy: Optional[NumpyPolicy] = None,
              seed: int = 0) -> Dict[str, Any]:
    """The RL policy's advice for the player to act: probabilities over the legal actions and the chosen one."""
    policy = policy or load_policy()
    if policy is None:
        return {"available": False, "reason_ja": "RLモデルがまだ学習・配置されていません（data/rl/policy.npz）。"}
    rng = random.Random(f"{obs.get('hand_id')}|{obs['street']}|{len(obs.get('history') or [])}|{seed}")
    x = encode(obs, styles, rng=rng)
    mask, targets = action_targets(obs)
    pr, value = policy.probs(x, mask)
    order = [int(i) for i in np.argsort(-pr) if mask[i]]
    best = order[0]
    bb = float(obs.get("min_bet") or 1.0)
    live = [p for p in obs["players"] if not p["folded"] and not p["is_hero"]]
    ranges = HR.predict_ranges(obs, [p["index"] for p in live])
    return {
        "ranges": [{"index": p["index"], "name": p["name"], "grid": HR.grid(ranges[p["index"]]),
                    "top": [{"hand": HR.CLASS_NAMES[k], "p": round(float(ranges[p["index"]][k]), 4)}
                            for k in np.argsort(-ranges[p["index"]])[:8]]}
                   for p in live if p["index"] in ranges],
        "available": True,
        "action_index": best, "action": ACTIONS[best], "action_ja": ACTIONS_JA[best], "engine_action": targets[best],
        "probs": [{"action": ACTIONS[i], "action_ja": ACTIONS_JA[i], "p": round(float(pr[i]), 4),
                   "amount": (targets[i] or {}).get("amount")} for i in order],
        "value_bb": round(value * 10.0, 2),  # training reward is net big blinds / 10
        "value_chips": round(value * 10.0 * bb, 2),
        "model": {k: policy.meta.get(k) for k in ("trained_at", "updates", "hands", "steps", "seconds", "device", "eval")},
    }


def act(obs: Dict[str, Any], styles: Dict[int, Dict[str, Any]], policy: NumpyPolicy, rng: random.Random,
        greedy: bool = False) -> Dict[str, Any]:
    x = encode(obs, styles, rng=rng)
    mask, targets = action_targets(obs)
    pr, _ = policy.probs(x, mask)
    i = int(np.argmax(pr)) if greedy else int(rng.choices(range(len(pr)), weights=pr)[0])
    return targets[i] or {"action": "call"}
