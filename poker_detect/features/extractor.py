"""Fixed-size feature vectors from sanitized Poker44 hand JSON (miner-visible schema)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np

FEATURE_SPEC_VERSION = 1
HAND_FEATURE_DIM = 20

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def hand_feature_vector(hand: Dict[str, Any]) -> np.ndarray:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []

    counts = Counter(
        str(a.get("action_type") or "other").strip().lower() for a in actions if isinstance(a, dict)
    )
    n_act = max(1, len(actions))

    def ratio(key: str) -> float:
        return counts.get(key, 0) / n_act

    vec: List[float] = [
        ratio("call"),
        ratio("check"),
        ratio("bet"),
        ratio("raise"),
        ratio("fold"),
        ratio("all_in"),
        ratio("small_blind"),
        ratio("big_blind"),
        ratio("ante"),
        ratio("other"),
        _clamp01(len(streets) / 4.0),
        _clamp01(min(len(players), 8) / 8.0),
    ]

    amts = [
        float(a.get("normalized_amount_bb") or 0.0)
        for a in actions
        if isinstance(a, dict)
    ]
    pots = [float(a.get("pot_after") or 0.0) for a in actions if isinstance(a, dict)]

    vec.append(_clamp01((np.mean(amts) if amts else 0.0) / 80.0))
    vec.append(_clamp01((np.std(amts) if amts else 0.0) / 80.0))
    vec.append(_clamp01((np.max(amts) if amts else 0.0) / 80.0))
    vec.append(_clamp01((np.mean(pots) if pots else 0.0) / 200.0))
    vec.append(_clamp01((np.max(pots) if pots else 0.0) / 200.0))
    vec.append(_clamp01(len(actions) / 12.0))

    while len(vec) < HAND_FEATURE_DIM:
        vec.append(0.0)
    vec = vec[:HAND_FEATURE_DIM]

    return np.asarray(vec, dtype=np.float32)

