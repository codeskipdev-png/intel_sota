"""Fixed-size feature vectors from sanitized Poker44 hand JSON (miner-visible schema)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np

FEATURE_SPEC_VERSION = 1
# HAND_FEATURE_DIM = 20
HAND_FEATURE_DIM = 17
EPS = 1e-10

CHUNK_FEATURE_DIM = HAND_FEATURE_DIM * 4 + 1


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _mean_std_over_max(vals: List[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = float(np.max(vals))
    d = max(m, EPS)
    return float(np.mean(vals) / d), float(np.std(vals) / d)


def hand_feature_vector(hand: Dict[str, Any]) -> np.ndarray:
    """
    Map one sanitized hand dict to a fixed-length float vector.
    Must stay in sync with training export and ONNX miner inference.
    """
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []

    counts = Counter(
        str(a.get("action_type") or "other").strip().lower() for a in actions if isinstance(a, dict)
    )
    n_act = max(1, len(actions))

    def ratio(key: str) -> float:
        return counts.get(key, 0) / n_act

    # vec: List[float] = [
    #     ratio("call"),
    #     ratio("check"),
    #     ratio("bet"),
    #     ratio("raise"),
    #     ratio("fold"),
    #     ratio("all_in"),
    #     ratio("small_blind"),
    #     ratio("big_blind"),
    #     ratio("ante"),
    #     ratio("other"),
    #     _clamp01(len(streets) / 4.0),
    #     _clamp01(min(len(players), 8) / 8.0),
    # ]

    vec: List[float] = [
        ratio("call"),
        ratio("check"),
        # ratio("bet"),
        ratio("raise"),
        ratio("fold"),
        # ratio("all_in"),
        # ratio("small_blind"),
        # ratio("big_blind"),
        # ratio("ante"),
        ratio("other"),
        _clamp01(len(counts) / 10.0),
        _clamp01(len(actions) / 20.0),
        _clamp01(len(streets) / 4.0),
        _clamp01(min(len(players), 8) / 8.0),
    ]

    # vec: List[float] = [
    #     _clamp01(len(counts) / 10.0),
    #     _clamp01(len(actions) / 20.0),
    #     _clamp01(len(streets) / 4.0),
    #     _clamp01(min(len(players), 8) / 8.0),
    # ]
    amts, pots_before, pots_after, amounts = [], [], [], []
    for a in actions:
        if isinstance(a, dict):
            amts.append(float(a.get("normalized_amount_bb") or 0.0))
            pots_before.append(float(a.get("pot_before") or 0.0))
            pots_after.append(float(a.get("pot_after") or 0.0))
            amounts.append(float(a.get("amount") or 0.0))


    mu, sd = _mean_std_over_max(amts)
    vec.append(_clamp01(mu))
    vec.append(_clamp01(sd))
    mu, sd = _mean_std_over_max(pots_before)
    vec.append(_clamp01(mu))
    vec.append(_clamp01(sd))
    mu, sd = _mean_std_over_max(pots_after)
    vec.append(_clamp01(mu))
    vec.append(_clamp01(sd))
    mu, sd = _mean_std_over_max(amounts)
    vec.append(_clamp01(mu))
    vec.append(_clamp01(sd))


    # amts = [
    #     float(a.get("normalized_amount_bb") or 0.0)
    #     for a in actions
    #     if isinstance(a, dict)
    # ]
    # pots = [float(a.get("pot_after") or 0.0) for a in actions if isinstance(a, dict)]

    # vec.append(_clamp01((np.mean(amts) if amts else 0.0) / 80.0))
    # vec.append(_clamp01((np.std(amts) if amts else 0.0) / 80.0))
    # vec.append(_clamp01((np.max(amts) if amts else 0.0) / 80.0))
    # vec.append(_clamp01((np.mean(pots) if pots else 0.0) / 200.0))
    # vec.append(_clamp01((np.max(pots) if pots else 0.0) / 200.0))
    # vec.append(_clamp01(len(actions) / 12.0))

    while len(vec) < HAND_FEATURE_DIM:
        vec.append(0.0)
    vec = vec[:HAND_FEATURE_DIM]

    return np.asarray(vec, dtype=np.float32)


def chunk_feature_vector(hands: List[Dict[str, Any]]) -> np.ndarray:
    """Aggregate hand vectors: mean, std, min, max per dim + normalized chunk size."""
    if not hands:
        return np.zeros(CHUNK_FEATURE_DIM, dtype=np.float32)

    mat = np.stack([hand_feature_vector(h) for h in hands], axis=0)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    vmin = mat.min(axis=0)
    vmax = mat.max(axis=0)
    n_norm = np.array([_clamp01(len(hands) / 120.0)], dtype=np.float32)

    out = np.concatenate([mean, std, vmin, vmax, n_norm]).astype(np.float32)
    assert out.shape[0] == CHUNK_FEATURE_DIM
    return out
