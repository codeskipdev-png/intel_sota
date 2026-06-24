"""Chunk-level bot detection for HTTP servers and local scripts.

Loads exported ``model.onnx`` (+ optional ``tree_ensemble.joblib`` sidecar) and scores
each chunk independently. Predictions follow subnet scoring: ``round(risk_score)``.

Environment:

- ``POKER44_ONNX_MODEL_PATH`` — path to ``model.onnx`` (default: ``poker_detect/dist/model.onnx``)
- ``POKER44_ONNX_PREPROCESS_PATH`` — optional ``*.preprocess.json``
- ``POKER44_TREE_BUNDLE_PATH`` — optional ``tree_ensemble.joblib`` (default: next to ONNX)

Requires: ``pip install -e ".[detect]"`` or at least ``onnxruntime`` (+ ``joblib``/``xgboost`` if blended).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

_scorer = None
_scorer_lock = threading.Lock()


def _get_scorer():
    global _scorer
    if _scorer is not None:
        return _scorer
    with _scorer_lock:
        if _scorer is None:
            from poker_detect.inference.onnx_scorer import load_detection_scorer_from_env

            _scorer = load_detection_scorer_from_env()
    return _scorer


def detect_bots(
    chunks: List[List[dict[str, Any]]],
) -> Tuple[List[float], List[bool]]:
    """
    Score each chunk (list of sanitized hand dicts).

    Returns:
        ``risk_scores`` — float in [0, 1] per chunk (blended + calibrated when configured).
        ``predictions`` — ``bool(round(risk_score))`` per chunk (subnet convention).
    """
    chunk_list = chunks or []
    if not chunk_list:
        return [], []

    scorer = _get_scorer()
    risk_scores = [
        float(scorer.score_chunk([h or {} for h in chunk]))
        for chunk in chunk_list
    ]
    predictions = [bool(round(score)) for score in risk_scores]
    return risk_scores, predictions


def reset_scorer() -> None:
    """Clear cached scorer (for tests or hot reload)."""
    global _scorer
    with _scorer_lock:
        _scorer = None


if __name__ == "__main__":
    import json

    from poker44.score.scoring import reward

    path = Path(
        os.environ.get(
            "POKER44_BENCHMARK_JSON",
            "C:/Users/admin/Documents/workspace/poker/bt_tool/dataset_maker/benchmark_out/benchmark_2026-06-23.json",
        )
    )
    if not path.exists():
        raise SystemExit(f"benchmark not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sub_data in data["data"]["chunks"]:
        chunks = sub_data["chunks"]
        ground_truth = sub_data["groundTruth"]
        risk_scores, predictions = detect_bots(chunks)
        rew, metrics = reward(np.asarray(risk_scores, dtype=float), np.asarray(ground_truth))
        print("=" * 40)
        print(f"reward={rew:.4f} fpr={metrics['fpr']:.4f} recall={metrics['bot_recall']:.4f}")
        print(f"predictions={predictions}")
        print(f"groundTruth={ground_truth}")
