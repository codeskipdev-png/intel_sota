"""ONNXRuntime scoring: chunk list -> bot risk in [0,1]."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from poker_detect.features.extractor import HAND_FEATURE_DIM, FEATURE_SPEC_VERSION, hand_feature_vector


class OnnxHandScorer:
    def __init__(self, onnx_path: Path, preprocess_path: Optional[Path] = None):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if preprocess_path is None:
            preprocess_path = onnx_path.with_suffix(".preprocess.json")
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        meta: Dict[str, Any] = {}
        if preprocess_path.exists():
            meta = json.loads(preprocess_path.read_text(encoding="utf-8"))
        self.feature_spec_version = int(meta.get("feature_spec_version", FEATURE_SPEC_VERSION))
        self.hand_feature_dim = int(meta.get("hand_feature_dim", HAND_FEATURE_DIM))
        if self.feature_spec_version != FEATURE_SPEC_VERSION:
            raise ValueError(
                f"ONNX preprocess feature_spec_version={self.feature_spec_version} "
                f"!= runtime {FEATURE_SPEC_VERSION}"
            )

    def score_hand(self, hand: Dict[str, Any]) -> float:
        x = hand_feature_vector(hand).astype(np.float32)
        if x.shape[0] != self.hand_feature_dim:
            raise ValueError(f"feature dim {x.shape[0]} != expected {self.hand_feature_dim}")
        x = x.reshape(1, -1)
        (logit,) = self._session.run(None, {"hand_features": x})
        y_score = torch.sigmoid(torch.tensor(logit)).numpy().reshape(-1)[0]
        return float(max(0.0, min(1.0, y_score)))
