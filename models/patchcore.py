# models/patchcore.py
"""
PatchCore — high-level wrapper combining extractor + bank + scorer.
===================================================================
Pure compute convenience class. No I/O loops.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from models.feature_extractor import ResNet50FeatureExtractor
from models.memory_bank import MemoryBank
from models.scorer import PatchCoreScorer


class PatchCore:
    """
    Facade over the three PatchCore components.

    Use this to build a complete model from config, or load a saved
    model for inference. The pipeline layer uses this class to avoid
    touching individual components directly.
    """

    def __init__(
        self,
        extractor: ResNet50FeatureExtractor,
        k_nearest: int = 3,
        score_method: str = "max",
        coreset_ratio: float = 0.25,
        seed: int = 42,
        fallback_threshold: float = 0.50,
    ):
        self.extractor = extractor
        self.k_nearest = k_nearest
        self.score_method = score_method
        self.coreset_ratio = coreset_ratio
        self.seed = seed
        self.fallback_threshold = fallback_threshold

        self.scorer = PatchCoreScorer(
            k_nearest=k_nearest,
            assume_normalized=False,
        )
        self.bank = MemoryBank()

    # ───────── Build from images (called by pipeline) ─────────
    def add_patches(self, patches: np.ndarray) -> None:
        """Accumulate patches into the memory bank."""
        self.bank.add(patches)

    def build(self) -> np.ndarray:
        """Build coreset from accumulated patches."""
        return self.bank.build(
            coreset_ratio=self.coreset_ratio,
            seed=self.seed,
        )

    def save(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save memory bank + metadata to disk."""
        self.bank.save(path, metadata=metadata)

    # ───────── Load for inference ─────────
    def load_index(self, path: Path) -> Tuple[float, Dict[str, Any]]:
        """
        Load a saved model file and build FAISS index.

        Returns (threshold, metadata).
        """
        bank_array, meta = MemoryBank.load(path)
        self.scorer.build_index(bank_array)
        threshold = float(meta.get("threshold", self.fallback_threshold))
        return threshold, meta

    # ───────── Score ─────────
    def score(self, patches: np.ndarray) -> float:
        """Score a single pill's patches."""
        return self.scorer.score_pill(
            patches, method=self.score_method,
        )

    def score_heatmap(
        self,
        patches: np.ndarray,
        grid_size: int,
        image_size: tuple = (256, 256),
    ) -> np.ndarray:
        """Get anomaly heatmap for visualization."""
        return self.scorer.score_heatmap(patches, grid_size, image_size)

    # ───────── Metadata helper ─────────
    def make_metadata(self, **extra) -> Dict[str, Any]:
        """Build standard metadata dict from current config."""
        meta = {
            "img_size": self.extractor.img_size,
            "grid_size": self.extractor.grid_size,
            "feature_dim": self.extractor.feature_dim,
            "k_nearest": self.k_nearest,
            "score_method": self.score_method,
            "coreset_ratio": self.coreset_ratio,
            "use_color_features": self.extractor.use_color_features,
            "use_hsv": self.extractor.use_hsv,
            "color_weight": self.extractor.color_weight,
        }
        meta.update(extra)
        return meta
