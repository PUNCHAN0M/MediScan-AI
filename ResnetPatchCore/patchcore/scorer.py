"""
PatchCore Scorer
================

FAISS-based kNN anomaly scoring.

Score computation
-----------------
1. L2-normalize query patches
2. Search against ``IndexFlatIP`` (cosine similarity via inner product)
3. ``score_per_patch = 1 − mean(top-k similarity)``
4. ``score_per_pill  = max(patch_scores)``  *or*  ``top-1 % mean``
"""
from __future__ import annotations

import cv2
import faiss
import numpy as np
from typing import Optional


class PatchCoreScorer:
    """
    Anomaly scorer backed by a FAISS ``IndexFlatIP``.

    Parameters
    ----------
    k_nearest : int
        Number of nearest neighbours used for scoring.
    """

    def __init__(self, k_nearest: int = 3):
        self.k = k_nearest
        self._index: Optional[faiss.Index] = None

    # ── build ──
    def build_index(self, memory_bank: np.ndarray) -> faiss.Index:
        """
        Build FAISS inner-product index from an L2-normalised bank.

        Returns the index (also stored internally).
        """
        bank = np.ascontiguousarray(memory_bank.astype(np.float32))
        faiss.normalize_L2(bank)

        d = bank.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(bank)

        self._index = index
        return index

    # ── per-patch ──
    def score_patches(
        self,
        patches: np.ndarray,
        index: Optional[faiss.Index] = None,
    ) -> np.ndarray:
        """
        Per-patch anomaly scores.

        Returns ``(N_patches,)`` array — higher = more anomalous.
        """
        idx = index or self._index
        if idx is None:
            raise ValueError("No FAISS index — call build_index() first")

        patches = np.ascontiguousarray(patches.astype(np.float32))
        faiss.normalize_L2(patches)

        sim, _ = idx.search(patches, self.k)
        return 1.0 - np.mean(sim, axis=1)

    # ── per-pill ──
    def score_pill(
        self,
        patches: np.ndarray,
        index: Optional[faiss.Index] = None,
        method: str = "max",
    ) -> float:
        """
        Single anomaly score for one pill.

        Parameters
        ----------
        method : ``"max"`` | ``"top1_mean"`` | ``"top5_mean"``
            ``max``       — maximum patch score (standard PatchCore, sensitive).
            ``top1_mean`` — mean of the top 1 % patches  (~2 of 256).
            ``top5_mean`` — mean of the top 5 % patches  (~13 of 256).
                            Best balance: robust against single-patch noise
                            yet still catches localised defects.
        """
        if patches is None or patches.shape[0] == 0:
            return 0.0

        scores = self.score_patches(patches, index)

        if method == "top1_mean":
            k = max(1, int(len(scores) * 0.01))
            return float(np.sort(scores)[-k:].mean())

        if method == "top5_mean":
            k = max(1, int(len(scores) * 0.05))
            return float(np.sort(scores)[-k:].mean())
        if method == "top10_mean":
            k = max(1, int(len(scores) * 0.10))
            return float(np.sort(scores)[-k:].mean())
        return float(scores.max())

    # ── heatmap ──
    def score_heatmap(
        self,
        patches: np.ndarray,
        grid_size: int,
        image_size: tuple = (256, 256),
        index: Optional[faiss.Index] = None,
    ) -> np.ndarray:
        """
        Anomaly heatmap resized to *image_size*.

        Returns float32 (H, W) array.
        """
        scores = self.score_patches(patches, index)
        hm = scores.reshape(grid_size, grid_size).astype(np.float32)
        return cv2.resize(hm, (image_size[1], image_size[0]),
                          interpolation=cv2.INTER_LINEAR)
