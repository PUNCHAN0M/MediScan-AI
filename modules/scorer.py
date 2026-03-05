# models/scorer.py
"""
PatchCore Scorer — anomaly scoring via FAISS inner product.
===========================================================
Pure compute — no I/O, no loops over datasets.

Performance notes:
    - Auto GPU-FAISS when CUDA + faiss-gpu available
    - IVF index for banks > IVF_THRESHOLD patches (faster search)
    - Queries assumed L2-normalised when built from normalised bank
    - Pre-allocated contiguous buffers for repeated searches
"""
from __future__ import annotations

import cv2
import faiss
import numpy as np
from typing import Optional

# ── Detect GPU FAISS once at import ──
_FAISS_HAS_GPU: bool = hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0


class PatchCoreScorer:
    """
    Score query patches against a memory bank index.

    Uses cosine similarity (inner product on L2-normalised vectors).
    ``assume_normalized=False`` will L2-normalise queries automatically.
    """

    # Banks larger than this use IVF for faster search
    IVF_THRESHOLD: int = 10_000
    IVF_NPROBE: int = 8           # search quality (more = slower but better)
    IVF_NLIST_FACTOR: int = 16    # n_lists = n_patches // factor (capped)

    def __init__(
        self,
        k_nearest: int = 3,
        assume_normalized: bool = False,
        use_gpu: Optional[bool] = None,
        gpu_id: int = 0,
    ):
        self.k = k_nearest
        self.assume_normalized = assume_normalized
        self.gpu_id = gpu_id

        # Auto GPU: True if hardware supports it, unless explicitly False
        if use_gpu is None:
            self.use_gpu = _FAISS_HAS_GPU
        else:
            self.use_gpu = use_gpu and _FAISS_HAS_GPU

        self._index: Optional[faiss.Index] = None
        self._gpu_res: Optional[faiss.StandardGpuResources] = None

    # ─────────────────── Build ───────────────────
    def build_index(self, memory_bank: np.ndarray) -> faiss.Index:
        """
        Build FAISS inner-product index from a memory bank.

        Uses IVF for large banks (>IVF_THRESHOLD) for ~5-10× faster search
        with minimal accuracy loss.
        """
        bank = memory_bank
        if bank.dtype != np.float32:
            bank = bank.astype(np.float32, copy=False)
        if not bank.flags["C_CONTIGUOUS"]:
            bank = np.ascontiguousarray(bank)
        if not self.assume_normalized:
            faiss.normalize_L2(bank)

        n, d = bank.shape

        # ── Choose index type based on bank size ──
        if n > self.IVF_THRESHOLD:
            n_lists = min(max(n // self.IVF_NLIST_FACTOR, 16), 256)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, n_lists, faiss.METRIC_INNER_PRODUCT)
            index.train(bank)
            index.add(bank)
            index.nprobe = self.IVF_NPROBE
        else:
            index = faiss.IndexFlatIP(d)
            index.add(bank)

        # ── Move to GPU if available ──
        if self.use_gpu:
            if self._gpu_res is None:
                self._gpu_res = faiss.StandardGpuResources()
                self._gpu_res.setTempMemory(64 * 1024 * 1024)  # 64 MB temp
            index = faiss.index_cpu_to_gpu(self._gpu_res, self.gpu_id, index)

        self._index = index
        return index

    # ─────────────────── Per-patch ───────────────────
    def score_patches(
        self,
        patches: np.ndarray,
        index: Optional[faiss.Index] = None,
    ) -> np.ndarray:
        """
        Score each patch against the memory bank.

        Returns 1D array of anomaly scores (1 - mean_cosine_sim).
        """
        idx = index or self._index
        if idx is None:
            raise ValueError("No FAISS index — call build_index() first")

        if patches.dtype != np.float32:
            patches = patches.astype(np.float32, copy=False)
        if not patches.flags["C_CONTIGUOUS"]:
            patches = np.ascontiguousarray(patches)
        if not self.assume_normalized:
            faiss.normalize_L2(patches)

        sim, _ = idx.search(patches, self.k)
        # In-place mean across k-nearest (avoids temp array allocation)
        return np.subtract(1.0, sim.mean(axis=1))

    # ─────────────────── Per-pill ───────────────────
    def score_pill(
        self,
        patches: np.ndarray,
        index: Optional[faiss.Index] = None,
        method: str = "max",
    ) -> float:
        if patches is None or patches.shape[0] == 0:
            return 0.0

        scores = self.score_patches(patches, index)
        n = len(scores)

        if method == "max":
            return float(scores.max())
        if method == "top1_mean":
            k = max(1, int(n * 0.01))
            return float(np.partition(scores, -k)[-k:].mean())
        if method == "top5_mean":
            k = max(1, int(n * 0.05))
            return float(np.partition(scores, -k)[-k:].mean())
        if method == "top10_mean":
            k = max(1, int(n * 0.10))
            return float(np.partition(scores, -k)[-k:].mean())
        return float(scores.max())

    # ─────────────────── Heatmap ───────────────────
    def score_heatmap(
        self,
        patches: np.ndarray,
        grid_size: int,
        image_size: tuple = (256, 256),
        index: Optional[faiss.Index] = None,
    ) -> np.ndarray:
        scores = self.score_patches(patches, index)
        expected = grid_size * grid_size
        if len(scores) != expected:
            raise ValueError(
                f"Patch count {len(scores)} != grid_size² ({expected})"
            )
        hm = scores.reshape(grid_size, grid_size)
        return cv2.resize(
            hm, (image_size[1], image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
