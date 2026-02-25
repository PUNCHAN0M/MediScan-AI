"""
Memory Bank + Coreset Sampling
==============================

Accumulate patch features → coreset subsample → L2 normalize → save / load.

Usage
-----
::

    bank = MemoryBank()
    for img in good_images:
        patches = extractor.extract(img)   # (grid², D)
        bank.add(patches)

    bank.build(coreset_ratio=0.10)         # → ~15 k–20 k patches
    bank.save("model/patchcore_resnet/cls.pth", meta={...})

    # later
    array, meta = MemoryBank.load("model/patchcore_resnet/cls.pth")
"""
from __future__ import annotations

import faiss
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────
#  Coreset Sampler
# ─────────────────────────────────────────────────────────
class CoresetSampler:
    """Random coreset subsampling for a patch bank."""

    @staticmethod
    def subsample(
        bank: np.ndarray,
        ratio: float = 0.10,
        seed: int = 42,
        min_keep: int = 1_000,
        max_keep: int = 20_000,
    ) -> np.ndarray:
        """
        Randomly keep *ratio* of patches.

        Target patch count ≈ 15 k–20 k  (max_keep).
        If the bank is already ≤ *min_keep* patches, return as-is.
        """
        n = bank.shape[0]
        if n <= min_keep:
            return bank

        n_sel = int(n * ratio)
        n_sel = max(min_keep, min(n_sel, max_keep, n))

        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n_sel, replace=False)
        return np.ascontiguousarray(bank[idx])


# ─────────────────────────────────────────────────────────
#  Memory Bank
# ─────────────────────────────────────────────────────────
class MemoryBank:
    """
    PatchCore memory bank.

    Lifecycle
    ---------
    ``add()`` → ``build()`` → ``save()``

    ``load()`` returns a raw numpy array + metadata dict.
    """

    def __init__(self) -> None:
        self._patches: list[np.ndarray] = []
        self._bank: Optional[np.ndarray] = None
        self._built = False

    # ── accumulate ──
    def add(self, patches: np.ndarray) -> None:
        """Append patches from one pill / image."""
        if patches is not None and patches.shape[0] > 0:
            self._patches.append(patches.astype(np.float32))

    # ── properties ──
    @property
    def total_patches(self) -> int:
        if self._built:
            return self._bank.shape[0] if self._bank is not None else 0
        return sum(p.shape[0] for p in self._patches)

    @property
    def feature_dim(self) -> int:
        if self._built and self._bank is not None:
            return self._bank.shape[1]
        if self._patches:
            return self._patches[0].shape[1]
        return 0

    @property
    def bank(self) -> Optional[np.ndarray]:
        return self._bank

    # ── build ──
    def build(
        self,
        coreset_ratio: float = 0.10,
        seed: int = 42,
        min_keep: int = 1_000,
        max_keep: int = 20_000,
    ) -> np.ndarray:
        """
        Concat → coreset subsample → L2 normalize.

        Returns the finalized ``(N, D)`` memory bank.
        """
        if not self._patches:
            raise ValueError("No patches added — call add() first")

        raw = np.concatenate(self._patches, axis=0).astype(np.float32)
        raw = np.ascontiguousarray(raw)
        print(f"  Raw patches : {raw.shape[0]:,} × {raw.shape[1]}")

        bank = CoresetSampler.subsample(raw, coreset_ratio, seed,
                                        min_keep, max_keep)
        print(f"  After coreset (ratio={coreset_ratio}): {bank.shape[0]:,}")

        faiss.normalize_L2(bank)

        self._bank = bank
        self._built = True
        return bank

    # ── save / load ──
    def save(
        self,
        path: Path | str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save ``memory_bank`` tensor + metadata to ``.pth``."""
        if self._bank is None:
            raise ValueError("Call build() before save()")

        data: Dict[str, Any] = {
            "memory_bank": torch.from_numpy(self._bank),
            "dim": int(self._bank.shape[1]),
            "n_patches": int(self._bank.shape[0]),
        }
        if metadata:
            data.update(metadata)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(path))
        print(f"  Saved → {path}")

    @staticmethod
    def load(path: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load memory bank from ``.pth``.

        Returns
        -------
        bank : np.ndarray  (N, D) float32
        meta : dict         all remaining keys (threshold, backbone, …)
        """
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        bank = data["memory_bank"]
        if isinstance(bank, torch.Tensor):
            bank = bank.numpy()
        bank = np.ascontiguousarray(bank.astype(np.float32))

        meta = {k: v for k, v in data.items() if k != "memory_bank"}
        return bank, meta
