from __future__ import annotations

import faiss
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────
#  Coreset Sampler (Fast Random)
# ─────────────────────────────────────────────────────────
class CoresetSampler:
    @staticmethod
    def subsample(
        bank: np.ndarray,
        ratio: float = 0.10,
        seed: int = 42,
        min_keep: int = 1_000,
        max_keep: int = 20_000,
    ) -> np.ndarray:

        n = bank.shape[0]

        if n <= min_keep:
            return bank

        n_sel = int(n * ratio)
        n_sel = max(min_keep, min(n_sel, max_keep, n))

        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n_sel, replace=False)

        return bank[idx]  # already contiguous if input contiguous


# ─────────────────────────────────────────────────────────
#  Memory Bank (High Performance)
# ─────────────────────────────────────────────────────────
class MemoryBank:

    def __init__(self) -> None:
        self._patches: list[np.ndarray] = []
        self._bank: Optional[np.ndarray] = None
        self._built = False
        self._total_patches = 0
        self._feature_dim = 0

    # ─────────────────────────────────────────────
    # Add patches (NO COPY)
    # ─────────────────────────────────────────────
    def add(self, patches: np.ndarray) -> None:
        if patches is None or patches.size == 0:
            return

        if patches.dtype != np.float32:
            patches = patches.astype(np.float32, copy=False)

        if not patches.flags["C_CONTIGUOUS"]:
            patches = np.ascontiguousarray(patches)

        self._patches.append(patches)
        self._total_patches += patches.shape[0]
        self._feature_dim = patches.shape[1]

    # ─────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────
    @property
    def total_patches(self) -> int:
        return self._bank.shape[0] if self._built else self._total_patches

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def bank(self) -> Optional[np.ndarray]:
        return self._bank

    # ─────────────────────────────────────────────
    # Build (FAST + Low RAM)
    # ─────────────────────────────────────────────
    def build(
        self,
        coreset_ratio: float = 0.10,
        seed: int = 42,
        min_keep: int = 1_000,
        max_keep: int = 20_000,
    ) -> np.ndarray:

        if not self._patches:
            raise ValueError("No patches added — call add() first")

        print(f"  Raw patches : {self._total_patches:,} × {self._feature_dim}")

        # 🔥 Preallocate memory (faster than np.concatenate)
        raw = np.empty(
            (self._total_patches, self._feature_dim),
            dtype=np.float32
        )

        offset = 0
        for p in self._patches:
            n = p.shape[0]
            raw[offset:offset + n] = p
            offset += n

        # free list memory early
        self._patches.clear()

        # ── coreset ──
        bank = CoresetSampler.subsample(
            raw, coreset_ratio, seed, min_keep, max_keep
        )

        print(f"  After coreset: {bank.shape[0]:,}")

        # ── normalize L2 (inplace, ultra fast C++)
        faiss.normalize_L2(bank)

        self._bank = bank
        self._built = True

        return bank

    # ─────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────
    def save(
        self,
        path: Path | str,
        memory: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        bank = memory if memory is not None else self._bank
        if bank is None:
            raise ValueError("Call build() before save()")

        data: Dict[str, Any] = {
            "memory_bank": torch.from_numpy(bank),
            "dim": int(bank.shape[1]),
            "n_patches": int(bank.shape[0]),
        }

        if metadata:
            data.update(metadata)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(path))

        print(f"  Saved → {path}")

    # ─────────────────────────────────────────────
    # Load (Zero-Copy Safe)
    # ─────────────────────────────────────────────
    @staticmethod
    def load(path: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:

        data = torch.load(str(path), map_location="cpu")

        bank = data["memory_bank"]

        if isinstance(bank, torch.Tensor):
            bank = bank.numpy()

        if bank.dtype != np.float32:
            bank = bank.astype(np.float32, copy=False)

        if not bank.flags["C_CONTIGUOUS"]:
            bank = np.ascontiguousarray(bank)

        meta = {k: v for k, v in data.items() if k != "memory_bank"}

        return bank, meta