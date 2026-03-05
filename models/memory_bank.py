# models/memory_bank.py
"""
Memory Bank + Coreset Sampler for PatchCore
=============================================
Pure compute — no pipeline loops, no dataset logic.
"""
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
    """Subsample a large patch bank to a coreset via random selection."""

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
        return bank[idx]


# ─────────────────────────────────────────────────────────
#  Memory Bank
# ─────────────────────────────────────────────────────────
class MemoryBank:
    """
    Accumulate patches → build coreset → L2 normalize → save/load.

    Lifecycle:
        1. ``add()`` — accumulate patches (no copy)
        2. ``build()`` — coreset + L2 normalize
        3. ``save()`` — persist to disk
        4. ``load()`` — static, returns (bank_array, metadata_dict)
    """

    def __init__(self) -> None:
        self._patches: list[np.ndarray] = []
        self._bank: Optional[np.ndarray] = None
        self._built = False
        self._total_patches = 0
        self._feature_dim = 0

    # ───────── Add (no copy) ─────────
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

    # ───────── Properties ─────────
    @property
    def total_patches(self) -> int:
        return self._bank.shape[0] if self._built else self._total_patches

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def bank(self) -> Optional[np.ndarray]:
        return self._bank

    # ───────── Build ─────────
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

        raw = np.empty(
            (self._total_patches, self._feature_dim), dtype=np.float32
        )
        offset = 0
        for p in self._patches:
            n = p.shape[0]
            raw[offset:offset + n] = p
            offset += n

        self._patches.clear()

        bank = CoresetSampler.subsample(raw, coreset_ratio, seed, min_keep, max_keep)
        print(f"  After coreset: {bank.shape[0]:,}")

        faiss.normalize_L2(bank)

        self._bank = bank
        self._built = True
        return bank

    # ───────── Save ─────────
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

    # ───────── Load (static) ─────────
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

    # ══════════════════════════════════════════════════════
    #  Multi-subclass format
    # ══════════════════════════════════════════════════════
    #
    #  main_class.pth = {
    #      "format": "multi_subclass",
    #      "parent_class": "JANUMET",
    #      "shared_meta": { ... },
    #      "subclasses": {
    #          "JANUMET_back": { "memory_bank": tensor, "threshold": 0.45, ... },
    #          "JANUMET_front": { "memory_bank": tensor, "threshold": 0.52, ... },
    #      }
    #  }
    #

    @staticmethod
    def save_multi(
        path: Path | str,
        subclass_data: Dict[str, Dict[str, Any]],
        shared_meta: Optional[Dict[str, Any]] = None,
        parent_class: str = "",
    ) -> None:
        """
        Save multi-subclass memory banks into one .pth file.

        ``subclass_data`` maps subclass_name → {
            "memory_bank": np.ndarray,
            "threshold": float,
            ... (any per-subclass metadata)
        }
        """
        serialized: Dict[str, Dict[str, Any]] = {}
        for name, entry in subclass_data.items():
            s_entry = dict(entry)
            bank = s_entry.get("memory_bank")
            if bank is not None and isinstance(bank, np.ndarray):
                s_entry["memory_bank"] = torch.from_numpy(bank)
                s_entry["dim"] = int(bank.shape[1])
                s_entry["n_patches"] = int(bank.shape[0])
            serialized[name] = s_entry

        data: Dict[str, Any] = {
            "format": "multi_subclass",
            "parent_class": parent_class,
            "shared_meta": shared_meta or {},
            "subclasses": serialized,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(path))
        print(f"  Saved → {path} ({len(serialized)} subclasses)")

    @staticmethod
    def load_multi(
        path: Path | str,
    ) -> Tuple[Dict[str, Tuple[np.ndarray, Dict[str, Any]]], Dict[str, Any]]:
        """
        Load multi-subclass .pth file.

        Returns
        -------
        subclasses : dict  name → (bank_array, per_sub_meta)
        shared_meta : dict
        """
        data = torch.load(str(path), map_location="cpu")

        # ── backward compat: old single-bank format ──
        if data.get("format") != "multi_subclass":
            bank = data["memory_bank"]
            if isinstance(bank, torch.Tensor):
                bank = bank.numpy()
            if bank.dtype != np.float32:
                bank = bank.astype(np.float32, copy=False)
            if not bank.flags["C_CONTIGUOUS"]:
                bank = np.ascontiguousarray(bank)
            meta = {k: v for k, v in data.items() if k != "memory_bank"}
            sub_name = meta.get("parent_class", Path(path).stem)
            return {sub_name: (bank, meta)}, meta

        shared_meta = data.get("shared_meta", {})
        result: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

        for name, entry in data.get("subclasses", {}).items():
            bank = entry["memory_bank"]
            if isinstance(bank, torch.Tensor):
                bank = bank.numpy()
            if bank.dtype != np.float32:
                bank = bank.astype(np.float32, copy=False)
            if not bank.flags["C_CONTIGUOUS"]:
                bank = np.ascontiguousarray(bank)
            sub_meta = {k: v for k, v in entry.items() if k != "memory_bank"}
            result[name] = (bank, sub_meta)

        return result, shared_meta

    @staticmethod
    def is_multi(path: Path | str) -> bool:
        """Check if a .pth file uses multi-subclass format."""
        data = torch.load(str(path), map_location="cpu")
        return data.get("format") == "multi_subclass"
