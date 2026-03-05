# pipeline/train_pipeline.py
"""
Training Pipeline — per-subclass memory bank + threshold.
==========================================================
One ``main_class.pth`` holds *separate* memory banks for each subclass.

    main_class.pth = {
        "format": "multi_subclass",
        "subclasses": {
            "sub_class1": { memory_bank, threshold, … },
            "sub_class2": { memory_bank, threshold, … },
        },
        "shared_meta": { … },
    }

Re-training a single subclass keeps the others unchanged.

Performance notes:
    - Image loading via cv2.imread (no PIL roundtrip)
    - Batch calibration with extract_batch
    - Explicit del of large intermediates for memory pressure
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple

from core.utils import list_images_recursive
from modules.feature_extractor import ResNet50FeatureExtractor
from modules.memory_bank import MemoryBank
from modules.scorer import PatchCoreScorer


class TrainPipeline:
    """
    Per-subclass training pipeline.

    Usage::

        pipeline = TrainPipeline(extractor, coreset_ratio=0.25)

        # train ALL subclasses under one parent dir
        pipeline.train_parent(parent_dir, output_path)

        # re-train ONLY one subclass (keeps the rest untouched)
        pipeline.train_subclass(parent_dir, "JANUMET_front", output_path)
    """

    def __init__(
        self,
        extractor: ResNet50FeatureExtractor,
        coreset_ratio: float = 0.25,
        seed: int = 42,
        fallback_threshold: float = 0.50,
        k_nearest: int = 3,
        score_method: str = "max",
        batch_size: int = 32,
        bad_dir: Optional[Path] = None,
    ):
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.seed = seed
        self.fallback_threshold = fallback_threshold
        self.k_nearest = k_nearest
        self.score_method = score_method
        self.batch_size = batch_size
        self.bad_dir = bad_dir

    # ═══════════════════════════════════════════════════════
    #  Public: train ALL subclasses under one parent dir
    # ═══════════════════════════════════════════════════════
    def train_parent(
        self,
        parent_dir: Path,
        output_path: Path,
        extra_meta: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Discover subclass dirs inside *parent_dir*, train each one
        independently, and save them all into a single ``.pth``.
        """
        if not parent_dir.exists():
            print(f"[Skip] {parent_dir} not found")
            return None

        sub_dirs = self._discover_subclass_dirs(parent_dir)
        if not sub_dirs:
            print(f"[Skip] No subclass dirs in {parent_dir}")
            return None

        print(f"[Train] {parent_dir.name} | "
              f"{len(sub_dirs)} subclass(es): {[d.name for d in sub_dirs]}")

        subclass_data: Dict[str, Dict] = {}
        for sub_dir in sub_dirs:
            result = self._train_one_subclass(sub_dir)
            if result is not None:
                subclass_data[sub_dir.name] = result

        if not subclass_data:
            print("[Skip] No subclass produced valid patches")
            return None

        shared_meta = self._build_shared_meta(parent_dir, extra_meta)
        MemoryBank.save_multi(output_path, subclass_data, shared_meta,
                              parent_class=parent_dir.name)
        return output_path

    # ═══════════════════════════════════════════════════════
    #  Public: re-train ONE subclass (keep the rest)
    # ═══════════════════════════════════════════════════════
    def train_subclass(
        self,
        parent_dir: Path,
        subclass_name: str,
        output_path: Path,
        extra_meta: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Train *only* ``subclass_name`` and merge back into existing
        ``output_path``.  Other subclasses remain untouched.
        """
        sub_dir = parent_dir / subclass_name
        if not sub_dir.is_dir():
            print(f"[Skip] subclass dir not found: {sub_dir}")
            return None

        result = self._train_one_subclass(sub_dir)
        if result is None:
            print(f"[Skip] No patches for {subclass_name}")
            return None

        # load existing .pth (if any)
        existing: Dict[str, Dict] = {}
        if output_path.exists():
            loaded, _ = MemoryBank.load_multi(output_path)
            for name, (bank, meta) in loaded.items():
                entry = dict(meta)
                entry["memory_bank"] = bank
                existing[name] = entry

        # merge / replace
        existing[subclass_name] = result

        shared_meta = self._build_shared_meta(parent_dir, extra_meta)
        MemoryBank.save_multi(output_path, existing, shared_meta,
                              parent_class=parent_dir.name)
        print(f"[Updated] {subclass_name} in {output_path}")
        return output_path

    # ═══════════════════════════════════════════════════════
    #  Internal: train ONE subclass dir → dict
    # ═══════════════════════════════════════════════════════
    def _train_one_subclass(
        self,
        sub_dir: Path,
    ) -> Optional[Dict]:
        """
        Extract features + build coreset + calibrate threshold for a
        single subclass directory.

        Returns dict ready for ``MemoryBank.save_multi()``, or *None*.
        """
        image_paths = list_images_recursive(sub_dir)
        if not image_paths:
            print(f"  [Skip] No images in {sub_dir.name}")
            return None

        print(f"  [SubClass] {sub_dir.name} | {len(image_paths)} images")

        # ── Batch Feature Extraction ──
        bank = MemoryBank()

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images: List[np.ndarray] = []

            for p in batch_paths:
                try:
                    # Direct cv2 load — faster than PIL→numpy→BGR roundtrip
                    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        images.append(bgr)
                except Exception:
                    continue

            if not images:
                continue

            patches_batch = self.extractor.extract_batch(images)
            for patches in patches_batch:
                bank.add(patches)

        if bank.total_patches == 0:
            print(f"  [Skip] No patches for {sub_dir.name}")
            return None

        # ── Build Coreset ──
        memory = bank.build(
            coreset_ratio=self.coreset_ratio,
            seed=self.seed,
        )

        # ── Calibrate Threshold ──
        threshold = self._calibrate(memory)
        print(f"  [Threshold] {sub_dir.name}  →  {threshold:.4f}")

        return {
            "memory_bank": memory,
            "threshold": threshold,
            "num_images": len(image_paths),
        }

    # ═══════════════════════════════════════════════════════
    #  Calibration (batch mode)
    # ═══════════════════════════════════════════════════════
    def _calibrate(self, memory: np.ndarray) -> float:
        if self.bad_dir is None:
            print("  [Calibrate] No bad_dir → fallback")
            return self.fallback_threshold

        bad_paths = list_images_recursive(self.bad_dir)
        if not bad_paths:
            print("  [Calibrate] No BAD images → fallback")
            return self.fallback_threshold

        scorer = PatchCoreScorer(k_nearest=self.k_nearest, assume_normalized=False)
        scorer.build_index(memory)

        # ── Batch calibration — process in batches instead of one-by-one ──
        scores: List[float] = []
        for i in range(0, len(bad_paths), self.batch_size):
            batch_paths = bad_paths[i : i + self.batch_size]
            batch_imgs: List[np.ndarray] = []
            for p in batch_paths:
                try:
                    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        batch_imgs.append(bgr)
                except Exception:
                    continue

            if not batch_imgs:
                continue

            batch_feats = self.extractor.extract_batch(batch_imgs)
            for patches in batch_feats:
                score = scorer.score_pill(patches, method=self.score_method)
                scores.append(score)

        if not scores:
            return self.fallback_threshold

        arr = np.array(scores, dtype=np.float32)
        return float(np.percentile(arr, 5.0))

    # ═══════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════
    @staticmethod
    def _discover_subclass_dirs(parent_dir: Path) -> List[Path]:
        """Return sorted list of immediate child directories."""
        return sorted(
            d for d in parent_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _build_shared_meta(
        self,
        parent_dir: Path,
        extra_meta: Optional[Dict] = None,
    ) -> Dict:
        meta: Dict = {
            "parent_class": parent_dir.name,
            "coreset_ratio": self.coreset_ratio,
            "backbone": getattr(self.extractor, "backbone_path", None) or "resnet50",
            "img_size": self.extractor.img_size,
            "grid_size": self.extractor.grid_size,
            "feature_dim": self.extractor.feature_dim,
            "k_nearest": self.k_nearest,
            "score_method": self.score_method,
            "use_color_features": self.extractor.use_color_features,
            "use_hsv": self.extractor.use_hsv,
            "color_weight": self.extractor.color_weight,
        }
        if extra_meta:
            meta.update(extra_meta)
        return meta
