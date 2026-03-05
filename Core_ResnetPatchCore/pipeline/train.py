"""
Optimized Training Pipeline (Parent-Level Memory Build)
=======================================================
• รวม subclass เป็น parent memory เดียว
• Batch feature extraction
• Fast threshold calibration
• Production-ready
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Set

from Core_ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor
from Core_ResnetPatchCore.patchcore.memory_bank import MemoryBank
from Core_ResnetPatchCore.patchcore.scorer import PatchCoreScorer
from Core_ResnetPatchCore.utils.structure_manager import DatasetManager


class TrainPipeline:

    IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        extractor: ResNet50FeatureExtractor,
        coreset_ratio: float = 0.10,
        seed: int = 42,
        fallback_threshold: float = 0.50,
        k_nearest: int = 3,
        score_method: str = "max",
        dataset_mgr: Optional[DatasetManager] = None,
        batch_size: int = 32,
    ):
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.seed = seed
        self.fallback_threshold = fallback_threshold
        self.k_nearest = k_nearest
        self.score_method = score_method
        self.dataset_mgr = dataset_mgr
        self.batch_size = batch_size

    # ═══════════════════════════════════════
    # Train Parent Class (รวมทุก subclass)
    # ═══════════════════════════════════════
    def train_parent(
        self,
        parent_dir: Path,
        output_path: Path,
        extra_meta: Optional[Dict] = None,
    ) -> Optional[Path]:

        if not parent_dir.exists():
            print(f"[Skip] {parent_dir} not found")
            return None

        # ── collect all GOOD images recursively ──
        if self.dataset_mgr:
            image_paths = self.dataset_mgr.list_images_recursive(parent_dir)
        else:
            image_paths = self._list_images_recursive(parent_dir)

        if not image_paths:
            print(f"[Skip] No images in {parent_dir}")
            return None

        print(f"[Train] {parent_dir.name} | {len(image_paths)} images")

        # ── Batch Feature Extraction ──
        bank = MemoryBank()
        total = len(image_paths)

        for i in range(0, total, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]

            images = []
            valid_paths = []

            for p in batch_paths:
                try:
                    pil_img = Image.open(p).convert("RGB")
                    bgr = np.array(pil_img)[:, :, ::-1]  # RGB → BGR for extractor
                    images.append(bgr)
                    valid_paths.append(p)
                except Exception:
                    continue

            if not images:
                continue

            patches_batch = self.extractor.extract_batch(images)

            for patches in patches_batch:
                bank.add(patches)

        if bank.total_patches == 0:
            print("[Skip] No patches extracted")
            return None

        # ── Build Coreset ──
        memory = bank.build(
            coreset_ratio=self.coreset_ratio,
            seed=self.seed,
        )

        # ── Calibrate Threshold ──
        threshold = self._calibrate(memory)
        print(f"[Threshold] {threshold:.4f}")

        # ── Metadata ──
        meta: Dict = {
            "threshold": threshold,
            "parent_class": parent_dir.name,
            "num_images": len(image_paths),
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

        bank.save(output_path, memory, meta)
        print(f"[Saved] {output_path}")

        return output_path

    # ═══════════════════════════════════════
    # Threshold Calibration (BAD-only)
    # ═══════════════════════════════════════
    def _calibrate(self, memory: np.ndarray) -> float:

        from config.base import BAD_DIR

        bad_dir = Path(BAD_DIR)
        bad_paths = self._list_images_recursive(bad_dir)

        if not bad_paths:
            print("[Calibrate] No BAD images → fallback")
            return self.fallback_threshold

        scorer = PatchCoreScorer(k_nearest=self.k_nearest, assume_normalized=False)
        index = scorer.build_index(memory)

        scores = []

        for p in bad_paths:
            try:
                img = Image.open(p).convert("RGB")
                patches = self.extractor.extract(img)
                score = scorer.score_pill(
                    patches,
                    index,
                    method=self.score_method,
                )
                scores.append(score)
            except Exception:
                continue

        if not scores:
            return self.fallback_threshold

        arr = np.array(scores, dtype=np.float32)

        # robust threshold (5th percentile)
        threshold = float(np.percentile(arr, 5.0))
        return threshold

    # ═══════════════════════════════════════
    # Utils
    # ═══════════════════════════════════════
    def _list_images_recursive(self, directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        return sorted(
            p for p in directory.rglob("*")
            if p.suffix.lower() in self.IMAGE_EXTS
        )