#Core_ResnetPatchCore\pipeline\train.py
"""
Training Pipeline — ใช้ DatasetManager สำหรับจัดการ folder
=========================================================
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Set
from Core_ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor
from Core_ResnetPatchCore.patchcore.memory_bank import MemoryBank
from Core_ResnetPatchCore.patchcore.scorer import PatchCoreScorer
from Core_ResnetPatchCore.utils.structure_manager import DatasetManager  # ✅ เพิ่ม import


class TrainPipeline:
    """
    Train PatchCore model for one (or many) pill classes.
    """
    IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        extractor: ResNet50FeatureExtractor,
        coreset_ratio: float = 0.10,
        seed: int = 42,
        fallback_threshold: float = 0.50,
        k_nearest: int = 3,
        score_method: str = "max",
        dataset_mgr: Optional[DatasetManager] = None,  # ✅ เพิ่ม parameter
    ):
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.seed = seed
        self.fallback_threshold = fallback_threshold
        self.k_nearest = k_nearest
        self.score_method = score_method
        self.dataset_mgr = dataset_mgr  # ✅ เก็บ reference

    # ─────────────────── helpers ───────────────────
    @classmethod
    def list_images(cls, directory: Path) -> List[Path]:
        """List image files sorted alphabetically."""
        if not directory.is_dir():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in cls.IMAGE_EXTS
        )

    # ─────────────────── train one class ───────────────────
    def train_class(
        self,
        good_dir: Path,
        output_path: Path,
        test_dir: Optional[Path] = None,
        extra_meta: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Train memory bank for a single class / subclass.
        """
        # ✅ ใช้ DatasetManager.list_images() ถ้ามี
        if self.dataset_mgr:
            images = self.dataset_mgr.list_images(
                main_class=good_dir.parent.parent.name if good_dir.parent.parent != good_dir else "",
                sub_class=good_dir.parent.name if good_dir.parent != good_dir else "",
                split="train",
                label="good",
            )
        else:
            images = self.list_images(good_dir)

        if not images:
            print(f"    [Skip] No images in {good_dir}")
            return None

        print(f"    Extracting features from {len(images)} images …")

        # ── build memory bank ──
        bank = MemoryBank()
        for img_path in images:
            try:
                pil = Image.open(img_path).convert("RGB")
                patches = self.extractor.extract(pil)
                bank.add(patches)
            except Exception as exc:
                print(f"    Skip {img_path.name}: {exc}")

        if bank.total_patches == 0:
            print("    [Skip] No patches extracted")
            return None

        memory = bank.build(
            coreset_ratio=self.coreset_ratio,
            seed=self.seed,
        )

        # ── threshold ──
        threshold = self._calibrate(memory, test_dir)
        print(f"    Threshold: {threshold:.4f}")

        # ── metadata ──
        meta: Dict = {
            "threshold": threshold,
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

        bank.save(output_path, meta)
        return output_path

    # ─────────────────── threshold calibration ───────────────────
    def _calibrate(
        self,
        memory_bank: np.ndarray,
        test_dir: Optional[Path] = None,
    ) -> float:
        """Calibrate using BAD_DIR only."""
        from config.base import BAD_DIR
        bad_dir = Path(BAD_DIR)

        # ✅ ใช้ DatasetManager.list_images_recursive() ถ้ามี
        if self.dataset_mgr:
            bad_paths = self.dataset_mgr.list_images_recursive(bad_dir)
        else:
            bad_paths = self.list_images_recursive(bad_dir)

        if bad_paths:
            print(f"    [Calibrate] Found {len(bad_paths)} BAD images from {bad_dir}")
        else:
            print("    [Calibrate] No BAD images → fallback")
            return self.fallback_threshold

        scorer = PatchCoreScorer(k_nearest=self.k_nearest)
        index = scorer.build_index(memory_bank)
        bad_scores = self._score_paths(bad_paths, scorer, index)

        if not bad_scores:
            print("    [Calibrate] No bad scores → fallback")
            return self.fallback_threshold

        arr = np.array(bad_scores, dtype=np.float32)
        threshold = float(np.percentile(arr, 5.0))
        print(f"    [Calibrate] BAD-only threshold (p5): {threshold:.4f}")
        return threshold

    def _score_paths(
        self,
        paths: List[Path],
        scorer: PatchCoreScorer,
        index,
    ) -> List[float]:
        scores: list = []
        for p in paths:
            try:
                pil = Image.open(p).convert("RGB")
                patches = self.extractor.extract(pil)
                scores.append(scorer.score_pill(patches, index, method=self.score_method))
            except Exception:
                pass
        return scores

    @staticmethod
    def list_images_recursive(directory: Path) -> List[Path]:
        """List image files recursively through all subdirectories."""
        if not directory.is_dir():
            return []
        return sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in TrainPipeline.IMAGE_EXTS
        )