# DINOv2PatchCore/trainer_dinov2.py
"""
DINOv2 PatchCore Training Utilities.

Single Responsibility:
- Build memory banks from training images using DINOv2 features
- Perform coreset subsampling
- Calibrate thresholds
"""
import faiss
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Sequence, Iterable

from .core_dinov2 import DINOv2PatchCore


class DINOv2PatchCoreTrainer:
    """Training utilities for DINOv2-based PatchCore."""
    
    def __init__(self, patchcore: DINOv2PatchCore):
        """
        Args:
            patchcore: DINOv2PatchCore feature extractor instance
        """
        self.patchcore = patchcore
    
    # =========================================================
    # Utilities
    # =========================================================
    @staticmethod
    def iter_images(dir_path: Path, image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
        """List image files in directory."""
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        exts = {e.lower() for e in image_exts}
        paths = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        paths.sort()
        return paths
    
    # =========================================================
    # Memory Bank Building
    # =========================================================
    def extract_patch_bank(self, image_paths: Iterable[Path]) -> Optional[np.ndarray]:
        """Extract and concat patch features from images."""
        all_patches: list[np.ndarray] = []
        for img_path in image_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                all_patches.append(self.patchcore.extract_features(pil_img))
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")

        if not all_patches:
            return None

        bank = np.concatenate(all_patches, axis=0).astype(np.float32)
        return np.ascontiguousarray(bank)

    @staticmethod
    def coreset_subsample(
        bank: np.ndarray,
        ratio: float,
        rng: np.random.Generator,
        min_keep: int = 1000,
        keep_full_if_leq: int = 5000,
    ) -> np.ndarray:
        """Random coreset subsampling."""
        n = int(bank.shape[0])
        if n <= keep_full_if_leq:
            return bank

        n_select = max(min_keep, int(n * ratio))
        n_select = min(n_select, n)
        idx = rng.choice(n, size=n_select, replace=False)
        return np.ascontiguousarray(bank[idx])

    def build_memory_bank_from_dir(
        self,
        good_dir: Path,
        rng: np.random.Generator,
        coreset_ratio: float = 0.12,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> Optional[np.ndarray]:
        """Build normalized memory bank from train/good images."""
        image_paths = self.iter_images(good_dir, image_exts=image_exts)
        if not image_paths:
            return None

        bank = self.extract_patch_bank(image_paths)
        if bank is None or bank.shape[0] == 0:
            return None

        bank = self.coreset_subsample(bank, coreset_ratio, rng)
        faiss.normalize_L2(bank)
        return bank

    # =========================================================
    # Threshold Calibration
    # =========================================================
    def calibrate_threshold_from_dir(
        self,
        memory_bank: np.ndarray,
        test_good_dir: Path,
        fallback_threshold: float = 0.35,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        percentile: float = 98,
        sigma: float = 3.0,
    ) -> float:
        """Calibrate threshold from good images."""
        image_paths = self.iter_images(test_good_dir, image_exts=image_exts)
        if not image_paths:
            return fallback_threshold

        max_scores: list[float] = []
        temp_index = self.patchcore.build_faiss_index(memory_bank)

        for img_path in image_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                feats = self.patchcore.extract_features(pil_img)
                score = self.patchcore.get_max_anomaly_score(feats, temp_index)
                max_scores.append(score)
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")

        if not max_scores:
            return fallback_threshold

        scores_np = np.array(max_scores, dtype=np.float32)
        mean_val = float(scores_np.mean())
        std_val = float(scores_np.std())
        p = float(np.percentile(scores_np, percentile))
        return max(p, mean_val + sigma * std_val)

    def calibrate_threshold_from_test_dir(
        self,
        memory_bank: np.ndarray,
        test_dir: Path,
        good_folder: str = "good",
        fallback_threshold: float = 0.35,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        n_quantiles: int = 301,
    ) -> float:
        """Calibrate threshold using good vs anomaly images."""
        if not test_dir.exists() or not test_dir.is_dir():
            return fallback_threshold

        good_dir = test_dir / good_folder
        good_paths = self.iter_images(good_dir, image_exts=image_exts)
        if not good_paths:
            return fallback_threshold

        bad_paths: list[Path] = []
        for d in sorted([p for p in test_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            if d.name == good_folder:
                continue
            bad_paths.extend(self.iter_images(d, image_exts=image_exts))

        if not bad_paths:
            return self.calibrate_threshold_from_dir(
                memory_bank, good_dir, fallback_threshold, image_exts
            )

        index = self.patchcore.build_faiss_index(memory_bank)

        def _score(paths: Sequence[Path]) -> np.ndarray:
            scores: list[float] = []
            for img_path in paths:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    feats = self.patchcore.extract_features(pil_img)
                    scores.append(self.patchcore.get_max_anomaly_score(feats, index))
                except Exception:
                    pass
            return np.array(scores, dtype=np.float32)

        good_scores = _score(good_paths)
        bad_scores = _score(bad_paths)

        if good_scores.size == 0 or bad_scores.size == 0:
            return self.calibrate_threshold_from_dir(
                memory_bank, good_dir, fallback_threshold, image_exts
            )

        all_scores = np.concatenate([good_scores, bad_scores], axis=0)
        qs = np.linspace(0.0, 1.0, max(3, int(n_quantiles)))
        thresholds = np.unique(np.quantile(all_scores, qs))

        best_thr = float(thresholds[0])
        best_f1 = -1.0
        n_good = int(good_scores.size)
        n_bad = int(bad_scores.size)

        for thr in thresholds:
            fp = int((good_scores > thr).sum())
            tp = int((bad_scores > thr).sum())
            fn = n_bad - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        return best_thr
