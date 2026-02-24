# core_train/trainer.py
"""
PatchCore + SIFE Training Utilities with CUDA + Threading.

CUDA Optimized:
- ThreadPoolExecutor for parallel image loading (I/O bound)
- Batch feature extraction (GPU bound)
- Configurable batch size with auto GPU memory estimation

Responsibilities:
- Build memory banks from training images
- Coreset subsampling
- Threshold calibration (good-only / good-vs-bad)
"""
from __future__ import annotations

import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from typing import Iterable, List, Optional, Sequence

from ..core_shared.patchcore_sife import PatchCoreSIFE


class PatchCoreSIFETrainer:
    """
    Training utilities for PatchCoreSIFE.
    
    Optimizations:
    - Parallel image loading with ThreadPoolExecutor
    - Batch feature extraction (single GPU forward pass per batch)
    - Memory-efficient processing of large datasets
    """

    def __init__(
        self,
        patchcore: PatchCoreSIFE,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        self.patchcore = patchcore
        self.num_workers = num_workers
        self.batch_size = batch_size

    # =========================================================
    # UTILITIES
    # =========================================================

    @staticmethod
    def iter_images(
        dir_path: Path,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> list[Path]:
        """List image files in directory, sorted by name."""
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        exts = {e.lower() for e in image_exts}
        paths = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        paths.sort()
        return paths

    @staticmethod
    def _load_image(path: Path) -> Optional[Image.Image]:
        """Load a single image. Thread-safe (no shared state)."""
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    # =========================================================
    # MEMORY BANK BUILDING
    # =========================================================

    def extract_patch_bank(self, image_paths: Iterable[Path]) -> Optional[np.ndarray]:
        """
        Extract and concatenate patch features from images.
        
        Uses ThreadPool for parallel I/O + batch GPU extraction.
        """
        paths = list(image_paths)
        if not paths:
            return None

        all_patches: List[np.ndarray] = []
        total = len(paths)
        processed = 0

        # Process in batches
        for batch_start in range(0, total, self.batch_size):
            batch_paths = paths[batch_start:batch_start + self.batch_size]

            # --- Parallel image loading (I/O bound → threads) ---
            images: List[Image.Image] = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                future_to_path = {
                    pool.submit(self._load_image, p): p for p in batch_paths
                }
                for future in as_completed(future_to_path):
                    result = future.result()
                    if result is not None:
                        images.append(result)
                    else:
                        path = future_to_path[future]
                        print(f"  Skip {path.name}: failed to load")

            if not images:
                continue

            # --- Batch feature extraction (GPU) ---
            features_list = self.patchcore.extract_features_batch(images)
            all_patches.extend(features_list)

            processed += len(batch_paths)
            print(f"    Progress: {processed}/{total} images", end="\r")

        print()  # newline after progress

        if not all_patches:
            return None

        bank = np.concatenate(all_patches, axis=0).astype(np.float32)
        return np.ascontiguousarray(bank)

    @staticmethod
    def coreset_subsample(
        bank: np.ndarray,
        ratio: float,
        rng: np.random.Generator,
        min_keep: int = 10000,
        keep_full_if_leq: int = 15000,
    ) -> np.ndarray:
        """Random coreset subsampling for memory efficiency."""
        n = bank.shape[0]
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

        print(f"    Extracting features from {len(image_paths)} images (batch={self.batch_size}, workers={self.num_workers})...")
        bank = self.extract_patch_bank(image_paths)
        if bank is None or bank.shape[0] == 0:
            return None

        print(f"    Raw bank: {bank.shape[0]:,} patches, dim={bank.shape[1]}")
        bank = self.coreset_subsample(bank, coreset_ratio, rng)
        print(f"    After coreset: {bank.shape[0]:,} patches")

        faiss.normalize_L2(bank)
        return bank

    # =========================================================
    # THRESHOLD CALIBRATION
    # =========================================================

    def _score_images(
        self,
        image_paths: List[Path],
        index: faiss.Index,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        use_detailed: bool = True,
    ) -> np.ndarray:
        """
        Score images in batches with parallel loading.

        use_detailed=True  → same scoring as inspector (get_anomaly_score_detailed)
        use_detailed=False → simple max score

        Returns:
            Array of anomaly scores per image
        """
        paths = list(image_paths)
        scores: List[float] = []

        for batch_start in range(0, len(paths), self.batch_size):
            batch_paths = paths[batch_start:batch_start + self.batch_size]

            # Parallel load
            images: List[Image.Image] = []
            valid_indices: List[int] = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                future_map = {
                    pool.submit(self._load_image, p): i
                    for i, p in enumerate(batch_paths)
                }
                for future in as_completed(future_map):
                    img = future.result()
                    if img is not None:
                        images.append(img)

            if not images:
                continue

            # Batch extract
            features_list = self.patchcore.extract_features_batch(images)

            # Score each — use same method as inspector — use same method as inspector
            for feats in features_list:
                if use_detailed:
                    s = self.patchcore.get_anomaly_score_detailed(feats, index)["score"]
                else:
                    s = self.patchcore.get_max_anomaly_score(feats, index)
                scores.append(s)

        return np.array(scores, dtype=np.float32)

    def calibrate_threshold_from_dir(
        self,
        memory_bank: np.ndarray,
        test_good_dir: Path,
        fallback_threshold: float = 0.35,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        percentile: float = 97.0,
        sigma: float = 3,
    ) -> float:
        """Calibrate threshold from good images only (percentile + sigma).
        Uses the same scoring method as the inspector (detailed scoring).
        """
        image_paths = self.iter_images(test_good_dir, image_exts=image_exts)
        if not image_paths:
            return fallback_threshold

        index = self.patchcore.build_faiss_index(memory_bank)
        scores = self._score_images(image_paths, index, use_detailed=True)

        if scores.size == 0:
            return fallback_threshold

        mean_val = float(scores.mean())
        std_val  = float(scores.std())
        p        = float(np.percentile(scores, percentile))
        threshold = max(p, mean_val + sigma * std_val)
        print(f"    Good scores  mean={mean_val:.4f} std={std_val:.4f} p{int(percentile)}={p:.4f} → thr={threshold:.4f}")
        return threshold

    def calibrate_threshold_from_test_dir(
        self,
        memory_bank: np.ndarray,
        test_dir: Path,
        good_folder: str = "good",
        fallback_threshold: float = 0.35,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        n_quantiles: int = 351,
    ) -> float:
        """Calibrate threshold using good vs anomaly images with F1 optimization."""
        if not test_dir.exists() or not test_dir.is_dir():
            return fallback_threshold

        good_dir = test_dir / good_folder
        good_paths = self.iter_images(good_dir, image_exts=image_exts)
        if not good_paths:
            return fallback_threshold

        # Collect anomaly images
        bad_paths: list[Path] = []
        for d in sorted(test_dir.iterdir(), key=lambda p: p.name):
            if d.is_dir() and d.name != good_folder:
                bad_paths.extend(self.iter_images(d, image_exts=image_exts))

        if not bad_paths:
            return self.calibrate_threshold_from_dir(
                memory_bank, good_dir, fallback_threshold, image_exts
            )

        index = self.patchcore.build_faiss_index(memory_bank)

        print(f"    Scoring {len(good_paths)} good images...")
        good_scores = self._score_images(good_paths, index, use_detailed=True)
        print(f"    Scoring {len(bad_paths)} anomaly images...")
        bad_scores = self._score_images(bad_paths, index, use_detailed=True)

        if good_scores.size == 0 or bad_scores.size == 0:
            return self.calibrate_threshold_from_dir(
                memory_bank, good_dir, fallback_threshold, image_exts
            )

        # F1-optimal threshold search
        all_scores = np.concatenate([good_scores, bad_scores])
        qs = np.linspace(0.0, 1.0, max(3, n_quantiles))
        thresholds = np.unique(np.quantile(all_scores, qs))

        best_thr = float(thresholds[0])
        best_f1 = -1.0
        n_bad = bad_scores.size

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

        print(f"    Best threshold: {best_thr:.4f} (F1={best_f1:.4f})")
        return best_thr
