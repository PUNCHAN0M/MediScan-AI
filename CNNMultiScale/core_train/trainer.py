# CNNMultiScale/core_train/trainer.py
"""
CNN Multi-Scale PatchCore Training Utilities.

🔥 Key differences from standard PatchCore training:
1. Builds SEPARATE memory banks per scale (layer1, layer2, layer3)
2. Calibrates threshold using adaptive method (mean + k*std or percentile)
3. Supports multi-resolution feature extraction during training
4. Per-scale coreset subsampling

Single Responsibility:
- Build per-scale memory banks from training images
- Perform coreset subsampling per scale
- Calibrate adaptive thresholds
"""
import faiss
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Sequence, Iterable, Dict, List

from CNNMultiScale.core_shared.patchcore_multiscale import CNNMultiScalePatchCore


class CNNMultiScaleTrainer:
    """Training utilities for CNN Multi-Scale PatchCore."""

    def __init__(self, patchcore: CNNMultiScalePatchCore):
        self.patchcore = patchcore

    # =========================================================
    # Utilities
    # =========================================================
    @staticmethod
    def iter_images(
        dir_path: Path,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> list[Path]:
        """List image files in directory."""
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        exts = {e.lower() for e in image_exts}
        paths = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        paths.sort()
        return paths

    # =========================================================
    # Per-Scale Memory Bank Building
    # =========================================================
    def extract_multiscale_patch_bank(
        self,
        image_paths: Iterable[Path],
        use_multi_resolution: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract per-scale patch features from all images.
        
        Returns:
            Dict[layer_name → np.ndarray of shape (total_patches, feature_dim)]
            or None if no valid images
        """
        all_scale_patches: Dict[str, List[np.ndarray]] = {
            layer: [] for layer in self.patchcore.selected_layers
        }

        for img_path in image_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                if use_multi_resolution:
                    features = self.patchcore.extract_multiscale_features_multiresolution(pil_img)
                else:
                    features = self.patchcore.extract_multiscale_features(pil_img)

                for layer_name, feat in features.items():
                    all_scale_patches[layer_name].append(feat)
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")

        # Check if any data was extracted
        has_data = any(len(patches) > 0 for patches in all_scale_patches.values())
        if not has_data:
            return None

        # Concatenate patches per scale
        result = {}
        for layer_name, patches_list in all_scale_patches.items():
            if patches_list:
                bank = np.concatenate(patches_list, axis=0).astype(np.float32)
                result[layer_name] = np.ascontiguousarray(bank)

        return result if result else None

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

    def build_multiscale_memory_banks(
        self,
        good_dir: Path,
        rng: np.random.Generator,
        coreset_ratio: float = 0.15,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        use_multi_resolution: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Build normalized memory banks per scale from train/good images.
        
        Returns:
            Dict[layer_name → normalized memory bank]
        """
        image_paths = self.iter_images(good_dir, image_exts=image_exts)
        if not image_paths:
            return None

        print(f"    Extracting features from {len(image_paths)} images...")
        scale_banks = self.extract_multiscale_patch_bank(
            image_paths, use_multi_resolution=use_multi_resolution
        )
        if scale_banks is None:
            return None

        # Coreset subsampling and normalization per scale
        result = {}
        for layer_name, bank in scale_banks.items():
            print(f"    {layer_name}: {bank.shape[0]} patches → ", end="")
            bank = self.coreset_subsample(bank, coreset_ratio, rng)
            faiss.normalize_L2(bank)
            result[layer_name] = bank
            print(f"{bank.shape[0]} patches (dim={bank.shape[1]})")

        return result if result else None

    # =========================================================
    # Adaptive Threshold Calibration
    # =========================================================
    def calibrate_adaptive_threshold(
        self,
        scale_memory_banks: Dict[str, np.ndarray],
        test_good_dir: Path,
        method: str = "sigma",
        sigma: float = 3.0,
        percentile: float = 99.5,
        fallback_threshold: float = 0.50,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        use_multi_resolution: bool = False,
    ) -> float:
        """
        Calibrate threshold using adaptive method.
        
        Methods:
        - "sigma": mean + k*std — good for low-variance surfaces
        - "percentile": top percentile — good for consistent good samples
        
        Args:
            scale_memory_banks: Per-scale memory banks
            test_good_dir: Directory with known-good test images
            method: "sigma" or "percentile"
            sigma: k value for sigma method
            percentile: percentile value for percentile method
        """
        image_paths = self.iter_images(test_good_dir, image_exts=image_exts)
        if not image_paths:
            return fallback_threshold

        # Build FAISS indices per scale
        scale_indices = {}
        for layer_name, bank in scale_memory_banks.items():
            scale_indices[layer_name] = self.patchcore.build_faiss_index(bank)

        # Score all good images
        max_scores: List[float] = []
        per_scale_score_lists: Dict[str, List[float]] = {
            l: [] for l in self.patchcore.selected_layers
        }

        for img_path in image_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                if use_multi_resolution:
                    features = self.patchcore.extract_multiscale_features_multiresolution(pil_img)
                else:
                    features = self.patchcore.extract_multiscale_features(pil_img)

                per_scale = self.patchcore.get_per_scale_anomaly_scores(
                    features, scale_indices
                )
                fused = self.patchcore.fuse_scores(per_scale)
                max_scores.append(fused)

                for layer_name, score in per_scale.items():
                    per_scale_score_lists[layer_name].append(score)

            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")

        if not max_scores:
            return fallback_threshold

        scores_np = np.array(max_scores, dtype=np.float32)

        if method == "sigma":
            mean_val = float(scores_np.mean())
            std_val = float(scores_np.std())
            threshold = mean_val + sigma * std_val
            print(f"    Adaptive threshold (sigma): mean={mean_val:.4f}, "
                  f"std={std_val:.4f}, threshold={threshold:.4f}")
        elif method == "percentile":
            threshold = float(np.percentile(scores_np, percentile))
            print(f"    Adaptive threshold (percentile {percentile}%): {threshold:.4f}")
        else:
            threshold = fallback_threshold

        # Print per-scale statistics
        for layer_name, scores in per_scale_score_lists.items():
            if scores:
                arr = np.array(scores)
                print(f"    {layer_name}: mean={arr.mean():.4f}, "
                      f"std={arr.std():.4f}, max={arr.max():.4f}")

        return max(threshold, 0.01)  # Ensure positive threshold

    def calibrate_threshold_from_test_dir(
        self,
        scale_memory_banks: Dict[str, np.ndarray],
        test_dir: Path,
        good_folder: str = "good",
        method: str = "sigma",
        sigma: float = 3.0,
        percentile: float = 99.5,
        fallback_threshold: float = 0.50,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        use_multi_resolution: bool = False,
        n_quantiles: int = 301,
    ) -> float:
        """
        Calibrate threshold using good vs anomaly images (F1 optimization).
        Falls back to adaptive if no anomaly images available.
        """
        if not test_dir.exists() or not test_dir.is_dir():
            return fallback_threshold

        good_dir = test_dir / good_folder
        good_paths = self.iter_images(good_dir, image_exts=image_exts)
        if not good_paths:
            return fallback_threshold

        # Find anomaly directories
        bad_paths: List[Path] = []
        for d in sorted([p for p in test_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            if d.name == good_folder:
                continue
            bad_paths.extend(self.iter_images(d, image_exts=image_exts))

        # If no anomaly data, use adaptive threshold
        if not bad_paths:
            return self.calibrate_adaptive_threshold(
                scale_memory_banks, good_dir, method=method,
                sigma=sigma, percentile=percentile,
                fallback_threshold=fallback_threshold,
                image_exts=image_exts,
                use_multi_resolution=use_multi_resolution,
            )

        # Build indices
        scale_indices = {}
        for layer_name, bank in scale_memory_banks.items():
            scale_indices[layer_name] = self.patchcore.build_faiss_index(bank)

        def _score_paths(paths: Sequence[Path]) -> np.ndarray:
            scores = []
            for img_path in paths:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    if use_multi_resolution:
                        features = self.patchcore.extract_multiscale_features_multiresolution(pil_img)
                    else:
                        features = self.patchcore.extract_multiscale_features(pil_img)
                    fused = self.patchcore.get_max_anomaly_score(features, scale_indices)
                    scores.append(fused)
                except Exception:
                    pass
            return np.array(scores, dtype=np.float32)

        good_scores = _score_paths(good_paths)
        bad_scores = _score_paths(bad_paths)

        if good_scores.size == 0 or bad_scores.size == 0:
            return self.calibrate_adaptive_threshold(
                scale_memory_banks, good_dir, method=method,
                sigma=sigma, percentile=percentile,
                fallback_threshold=fallback_threshold,
                image_exts=image_exts,
                use_multi_resolution=use_multi_resolution,
            )

        # F1 optimization
        all_scores = np.concatenate([good_scores, bad_scores], axis=0)
        qs = np.linspace(0.0, 1.0, max(3, int(n_quantiles)))
        thresholds = np.unique(np.quantile(all_scores, qs))

        best_thr = float(thresholds[0])
        best_f1 = -1.0
        n_bad = int(bad_scores.size)

        for thr in thresholds:
            fp = int((good_scores > thr).sum())
            tp = int((bad_scores > thr).sum())
            fn = n_bad - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        print(f"    F1-optimized threshold: {best_thr:.4f} (F1={best_f1:.4f})")
        print(f"    Good scores: mean={good_scores.mean():.4f}, max={good_scores.max():.4f}")
        print(f"    Bad scores: mean={bad_scores.mean():.4f}, min={bad_scores.min():.4f}")

        return best_thr
