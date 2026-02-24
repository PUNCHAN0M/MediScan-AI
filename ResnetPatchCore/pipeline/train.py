"""
Training Pipeline
=================

Train PatchCore memory bank from **good-only** pill images.

Flow
----
::

    GOOD images
        ↓  (pre-cropped pills — per subclass folder)
    ResNet50 feature extraction   (batch)
        ↓
    accumulate patches → MemoryBank
        ↓
    coreset subsample  (target ≈ 15 k–20 k)
        ↓
    L2 normalize
        ↓
    calibrate threshold  (percentile 99.5 / mean+3σ / F1 sweep)
        ↓
    save  →  .pth  {memory_bank, threshold, meta…}

No back-propagation — training completes in minutes.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Set

from ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor
from ResnetPatchCore.patchcore.memory_bank import MemoryBank
from ResnetPatchCore.patchcore.scorer import PatchCoreScorer


class TrainPipeline:
    """
    Train PatchCore model for one (or many) pill classes.

    Parameters
    ----------
    extractor : ResNet50FeatureExtractor
    coreset_ratio : float
        Fraction of patches to keep after coreset sampling.
    seed : int
        Random seed for reproducibility.
    fallback_threshold : float
        Used when calibration data is absent.
    k_nearest : int
        ``k`` for kNN scoring during threshold calibration.
    """

    IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        extractor: ResNet50FeatureExtractor,
        coreset_ratio: float = 0.10,
        seed: int = 42,
        fallback_threshold: float = 0.50,
        k_nearest: int = 3,
    ):
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.seed = seed
        self.fallback_threshold = fallback_threshold
        self.k_nearest = k_nearest

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

        Parameters
        ----------
        good_dir : Path
            Folder of good pill crops (``train/good``).
        output_path : Path
            Where to save the ``.pth`` model file.
        test_dir : Path, optional
            ``test/`` folder with ``good/`` and optionally ``bad/`` sub-dirs.
        extra_meta : dict, optional
            Extra metadata to persist (overrides defaults).

        Returns
        -------
        Path to saved ``.pth``, or ``None`` on failure.
        """
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
            "backbone": "resnet50",
            "img_size": self.extractor.img_size,
            "grid_size": self.extractor.grid_size,
            "feature_dim": self.extractor.feature_dim,
            "k_nearest": self.k_nearest,
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
        test_dir: Optional[Path],
    ) -> float:
        """
        Calibrate decision threshold.

        Strategy
        --------
        1. ``test_dir`` has ``good/`` + other dirs (bad) → F1-optimal sweep
        2. ``test_dir`` has ``good/`` only → percentile 99.5  or  mean + 3σ
        3. Fallback
        """
        if test_dir is None or not test_dir.is_dir():
            return self.fallback_threshold

        good_dir = test_dir / "good"
        good_paths = self.list_images(good_dir)
        if not good_paths:
            return self.fallback_threshold

        scorer = PatchCoreScorer(k_nearest=self.k_nearest)
        index = scorer.build_index(memory_bank)

        good_scores = self._score_paths(good_paths, scorer, index)

        # look for bad images
        bad_paths: List[Path] = []
        for d in sorted(test_dir.iterdir()):
            if d.is_dir() and d.name != "good":
                bad_paths.extend(self.list_images(d))

        if bad_paths:
            bad_scores = self._score_paths(bad_paths, scorer, index)
            if good_scores and bad_scores:
                return self._f1_sweep(np.array(good_scores),
                                      np.array(bad_scores))

        # good-only calibration
        if good_scores:
            arr = np.array(good_scores, dtype=np.float32)
            pctl = float(np.percentile(arr, 99.5))
            sigma = float(arr.mean() + 3.0 * arr.std())
            return max(pctl, sigma)

        return self.fallback_threshold

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
                scores.append(scorer.score_pill(patches, index))
            except Exception:
                pass
        return scores

    @staticmethod
    def _f1_sweep(
        good: np.ndarray,
        bad: np.ndarray,
        n_steps: int = 351,
    ) -> float:
        """Find threshold that maximises F1."""
        all_s = np.concatenate([good, bad])
        thrs = np.unique(np.quantile(all_s, np.linspace(0, 1, n_steps)))

        best_thr = float(thrs[0])
        best_f1 = -1.0

        for thr in thrs:
            tp = int((bad > thr).sum())
            fp = int((good > thr).sum())
            fn = int(bad.size) - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        return best_thr
