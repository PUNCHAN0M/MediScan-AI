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
import time
import numpy as np
from pathlib import Path
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
        calib_bad_percentile: float = 5.0,
        calib_good_cap_percentile: float = 99.0,
    ):
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.seed = seed
        self.fallback_threshold = fallback_threshold
        self.k_nearest = k_nearest
        self.score_method = score_method
        self.batch_size = batch_size
        self.bad_dir = bad_dir
        self.calib_bad_percentile = float(calib_bad_percentile)
        self.calib_good_cap_percentile = float(calib_good_cap_percentile)

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

        t_sub_start = time.perf_counter()
        print(f"  [SubClass] {sub_dir.name} | {len(image_paths)} images")

        # ── Batch Feature Extraction ──
        bank = MemoryBank()
        t_extract_start = time.perf_counter()

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

        t_extract_elapsed = time.perf_counter() - t_extract_start
        print(f"  [Extract] done in {t_extract_elapsed:.1f}s | elapsed: {time.perf_counter() - t_sub_start:.1f}s")

        # ── Build Coreset ──
        memory = bank.build(
            coreset_ratio=self.coreset_ratio,
            seed=self.seed,
        )

        # ── Calibrate Threshold ──
        threshold = self._calibrate(memory, good_paths=image_paths)

        t_sub_elapsed = time.perf_counter() - t_sub_start
        print(f"  [Threshold] {sub_dir.name}  →  {threshold:.4f}  | total: {t_sub_elapsed:.1f}s")

        return {
            "memory_bank": memory,
            "threshold": threshold,
            "num_images": len(image_paths),
        }

    # ═══════════════════════════════════════════════════════
    #  Calibration (batch mode)
    # ═══════════════════════════════════════════════════════
    def _calibrate(self, memory: np.ndarray, good_paths: Optional[List[Path]] = None) -> float:
        t_calib_start = time.perf_counter()

        if self.bad_dir is None:
            print(f"  [Calibrate] mode=FALLBACK  reason=no bad_dir  value={self.fallback_threshold:.4f}")
            return self.fallback_threshold

        bad_paths = list_images_recursive(self.bad_dir)
        if not bad_paths:
            print(f"  [Calibrate] mode=FALLBACK  reason=no images in bad_dir  value={self.fallback_threshold:.4f}")
            return self.fallback_threshold

        print(f"  [Calibrate] mode=BAD_DATA  bad_images={len(bad_paths)}  bad_dir={self.bad_dir}")

        # Use stable old-style scorer path for calibration:
        # - FlatIP index (no IVF clustering)
        # - CPU index (avoid GPU FAISS numeric edge-cases)
        scorer = PatchCoreScorer(
            k_nearest=self.k_nearest,
            assume_normalized=False,
            use_gpu=False,
            use_ivf=False,
        )
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
            print(f"  [Calibrate] mode=FALLBACK  reason=all bad images failed to load  value={self.fallback_threshold:.4f}")
            return self.fallback_threshold

        arr = np.array(scores, dtype=np.float32)
        finite = np.isfinite(arr)
        n_bad = int((~finite).sum())
        if n_bad > 0:
            arr = arr[finite]
            print(f"  [Calibrate] dropped_non_finite={n_bad}")

        if arr.size == 0:
            print(f"  [Calibrate] mode=FALLBACK  reason=all scores non-finite  value={self.fallback_threshold:.4f}")
            return self.fallback_threshold

        bad_thr = float(np.percentile(arr, self.calib_bad_percentile))
        result = bad_thr
        reason = [f"bad_p{self.calib_bad_percentile:g}={bad_thr:.4f}"]

        # Adaptive calibration with GOOD scores (no fixed cap):
        # 1) If distributions are separable, use midpoint between
        #    good edge and bad edge.
        # 2) If overlapping, use F1 sweep on good vs bad score sets.
        if good_paths:
            good_scores: List[float] = []
            for i in range(0, len(good_paths), self.batch_size):
                batch_paths = good_paths[i : i + self.batch_size]
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
                    good_scores.append(score)

            if good_scores:
                good_arr = np.array(good_scores, dtype=np.float32)
                good_arr = good_arr[np.isfinite(good_arr)]
                if good_arr.size > 0:
                    good_edge = float(np.percentile(good_arr, self.calib_good_cap_percentile))

                    if good_edge < bad_thr:
                        result = 0.5 * (good_edge + bad_thr)
                        reason.append(
                            f"midpoint(good_p{self.calib_good_cap_percentile:g}={good_edge:.4f},"
                            f" bad_p{self.calib_bad_percentile:g}={bad_thr:.4f})"
                        )
                    else:
                        result = self._f1_sweep(good_arr, arr, n_steps=351)
                        reason.append("f1_sweep(good_vs_bad)")

                    print(
                        f"  [Calibrate][GOOD] scored={good_arr.size} "
                        f"min={good_arr.min():.4f} max={good_arr.max():.4f} "
                        f"mean={good_arr.mean():.4f} p95={np.percentile(good_arr, 95):.4f} "
                        f"p99={np.percentile(good_arr, 99):.4f}"
                    )

        t_calib_elapsed = time.perf_counter() - t_calib_start
        print(
            f"  [Calibrate] scored={len(scores)}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}  "
            f"p5={bad_thr:.4f}  final={result:.4f}  "
            f"reason={' | '.join(reason)}  time={t_calib_elapsed:.1f}s"
        )
        return result

    @staticmethod
    def _f1_sweep(good: np.ndarray, bad: np.ndarray, n_steps: int = 351) -> float:
        all_s = np.concatenate([good, bad])
        if all_s.size == 0:
            return 0.5

        thrs = np.unique(np.quantile(all_s, np.linspace(0.0, 1.0, n_steps)))
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
