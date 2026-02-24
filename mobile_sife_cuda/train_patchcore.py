#!/usr/bin/env python3
# mobile_sife_cuda/train_patchcore.py
"""
Train MobileNetV3 + SIFE PatchCore — CUDA Optimized + Parallel Training.

Architecture:
- ThreadPoolExecutor trains multiple subclasses concurrently
- GPU lock ensures safe shared backbone access
- I/O, coreset, calibration overlap with GPU extraction
- Pipeline: load_images → [GPU lock] extract → coreset → calibrate → save

Usage:
    python run_train_sife_cuda.py
    python mobile_sife_cuda/train_patchcore.py
"""
import sys
from pathlib import Path

# Add parent directory for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import threading
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from mobile_sife_cuda.core_shared.patchcore_sife import PatchCoreSIFE
from mobile_sife_cuda.core_shared.cuda_utils import gpu_memory_info, clear_gpu_cache
from mobile_sife_cuda.core_train.trainer import PatchCoreSIFETrainer

# Import configurations
from config.base import DATA_ROOT, SELECTED_CLASSES, SEED, IMAGE_EXTS
from config.sife import (
    IMG_SIZE,
    GRID_SIZE,
    CORESET_RATIO,
    K_NEAREST,
    FALLBACK_THRESHOLD,
    MODEL_OUTPUT_DIR,
    FINETUNED_BACKBONE_PATH,
    USE_SIFE,
    SIFE_DIM,
    SIFE_ENCODING_TYPE,
    SIFE_WEIGHT,
    USE_CENTER_DISTANCE,
    USE_LOCAL_GRADIENT,
    USE_MULTI_SCALE,
    MULTI_SCALE_GRIDS,
    USE_EDGE_ENHANCEMENT,
    EDGE_WEIGHT,
    USE_COLOR_FEATURES,
    USE_HSV,
    COLOR_WEIGHT,
)


# =============================================================================
#                              RUNTIME SETTINGS
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16       # Images per batch for feature extraction
NUM_WORKERS = 4       # Threads for parallel image loading
MAX_PARALLEL_CLASSES = 3  # Max subclasses training concurrently


# =============================================================================
#                              TRAINING RESULT
# =============================================================================

@dataclass
class SubclassResult:
    """Result from training a single subclass."""
    parent_name: str
    subclass_name: str
    success: bool
    n_patches: int = 0
    feature_dim: int = 0
    threshold: float = 0.0
    elapsed_sec: float = 0.0
    error: str = ""


# =============================================================================
#                              HELPERS
# =============================================================================

def save_class_pth(parent_class: str, subclass_name: str, class_data: dict) -> Path:
    """Save subclass data as .pth file under parent class folder."""
    output_dir = MODEL_OUTPUT_DIR / parent_class
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subclass_name}.pth"
    torch.save(class_data, output_path)
    print(f"  Saved -> {output_path}")
    return output_path


def _resolve_parent_classes(
    data_root: Path, selected_classes, trainer: PatchCoreSIFETrainer
) -> list[Path]:
    """Find parent class folders with valid subclass folders."""
    if selected_classes:
        dirs = [data_root / name for name in selected_classes]
    else:
        dirs = [d for d in data_root.iterdir() if d.is_dir()]

    resolved: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            if sub.is_dir():
                good_dir = sub / "train" / "good"
                if good_dir.exists() and trainer.iter_images(good_dir, image_exts=tuple(IMAGE_EXTS)):
                    resolved.append(d)
                    break

    resolved.sort(key=lambda p: p.name)
    return resolved


def _get_subclass_dirs(parent_dir: Path, trainer: PatchCoreSIFETrainer) -> list[Path]:
    """Find subclass folders with train/good images."""
    subclasses = []
    for sub in parent_dir.iterdir():
        if sub.is_dir():
            good_dir = sub / "train" / "good"
            if good_dir.exists() and trainer.iter_images(good_dir, image_exts=tuple(IMAGE_EXTS)):
                subclasses.append(sub)
    subclasses.sort(key=lambda p: p.name)
    return subclasses


# =============================================================================
#                              MAIN
# =============================================================================

# GPU lock: only one thread does backbone forward pass at a time
_gpu_lock = threading.Lock()


def _train_one_subclass(
    subclass_dir: Path,
    parent_name: str,
    trainer: PatchCoreSIFETrainer,
    rng_seed: int,
) -> SubclassResult:
    """
    Train a single subclass end-to-end. Thread-safe via GPU lock.
    
    Pipeline per subclass:
        1. Load images from disk (parallel I/O, no lock)
        2. [GPU LOCK] Extract features via backbone
        3. Coreset subsampling (CPU, no lock)
        4. [GPU LOCK] Calibrate threshold (uses backbone for scoring)
        5. Save .pth file (I/O, no lock)
    
    Multiple subclasses run concurrently — CPU/IO phases overlap with
    other threads' GPU phases.
    """
    subclass_name = subclass_dir.name
    t0 = time.perf_counter()
    tag = f"[{parent_name}/{subclass_name}]"

    try:
        train_good = subclass_dir / "train" / "good"
        if not train_good.exists() or not trainer.iter_images(train_good, image_exts=tuple(IMAGE_EXTS)):
            print(f"  {tag} No train/good images -> skip")
            return SubclassResult(parent_name, subclass_name, success=False, error="no images")

        # Each thread gets its own RNG (deterministic per subclass)
        local_rng = np.random.default_rng(rng_seed)

        # --- 1. Build memory bank (GPU locked internally by trainer) ---
        print(f"  {tag} Building memory bank...")
        with _gpu_lock:
            bank = trainer.build_memory_bank_from_dir(
                train_good,
                local_rng,
                coreset_ratio=CORESET_RATIO,
                image_exts=tuple(IMAGE_EXTS),
            )
        if bank is None:
            print(f"  {tag} Failed to build memory bank -> skip")
            return SubclassResult(parent_name, subclass_name, success=False, error="bank failed")

        # --- 2. Calibrate threshold (GPU locked for scoring) ---
        print(f"  {tag} Calibrating threshold...")
        test_dir = subclass_dir / "test"
        if test_dir.exists():
            with _gpu_lock:
                threshold = trainer.calibrate_threshold_from_test_dir(
                    bank,
                    test_dir,
                    good_folder="good",
                    fallback_threshold=FALLBACK_THRESHOLD,
                    image_exts=tuple(IMAGE_EXTS),
                )
        else:
            threshold = FALLBACK_THRESHOLD

        # --- 3. Save (I/O, no lock) ---
        bank_tensor = torch.from_numpy(bank).contiguous().cpu()
        class_data = {
            "memory_bank": bank_tensor,
            "threshold": threshold,
            "meta": {
                "n_patches": int(bank.shape[0]),
                "feature_dim": int(bank.shape[1]),
                "created_at": datetime.now().isoformat(),
                "model_size": IMG_SIZE,
                "grid_size": GRID_SIZE,
                "k_nearest": K_NEAREST,
                "coreset_ratio": CORESET_RATIO,
                "seed": rng_seed,
                "parent_class": parent_name,
                "backbone": "MobileNetV3_SIFE_CUDA",
                "use_sife": USE_SIFE,
                "sife_dim": SIFE_DIM,
                "sife_encoding_type": SIFE_ENCODING_TYPE,
                "sife_weight": SIFE_WEIGHT,
                "use_center_distance": USE_CENTER_DISTANCE,
                "use_local_gradient": USE_LOCAL_GRADIENT,
                "use_color_features": USE_COLOR_FEATURES,
                "use_hsv": USE_HSV,
                "color_weight": COLOR_WEIGHT,
            },
        }
        save_class_pth(parent_name, subclass_name, class_data)

        elapsed = time.perf_counter() - t0
        print(f"  {tag} Done: patches={bank.shape[0]:,} dim={bank.shape[1]} thr={threshold:.4f} ({elapsed:.1f}s)")

        return SubclassResult(
            parent_name, subclass_name, success=True,
            n_patches=bank.shape[0], feature_dim=bank.shape[1],
            threshold=threshold, elapsed_sec=elapsed,
        )

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  {tag} ERROR: {e} ({elapsed:.1f}s)")
        return SubclassResult(parent_name, subclass_name, success=False, error=str(e), elapsed_sec=elapsed)


def main():
    print("=" * 70)
    print("  MobileNet + SIFE PatchCore Training [CUDA + Parallel]")
    print("=" * 70)
    print(f"Device          : {DEVICE}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"I/O Workers     : {NUM_WORKERS}")
    print(f"Parallel classes: {MAX_PARALLEL_CLASSES}")
    print(f"Image size      : {IMG_SIZE}x{IMG_SIZE}")
    print(f"Grid size       : {GRID_SIZE}x{GRID_SIZE}")
    print(f"Coreset ratio   : {CORESET_RATIO}")
    print(f"k-nearest       : {K_NEAREST}")
    print("-" * 70)

    # GPU info
    mem = gpu_memory_info()
    if mem.get("available"):
        print(f"GPU             : {mem['device_name']}")
        print(f"GPU Memory      : {mem['total_mb']:.0f} MB total, {mem['free_mb']:.0f} MB free")
    print("-" * 70)

    print("SIFE Settings:")
    print(f"  Enabled       : {USE_SIFE} (dim={SIFE_DIM}, type={SIFE_ENCODING_TYPE})")
    print(f"  Weight        : {SIFE_WEIGHT} | CenterDist={USE_CENTER_DISTANCE} | Gradient={USE_LOCAL_GRADIENT}")
    print(f"  Multi-scale   : {USE_MULTI_SCALE} {MULTI_SCALE_GRIDS if USE_MULTI_SCALE else ''}")
    print(f"  Edge enhance  : {USE_EDGE_ENHANCEMENT} (weight={EDGE_WEIGHT})")
    print("-" * 70)
    print(f"Data root       : {DATA_ROOT}")
    print(f"Output dir      : {MODEL_OUTPUT_DIR}")
    print("-" * 70)

    # Initialize shared PatchCore (CUDA)
    patchcore = PatchCoreSIFE(
        model_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        k_nearest=K_NEAREST,
        device=DEVICE,
        use_sife=USE_SIFE,
        sife_dim=SIFE_DIM,
        sife_encoding_type=SIFE_ENCODING_TYPE,
        sife_weight=SIFE_WEIGHT,
        use_center_distance=USE_CENTER_DISTANCE,
        use_local_gradient=USE_LOCAL_GRADIENT,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
        use_multi_scale=USE_MULTI_SCALE,
        multi_scale_grids=MULTI_SCALE_GRIDS,
        use_edge_enhancement=USE_EDGE_ENHANCEMENT,
        edge_weight=EDGE_WEIGHT,
        finetuned_backbone_path=FINETUNED_BACKBONE_PATH,
    )

    trainer = PatchCoreSIFETrainer(
        patchcore,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
    )

    # Resolve classes
    parent_dirs = _resolve_parent_classes(DATA_ROOT, SELECTED_CLASSES, trainer)
    if not parent_dirs:
        print("No classes found with valid subclass folders")
        return

    # Collect all (parent, subclass) pairs
    all_tasks: List[tuple] = []
    base_rng = np.random.default_rng(SEED)

    for parent_dir in parent_dirs:
        parent_name = parent_dir.name
        subclass_dirs = _get_subclass_dirs(parent_dir, trainer)
        if not subclass_dirs:
            print(f"[Skip] {parent_name}: no subclass with train/good")
            continue
        for subclass_dir in subclass_dirs:
            # Deterministic seed per subclass
            sub_seed = int(base_rng.integers(0, 2**31))
            all_tasks.append((subclass_dir, parent_name, sub_seed))

    total_tasks = len(all_tasks)
    print(f"\nTotal subclasses to train: {total_tasks}")
    for subdir, pname, _ in all_tasks:
        print(f"  {pname}/{subdir.name}")
    print("=" * 70)

    # --- Parallel training ---
    t_start = time.perf_counter()
    results: List[SubclassResult] = []

    with ThreadPoolExecutor(
        max_workers=MAX_PARALLEL_CLASSES,
        thread_name_prefix="train",
    ) as pool:
        futures = {
            pool.submit(_train_one_subclass, subdir, pname, trainer, seed): (pname, subdir.name)
            for subdir, pname, seed in all_tasks
        }

        for future in as_completed(futures):
            pname, sname = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"  [{pname}/{sname}] Unexpected error: {e}")
                results.append(SubclassResult(pname, sname, success=False, error=str(e)))

    # Clean up GPU memory
    clear_gpu_cache()

    total_elapsed = time.perf_counter() - t_start

    # --- Summary ---
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 70)
    print(f"  Training Complete! ({total_elapsed:.1f}s total)")
    print("=" * 70)

    if succeeded:
        print(f"\n  Saved {len(succeeded)} subclasses:")
        for r in sorted(succeeded, key=lambda x: (x.parent_name, x.subclass_name)):
            print(f"    {r.parent_name}/{r.subclass_name}.pth"
                  f"  patches={r.n_patches:,} dim={r.feature_dim} thr={r.threshold:.4f} ({r.elapsed_sec:.1f}s)")

    if failed:
        print(f"\n  Failed {len(failed)} subclasses:")
        for r in sorted(failed, key=lambda x: (x.parent_name, x.subclass_name)):
            print(f"    {r.parent_name}/{r.subclass_name}: {r.error}")

    print("=" * 70)


if __name__ == "__main__":
    main()
