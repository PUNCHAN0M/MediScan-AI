#!/usr/bin/env python3
# ResnetPatchCore/train_patchcore.py
"""
ResNet50 PatchCore Training
===========================

Flow
----
::

    data/{class}/{sub}/train/good/  (pre-cropped pills)
        ↓
    ResNet50 feature extraction  (layer2 + layer3 → 1536-dim)
        ↓
    memory bank  →  coreset subsample  →  L2 normalize
        ↓
    threshold calibration  (F1 sweep / percentile 99.5 / mean+3σ)
        ↓
    save  →  model/patchcore_resnet/{class}/{sub}.pth

No backprop — completes in minutes.

Usage
-----
::

    python run_train.py --model=resnet
    python ResnetPatchCore/train_patchcore.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datetime import datetime

from ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor
from ResnetPatchCore.pipeline.train import TrainPipeline

from config.base import DATA_ROOT, SELECTED_CLASSES, SEED, IMAGE_EXTS
from config.resnet import (
    IMG_SIZE,
    GRID_SIZE,
    CORESET_RATIO,
    K_NEAREST,
    FALLBACK_THRESHOLD,
    MODEL_OUTPUT_DIR,
    USE_COLOR_FEATURES,
    USE_HSV,
    COLOR_WEIGHT,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────── directory helpers ───────────────────
def _find_subclass_dirs(parent_dir: Path) -> list[Path]:
    """Return subclass dirs that contain train/good/ with images."""
    subs = []
    for d in sorted(parent_dir.iterdir()):
        if d.is_dir():
            good = d / "train" / "good"
            if good.is_dir() and TrainPipeline.list_images(good):
                subs.append(d)
    return subs


def _resolve_parents(data_root: Path) -> list[Path]:
    """Return parent class dirs that have valid subclass dirs."""
    if SELECTED_CLASSES:
        candidates = [data_root / c for c in SELECTED_CLASSES
                       if (data_root / c).is_dir()]
    else:
        candidates = [d for d in data_root.iterdir() if d.is_dir()]

    resolved = []
    for d in sorted(candidates, key=lambda p: p.name):
        if _find_subclass_dirs(d):
            resolved.append(d)
        else:
            print(f"[Skip] {d.name}: no valid subclass folders")
    return resolved


# ─────────────────── main ───────────────────
def main():
    print("=" * 70)
    print("      ResNet50 PatchCore Training")
    print("=" * 70)
    print(f"  Device        : {DEVICE}")
    print(f"  Backbone      : resnet50  (layer2 + layer3 → 1536-dim)")
    print(f"  Image size    : {IMG_SIZE} × {IMG_SIZE}")
    print(f"  Grid size     : {GRID_SIZE} × {GRID_SIZE}")
    print(f"  Color features: {USE_COLOR_FEATURES}  HSV: {USE_HSV}  weight: {COLOR_WEIGHT}")
    print(f"  Coreset ratio : {CORESET_RATIO}")
    print(f"  k-nearest     : {K_NEAREST}")
    print(f"  Fallback thr  : {FALLBACK_THRESHOLD}")
    print(f"  Data root     : {DATA_ROOT}")
    print(f"  Model output  : {MODEL_OUTPUT_DIR}")
    print("=" * 70)

    # ── init ──
    extractor = ResNet50FeatureExtractor(
        img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        device=DEVICE,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
    )

    pipeline = TrainPipeline(
        extractor=extractor,
        coreset_ratio=CORESET_RATIO,
        seed=SEED,
        fallback_threshold=FALLBACK_THRESHOLD,
        k_nearest=K_NEAREST,
    )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── resolve classes ──
    parents = _resolve_parents(Path(DATA_ROOT))
    if not parents:
        print("No valid parent class folders found.")
        return

    print(f"\nFound {len(parents)} parent classes:")
    for p in parents:
        subs = _find_subclass_dirs(p)
        print(f"  • {p.name}: {len(subs)} subclasses")
    print()

    # ── train ──
    total = 0
    t0 = datetime.now()

    for parent_dir in parents:
        parent_name = parent_dir.name
        print(f"\n{'=' * 60}")
        print(f"Parent Class: {parent_name}")
        print(f"{'=' * 60}")

        for sub_dir in _find_subclass_dirs(parent_dir):
            sub_name = sub_dir.name
            print(f"\n  Subclass: {sub_name}")
            print(f"  {'-' * 50}")

            good_dir = sub_dir / "train" / "good"
            test_dir = sub_dir / "test" if (sub_dir / "test").is_dir() else None
            out_path = MODEL_OUTPUT_DIR / parent_name / f"{sub_name}.pth"

            result = pipeline.train_class(
                good_dir=good_dir,
                output_path=out_path,
                test_dir=test_dir,
            )
            if result:
                total += 1

    elapsed = datetime.now() - t0
    print(f"\n{'=' * 70}")
    print(f"  Training Complete!")
    print(f"  Subclasses trained : {total}")
    print(f"  Time               : {elapsed}")
    print(f"  Models saved to    : {MODEL_OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
