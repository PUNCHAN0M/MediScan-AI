#!/usr/bin/env python3
# CNNMultiScale/train_patchcore.py
"""
สคริปต์สำหรับฝึก CNN Multi-Scale PatchCore

🔥 Optimized for tiny defect detection (2-5px cracks):
- Modified ResNet34 (no maxpool, stride=1 conv1) → 4x more resolution
- Separate memory bank per scale (layer1, layer2, layer3)
- CLAHE preprocessing for micro-crack contrast
- Multi-resolution input (512 + 768)
- Adaptive threshold (mean + k*std)

Usage:
    python run_train_cnnmultiscale.py
    # หรือ
    python CNNMultiScale/train_patchcore.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

from CNNMultiScale.core_shared.patchcore_multiscale import CNNMultiScalePatchCore
from CNNMultiScale.core_train.trainer import CNNMultiScaleTrainer

# Import configurations
from config.base import DATA_ROOT, SELECTED_CLASSES, SEED, IMAGE_EXTS
from config.cnnmultiscale import (
    IMG_SIZE,
    IMG_SIZE_SECONDARY,
    ENABLE_MULTI_RESOLUTION,
    GRID_SIZE,
    CORESET_RATIO,
    K_NEAREST,
    FALLBACK_THRESHOLD,
    MODEL_OUTPUT_DIR,
    # Backbone
    BACKBONE,
    REMOVE_MAXPOOL,
    STRIDE1_CONV1,
    USE_DILATED_LAYER3,
    SELECTED_LAYERS,
    # Fusion
    SCORE_FUSION,
    SCALE_WEIGHTS,
    SEPARATE_MEMORY_PER_SCALE,
    # Preprocessing
    USE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    USE_LAPLACIAN_BOOST,
    LAPLACIAN_WEIGHT,
    # Attention
    USE_SE_ATTENTION,
    SE_REDUCTION,
    # Color
    USE_COLOR_FEATURES,
    USE_HSV,
    COLOR_WEIGHT,
    # Threshold
    THRESHOLD_METHOD,
    THRESHOLD_SIGMA,
    THRESHOLD_PERCENTILE,
)


# =============================================================================
#                              RUNTIME SETTINGS
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_class_pth(parent_class: str, subclass_name: str, class_data: dict) -> Path:
    """บันทึกข้อมูล subclass เป็นไฟล์ .pth ภายใต้โฟลเดอร์ parent class"""
    output_dir = MODEL_OUTPUT_DIR / parent_class
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subclass_name}.pth"
    torch.save(class_data, output_path)
    print(f"  บันทึก → {output_path}")
    return output_path


def _resolve_parent_classes(data_root: Path, selected_classes, trainer: CNNMultiScaleTrainer) -> list[Path]:
    """หา parent class folders ที่มี subclass folders ภายใน"""
    if selected_classes:
        dirs = [data_root / c for c in selected_classes if (data_root / c).is_dir()]
    else:
        dirs = [d for d in data_root.iterdir() if d.is_dir()]

    resolved: list[Path] = []
    for d in dirs:
        subs = _get_subclass_dirs(d, trainer)
        if subs:
            resolved.append(d)
        else:
            print(f"[Skip] {d.name}: ไม่พบ subclass folder หรือไม่มีภาพ train/good")

    resolved.sort(key=lambda p: p.name)
    return resolved


def _get_subclass_dirs(parent_dir: Path, trainer: CNNMultiScaleTrainer) -> list[Path]:
    """หา subclass folders ภายใน parent class folder"""
    subclasses = []
    for sub in parent_dir.iterdir():
        if sub.is_dir():
            good_dir = sub / "train" / "good"
            if good_dir.is_dir() and trainer.iter_images(good_dir, IMAGE_EXTS):
                subclasses.append(sub)
    subclasses.sort(key=lambda p: p.name)
    return subclasses


def main():
    print("=" * 70)
    print("    CNN Multi-Scale PatchCore Training — Tiny Defect Edition")
    print("=" * 70)
    print(f"Device            : {DEVICE}")
    print(f"Backbone          : Modified {BACKBONE}")
    print(f"  stride1_conv1   : {STRIDE1_CONV1}")
    print(f"  remove_maxpool  : {REMOVE_MAXPOOL}")
    print(f"  dilated_layer3  : {USE_DILATED_LAYER3}")
    print(f"  SE attention    : {USE_SE_ATTENTION}")
    print(f"Input size        : {IMG_SIZE}×{IMG_SIZE}" +
          (f" + {IMG_SIZE_SECONDARY}×{IMG_SIZE_SECONDARY}" if ENABLE_MULTI_RESOLUTION else ""))
    print(f"Grid size         : {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE**2} patches")
    print(f"Selected layers   : {SELECTED_LAYERS}")
    print(f"Score fusion      : {SCORE_FUSION}")
    print(f"Separate memory   : {SEPARATE_MEMORY_PER_SCALE}")
    print(f"CLAHE             : {'ON' if USE_CLAHE else 'OFF'} (clip={CLAHE_CLIP_LIMIT})")
    print(f"Multi-resolution  : {'ON' if ENABLE_MULTI_RESOLUTION else 'OFF'}")
    print(f"Coreset ratio     : {CORESET_RATIO}")
    print(f"k-nearest         : {K_NEAREST}")
    print(f"Threshold method  : {THRESHOLD_METHOD} (sigma={THRESHOLD_SIGMA})")
    print(f"Fallback thr      : {FALLBACK_THRESHOLD}")
    print(f"Color features    : {USE_COLOR_FEATURES} (weight={COLOR_WEIGHT})")
    print(f"Data root         : {DATA_ROOT}")
    print(f"Model output      : {MODEL_OUTPUT_DIR}")
    print("=" * 70)

    # Create PatchCore and Trainer
    patchcore = CNNMultiScalePatchCore(
        model_size=IMG_SIZE,
        model_size_secondary=IMG_SIZE_SECONDARY,
        enable_multi_resolution=ENABLE_MULTI_RESOLUTION,
        grid_size=GRID_SIZE,
        k_nearest=K_NEAREST,
        device=DEVICE,
        backbone=BACKBONE,
        remove_maxpool=REMOVE_MAXPOOL,
        stride1_conv1=STRIDE1_CONV1,
        use_dilated_layer3=USE_DILATED_LAYER3,
        selected_layers=list(SELECTED_LAYERS),
        score_fusion=SCORE_FUSION,
        scale_weights=list(SCALE_WEIGHTS),
        separate_memory_per_scale=SEPARATE_MEMORY_PER_SCALE,
        use_clahe=USE_CLAHE,
        clahe_clip_limit=CLAHE_CLIP_LIMIT,
        clahe_tile_size=CLAHE_TILE_SIZE,
        use_laplacian_boost=USE_LAPLACIAN_BOOST,
        laplacian_weight=LAPLACIAN_WEIGHT,
        use_se_attention=USE_SE_ATTENTION,
        se_reduction=SE_REDUCTION,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
    )
    trainer = CNNMultiScaleTrainer(patchcore)

    # Create output directory
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # RNG for reproducibility
    rng = np.random.default_rng(SEED)

    # Resolve parent classes
    parent_dirs = _resolve_parent_classes(Path(DATA_ROOT), SELECTED_CLASSES, trainer)

    if not parent_dirs:
        print("ไม่พบ parent class folder ที่มี subclass folder ที่ valid")
        return

    print(f"\nพบ {len(parent_dirs)} parent classes:")
    for p in parent_dirs:
        subs = _get_subclass_dirs(p, trainer)
        print(f"  • {p.name}: {len(subs)} subclasses")
    print()

    # Process each parent class
    total_subclasses = 0
    start_time = datetime.now()

    for parent_dir in parent_dirs:
        parent_name = parent_dir.name
        print(f"\n{'='*60}")
        print(f"Parent Class: {parent_name}")
        print(f"{'='*60}")

        subclass_dirs = _get_subclass_dirs(parent_dir, trainer)

        for sub_dir in subclass_dirs:
            sub_name = sub_dir.name
            print(f"\n  Subclass: {sub_name}")
            print(f"  {'-'*50}")

            good_dir = sub_dir / "train" / "good"
            test_dir = sub_dir / "test"

            # Build per-scale memory banks
            print(f"    Building multi-scale memory banks from {good_dir}...")
            scale_banks = trainer.build_multiscale_memory_banks(
                good_dir,
                rng,
                coreset_ratio=CORESET_RATIO,
                image_exts=IMAGE_EXTS,
                use_multi_resolution=ENABLE_MULTI_RESOLUTION,
            )

            if scale_banks is None:
                print(f"    [Skip] ไม่สามารถสร้าง memory bank ได้")
                continue

            print(f"    Memory banks created for {len(scale_banks)} scales:")
            for layer_name, bank in scale_banks.items():
                print(f"      {layer_name}: {bank.shape}")

            # Calibrate threshold
            if test_dir.exists():
                print(f"    Calibrating threshold from {test_dir}...")
                threshold = trainer.calibrate_threshold_from_test_dir(
                    scale_banks,
                    test_dir,
                    good_folder="good",
                    method=THRESHOLD_METHOD,
                    sigma=THRESHOLD_SIGMA,
                    percentile=THRESHOLD_PERCENTILE,
                    fallback_threshold=FALLBACK_THRESHOLD,
                    image_exts=IMAGE_EXTS,
                    use_multi_resolution=ENABLE_MULTI_RESOLUTION,
                )
            else:
                # Try calibrate from training good images
                threshold = trainer.calibrate_adaptive_threshold(
                    scale_banks,
                    good_dir,
                    method=THRESHOLD_METHOD,
                    sigma=THRESHOLD_SIGMA,
                    percentile=THRESHOLD_PERCENTILE,
                    fallback_threshold=FALLBACK_THRESHOLD,
                    image_exts=IMAGE_EXTS,
                    use_multi_resolution=ENABLE_MULTI_RESOLUTION,
                )
                print(f"    No test dir, adaptive threshold: {threshold:.4f}")

            print(f"    Final threshold: {threshold:.4f}")

            # Save model with per-scale memory banks
            class_data = {
                # Per-scale memory banks
                "scale_memory_banks": {
                    layer_name: torch.from_numpy(bank)
                    for layer_name, bank in scale_banks.items()
                },
                "threshold": threshold,
                # Config for reproducibility
                "model_size": IMG_SIZE,
                "model_size_secondary": IMG_SIZE_SECONDARY,
                "enable_multi_resolution": ENABLE_MULTI_RESOLUTION,
                "grid_size": GRID_SIZE,
                "k_nearest": K_NEAREST,
                "backbone": BACKBONE,
                "remove_maxpool": REMOVE_MAXPOOL,
                "stride1_conv1": STRIDE1_CONV1,
                "use_dilated_layer3": USE_DILATED_LAYER3,
                "selected_layers": list(SELECTED_LAYERS),
                "score_fusion": SCORE_FUSION,
                "scale_weights": list(SCALE_WEIGHTS),
                "separate_memory_per_scale": SEPARATE_MEMORY_PER_SCALE,
                "use_clahe": USE_CLAHE,
                "clahe_clip_limit": CLAHE_CLIP_LIMIT,
                "use_se_attention": USE_SE_ATTENTION,
                "use_color_features": USE_COLOR_FEATURES,
                "use_hsv": USE_HSV,
                "color_weight": COLOR_WEIGHT,
                "threshold_method": THRESHOLD_METHOD,
                "threshold_sigma": THRESHOLD_SIGMA,
            }

            save_class_pth(parent_name, sub_name, class_data)
            total_subclasses += 1

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Total subclasses trained: {total_subclasses}")
    print(f"Total time: {elapsed}")
    print(f"Models saved to: {MODEL_OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
