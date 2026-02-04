#!/usr/bin/env python3
# ResnetPatchCore/train_patchcore.py
"""
สคริปต์สำหรับฝึก ResNet18 PatchCore พร้อม Color Features

🎯 Best for:
- Color anomaly detection (white vs black pills)
- Small colored defects (black spots, discoloration)
- Pills with color-based quality criteria

Usage:
    python run_train_resnet.py
    # หรือ
    python ResnetPatchCore/train_patchcore.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

from ResnetPatchCore.core_shared.patchcore import ResNetPatchCore
from ResnetPatchCore.core_train.trainer import ResNetPatchCoreTrainer

# Import configurations
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

# Alias for compatibility
RESNET_IMG_SIZE = IMG_SIZE
RESNET_GRID_SIZE = GRID_SIZE
RESNET_CORESET_RATIO = CORESET_RATIO
RESNET_K_NEAREST = K_NEAREST
RESNET_FALLBACK_THRESHOLD = FALLBACK_THRESHOLD


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


def _resolve_parent_classes(data_root: Path, selected_classes, trainer: ResNetPatchCoreTrainer) -> list[Path]:
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


def _get_subclass_dirs(parent_dir: Path, trainer: ResNetPatchCoreTrainer) -> list[Path]:
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
    print("      ResNet18 PatchCore Training - Color-Aware Edition")
    print("=" * 70)
    print(f"Device          : {DEVICE}")
    print(f"Image size      : {RESNET_IMG_SIZE} × {RESNET_IMG_SIZE}")
    print(f"Grid size       : {RESNET_GRID_SIZE} × {RESNET_GRID_SIZE}")
    print(f"Color features  : {USE_COLOR_FEATURES}")
    print(f"HSV features    : {USE_HSV}")
    print(f"Color weight    : {COLOR_WEIGHT}")
    print(f"Coreset ratio   : {RESNET_CORESET_RATIO}")
    print(f"k-nearest       : {RESNET_K_NEAREST}")
    print(f"Fallback thr    : {RESNET_FALLBACK_THRESHOLD}")
    print(f"Data root       : {DATA_ROOT}")
    print(f"Model output    : {MODEL_OUTPUT_DIR}")
    print("=" * 70)
    
    # Create PatchCore and Trainer
    patchcore = ResNetPatchCore(
        model_size=RESNET_IMG_SIZE,
        grid_size=RESNET_GRID_SIZE,
        k_nearest=RESNET_K_NEAREST,
        device=DEVICE,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
    )
    trainer = ResNetPatchCoreTrainer(patchcore)
    
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
            
            # Build memory bank
            print(f"    Building memory bank from {good_dir}...")
            bank = trainer.build_memory_bank_from_dir(
                good_dir,
                rng,
                coreset_ratio=RESNET_CORESET_RATIO,
                image_exts=IMAGE_EXTS,
            )
            
            if bank is None or bank.shape[0] == 0:
                print(f"    [Skip] ไม่สามารถสร้าง memory bank ได้")
                continue
            
            print(f"    Memory bank shape: {bank.shape}")
            print(f"    Feature dimension: {bank.shape[1]}")
            
            # Calibrate threshold
            if test_dir.exists():
                print(f"    Calibrating threshold from {test_dir}...")
                threshold = trainer.calibrate_threshold_from_test_dir(
                    bank,
                    test_dir,
                    good_folder="good",
                    fallback_threshold=RESNET_FALLBACK_THRESHOLD,
                    image_exts=IMAGE_EXTS,
                )
            else:
                threshold = RESNET_FALLBACK_THRESHOLD
                print(f"    No test dir, using fallback threshold: {threshold}")
            
            print(f"    Threshold: {threshold:.4f}")
            
            # Save model
            class_data = {
                "memory_bank": torch.from_numpy(bank),
                "threshold": threshold,
                "model_size": RESNET_IMG_SIZE,
                "grid_size": RESNET_GRID_SIZE,
                "k_nearest": RESNET_K_NEAREST,
                "use_color_features": USE_COLOR_FEATURES,
                "use_hsv": USE_HSV,
                "color_weight": COLOR_WEIGHT,
                "backbone": "resnet50",
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
