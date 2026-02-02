# train_patchcore_dinov2.py
"""
สคริปต์สำหรับฝึก PatchCore ด้วย DINOv2 backbone
- ใช้ DINOv2 (ViT) แทน MobileNetV3
- ดีกว่าในการแยกสีและ texture
- สร้าง memory bank จากภาพ good (train set)
- บันทึกแยกไฟล์ต่อ subclass
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

from DINOv2PatchCore.core_dinov2 import DINOv2PatchCore
from DINOv2PatchCore.trainer_dinov2 import DINOv2PatchCoreTrainer


# =========================================================
#               CONFIGURATION
# =========================================================
DATA_ROOT = Path("/home/punchan0m/project/MediScan-AI/MediScan-AI/data")
MODEL_OUTPUT_DIR = Path("./model/patchcore_dinov2")  # แยกโฟลเดอร์จาก MobileNet

# DINOv2 Parameters
IMG_SIZE = 252          # Must be divisible by 14 (DINOv2 patch size)
GRID_SIZE = 18          # 18x18 = 324 patches
CORESET_RATIO = 0.18    # Memory bank ratio
K_NEAREST = 19          # k-nearest neighbors
BACKBONE_SIZE = "base"  # "small" (fast), "base" (balanced), "large" (best quality)
MULTI_SCALE = True      # Extract multi-scale features

# Selected classes to train
SELECTED_CLASSES = [
    'vitaminc',
    # 'oval_orange',  # ถ้ามี - แยกสีส้ม
    # 'oval_white',   # ถ้ามี - แยกสีขาว
    # 'white',
    # 'paracap',
]

SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FALLBACK_THRESHOLD = 0.40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_class_pth(parent_class: str, subclass_name: str, class_data: dict) -> Path:
    """บันทึกข้อมูล subclass เป็นไฟล์ .pth"""
    output_dir = MODEL_OUTPUT_DIR / parent_class
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subclass_name}.pth"
    torch.save(class_data, output_path)
    print(f"  บันทึก → {output_path}")
    return output_path


def _resolve_parent_classes(data_root: Path, selected_classes, trainer: DINOv2PatchCoreTrainer) -> list[Path]:
    """หา parent class folders"""
    if selected_classes:
        dirs = [data_root / name for name in selected_classes]
    else:
        dirs = [d for d in data_root.iterdir() if d.is_dir()]

    resolved: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        has_valid_subclass = False
        for sub in d.iterdir():
            if sub.is_dir():
                good_dir = sub / "train" / "good"
                if good_dir.exists() and trainer.iter_images(good_dir, image_exts=tuple(IMAGE_EXTS)):
                    has_valid_subclass = True
                    break
        if has_valid_subclass:
            resolved.append(d)

    resolved.sort(key=lambda p: p.name)
    return resolved


def _get_subclass_dirs(parent_dir: Path, trainer: DINOv2PatchCoreTrainer) -> list[Path]:
    """หา subclass folders"""
    subclasses = []
    for sub in parent_dir.iterdir():
        if sub.is_dir():
            good_dir = sub / "train" / "good"
            if good_dir.exists() and trainer.iter_images(good_dir, image_exts=tuple(IMAGE_EXTS)):
                subclasses.append(sub)
    subclasses.sort(key=lambda p: p.name)
    return subclasses


def main():
    print("=" * 70)
    print("     DINOv2 PatchCore Training - Better Color/Texture Detection")
    print("=" * 70)
    print(f"Device          : {DEVICE}")
    print(f"Backbone        : DINOv2 {BACKBONE_SIZE}")
    print(f"Image size      : {IMG_SIZE} × {IMG_SIZE}")
    print(f"Grid size       : {GRID_SIZE} × {GRID_SIZE}")
    print(f"Coreset ratio   : {CORESET_RATIO}")
    print(f"k-nearest       : {K_NEAREST}")
    print(f"Multi-scale     : {MULTI_SCALE}")
    print(f"Data root       : {DATA_ROOT}")
    print(f"Output dir      : {MODEL_OUTPUT_DIR}")
    print("-" * 70)

    rng = np.random.default_rng(SEED)

    # Initialize DINOv2 PatchCore
    patchcore = DINOv2PatchCore(
        model_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        k_nearest=K_NEAREST,
        device=DEVICE,
        backbone_size=BACKBONE_SIZE,
        multi_scale=MULTI_SCALE,
    )
    
    trainer = DINOv2PatchCoreTrainer(patchcore)

    parent_dirs = _resolve_parent_classes(DATA_ROOT, SELECTED_CLASSES, trainer)
    if not parent_dirs:
        print("ไม่พบ class ใด ๆ ที่มี subclass พร้อมภาพ")
        return

    print("Parent classes ที่จะ train:", [d.name for d in parent_dirs])
    saved_classes = {}

    for parent_dir in parent_dirs:
        parent_name = parent_dir.name
        subclass_dirs = _get_subclass_dirs(parent_dir, trainer)
        
        if not subclass_dirs:
            print(f"\n[Skip] {parent_name}: ไม่พบ subclass ที่มี train/good")
            continue
        
        print(f"\n{'='*70}")
        print(f"Training parent class → {parent_name}")
        print(f"  Subclasses: {[s.name for s in subclass_dirs]}")
        
        saved_classes[parent_name] = []

        for subclass_dir in subclass_dirs:
            subclass_name = subclass_dir.name
            print(f"\n  --- Training subclass: {subclass_name} ---")

            train_good = subclass_dir / "train" / "good"
            if not train_good.exists() or not trainer.iter_images(train_good, image_exts=tuple(IMAGE_EXTS)):
                print("    ไม่พบ train/good หรือไม่มีภาพ → ข้าม")
                continue

            # 1. สร้าง memory bank
            print("    1. Building memory bank with DINOv2...")
            bank = trainer.build_memory_bank_from_dir(
                train_good,
                rng,
                coreset_ratio=CORESET_RATIO,
                image_exts=tuple(IMAGE_EXTS),
            )
            if bank is None:
                print("    ไม่สามารถสร้าง memory bank ได้ → ข้าม")
                continue

            # 2. Calibrate threshold
            print("    2. Calibrating threshold...")
            test_dir = subclass_dir / "test"
            if test_dir.exists():
                threshold = trainer.calibrate_threshold_from_test_dir(
                    bank,
                    test_dir,
                    good_folder="good",
                    fallback_threshold=FALLBACK_THRESHOLD,
                    image_exts=tuple(IMAGE_EXTS),
                )
            else:
                threshold = FALLBACK_THRESHOLD

            # เก็บข้อมูล
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
                    "seed": SEED,
                    "parent_class": parent_name,
                    "backbone": f"DINOv2_{BACKBONE_SIZE}",
                    "multi_scale": MULTI_SCALE,
                }
            }

            save_class_pth(parent_name, subclass_name, class_data)
            saved_classes[parent_name].append(subclass_name)
            print(f"    {subclass_name}: patches={bank.shape[0]:,} dim={bank.shape[1]} threshold={threshold:.4f}")

    # สรุปผล
    total_saved = sum(len(subs) for subs in saved_classes.values())
    if total_saved > 0:
        print(f"\nบันทึกสำเร็จ {total_saved} subclasses:")
        for parent, subs in saved_classes.items():
            if subs:
                print(f"  {parent}/")
                for sub in subs:
                    print(f"    - {sub}.pth")
    else:
        print("\nไม่มี class ใดถูกบันทึกสำเร็จ")

    print("\n" + "="*70)
    print("            การฝึก DINOv2 PatchCore เสร็จสมบูรณ์!")
    print("="*70)


if __name__ == "__main__":
    main()
