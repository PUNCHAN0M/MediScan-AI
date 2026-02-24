#!/usr/bin/env python3
"""
Augment images from result/ → augmentation_result/

- รักษา folder structure เดิม (class folder ทุก folder)
- Copy ภาพต้นฉบับมาด้วย
- Augmentation: rotate (step 5°), flip, brightness
- ข้าม _aug files ที่มีอยู่แล้ว (idempotent)

Usage:
    python run_augmentation_result.py
    python run_augmentation_result.py --input result --output augmentation_result_bright --count 10 --rotation_step 5
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2


# =============================================================================
#                              CONFIG
# =============================================================================

DEFAULT_INPUT        = Path("result")
DEFAULT_OUTPUT       = Path("augmentation_result")
DEFAULT_AUG_COUNT    = 10       # augmented images per original
DEFAULT_ROTATION_STEP = 5       # degrees per step  (0, 5, 10, ..., 355)
DEFAULT_BRIGHTNESS   = 30       # max ±brightness delta

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# =============================================================================
#                              CORE AUGMENTATION
# =============================================================================

def augment_images(
    input_dir: Path,
    output_dir: Path,
    augment_count: int = 10,
    rotation_step: int = 5,
    brightness_max: int = 30,
    seed: int = None,
) -> dict:
    """
    Augment every original image in input_dir and save to output_dir.

    - Copies the original image first
    - Creates augment_count augmented variants per original
    - Skips files whose stem already contains '_aug'

    Returns dict with stats.
    """
    if seed is not None:
        random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect original images (not previously augmented ones)
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTS
        and "_aug" not in f.stem
    ])

    stats = {"originals": 0, "augmented": 0, "skipped": 0, "failed": 0}
    rotation_options = list(range(0, 360, rotation_step))   # [0, 5, 10, ..., 355]

    for img_path in image_files:
        stem   = img_path.stem
        suffix = img_path.suffix.lower()

        # ── Copy original ────────────────────────────────────────
        dest_orig = output_dir / img_path.name
        if not dest_orig.exists():
            shutil.copy2(img_path, dest_orig)
        stats["originals"] += 1

        # ── Read once for augmentation ───────────────────────────
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read: {img_path.name}")
            stats["failed"] += 1
            continue

        h, w   = img.shape[:2]
        center = (w // 2, h // 2)

        for aug_idx in range(1, augment_count + 1):
            out_path = output_dir / f"{stem}_aug{aug_idx}{suffix}"
            if out_path.exists():
                stats["skipped"] += 1
                continue

            aug = img.copy()

            # 1. Rotation (random multiple of rotation_step)
            angle = random.choice(rotation_options)
            if angle != 0:
                M   = cv2.getRotationMatrix2D(center, angle, 1.0)
                aug = cv2.warpAffine(
                    aug, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

            # 2. Random vertical flip
            if random.random() > 0.5:
                aug = cv2.flip(aug, 0)

            # 3. Random horizontal flip
            if random.random() > 0.5:
                aug = cv2.flip(aug, 1)

            # 4. Brightness adjustment
            delta = random.uniform(-brightness_max, brightness_max)
            aug   = cv2.convertScaleAbs(aug, alpha=1.0, beta=delta)

            cv2.imwrite(str(out_path), aug)
            stats["augmented"] += 1

    return stats


# =============================================================================
#                              RECURSIVE RUNNER
# =============================================================================

def process_all_classes(
    input_root: Path,
    output_root: Path,
    augment_count: int,
    rotation_step: int,
    brightness_max: int,
):
    """
    Walk every immediate subdirectory (class folder) in input_root,
    augment images, and mirror the structure under output_root.
    """
    class_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"No class folders found in {input_root}")
        return

    print(f"Found {len(class_dirs)} class folders")
    print(f"Input  : {input_root}")
    print(f"Output : {output_root}")
    print(f"Aug/img: {augment_count}  |  Rotation step: {rotation_step}°  |  Brightness ±{brightness_max}")
    print("=" * 65)

    total = {"originals": 0, "augmented": 0, "skipped": 0, "failed": 0}

    for i, class_dir in enumerate(class_dirs, 1):
        out_dir = output_root / class_dir.name
        stats   = augment_images(
            input_dir      = class_dir,
            output_dir     = out_dir,
            augment_count  = augment_count,
            rotation_step  = rotation_step,
            brightness_max = brightness_max,
        )

        orig = stats["originals"]
        aug  = stats["augmented"]
        skip = stats["skipped"]
        print(
            f"[{i:3d}/{len(class_dirs)}] {class_dir.name[:55]:<55s}  "
            f"orig={orig:2d}  new_aug={aug:3d}  skipped={skip}"
        )
        for k in total:
            total[k] += stats[k]

    print("=" * 65)
    print(f"  Total originals  : {total['originals']}")
    print(f"  Total augmented  : {total['augmented']}")
    print(f"  Skipped (existed): {total['skipped']}")
    print(f"  Failed           : {total['failed']}")
    print(f"\n  Output saved to  : {output_root.resolve()}")


# =============================================================================
#                              MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Augment result/ images → augmentation_result/")
    parser.add_argument("--input",          type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output",         type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count",          type=int,  default=DEFAULT_AUG_COUNT,
                        help="Augmented images per original")
    parser.add_argument("--rotation_step",  type=int,  default=DEFAULT_ROTATION_STEP,
                        help="Rotation step in degrees (e.g. 5 → 0,5,10,...,355)")
    parser.add_argument("--brightness",     type=int,  default=DEFAULT_BRIGHTNESS,
                        help="Max brightness delta ±")
    parser.add_argument("--seed",           type=int,  default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.input.exists():
        print(f"[ERROR] Input directory not found: {args.input}")
        return

    process_all_classes(
        input_root     = args.input,
        output_root    = args.output,
        augment_count  = args.count,
        rotation_step  = args.rotation_step,
        brightness_max = args.brightness,
    )


if __name__ == "__main__":
    #python run_augmentation_result.py --input result --output augmentation_result --count 10 --rotation_step 5 --brightness 30
    main()
