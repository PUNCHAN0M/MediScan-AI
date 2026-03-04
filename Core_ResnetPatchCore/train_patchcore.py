# Core_ResnetPatchCore/train_patchcore.py
"""
ResNet50 PatchCore Training — ใช้ DatasetManager + SELECTED_CLASSES
====================================================================
Usage
-----
::
python Core_ResnetPatchCore/train_patchcore.py
python Core_ResnetPatchCore/train_patchcore.py --backbone model/backbone/resnet_last.pth
python Core_ResnetPatchCore/train_patchcore.py --classes paracap vitaminc  # override SELECTED_CLASSES
python Core_ResnetPatchCore/train_patchcore.py --dry-run
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datetime import datetime
from Core_ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor
from Core_ResnetPatchCore.pipeline.train import TrainPipeline
from Core_ResnetPatchCore.utils.structure_manager import DatasetManager
from config.base import (
    DATA_ROOT, SELECTED_CLASSES, SEED, IMAGE_EXTS,
    MODEL_OUTPUT_DIR, DEVICE,
)
from config.resnet import (
    BACKBONE,
    IMG_SIZE,
    GRID_SIZE,
    CORESET_RATIO,
    CORESET_MIN_KEEP,
    CORESET_MAX_KEEP,
    K_NEAREST,
    FALLBACK_THRESHOLD,
    USE_COLOR_FEATURES,
    USE_HSV,
    COLOR_WEIGHT,
    SCORE_METHOD,
)


# ─────────────────── main ───────────────────
def main():
    parser = argparse.ArgumentParser(description="ResNet50 PatchCore Training")
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a custom backbone .pth file.",
    )
    parser.add_argument(
        "--classes", "-c",
        nargs="*",
        default=None,
        help="Train specific main classes only (override SELECTED_CLASSES from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be trained without executing",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="auto",
        choices=["auto", "nested", "flat"],
        help="Dataset structure type (default: auto-detect)",
    )
    args, _ = parser.parse_known_args()

    # CLI --backbone overrides config
    backbone_path = args.backbone or BACKBONE
    if backbone_path and not Path(backbone_path).suffix == ".pth":
        backbone_path = None
    if backbone_path and not Path(backbone_path).exists():
        print(f"  [Warning] Backbone not found: {backbone_path}")
        print(f"  → ใช้ resnet50 (ImageNet pretrained) แทน")
        backbone_path = None

    backbone_label = backbone_path if backbone_path else "resnet50 (ImageNet pretrained)"

    # ── Initialize DatasetManager ──
    mgr = DatasetManager(root=DATA_ROOT, auto_create=False)

    # ── Discover classes to train ──
    # ✅ ใช้ SELECTED_CLASSES จาก config ถ้าไม่ระบุ --classes ทาง CLI
    if args.classes:
        # CLI override: ใช้ classes จาก command line
        selected = args.classes
        print(f"\n📋 Training specific classes (CLI override): {selected}")
    elif SELECTED_CLASSES:
        # ใช้ SELECTED_CLASSES จาก config/base.py
        selected = SELECTED_CLASSES
        print(f"\n📋 Training specific classes (from config): {selected}")
    else:
        # Train ทุก class
        selected = None
        print(f"\n📋 Training ALL classes (SELECTED_CLASSES is empty)")

    # List classes จาก DatasetManager
    all_classes = mgr.list_classes(require_train_good=True, structure=args.structure)

    # ✅ Filter ด้วย SELECTED_CLASSES (ถ้ามี)
    if selected:
        classes = [
            (m, s) for m, s in all_classes
            if m in selected or any(m in c for c in selected)
        ]
        print(f"   Found {len(classes)} matching subclasses from {len(all_classes)} total")
        
        # เตือนถ้าไม่พบ class ที่ระบุ
        matched_names = {m for m, s in classes}
        unmatched = [c for c in selected if c not in matched_names and not any(c in m for m in matched_names)]
        if unmatched:
            print(f"   ⚠️  Warning: These classes not found: {unmatched}")
    else:
        classes = all_classes
        print(f"   Total: {len(classes)} subclasses")

    if not classes:
        print("❌ No valid class folders found.")
        print(f"   Data root: {DATA_ROOT}")
        print(f"   Structure: {args.structure}")
        if selected:
            print(f"   Selected classes: {selected}")
        return

    # ── Print config ──
    print("=" * 70)
    print("      ResNet50 PatchCore Training")
    print("=" * 70)
    print(f"  Device        : {DEVICE}")
    print(f"  Backbone      : {backbone_label}")
    print(f"  Image size    : {IMG_SIZE} × {IMG_SIZE}")
    print(f"  Grid size     : {GRID_SIZE} × {GRID_SIZE}")
    print(f"  Color features: {USE_COLOR_FEATURES}  HSV: {USE_HSV}  weight: {COLOR_WEIGHT}")
    print(f"  Score method  : {SCORE_METHOD}")
    print(f"  Coreset ratio : {CORESET_RATIO}")
    print(f"  k-nearest     : {K_NEAREST}")
    print(f"  Fallback thr  : {FALLBACK_THRESHOLD}")
    print(f"  Data root     : {DATA_ROOT}")
    print(f"  Model output  : {MODEL_OUTPUT_DIR}")
    print(f"  Structure     : {args.structure}")
    print(f"  Selected      : {selected if selected else 'ALL'}")
    print("=" * 70)

    # ── Dry run ──
    if args.dry_run:
        print(f"\n📋 Training Plan: {len(classes)} subclasses")
        for main_class, sub_class in classes:
            cp = mgr.get_class_path(main_class, sub_class, structure=args.structure)
            if cp:
                n_imgs = cp.count_images("train", "good")
                print(f"   • {main_class}/{sub_class}: {n_imgs} images")
            else:
                print(f"   • {main_class}/{sub_class}: [path not found]")
        print("\n✅ Dry run complete (no training executed)")
        return

    # ── init components ──
    extractor = ResNet50FeatureExtractor(
        img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        device=DEVICE,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
        backbone_path=backbone_path,
    )

    pipeline = TrainPipeline(
        extractor=extractor,
        coreset_ratio=CORESET_RATIO,
        seed=SEED,
        fallback_threshold=FALLBACK_THRESHOLD,
        k_nearest=K_NEAREST,
        score_method=SCORE_METHOD,
    )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── train ──
    total = 0
    failed = 0
    t0 = datetime.now()

    print(f"\n🚀 Starting training: {len(classes)} subclasses")
    print("=" * 70)

    for main_class, sub_class in classes:
        # ✅ ใช้ DatasetManager เข้าถึง path
        cp = mgr.get_class_path(main_class, sub_class, structure=args.structure)
        if not cp:
            print(f"\n[Skip] {main_class}/{sub_class}: path not found")
            failed += 1
            continue

        good_dir = cp.train_good
        out_path = MODEL_OUTPUT_DIR / main_class / f"{sub_class}.pth"

        print(f"\n{'=' * 60}")
        print(f"Parent Class: {main_class}")
        print(f"Subclass    : {sub_class}")
        print(f"{'=' * 60}")
        print(f"  Good dir : {good_dir}")
        print(f"  Images   : {cp.count_images('train', 'good')}")
        print(f"  Output   : {out_path}")

        result = pipeline.train_class(
            good_dir=good_dir,
            output_path=out_path,
        )

        if result:
            total += 1
            print(f"  ✅ Saved: {out_path}")
        else:
            failed += 1
            print(f"  ❌ Failed: {out_path}")

    elapsed = datetime.now() - t0

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Training Complete!")
    print(f"  Subclasses trained : {total}")
    print(f"  Subclasses failed  : {failed}")
    print(f"  Time               : {elapsed}")
    print(f"  Models saved to    : {MODEL_OUTPUT_DIR}")
    print(f"{'=' * 70}")

    # ── Dataset Summary ──
    summary = mgr.summary()
    print(f"\n📊 Dataset Summary:")
    print(f"   Total classes: {summary['total_classes']}")
    print(f"   Total images:  {summary['total_images']:,}")


if __name__ == "__main__":
    main()