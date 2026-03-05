"""
ResNet50 PatchCore Training (Optimized Production)
==================================================
• Deterministic
• Faster feature extraction
• Safer class filtering
• Clean logging
"""

import argparse
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

from Core_ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor
from Core_ResnetPatchCore.pipeline.train import TrainPipeline
from Core_ResnetPatchCore.utils.structure_manager import DatasetManager

from config.base import (
    DATA_ROOT, SELECTED_CLASSES, SEED,
    MODEL_OUTPUT_DIR, DEVICE,
)
from config.resnet import (
    BACKBONE,
    IMG_SIZE,
    GRID_SIZE,
    CORESET_RATIO,
    K_NEAREST,
    FALLBACK_THRESHOLD,
    USE_COLOR_FEATURES,
    USE_HSV,
    COLOR_WEIGHT,
    SCORE_METHOD,
)


# ─────────────────────────────────────────────
# Performance Setup
# ─────────────────────────────────────────────
def setup_environment():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        if torch.__version__ >= "2.0":
            torch.set_float32_matmul_precision("high")

        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# Safe class filtering
# ─────────────────────────────────────────────
def filter_classes(all_classes, selected):
    if not selected:
        return all_classes

    selected_set = set(selected)
    filtered = []

    for main_class, sub_class in all_classes:
        if main_class in selected_set or sub_class in selected_set:
            filtered.append((main_class, sub_class))

    return filtered


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():

    parser = argparse.ArgumentParser(description="ResNet50 PatchCore Training")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--classes", "-c", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--structure", type=str, default="auto",
                        choices=["auto", "nested", "flat"])

    args = parser.parse_args()

    setup_environment()

    # ── Backbone Handling ──
    backbone_path = args.backbone or BACKBONE
    if backbone_path:
        backbone_path = Path(backbone_path)
        if not backbone_path.exists() or backbone_path.suffix != ".pth":
            print(f"[Warning] Invalid backbone → fallback to ImageNet")
            backbone_path = None

    backbone_label = str(backbone_path) if backbone_path else "resnet50 (ImageNet)"

    # ── Dataset Manager ──
    mgr = DatasetManager(root=DATA_ROOT, auto_create=False)

    # ── Discover Classes ──
    all_classes = mgr.list_classes(require_train_good=True, structure=args.structure)

    selected = args.classes if args.classes else SELECTED_CLASSES
    classes = filter_classes(all_classes, selected)

    if not classes:
        print("❌ No valid classes found.")
        return

    # ── Print Config ──
    print("=" * 70)
    print("      ResNet50 PatchCore Training (Optimized)")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Backbone      : {backbone_label}")
    print(f"Image size    : {IMG_SIZE}")
    print(f"Grid size     : {GRID_SIZE}")
    print(f"Coreset ratio : {CORESET_RATIO}")
    print(f"k-nearest     : {K_NEAREST}")
    print(f"Score method  : {SCORE_METHOD}")
    print(f"Color feature : {USE_COLOR_FEATURES} | HSV: {USE_HSV}")
    print(f"Classes       : {selected if selected else 'ALL'}")
    print("=" * 70)

    # ── Dry Run ──
    if args.dry_run:
        print("\n📋 Dry Run:")
        for main_class, sub_class in classes:
            cp = mgr.get_class_path(main_class, sub_class, structure=args.structure)
            n_imgs = cp.count_images("train", "good") if cp else 0
            print(f"  • {main_class}/{sub_class} → {n_imgs} images")
        return

    # ── Init Extractor ──
    extractor = ResNet50FeatureExtractor(
        img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        device=DEVICE,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
        backbone_path=str(backbone_path) if backbone_path else None,
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

    # ── Training Loop ──
    total, failed = 0, 0
    t0 = datetime.now()

    print(f"\n🚀 Training {len(classes)} subclasses")
    print("=" * 70)

    with torch.inference_mode():

        for main_class, sub_class in classes:

            cp = mgr.get_class_path(main_class, sub_class, structure=args.structure)
            if not cp:
                failed += 1
                continue

            good_dir = cp.train_good
            out_path = MODEL_OUTPUT_DIR / main_class / f"{sub_class}.pth"

            print(f"\n▶ {main_class}/{sub_class}")
            print(f"  Images : {cp.count_images('train', 'good')}")
            print(f"  Output : {out_path}")

            result = pipeline.train_parent(
                parent_dir=good_dir,
                output_path=out_path,
            )

            if result:
                total += 1
                print("  ✅ Success")
            else:
                failed += 1
                print("  ❌ Failed")

    elapsed = datetime.now() - t0

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Training Complete")
    print(f"Success : {total}")
    print(f"Failed  : {failed}")
    print(f"Time    : {elapsed}")
    print(f"Saved   : {MODEL_OUTPUT_DIR}")
    print("=" * 70)

    summary = mgr.summary()
    print(f"\n📊 Dataset Summary")
    print(f"Classes : {summary['total_classes']}")
    print(f"Images  : {summary['total_images']:,}")


if __name__ == "__main__":
    main()