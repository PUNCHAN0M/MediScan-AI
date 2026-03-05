# scripts/train_patchcore.py
"""
Entry point — Train PatchCore models (per-subclass).
=====================================================
Usage::

    python -m scripts.train_patchcore
    python -m scripts.train_patchcore --classes Androxsil JANUMET
    python -m scripts.train_patchcore --classes JANUMET --subclass JANUMET_front
    python -m scripts.train_patchcore --dry-run
"""
from __future__ import annotations

import argparse
import torch
from datetime import datetime
from pathlib import Path

from core.config import Config
from core.device import get_device, setup_cuda, setup_seed
from modules.feature_extractor import ResNet50FeatureExtractor
from pipeline.train_pipeline import TrainPipeline
from core.utils import list_images_recursive


def main():
    parser = argparse.ArgumentParser(description="ResNet50 PatchCore Training")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--classes", "-c", nargs="*", default=None)
    parser.add_argument(
        "--subclass", "-s", type=str, default=None,
        help="Re-train ONLY this subclass (requires exactly one --classes)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Validate --subclass usage
    if args.subclass and (not args.classes or len(args.classes) != 1):
        parser.error("--subclass requires exactly one --classes")

    cfg = Config()
    device = get_device()
    setup_cuda()
    setup_seed(cfg.train.seed)

    # ── Backbone ──
    backbone_path = args.backbone or str(cfg.paths.backbone)
    if backbone_path:
        bp = Path(backbone_path)
        if not bp.exists() or bp.suffix != ".pth":
            print(f"[Warning] Invalid backbone '{backbone_path}' → fallback to ImageNet")
            backbone_path = None

    backbone_label = backbone_path or "resnet50 (ImageNet)"

    # ── Discover classes ──
    data_root = cfg.paths.data_root
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        return

    all_class_dirs = sorted(
        d for d in data_root.iterdir() if d.is_dir()
    )

    selected = args.classes or list(cfg.train.selected_classes)
    if selected:
        selected_set = set(selected)
        class_dirs = [d for d in all_class_dirs if d.name in selected_set]
    else:
        class_dirs = all_class_dirs

    if not class_dirs:
        print("No valid classes found.")
        return

    # ── Config summary ──
    print("=" * 70)
    print("      ResNet50 PatchCore Training")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Backbone      : {backbone_label}")
    print(f"Image size    : {cfg.image.image_size}")
    print(f"Grid size     : {cfg.backbone.grid_size}")
    print(f"Coreset ratio : {cfg.backbone.coreset_ratio}")
    print(f"k-nearest     : {cfg.backbone.k_nearest}")
    print(f"Score method  : {cfg.backbone.score_method}")
    print(f"Color feature : {cfg.backbone.use_color_features} | HSV: {cfg.backbone.use_hsv}")
    print(f"Classes       : {[d.name for d in class_dirs]}")
    print("=" * 70)

    if args.dry_run:
        print("\nDry Run:")
        for d in class_dirs:
            n = len(list_images_recursive(d))
            print(f"  {d.name} → {n} images")
        return

    # ── Init ──
    extractor = ResNet50FeatureExtractor(
        img_size=cfg.image.image_size,
        grid_size=cfg.backbone.grid_size,
        device=device,
        use_color_features=cfg.backbone.use_color_features,
        use_hsv=cfg.backbone.use_hsv,
        color_weight=cfg.backbone.color_weight,
        backbone_path=backbone_path,
    )

    bad_dir = Path(cfg.paths.bad_dir) if cfg.paths.bad_dir else None

    pipeline = TrainPipeline(
        extractor=extractor,
        coreset_ratio=cfg.backbone.coreset_ratio,
        seed=cfg.train.seed,
        fallback_threshold=cfg.backbone.fallback_threshold,
        k_nearest=cfg.backbone.k_nearest,
        score_method=cfg.backbone.score_method,
        batch_size=cfg.train.batch_size,
        bad_dir=bad_dir,
    )

    output_dir = cfg.paths.model_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ──
    total, failed = 0, 0
    t0 = datetime.now()

    if args.subclass:
        # ── Re-train single subclass ──
        class_dir = class_dirs[0]
        out_path = output_dir / f"{class_dir.name}.pth"

        print(f"\nRe-training subclass '{args.subclass}' of {class_dir.name}")
        print("=" * 70)

        with torch.inference_mode():
            result = pipeline.train_subclass(
                parent_dir=class_dir,
                subclass_name=args.subclass,
                output_path=out_path,
            )

        if result:
            total = 1
            print("  OK")
        else:
            failed = 1
            print("  FAILED")
    else:
        # ── Train all subclasses for each parent class ──
        print(f"\nTraining {len(class_dirs)} classes")
        print("=" * 70)

        with torch.inference_mode():
            for class_dir in class_dirs:
                out_path = output_dir / f"{class_dir.name}.pth"
                n_images = len(list_images_recursive(class_dir))

                print(f"\n> {class_dir.name}")
                print(f"  Images : {n_images}")
                print(f"  Output : {out_path}")

                result = pipeline.train_parent(
                    parent_dir=class_dir,
                    output_path=out_path,
                )

                if result:
                    total += 1
                    print("  OK")
                else:
                    failed += 1
                    print("  FAILED")

    elapsed = datetime.now() - t0
    print("\n" + "=" * 70)
    print(f"Done | Success: {total} | Failed: {failed} | Time: {elapsed}")
    print(f"Saved: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
