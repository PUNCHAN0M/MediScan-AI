# scripts/run_infer.py
"""
Entry point — Single-image / folder inference (no camera).
============================================================
Usage::

    python -m scripts.run_infer --image path/to/image.jpg --classes Androxsil
    python -m scripts.run_infer --folder path/to/folder --classes Androxsil ANTA1mg
"""
from __future__ import annotations

import argparse
import cv2
from pathlib import Path

from core.config import Config
from core.device import get_device, setup_cuda, setup_seed
from core.utils import list_images
from pipeline.infer_pipeline import PillInspector, InspectorConfig


def main():
    parser = argparse.ArgumentParser(description="Single-image / folder inference")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--classes", "-c", nargs="*", required=True)
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder")

    cfg = Config()
    device = get_device()
    setup_cuda()
    setup_seed(cfg.train.seed)

    inspector_cfg = InspectorConfig.from_config(cfg, args.classes)
    inspector_cfg.device = device
    inspector = PillInspector(inspector_cfg)

    # gather images
    images = []
    if args.image:
        images.append(Path(args.image))
    if args.folder:
        images.extend(list_images(Path(args.folder)))

    print(f"Inspecting {len(images)} images for classes: {args.classes}\n")

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[Skip] Cannot read {img_path}")
            continue

        annotated = inspector.classify_anomaly(frame, class_names=args.classes)
        result = inspector.summarize()

        print(
            f"{img_path.name} | "
            f"count={result['count']} "
            f"good={result['good_count']} "
            f"bad={result['bad_count']}"
        )

        inspector.reset()

    print("\nDone.")


if __name__ == "__main__":
    main()
