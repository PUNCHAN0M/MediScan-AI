#!/usr/bin/env python3
"""
ResNet50 PatchCore — Folder-based Prediction
=============================================
Predict pill anomalies from a folder of images.

Usage:
    python run_predict.py
    python run_predict.py --input=data_yolo/test --output=result_resnet
"""
import argparse
import sys
import cv2
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# =============================================================================
#  ResNet PatchCore prediction
# =============================================================================

def run_predict(input_dir: Path, output_dir: Path):
    """
    Folder-based prediction using ResNet50 PatchCore.

    Pipeline:
      1. Load inspector (same as realtime camera uses)
      2. Read each image from input folder
      3. Run YOLO detect + PatchCore classify
      4. Save annotated result image + print scores
    """
    from Core_ResnetPatchCore.pipeline.infer import PillInspector, InspectorConfig
    from config.base import COMPARE_CLASSES

    # InspectorConfig defaults already come from config/base.py + config/resnet.py
    config = InspectorConfig(
        compare_classes=list(COMPARE_CLASSES),
    )

    print("=" * 60)
    print("  Folder Prediction — ResNet50 PatchCore (Color-Aware)")
    print("=" * 60)
    print(f"Input   : {input_dir}")
    print(f"Output  : {output_dir}")
    print("-" * 60)

    # Collect images
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init inspector
    print("Loading model...")
    inspector = PillInspector(config)

    # Process each image
    total_pills = 0
    total_anomalies = 0
    total_normal = 0

    for idx, img_path in enumerate(images):
        print(f"\n[{idx+1}/{len(images)}] {img_path.name}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Warning: Could not load {img_path}")
            continue

        inspector.reset()
        preview = inspector.classify_anomaly(frame)
        results = inspector._last_results

        n_pills = len(results)
        n_anomalies = 0

        for r in results:
            tid = r.get("id", -1)
            status = r.get("status", "UNKNOWN")
            scores = r.get("class_scores", {})
            normal_from = r.get("normal_from", [])

            if status == "ANOMALY":
                n_anomalies += 1

            score_strs = []
            for cls, score in scores.items():
                thr = inspector._thresholds.get(cls, 0)
                tag = "NORMAL" if score <= thr else "ANOMALY"
                score_strs.append(f"    {cls}: {score:.4f} {'<=' if score <= thr else '>'} {thr:.4f} -> {tag}")

            mark = "O" if status == "NORMAL" else "X"
            print(f"  [ID:{tid}] {mark} {status}")
            for s in score_strs:
                print(s)
            if normal_from:
                print(f"    Normal from: {', '.join(normal_from)}")

        total_pills += n_pills
        total_anomalies += n_anomalies
        total_normal += (n_pills - n_anomalies)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), preview)
        print(f"  Saved: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Results Summary")
    print(f"  Model:             ResNet50 PatchCore (Color-Aware)")
    print(f"  Images processed:  {len(images)}")
    print(f"  Total pills:       {total_pills}")
    print(f"  Total anomalies:   {total_anomalies}")
    print(f"  Total normal:      {total_normal}")
    print(f"  Output directory:  {output_dir}")
    print("=" * 60)


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ResNet50 PatchCore — Folder Prediction — MediScan-AI",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input folder of images (default: data_yolo/test)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output folder for results (default: result_resnet)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input) if args.input else ROOT / "data_yolo" / "test"
    output_dir = Path(args.output) if args.output else ROOT / "result_resnet"

    run_predict(input_dir, output_dir)


if __name__ == "__main__":
    main()
