#!/usr/bin/env python3
# ResnetPatchCore/predict_camera.py
"""
Realtime Pill Inspection — ResNet50 PatchCore
=============================================

Flow (per frame)
----------------
::

    camera frame
        ↓
    YOLOv12-seg  → instance masks + bboxes
        ↓
    per-pill crop (mask × frame, pad to 256×256)
        ↓
    batch ResNet50 feature extraction  (layer2+layer3)
        ↓
    PatchCore kNN scoring  (compare against COMPARE_CLASSES)
        ↓
    vote accumulation → majority vote → summary
        ↓
    output JSON:
        { count, bad_count, bad_pills, good_count, good_pills }

Console output per pill
-----------------------
::

    [ID:1] ✗ ANOMALY
        vitaminc_front: 0.3841 > 0.3628 → ANOMALY
        vitaminc_back:  0.3781 > 0.3278 → ANOMALY
        paracap:        0.6313 > 0.3663 → ANOMALY
    [ID:2] ✓ NORMAL
        paracap: 0.3159 ≤ 0.3663 → NORMAL
        Normal from: paracap

Hotkeys
-------
    s      Save current crops
    r      Reset votes and tracking
    Enter  Force summarize
    ESC/q  Quit

Usage
-----
::

    python run_realtime.py --model=resnet
    python ResnetPatchCore/predict_camera.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import json
import numpy as np
from datetime import datetime

from ResnetPatchCore.pipeline.infer import PillInspector, InspectorConfig
from ResnetPatchCore.pipeline.visualizer import (
    draw_summary, put_text_top_left, put_text_top_right,
    COLOR_ANOMALY, COLOR_BLACK,
)

from config.base import (
    SEGMENTATION_MODEL_PATH,
    DETECTION_MODEL_PATH,
    SAVE_DIR,
    COMPARE_CLASSES,
    CAMERA_INDEX,
    FRAMES_BEFORE_SUMMARY,
    WINDOW_NAME,
)
from config.resnet import (
    MODEL_OUTPUT_DIR,
    IMG_SIZE,
    GRID_SIZE,
    K_NEAREST,
    USE_COLOR_FEATURES,
    USE_HSV,
    COLOR_WEIGHT,
)


# ─────────────────── helpers ───────────────────
def save_crops(crops: dict, out_dir: Path) -> int:
    if not crops:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for tid, crop in crops.items():
        cv2.imwrite(str(out_dir / f"crop_{ts}_id{tid}.jpg"), crop)
    return len(crops)


def print_frame_scores(results: list, thresholds: dict) -> None:
    """Print per-pill console output matching the spec."""
    if not results:
        return
    print("-" * 60)
    for r in results:
        tid = r.get("id", -1)
        status = r.get("status", "UNKNOWN")
        scores = r.get("class_scores", {})
        normal_from = r.get("normal_from", [])

        mark = "✓" if status == "NORMAL" else "✗"
        print(f"[ID:{tid}] {mark} {status}")

        for cls, score in scores.items():
            thr = thresholds.get(cls, 0.50)
            if score <= thr:
                print(f"    {cls}: {score:.4f} ≤ {thr:.4f} → NORMAL")
            else:
                print(f"    {cls}: {score:.4f} > {thr:.4f} → ANOMALY")

        if normal_from:
            print(f"    Normal from: {', '.join(normal_from)}")
    print("-" * 60)


def draw_overlay(
    frame: np.ndarray,
    frame_count: int,
    total_frames: int,
    anomaly_count: int,
) -> np.ndarray:
    vis = frame.copy()
    put_text_top_left(vis, f"Frame: {frame_count}/{total_frames}")
    if anomaly_count > 0:
        put_text_top_right(vis, f"Anomaly frames: {anomaly_count}",
                           bg_color=COLOR_ANOMALY)
    return vis


# ─────────────────── camera loop ───────────────────
def run_camera(inspector: PillInspector) -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}")
        return

    print(f"\n{'=' * 50}")
    print("  ResNet50 PatchCore — Realtime Camera Inspection")
    print(f"{'=' * 50}")
    print(f"  Backbone       : resnet50 (layer2+layer3)")
    print(f"  Color features : {USE_COLOR_FEATURES}  HSV: {USE_HSV}")
    print(f"  Grid size      : {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Compare classes: {list(COMPARE_CLASSES)}")
    print(f"{'=' * 50}")
    print("  Hotkeys:  s=save  r=reset  Enter=summarize  ESC/q=quit")
    print(f"{'=' * 50}\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # classify
        preview = inspector.classify_anomaly(frame)
        frame_count += 1

        # overlay
        total_anomaly = sum(inspector.anomaly_counts.values())
        vis = draw_overlay(preview, frame_count,
                           FRAMES_BEFORE_SUMMARY, total_anomaly)

        # console output
        if inspector._last_results:
            print_frame_scores(inspector._last_results,
                               inspector._thresholds)

        # auto-summarize after N frames
        if frame_count >= FRAMES_BEFORE_SUMMARY:
            result = inspector.summarize()
            if result["image"] is not None:
                cv2.imshow(WINDOW_NAME, result["image"])

                # JSON output
                out = {
                    "count": result["count"],
                    "good_count": result["good_count"],
                    "bad_count": result["bad_count"],
                    "good_pills": result["good_pills"],
                    "bad_pills": result["bad_pills"],
                }
                print(f"\n{'=' * 50}")
                # print("SUMMARY:")
                # print(json.dumps(out, indent=2, ensure_ascii=False))
                print(f"{'=' * 50}\n")

                cv2.waitKey(2000)

            inspector.reset()
            frame_count = 0

        cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            n = save_crops(inspector.last_crops, SAVE_DIR)
            print(f"Saved {n} crops → {SAVE_DIR}")
        elif key == ord('r'):
            inspector.reset()
            frame_count = 0
            print("Reset!")
        elif key == 13:  # Enter
            result = inspector.summarize()
            if result["image"] is not None:
                cv2.imshow(WINDOW_NAME, result["image"])
                print(json.dumps({
                    "count": result["count"],
                    "good_count": result["good_count"],
                    "bad_count": result["bad_count"],
                }, indent=2))
                cv2.waitKey(2000)
            inspector.reset()
            frame_count = 0

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────── main ───────────────────
def main():
    print("Initializing ResNet50 PatchCore Inspector ...")

    config = InspectorConfig(
        compare_classes=list(COMPARE_CLASSES),
        model_dir=MODEL_OUTPUT_DIR,
        yolo_model_path=str(SEGMENTATION_MODEL_PATH),
        model_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        k_nearest=K_NEAREST,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
    )

    inspector = PillInspector(config)

    print("Starting camera ...")
    run_camera(inspector)


if __name__ == "__main__":
    main()
