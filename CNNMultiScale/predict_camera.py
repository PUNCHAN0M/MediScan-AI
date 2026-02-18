#!/usr/bin/env python3
# CNNMultiScale/predict_camera.py
"""
Realtime Pill Inspection with Camera using CNN Multi-Scale PatchCore.

🔥 Optimized for tiny defect detection (2-5px cracks):
- Modified ResNet34 preserves 4x more spatial resolution
- Per-scale anomaly scoring → defect in ANY scale triggers alarm
- CLAHE contrast boost for micro-cracks
- Multi-resolution input (512 + 768)

Usage:
    python run_predict_cnnmultiscale.py
    # หรือ
    python CNNMultiScale/predict_camera.py

Hotkeys:
    s     - Save current crops
    r     - Reset votes and tracking
    Enter - Force summarize
    ESC/q - Quit
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from datetime import datetime

from CNNMultiScale.core_predict.inspector import PillInspectorCNNMultiScale, InspectorConfig
from CNNMultiScale.core_predict.visualizer import (
    draw_summary,
    put_text_top_left,
    put_text_top_right,
    COLOR_ANOMALY,
)

# Import configurations
from config.base import (
    SEGMENTATION_MODEL_PATH,
    DETECTION_MODEL_PATH,
    SAVE_DIR,
    COMPARE_CLASSES,
    CAMERA_INDEX,
    FRAMES_BEFORE_SUMMARY,
    WINDOW_NAME,
)
from config.cnnmultiscale import (
    MODEL_OUTPUT_DIR,
    IMG_SIZE,
    IMG_SIZE_SECONDARY,
    ENABLE_MULTI_RESOLUTION,
    GRID_SIZE,
    K_NEAREST,
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
)


# =============================================================================
#                              HELPER FUNCTIONS
# =============================================================================

def save_crops(crops: dict, output_dir: Path) -> int:
    """Save crops to directory."""
    if not crops:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for tid, crop in crops.items():
        cv2.imwrite(str(output_dir / f"crop_{timestamp}_id{tid}.jpg"), crop)

    return len(crops)


def print_report(anomaly_counts: dict) -> None:
    """Print anomaly report."""
    if not anomaly_counts:
        return

    print("\n" + "=" * 50)
    print("Anomaly Frame Counts:")
    print("-" * 50)

    for tid, count in sorted(anomaly_counts.items()):
        print(f"  Track ID {tid}: {count} anomaly frame(s)")

    print("=" * 50 + "\n")


def print_frame_scores(results: list, thresholds: dict) -> None:
    """Print scores for each pill in current frame with per-scale detail."""
    if not results:
        return

    print("-" * 70)
    for r in results:
        tid = r.get("id", -1)
        status = r.get("status", "UNKNOWN")
        scores = r.get("class_scores", {})
        normal_from = r.get("normal_from", [])
        per_scale = r.get("per_scale_scores", {})

        print(f"ID:{tid} | {status}")
        if scores:
            for cls_name, score in scores.items():
                thr = thresholds.get(cls_name, 0.50)
                marker = "✓" if score <= thr else "✗"
                print(f"  {cls_name}: {score:.4f} (thr={thr:.4f}) {marker}")

                # Per-scale breakdown
                if cls_name in per_scale:
                    scale_scores = per_scale[cls_name]
                    for layer_name, s_score in scale_scores.items():
                        print(f"    └─ {layer_name}: {s_score:.4f}")

        if normal_from:
            print(f"  Normal from: {', '.join(normal_from)}")
    print("-" * 70)


def draw_overlay(
    frame: np.ndarray,
    frame_count: int,
    total_frames: int,
    anomaly_count: int,
) -> np.ndarray:
    """Draw status overlay."""
    vis = frame.copy()
    put_text_top_left(vis, f"Frame: {frame_count}/{total_frames}")
    if anomaly_count > 0:
        put_text_top_right(vis, f"Anomaly frames: {anomaly_count}", bg_color=COLOR_ANOMALY)
    return vis


def run_camera(inspector: PillInspectorCNNMultiScale) -> None:
    """Run camera loop."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}")
        return

    print(f"\n{'='*60}")
    print(f"CNN Multi-Scale PatchCore Camera Inspection")
    print(f"{'='*60}")
    print(f"Backbone: Modified {BACKBONE}")
    print(f"Input: {IMG_SIZE}×{IMG_SIZE}" +
          (f" + {IMG_SIZE_SECONDARY}×{IMG_SIZE_SECONDARY}" if ENABLE_MULTI_RESOLUTION else ""))
    print(f"Grid: {GRID_SIZE}×{GRID_SIZE}")
    print(f"Layers: {SELECTED_LAYERS}")
    print(f"Fusion: {SCORE_FUSION}")
    print(f"CLAHE: {'ON' if USE_CLAHE else 'OFF'}")
    print(f"{'='*60}")
    print(f"Hotkeys:")
    print(f"  s     - Save crops")
    print(f"  r     - Reset")
    print(f"  Enter - Summarize")
    print(f"  ESC/q - Quit")
    print(f"{'='*60}\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Process frame
        preview = inspector.classify_anomaly(frame)
        frame_count += 1

        # Draw overlay
        total_anomaly = sum(inspector.anomaly_counts.values())
        vis = draw_overlay(preview, frame_count, FRAMES_BEFORE_SUMMARY, total_anomaly)

        # Print scores
        if inspector._last_results:
            print_frame_scores(inspector._last_results, inspector._thresholds)

        # Auto summarize
        if frame_count >= FRAMES_BEFORE_SUMMARY:
            result = inspector.summarize()
            if result["image"] is not None:
                cv2.imshow(WINDOW_NAME, result["image"])
                print(f"\n{'='*50}")
                print(f"SUMMARY: Good={result['good']}, Bad={result['bad']}")
                print(f"{'='*50}\n")
                cv2.waitKey(2000)

            inspector.reset()
            frame_count = 0

        cv2.imshow(WINDOW_NAME, vis)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # ESC
            break
        elif key == ord('s'):
            count = save_crops(inspector.last_crops, SAVE_DIR)
            print(f"Saved {count} crops to {SAVE_DIR}")
        elif key == ord('r'):
            inspector.reset()
            frame_count = 0
            print("Reset!")
        elif key == 13:  # Enter
            result = inspector.summarize()
            if result["image"] is not None:
                cv2.imshow(WINDOW_NAME, result["image"])
                print_report(inspector.anomaly_counts)
                cv2.waitKey(2000)
            inspector.reset()
            frame_count = 0

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Initializing CNN Multi-Scale PatchCore Inspector...")

    config = InspectorConfig(
        compare_classes=list(COMPARE_CLASSES),
        model_dir=MODEL_OUTPUT_DIR,
        yolo_model_path=SEGMENTATION_MODEL_PATH,
        yolo_det_model_path=DETECTION_MODEL_PATH,
        model_size=IMG_SIZE,
        model_size_secondary=IMG_SIZE_SECONDARY,
        enable_multi_resolution=ENABLE_MULTI_RESOLUTION,
        grid_size=GRID_SIZE,
        k_nearest=K_NEAREST,
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

    inspector = PillInspectorCNNMultiScale(config)

    print("Starting camera...")
    run_camera(inspector)


if __name__ == "__main__":
    main()
