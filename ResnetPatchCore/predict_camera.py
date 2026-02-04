#!/usr/bin/env python3
# ResnetPatchCore/predict_camera.py
"""
Realtime Pill Inspection with Camera using ResNet18 + Color Features.

🎯 Best for:
- Color anomaly detection (white vs black pills)
- Small colored defects (black spots, discoloration)
- Pills with color-based quality criteria

Usage:
    python run_predict_resnet.py
    # หรือ
    python ResnetPatchCore/predict_camera.py

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

from ResnetPatchCore.core_predict.inspector import PillInspectorResNet, InspectorConfig
from ResnetPatchCore.core_predict.visualizer import draw_summary, put_text_top_left, put_text_top_right, COLOR_ANOMALY

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
from config.resnet import (
    MODEL_OUTPUT_DIR as RESNET_MODEL_DIR,
    IMG_SIZE,
    GRID_SIZE as RESNET_GRID_SIZE,
    K_NEAREST,
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
    """Print scores for each pill in current frame."""
    if not results:
        return
    
    print("-" * 60)
    for r in results:
        tid = r.get("id", -1)
        status = r.get("status", "UNKNOWN")
        scores = r.get("class_scores", {})
        normal_from = r.get("normal_from", [])
        
        print(f"ID:{tid} | {status}")
        if scores:
            for cls_name, score in scores.items():
                thr = thresholds.get(cls_name, 0.35)
                marker = "✓" if score <= thr else "✗"
                print(f"  {cls_name}: {score:.4f} (thr={thr:.4f}) {marker}")
        if normal_from:
            print(f"  Normal from: {', '.join(normal_from)}")
    print("-" * 60)


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


def run_camera(inspector: PillInspectorResNet) -> None:
    """Run camera loop."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}")
        return
    
    print(f"\n{'='*50}")
    print(f"ResNet18 PatchCore Camera Inspection")
    print(f"{'='*50}")
    print(f"Color features: {USE_COLOR_FEATURES}")
    print(f"HSV features: {USE_HSV}")
    print(f"Grid size: {RESNET_GRID_SIZE}x{RESNET_GRID_SIZE}")
    print(f"{'='*50}")
    print(f"Hotkeys:")
    print(f"  s     - Save crops")
    print(f"  r     - Reset")
    print(f"  Enter - Summarize")
    print(f"  ESC/q - Quit")
    print(f"{'='*50}\n")
    
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
    print("Initializing ResNet18 PatchCore Inspector...")
    
    config = InspectorConfig(
        compare_classes=list(COMPARE_CLASSES),
        model_dir=RESNET_MODEL_DIR,
        yolo_model_path=SEGMENTATION_MODEL_PATH,
        yolo_det_model_path=DETECTION_MODEL_PATH,
        model_size=IMG_SIZE,
        grid_size=RESNET_GRID_SIZE,
        k_nearest=K_NEAREST,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
    )
    
    inspector = PillInspectorResNet(config)
    
    print("Starting camera...")
    run_camera(inspector)


if __name__ == "__main__":
    main()
