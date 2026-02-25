#!/usr/bin/env python3
# DINOv2PatchCore/predict_camera_dinov2.py
"""
Realtime Pill Inspection with DINOv2 PatchCore.

🎯 Best for:
- Complex texture + shape anomalies
- High-quality offline inspection
- Best semantic understanding

Usage:
    python run_predict_dinov2.py
    # หรือ
    python DINOv2PatchCore/predict_camera_dinov2.py

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

from DINOv2PatchCore.inspector_dinov2 import DINOv2PillInspector, DINOv2InspectorConfig

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
from config.dinov2 import MODEL_OUTPUT_DIR as MODEL_DIR, BACKBONE_SIZE


# =============================================================================
#                              VISUALIZER HELPERS
# =============================================================================

COLOR_ANOMALY = (0, 0, 255)


def put_text_top_left(vis, text, bg_color=(50, 50, 50)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(vis, (5, 5), (15 + tw, 15 + th), bg_color, -1)
    cv2.putText(vis, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def put_text_top_right(vis, text, bg_color=(50, 50, 50)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x = vis.shape[1] - tw - 15
    cv2.rectangle(vis, (x - 5, 5), (x + tw + 5, 15 + th), bg_color, -1)
    cv2.putText(vis, text, (x, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def save_crops(crops: dict, output_dir: Path) -> int:
    """Save crops to directory."""
    if not crops:
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for tid, crop in crops.items():
        filename = output_dir / f"crop_id{tid}_{timestamp}.png"
        cv2.imwrite(str(filename), crop)
    
    return len(crops)


def print_report(anomaly_counts: dict) -> None:
    """Print anomaly report."""
    if not anomaly_counts:
        print("[Report] No pills detected")
        return
    
    print("\n" + "=" * 50)
    print("Anomaly Frame Counts:")
    print("-" * 50)
    
    for tid, count in sorted(anomaly_counts.items()):
        status = "⚠️ ANOMALY" if count > 0 else "✓ NORMAL"
        print(f"  ID {tid:3d}: {count:2d} anomaly frames  {status}")
    
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
        
        status_icon = "✓" if status == "NORMAL" else "✗"
        print(f"[ID:{tid}] {status_icon} {status}")
        
        for cls, score in scores.items():
            thr = thresholds.get(cls, 0.35)
            mark = "≤" if score <= thr else ">"
            result = "NORMAL" if score <= thr else "ANOMALY"
            print(f"    {cls}: {score:.4f} {mark} {thr:.4f} → {result}")
        
        if normal_from:
            print(f"    Normal from: {', '.join(normal_from)}")
    print("-" * 60)


def draw_overlay(frame, frame_count, total_frames, anomaly_count):
    """Draw status overlay."""
    vis = frame.copy()
    put_text_top_left(vis, f"Frame: {frame_count}/{total_frames}")
    
    bg = COLOR_ANOMALY if anomaly_count > 0 else (50, 50, 50)
    put_text_top_right(vis, f"Anomaly: {anomaly_count}", bg_color=bg)
    
    return vis


def run_camera(inspector: DINOv2PillInspector) -> None:
    """Main camera loop."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")
    
    print(f"[INFO] Camera opened")
    print(f"[INFO] Hotkeys: s=save, r=reset, Enter=summarize, ESC=quit")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            preview = inspector.classify_anomaly(frame, COMPARE_CLASSES)
            frame_count += 1
            
            # Print scores for this frame
            print_frame_scores(inspector._last_results, inspector._thresholds)
            
            # Count anomalies
            anomaly_count = sum(1 for c in inspector.anomaly_counts.values() if c > 0)
            
            # Draw overlay
            display = draw_overlay(preview, frame_count, FRAMES_BEFORE_SUMMARY, anomaly_count)
            cv2.imshow(WINDOW_NAME, display)
            
            # Auto-summarize
            if frame_count >= FRAMES_BEFORE_SUMMARY:
                result = inspector.summarize()
                
                print_report(inspector.anomaly_counts)
                print(f"[Summary] Good: {result['good']}, Bad: {result['bad']}")
                
                if result['image'] is not None:
                    cv2.imshow(WINDOW_NAME, result['image'])
                    cv2.waitKey(1500)
                
                inspector.reset()
                frame_count = 0
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):
                break
            
            elif key == ord('s'):
                count = save_crops(inspector.last_crops, SAVE_DIR)
                print(f"[Saved] {count} crops to {SAVE_DIR}")
            
            elif key == ord('r'):
                inspector.reset()
                frame_count = 0
                print("[Reset] Cleared votes and tracking")
            
            elif key == 13:  # Enter
                result = inspector.summarize()
                print_report(inspector.anomaly_counts)
                print(f"[Summary] Good: {result['good']}, Bad: {result['bad']}")
                
                if result['image'] is not None:
                    cv2.imshow(WINDOW_NAME, result['image'])
                    cv2.waitKey(1500)
                
                inspector.reset()
                frame_count = 0
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("DINOv2 Pill Inspector - Better Color/Texture Detection")
    print("=" * 60)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Backbone: DINOv2 {BACKBONE_SIZE}")
    print(f"Segmentation model: {SEGMENTATION_MODEL_PATH}")
    print(f"Detection model: {DETECTION_MODEL_PATH or 'None'}")
    print(f"Parent classes: {COMPARE_CLASSES}")
    
    # แสดง subclasses ที่จะโหลด
    for cls in COMPARE_CLASSES:
        cls_dir = MODEL_DIR / cls
        if cls_dir.exists():
            subclasses = [f.stem for f in cls_dir.glob("*.pth")]
            print(f"  {cls}: {subclasses}")
        else:
            print(f"  {cls}: (not found)")
    print("=" * 60)
    
    config = DINOv2InspectorConfig(
        compare_classes=COMPARE_CLASSES,
        model_dir=MODEL_DIR,
        yolo_model_path=SEGMENTATION_MODEL_PATH,
        yolo_det_model_path=DETECTION_MODEL_PATH,
        backbone_size=BACKBONE_SIZE,
    )
    inspector = DINOv2PillInspector(config)
    
    run_camera(inspector)


if __name__ == "__main__":
    main()
