#!/usr/bin/env python3
# predict_camera.py
"""
Realtime Pill Inspection with Camera.

Usage:
    python predict_camera.py

Hotkeys:
    s     - Save current crops
    r     - Reset votes and tracking
    Enter - Force summarize
    ESC/q - Quit
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from core_predict import PillInspector, InspectorConfig
from core_predict import draw_summary, put_text_top_left, put_text_top_right
from core_predict import COLOR_ANOMALY


# =========================================================
# CONFIGURATION
# =========================================================
WINDOW_NAME = "Pill Inspector"
CAMERA_INDEX = 0
FRAMES_BEFORE_SUMMARY = 5
SAVE_DIR = Path("./data/inspected")

# Parent classes to compare (จะโหลดทุก subclass จาก ./model/patchcore/{class_name}/)
# เช่น "vitaminc" จะโหลด vitaminc_front.pth, vitaminc_back.pth, ...
COMPARE_CLASSES = [
    "vitaminc",
    # "white",
    # "yellow",
    # "paracap"
]

# Path to model directory
MODEL_DIR = Path("./model/patchcore")

# YOLO Models
SEGMENTATION_MODEL_PATH = "model/yolo12-seg.pt"       # Segmentation model (works!)
DETECTION_MODEL_PATH = "model/best(2).pt"             # Detection model for initial bbox detection       


# =========================================================
# FUNCTIONS
# =========================================================
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


def draw_overlay(
    frame: np.ndarray,
    frame_count: int,
    total_frames: int,
    anomaly_count: int,
) -> np.ndarray:
    """Draw status overlay."""
    vis = frame.copy()
    put_text_top_left(vis, f"Frame: {frame_count}/{total_frames}")
    
    bg = COLOR_ANOMALY if anomaly_count > 0 else (50, 50, 50)
    put_text_top_right(vis, f"Anomaly: {anomaly_count}", bg_color=bg)
    
    return vis


def run_camera(inspector: PillInspector) -> None:
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
            elif frame_count == 5:
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("Pill Inspector - Camera Mode")
    print("=" * 60)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Segmentation model: {SEGMENTATION_MODEL_PATH}")
    print(f"Detection model: {DETECTION_MODEL_PATH or 'None (using segmentation only)'}")
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
    
    config = InspectorConfig(
        compare_classes=COMPARE_CLASSES,
        model_dir=MODEL_DIR,
        yolo_model_path=SEGMENTATION_MODEL_PATH,
        yolo_det_model_path=DETECTION_MODEL_PATH,
    )
    inspector = PillInspector(config)
    
    run_camera(inspector)


if __name__ == "__main__":
    main()
