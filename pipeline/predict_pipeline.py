# pipeline/predict_pipeline.py
"""
Predict Pipeline — realtime camera loop.
==========================================
Layer 4 — camera capture + inference orchestration.
"""
from __future__ import annotations

import cv2
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline.infer_pipeline import PillInspector, InspectorConfig
from visualization.visualizer import draw_summary


# ─────────────────────────────────────────────
# FPS Counter
# ─────────────────────────────────────────────
class FPSCounter:
    def __init__(self):
        self.prev = time.time()
        self.fps = 0.0

    def update(self) -> float:
        now = time.time()
        dt = now - self.prev
        self.prev = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps


# ─────────────────────────────────────────────
# Digital Zoom
# ─────────────────────────────────────────────
def digital_zoom(frame: np.ndarray, zoom: float = 1.0) -> np.ndarray:
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom), int(h / zoom)
    x1, y1 = (w - new_w) // 2, (h - new_h) // 2
    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────
# Save Crops
# ─────────────────────────────────────────────
def save_crops(save_dir: Path, crops: dict) -> int:
    if not crops:
        return 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for tid, crop in crops.items():
        out_dir = save_dir / "crops" / f"track_{tid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"crop_{ts}.jpg"), crop)
    return len(crops)


# ─────────────────────────────────────────────
# Main Camera Loop
# ─────────────────────────────────────────────
def run_camera(
    inspector: PillInspector,
    compare_classes: list,
    camera_index: int = 0,
    frames_before_summary: int = 3,
    save_dir: Optional[Path] = None,
    window_name: str = "Pill Inspector",
    zoom: float = 1.2,
) -> None:
    """
    Run the realtime camera inspection loop.

    Hotkeys: s=save crops | r=reset | Enter=summarize | q/ESC=quit
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    print(f"\nRealtime Inspection Started")
    print(f"Compare classes: {compare_classes}")
    print("Hotkeys: s=save | r=reset | Enter=summarize | q/ESC=quit\n")

    fps_counter = FPSCounter()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = digital_zoom(frame, zoom=zoom)

        # inference
        preview = inspector.classify_anomaly(frame, class_names=compare_classes)
        frame_count += 1

        fps = fps_counter.update()

        vis = draw_summary(
            preview,
            inspector._last_results,
            show_overlay=True,
            alpha_overlay=False,
        )

        cv2.putText(
            vis, f"FPS: {fps:.1f}", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            lineType=cv2.LINE_AA,
        )

        cv2.imshow(window_name, vis)

        # auto summarize
        if frame_count >= frames_before_summary:
            result = inspector.summarize()
            print(json.dumps({
                "count": result["count"],
                "good": result["good_count"],
                "bad": result["bad_count"],
            }, indent=2))
            inspector.reset()
            frame_count = 0

        # key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('s') and save_dir:
            n = save_crops(save_dir, inspector.last_crops)
            print(f"Saved {n} crops")
        elif key == ord('r'):
            inspector.reset()
            frame_count = 0
            print("Reset")
        elif key == 13:  # Enter
            result = inspector.summarize()
            print(json.dumps({
                "count": result["count"],
                "good": result["good_count"],
                "bad": result["bad_count"],
            }, indent=2))
            inspector.reset()
            frame_count = 0

    cap.release()
    cv2.destroyAllWindows()
