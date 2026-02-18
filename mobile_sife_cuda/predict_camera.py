#!/usr/bin/env python3
# mobile_sife_cuda/predict_camera.py
"""
Realtime Pill Inspection with Camera — CUDA + Multithreaded.

Architecture (3 threads, fully decoupled):
    CameraThread   — reads frames at full camera FPS → FrameHolder
    ProcessThread  — picks latest frame → YOLO + PatchCore → ResultHolder
    MainThread     — always shows live camera feed + overlays latest results

Key design:
    - MainThread renders live video at camera FPS (smooth, never freezes)
    - Detection results are overlaid on the live feed whenever available
    - Frame drops are intentional: ProcessThread always processes the NEWEST
      frame, skipping any captured while the previous inference was running

Usage:
    python run_predict_sife_cuda.py
    python mobile_sife_cuda/predict_camera.py

Hotkeys:
    s     - Save current crops
    r     - Reset votes and tracking
    Enter - Force summarize
    ESC/q - Quit
"""
import sys
from pathlib import Path

# Add parent directory for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import threading
import cv2
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from mobile_sife_cuda.core_predict.inspector import PillInspectorSIFE, InspectorConfig
from mobile_sife_cuda.core_predict.visualizer import (
    draw_pill_results, draw_summary,
    put_text_top_left, put_text_top_right, put_text_with_bg,
    COLOR_NORMAL, COLOR_ANOMALY, COLOR_WHITE, COLOR_BLACK,
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
from config.sife import (
    MODEL_OUTPUT_DIR,
    IMG_SIZE,
    GRID_SIZE,
    K_NEAREST,
    USE_SIFE,
    SIFE_DIM,
    SIFE_ENCODING_TYPE,
    SIFE_WEIGHT,
    USE_CENTER_DISTANCE,
    USE_LOCAL_GRADIENT,
    USE_COLOR_FEATURES,
    USE_HSV,
    COLOR_WEIGHT,
    USE_MULTI_SCALE,
    MULTI_SCALE_GRIDS,
    USE_EDGE_ENHANCEMENT,
    EDGE_WEIGHT,
)


# =============================================================================
#                         THREAD-SAFE DATA HOLDERS
# =============================================================================

class FrameHolder:
    """
    Latest-frame holder. CameraThread writes, MainThread + ProcessThread read.
    Always stores the most recent frame (previous frame is overwritten).
    """

    __slots__ = ("_lock", "_frame", "_frame_id")

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def get(self) -> Tuple[Optional[np.ndarray], int]:
        """Returns (frame, frame_id). frame may be None if no frame yet."""
        with self._lock:
            return self._frame, self._frame_id


class ResultHolder:
    """
    Thread-safe detection result holder.
    ProcessThread writes detection DATA (not rendered image).
    MainThread reads and draws onto the live camera frame.
    """

    __slots__ = (
        "_lock", "_results", "_thresholds", "_anomaly_counts",
        "_result_id", "_proc_fps",
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._results: List[Dict[str, Any]] = []
        self._thresholds: Dict[str, float] = {}
        self._anomaly_counts: Dict[int, int] = {}
        self._result_id: int = 0
        self._proc_fps: float = 0.0

    def put(
        self,
        results: List[Dict[str, Any]],
        thresholds: Dict[str, float],
        anomaly_counts: Dict[int, int],
        proc_fps: float,
    ) -> None:
        with self._lock:
            self._results = results
            self._thresholds = thresholds
            self._anomaly_counts = dict(anomaly_counts)
            self._result_id += 1
            self._proc_fps = proc_fps

    def get(self) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[int, int], int, float]:
        """Returns (results, thresholds, anomaly_counts, result_id, proc_fps)."""
        with self._lock:
            return (
                list(self._results),
                dict(self._thresholds),
                dict(self._anomaly_counts),
                self._result_id,
                self._proc_fps,
            )


# =============================================================================
#                            CAMERA THREAD
# =============================================================================

class CameraThread(threading.Thread):
    """
    Reads frames from camera as fast as possible, stores latest into FrameHolder.
    Runs as daemon so it dies when main thread exits.
    """

    def __init__(
        self,
        camera_index: int,
        frame_holder: FrameHolder,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True, name="CameraThread")
        self.camera_index = camera_index
        self.frame_holder = frame_holder
        self.stop_event = stop_event
        self.camera_fps: float = 0.0

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[CameraThread] ERROR: cannot open camera {self.camera_index}")
            self.stop_event.set()
            return

        print(f"[CameraThread] Started (index={self.camera_index})")
        fps_counter = 0
        fps_timer = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.001)
                    continue

                self.frame_holder.put(frame)
                fps_counter += 1

                # Calculate camera FPS every second
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    self.camera_fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now
        finally:
            cap.release()
            print("[CameraThread] Stopped")


# =============================================================================
#                           PROCESS THREAD
# =============================================================================

class ProcessThread(threading.Thread):
    """
    Picks latest frame from FrameHolder, runs full inference pipeline
    (YOLO detection + PatchCore scoring), stores RESULT DATA into ResultHolder.

    Does NOT render preview images — that's MainThread's job.
    If inference is slower than camera, intermediate frames are skipped.
    """

    def __init__(
        self,
        inspector: PillInspectorSIFE,
        compare_classes: List[str],
        frame_holder: FrameHolder,
        result_holder: ResultHolder,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True, name="ProcessThread")
        self.inspector = inspector
        self.compare_classes = compare_classes
        self.frame_holder = frame_holder
        self.result_holder = result_holder
        self.stop_event = stop_event
        self.process_fps: float = 0.0
        self.frame_count: int = 0

    def run(self) -> None:
        print("[ProcessThread] Started — waiting for frames...")
        last_frame_id = -1
        fps_counter = 0
        fps_timer = time.perf_counter()

        while not self.stop_event.is_set():
            frame, frame_id = self.frame_holder.get()

            # No new frame yet — spin briefly
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = frame_id

            # --- Run full inference pipeline (YOLO + PatchCore) ---
            # classify_anomaly returns a rendered preview, but we only need
            # the result data stored in inspector._last_results
            self.inspector.classify_anomaly(frame, self.compare_classes)
            self.frame_count += 1
            fps_counter += 1

            # Calculate process FPS
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                self.process_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # Store detection RESULT DATA (not image)
            self.result_holder.put(
                results=list(self.inspector._last_results),
                thresholds=dict(self.inspector._thresholds),
                anomaly_counts=self.inspector.anomaly_counts,
                proc_fps=self.process_fps,
            )

        print("[ProcessThread] Stopped")


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
        filename = output_dir / f"crop_id{tid}_{timestamp}.png"
        cv2.imwrite(str(filename), crop)
    return len(crops)


def print_report(anomaly_counts: dict) -> None:
    """Print anomaly report table."""
    if not anomaly_counts:
        print("[Report] No pills detected")
        return
    print("\n" + "=" * 50)
    print("Anomaly Frame Counts:")
    print("-" * 50)
    for tid, count in sorted(anomaly_counts.items()):
        status = "ANOMALY" if count > 0 else "NORMAL"
        print(f"  ID {tid:3d}: {count:2d} anomaly frames  {status}")
    print("=" * 50 + "\n")


def print_frame_scores(results: list, thresholds: dict) -> None:
    """Print per-pill scores for current frame."""
    if not results:
        return
    print("-" * 60)
    for r in results:
        tid = r.get("id", -1)
        status = r.get("status", "UNKNOWN")
        scores = r.get("class_scores", {})
        normal_from = r.get("normal_from", [])

        icon = "O" if status == "NORMAL" else "X"
        print(f"[ID:{tid}] {icon} {status}")
        for cls, score in scores.items():
            thr = thresholds.get(cls, 0.35)
            mark = "<=" if score <= thr else ">"
            result = "NORMAL" if score <= thr else "ANOMALY"
            print(f"    {cls}: {score:.4f} {mark} {thr:.4f} -> {result}")
        if normal_from:
            print(f"    Normal from: {', '.join(normal_from)}")
    print("-" * 60)


def draw_live_overlay(
    frame: np.ndarray,
    results: List[Dict[str, Any]],
    process_count: int,
    total_frames: int,
    anomaly_count: int,
    cam_fps: float,
    proc_fps: float,
) -> np.ndarray:
    """
    Draw live overlay: bounding boxes from latest inference + status bar.
    Called on every camera frame (fast — just rectangle + text drawing).
    """
    vis = frame.copy()

    # --- Draw bounding boxes from latest detection results ---
    for r in results:
        bbox = r.get("bbox", r.get("padded_region", (0, 0, 0, 0)))
        x1, y1, x2, y2 = map(int, bbox)
        status = r.get("status", "UNKNOWN")
        tid = r.get("track_id", r.get("id", -1))
        normal_from = r.get("normal_from", [])

        color = COLOR_NORMAL if status == "NORMAL" else COLOR_ANOMALY
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        label = f"ID:{tid} | {status}"
        if normal_from:
            label += f" ({','.join(normal_from)})"
        y_text = max(20, y1 - 10)
        cv2.putText(vis, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Status bar ---
    put_text_top_left(
        vis,
        f"Frame: {process_count}/{total_frames} [SIFE-CUDA MT] "
        f"CamFPS:{cam_fps:.0f} ProcFPS:{proc_fps:.1f}",
    )
    bg = COLOR_ANOMALY if anomaly_count > 0 else (50, 50, 50)
    put_text_top_right(vis, f"Anomaly: {anomaly_count}", bg_color=bg)

    return vis


# =============================================================================
#                     MAIN LOOP (runs on main thread)
# =============================================================================

def run_camera(inspector: PillInspectorSIFE) -> None:
    """
    Main display loop running on the main thread (required by OpenCV).

    Thread architecture:
        CameraThread  → FrameHolder   (raw frames at camera FPS)
        ProcessThread → ResultHolder   (detection data, slower)
        MainThread    ← reads BOTH holders, draws live video + overlays

    The display ALWAYS shows the latest camera frame (smooth video).
    Detection bboxes/labels are overlaid and update when ProcessThread
    finishes the next inference cycle.
    """
    stop_event = threading.Event()
    frame_holder = FrameHolder()
    result_holder = ResultHolder()

    # Start worker threads
    cam_thread = CameraThread(CAMERA_INDEX, frame_holder, stop_event)
    proc_thread = ProcessThread(
        inspector, list(COMPARE_CLASSES),
        frame_holder, result_holder, stop_event,
    )

    cam_thread.start()
    proc_thread.start()

    print(f"[Main] Display loop started — live camera + async inference")
    print(f"[Main] Hotkeys: s=save, r=reset, Enter=summarize, ESC/q=quit")

    last_result_id = -1
    last_frame_id = -1

    # Cached latest detection results (drawn on every camera frame)
    cached_results: List[Dict[str, Any]] = []
    cached_anomaly_count: int = 0
    cached_proc_fps: float = 0.0

    try:
        while not stop_event.is_set():
            # ── 1. Read latest CAMERA FRAME (this updates every frame) ──
            raw_frame, frame_id = frame_holder.get()
            if raw_frame is None:
                time.sleep(0.001)
                continue

            # ── 2. Check for new DETECTION RESULTS (updates less often) ──
            results, thresholds, anomaly_counts, result_id, proc_fps = (
                result_holder.get()
            )

            if result_id != last_result_id:
                last_result_id = result_id
                cached_results = results
                cached_anomaly_count = sum(1 for c in anomaly_counts.values() if c > 0)
                cached_proc_fps = proc_fps

                # Print scores to console only when new results arrive
                print_frame_scores(results, thresholds)

                # Auto-summarize check
                if proc_thread.frame_count >= FRAMES_BEFORE_SUMMARY:
                    result = inspector.summarize()
                    print_report(inspector.anomaly_counts)
                    print(f"[Summary] Good: {result['good']}, Bad: {result['bad']}")

                    if result["image"] is not None:
                        cv2.imshow(WINDOW_NAME + " [SIFE-CUDA MT]", result["image"])
                        cv2.waitKey(1500)

                    inspector.reset()
                    proc_thread.frame_count = 0
                    cached_results = []
                    cached_anomaly_count = 0

            # ── 3. ALWAYS draw live frame with overlaid results ──
            display = draw_live_overlay(
                raw_frame,
                cached_results,
                proc_thread.frame_count,
                FRAMES_BEFORE_SUMMARY,
                cached_anomaly_count,
                cam_fps=cam_thread.camera_fps,
                proc_fps=cached_proc_fps,
            )
            cv2.imshow(WINDOW_NAME + " [SIFE-CUDA MT]", display)

            # ── 4. Handle keyboard (must be main thread for OpenCV) ──
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                break
            elif key == ord("s"):
                count = save_crops(inspector.last_crops, SAVE_DIR)
                print(f"[Saved] {count} crops to {SAVE_DIR}")
            elif key == ord("r"):
                inspector.reset()
                proc_thread.frame_count = 0
                cached_results = []
                cached_anomaly_count = 0
                print("[Reset] Cleared votes and tracking")
            elif key == 13:  # Enter
                result = inspector.summarize()
                print_report(inspector.anomaly_counts)
                print(f"[Summary] Good: {result['good']}, Bad: {result['bad']}")
                if result["image"] is not None:
                    cv2.imshow(WINDOW_NAME + " [SIFE-CUDA MT]", result["image"])
                    cv2.waitKey(1500)
                inspector.reset()
                proc_thread.frame_count = 0
                cached_results = []
                cached_anomaly_count = 0

    finally:
        stop_event.set()
        cam_thread.join(timeout=3.0)
        proc_thread.join(timeout=3.0)
        cv2.destroyAllWindows()
        print("[Main] Shutdown complete")


# =============================================================================
#                              MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Pill Inspector - MobileNet + SIFE [CUDA + Multithreaded]")
    print("=" * 60)
    print(f"Model dir     : {MODEL_OUTPUT_DIR}")
    print(f"Seg model     : {SEGMENTATION_MODEL_PATH}")
    print(f"Det model     : {DETECTION_MODEL_PATH or 'None'}")
    print(f"Classes       : {COMPARE_CLASSES}")
    print("-" * 60)
    print("SIFE Settings:")
    print(f"  Enabled     : {USE_SIFE} | Dim={SIFE_DIM} | Encoding={SIFE_ENCODING_TYPE}")
    print(f"  Weight      : {SIFE_WEIGHT} | CenterDist={USE_CENTER_DISTANCE} | Gradient={USE_LOCAL_GRADIENT}")
    print(f"  MultiScale  : {USE_MULTI_SCALE} {MULTI_SCALE_GRIDS if USE_MULTI_SCALE else ''}")
    print(f"  Edge        : {USE_EDGE_ENHANCEMENT} (w={EDGE_WEIGHT})")
    print("-" * 60)

    # Show available subclasses
    for cls in COMPARE_CLASSES:
        cls_dir = MODEL_OUTPUT_DIR / cls
        if cls_dir.exists():
            subclasses = [f.stem for f in cls_dir.glob("*.pth")]
            print(f"  {cls}: {subclasses}")
        else:
            print(f"  {cls}: (not found)")
    print("=" * 60)

    config = InspectorConfig(
        compare_classes=list(COMPARE_CLASSES),
        model_dir=MODEL_OUTPUT_DIR,
        yolo_model_path=SEGMENTATION_MODEL_PATH,
        yolo_det_model_path=DETECTION_MODEL_PATH,
        model_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        k_nearest=K_NEAREST,
        use_sife=USE_SIFE,
        sife_dim=SIFE_DIM,
        sife_encoding_type=SIFE_ENCODING_TYPE,
        sife_weight=SIFE_WEIGHT,
        use_center_distance=USE_CENTER_DISTANCE,
        use_local_gradient=USE_LOCAL_GRADIENT,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
        use_multi_scale=USE_MULTI_SCALE,
        multi_scale_grids=MULTI_SCALE_GRIDS,
        use_edge_enhancement=USE_EDGE_ENHANCEMENT,
        edge_weight=EDGE_WEIGHT,
        use_detailed_scoring=True,
    )

    inspector = PillInspectorSIFE(config)
    run_camera(inspector)


if __name__ == "__main__":
    main()
