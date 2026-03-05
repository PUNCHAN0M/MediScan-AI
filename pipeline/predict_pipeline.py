# pipeline/predict_pipeline.py
"""
Predict Pipeline — realtime camera loop.
==========================================
Layer 4 — camera capture + inference orchestration.

Architecture:
    Main thread  : cap.read() → YOLO detect (sync, fast ~20-50 ms)
                   → draw current bboxes + cached labels → imshow()
    Worker thread: feature extraction + scoring (async, ~100-500 ms)
                   → updates {track_id: status} map

Benefits:
    - Bboxes appear/disappear **instantly** with YOLO detections.
    - Labels (NORMAL/ANOMALY) update once scoring finishes (~100-500 ms lag).
    - Covering a pill → YOLO stops detecting → bbox gone immediately.
    - No stale bounding boxes from previous frames.
    - EMA-smoothed FPS counter for stable readout.
"""
from __future__ import annotations

import cv2
import json
import threading
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.infer_pipeline import PillInspector, InspectorConfig
from visualization.visualizer import draw_pill_results, draw_summary


# ─────────────────────────────────────────────
# FPS Counter (exponential moving average)
# ─────────────────────────────────────────────
class FPSCounter:
    """Smoothed FPS using exponential moving average."""

    def __init__(self, alpha: float = 0.1):
        self.prev = time.perf_counter()
        self.fps = 0.0
        self._alpha = alpha

    def update(self) -> float:
        now = time.perf_counter()
        dt = now - self.prev
        self.prev = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = (
                self._alpha * instant + (1.0 - self._alpha) * self.fps
                if self.fps > 0
                else instant
            )
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
# Async Scoring Worker (background thread)
# ─────────────────────────────────────────────
class _ScoringWorker:
    """
    Background thread for feature extraction + anomaly scoring.

    Only **scoring** runs here — YOLO detection stays in the main thread
    for instant bbox response.

    - Main thread calls ``submit(crops, infos)`` after YOLO detection.
    - Worker processes only the **latest** batch (auto frame-skip).
    - Results are read via ``status_map`` — {track_id: status_info}.
    """

    def __init__(self, inspector: PillInspector, classes: list):
        self._inspector = inspector
        self._classes = classes

        # Input slot (latest-only)
        self._input_lock = threading.Lock()
        self._input_slot: Optional[tuple] = None   # (crops, infos)

        # Output: per-track scorings
        self._result_lock = threading.Lock()
        self._status_map: Dict[int, Dict[str, Any]] = {}
        self._crops_map: Dict[int, np.ndarray] = {}

        self._new_input = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, crops: list, infos: list) -> None:
        """Submit a YOLO detection batch for scoring (replaces pending)."""
        with self._input_lock:
            self._input_slot = (list(crops), list(infos))
        self._new_input.set()

    @property
    def status_map(self) -> Dict[int, Dict[str, Any]]:
        """Thread-safe read: {track_id: {"status", "class_scores", "normal_from"}}."""
        with self._result_lock:
            return dict(self._status_map)

    @property
    def crops_map(self) -> Dict[int, np.ndarray]:
        """Thread-safe read: {track_id: crop_image}."""
        with self._result_lock:
            return dict(self._crops_map)

    def stop(self) -> None:
        self._stop_event.set()
        self._new_input.set()
        self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._new_input.wait()
            self._new_input.clear()
            if self._stop_event.is_set():
                break

            with self._input_lock:
                data = self._input_slot
                self._input_slot = None
            if data is None:
                continue

            crops, infos = data
            try:
                results = self._inspector.score_pills(
                    crops, infos, class_names=self._classes,
                )

                new_status: Dict[int, Dict[str, Any]] = {}
                new_crops: Dict[int, np.ndarray] = {}
                for r, crop in zip(results, crops):
                    tid = r["id"]
                    new_status[tid] = {
                        "status": r["status"],
                        "class_scores": r.get("class_scores", {}),
                        "normal_from": r.get("normal_from", []),
                    }
                    if tid >= 0:
                        new_crops[tid] = crop

                with self._result_lock:
                    self._status_map = new_status
                    self._crops_map = new_crops
            except Exception as e:
                print(f"[Scoring] Error: {e}")


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

    YOLO detection is **synchronous** (fast, ~20-50 ms) for instant bbox
    feedback.  Scoring runs in a background thread — labels update once
    scoring finishes.  Bboxes appear/disappear instantly regardless of
    scoring latency.

    Hotkeys: s=save crops | r=reset | Enter=summarize | q/ESC=quit
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    # Warmup YOLO (triggers ONNX compilation — avoids 4 s lag on first frame)
    print("[Warming up YOLO …]")
    inspector.warmup()
    print("[Ready]")

    print(f"\nRealtime Inspection Started")
    print(f"Compare classes: {compare_classes}")
    print("Hotkeys: s=save | r=reset | Enter=summarize | q/ESC=quit\n")

    worker = _ScoringWorker(inspector, compare_classes)
    fps_counter = FPSCounter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = digital_zoom(frame, zoom=zoom)

            # ── Phase 1: synchronous YOLO detection (fast) ──
            crops, infos = inspector.detect_frame(frame)

            # Submit crops to background scorer (non-blocking)
            if crops:
                worker.submit(crops, infos)

            # ── Merge: current YOLO bboxes + cached scoring labels ──
            scored = worker.status_map
            results: List[Dict[str, Any]] = []
            for info in infos:
                tid = info.get("track_id", -1)
                cached = scored.get(tid)
                results.append({
                    "id": tid,
                    "bbox": info["bbox"],
                    "conf": info["conf"],
                    "center": info.get("center"),
                    "status": cached["status"] if cached else "SCORING",
                    "class_scores": cached.get("class_scores", {}) if cached else {},
                    "normal_from": cached.get("normal_from", []) if cached else [],
                })

            # ── Draw ──
            vis = draw_pill_results(frame, results)
            vis = draw_summary(
                vis, results,
                show_overlay=True,
                alpha_overlay=False,
            )

            fps = fps_counter.update()
            cv2.putText(
                vis, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow(window_name, vis)

            # Key handling (summarize is manual-only to avoid log spam)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('s') and save_dir:
                n = save_crops(save_dir, worker.crops_map)
                print(f"Saved {n} crops")
            elif key == ord('r'):
                inspector.reset()
                print("Reset")
            elif key == 13:  # Enter
                good = sum(1 for r in results if r["status"] == "NORMAL")
                bad = sum(1 for r in results if r["status"] == "ANOMALY")
                pending = sum(1 for r in results if r["status"] not in ("NORMAL", "ANOMALY"))
                print(json.dumps({
                    "count": len(results),
                    "good": good,
                    "bad": bad,
                    "pending": pending,
                }, indent=2))

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
