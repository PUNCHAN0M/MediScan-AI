# visualization/visualizer.py
"""
Drawing utilities for detection and summary overlays.
=====================================================
Pure drawing — no model logic, no pipeline loops.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ─────────────── Colors (BGR) ───────────────
COLOR_NORMAL  = (0, 255, 100)
COLOR_ANOMALY = (0, 0, 255)
COLOR_PENDING = (200, 200, 0)   # Cyan-ish: detected but not yet scored
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)

def _status_color(status: str) -> tuple:
    if status == "NORMAL":
        return COLOR_NORMAL
    if status == "ANOMALY":
        return COLOR_ANOMALY
    return COLOR_PENDING


# ─────────────── Label helper ───────────────
def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 4), COLOR_BLACK, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)


# ─────────────── Draw realtime results ───────────────
def draw_pill_results(
    frame: np.ndarray,
    results: List[Dict[str, Any]],
    fps: Optional[float] = None,
    show_label: bool = True,
) -> np.ndarray:
    """Draw detection bboxes + status labels on frame."""
    vis = frame.copy()

    for r in results:
        bbox = r.get("bbox", (0, 0, 0, 0))
        x1, y1, x2, y2 = map(int, bbox)
        status = r.get("status", "UNKNOWN")
        tid = r.get("id", -1)
        normal_from = r.get("normal_from", [])

        color = _status_color(status)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if show_label:
            label = f"ID:{tid} | {status}"
            if normal_from:
                label += f" ({','.join(normal_from)})"
            _draw_label(vis, label, x1, max(20, y1 - 8), color)

    if fps is not None and show_label:
        _draw_label(vis, f"FPS: {fps:.1f}", 15, 30, COLOR_WHITE)

    return vis


# ─────────────── Draw summary ───────────────
def draw_summary(
    frame: np.ndarray,
    items: List[Dict[str, Any]],
    show_overlay: bool = True,
    alpha_overlay: bool = False,
    show_label: bool = True,
) -> np.ndarray:
    """Draw summary with anomaly counter overlay."""
    vis = frame.copy()
    anomaly_count = 0

    for it in items:
        bbox = it.get("bbox", (0, 0, 0, 0))
        x1, y1, x2, y2 = map(int, bbox)
        status = it.get("status", "UNKNOWN")
        tid = it.get("id", -1)

        color = _status_color(status)
        if status == "ANOMALY":
            anomaly_count += 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if show_label:
            _draw_label(vis, f"ID:{tid} {status}", x1, max(20, y1 - 8), color)

    if show_overlay:
        text = f"Anomaly: {anomaly_count}"
        bg_color = COLOR_ANOMALY if anomaly_count > 0 else COLOR_BLACK

        if alpha_overlay:
            overlay = vis.copy()
            cv2.rectangle(overlay, (vis.shape[1] - 220, 10), (vis.shape[1] - 10, 60), bg_color, -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            cv2.putText(vis, text, (vis.shape[1] - 200, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2, lineType=cv2.LINE_AA)
        elif show_label:
            _draw_label(vis, text, vis.shape[1] - 200, 40, COLOR_WHITE)

    return vis
