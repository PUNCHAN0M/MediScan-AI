"""
Visualization Utilities
=======================

Draw bboxes, labels, overlays for pill inspection.
No business logic — pure drawing.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple


# ── colours (BGR) ──
COLOR_NORMAL  = (0, 255, 100)   # green
COLOR_ANOMALY = (0, 0, 255)     # red
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)


# ─────────────────────────────────────────────────────────
#  Text helpers
# ─────────────────────────────────────────────────────────
def put_text_with_bg(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = COLOR_WHITE,
    bg_color: Tuple[int, int, int] = COLOR_BLACK,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 5, y - th - 5), (x + tw + 5, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


def put_text_top_left(
    img: np.ndarray,
    text: str,
    margin: int = 15,
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = COLOR_WHITE,
    bg_color: Tuple[int, int, int] = COLOR_BLACK,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = margin, th + margin
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


def put_text_top_right(
    img: np.ndarray,
    text: str,
    margin: int = 15,
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = COLOR_WHITE,
    bg_color: Tuple[int, int, int] = COLOR_ANOMALY,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = img.shape[:2]
    x = w - tw - margin
    y = th + margin
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


# ─────────────────────────────────────────────────────────
#  Drawing functions
# ─────────────────────────────────────────────────────────
def draw_pill_results(
    frame: np.ndarray,
    results: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw per-frame detection results: coloured bboxes + labels.
    """
    vis = frame.copy()
    for r in results:
        bbox = r.get("bbox", r.get("padded_region", (0, 0, 0, 0)))
        x1, y1, x2, y2 = map(int, bbox)
        status = r.get("status", "UNKNOWN")
        tid = r.get("id", r.get("track_id", -1))
        normal_from = r.get("normal_from", [])

        color = COLOR_NORMAL if status == "NORMAL" else COLOR_ANOMALY
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        label = f"ID:{tid} | {status}"
        if normal_from:
            label += f" ({','.join(normal_from)})"
        cv2.putText(vis, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis


def draw_summary(
    frame: np.ndarray,
    items: List[Dict[str, Any]],
    show_overlay: bool = True,
) -> np.ndarray:
    """
    Draw summary results with anomaly-count overlay.
    """
    vis = frame.copy()
    anomaly_count = 0

    for it in items:
        bbox = it.get("bbox", (0, 0, 0, 0))
        x1, y1, x2, y2 = map(int, bbox)
        status = it.get("status", "UNKNOWN")
        tid = it.get("id", -1)

        color = COLOR_NORMAL if status == "NORMAL" else COLOR_ANOMALY
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        if status == "ANOMALY":
            anomaly_count += 1

        label = f"ID:{tid} {status}"
        cv2.putText(vis, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if show_overlay:
        bg = COLOR_ANOMALY if anomaly_count > 0 else COLOR_BLACK
        put_text_top_right(vis, f"Anomaly: {anomaly_count}", bg_color=bg)

    return vis
