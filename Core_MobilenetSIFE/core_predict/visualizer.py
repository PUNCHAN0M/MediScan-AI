# core_predict/visualizer.py
"""
Visualization utilities for SIFE pill inspection.

Single Responsibility:
- Draw bboxes, labels, overlays
- No business logic
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any


# Colors (BGR)
COLOR_NORMAL = (0, 255, 100)   # Green
COLOR_ANOMALY = (0, 0, 255)    # Red
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def put_text_with_bg(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = COLOR_WHITE,
    bg_color: Tuple[int, int, int] = COLOR_BLACK,
    thickness: int = 2,
) -> None:
    """Draw text with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    cv2.rectangle(img, (x - 5, y - th - 5), (x + tw + 5, y + 5), bg_color, -1)
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
    """Draw text at top-right corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    h, w = img.shape[:2]
    x = w - tw - margin
    y = th + margin
    
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), bg_color, -1)
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
    """Draw text at top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = margin
    y = th + margin
    
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


def draw_bbox(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw bounding box."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_pill_results(
    frame: np.ndarray,
    results: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw detection results on frame."""
    vis = frame.copy()
    
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
    
    return vis


def draw_summary(
    frame: np.ndarray,
    items: List[Dict[str, Any]],
    show_overlay: bool = True,
) -> np.ndarray:
    """Draw summary results with anomaly count overlay."""
    vis = frame.copy()
    
    anomaly_count = 0
    for it in items:
        bbox = it.get("bbox", it.get("position", (0, 0, 0, 0)))
        x1, y1, x2, y2 = map(int, bbox)
        status = it.get("status", "UNKNOWN")
        tid = it.get("track_id", it.get("id", -1))
        
        color = COLOR_NORMAL if status == "NORMAL" else COLOR_ANOMALY
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        
        if status == "ANOMALY":
            anomaly_count += 1
        
        label = f"ID:{tid} {status}"
        y_text = max(20, y1 - 10)
        cv2.putText(vis, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if show_overlay:
        bg = COLOR_ANOMALY if anomaly_count > 0 else COLOR_BLACK
        put_text_top_right(vis, f"Anomaly: {anomaly_count}", bg_color=bg)
    
    return vis
