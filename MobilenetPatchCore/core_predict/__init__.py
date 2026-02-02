# core_predict/__init__.py
"""
Prediction modules for realtime pill inspection.

Usage:
    from MobilenetPatchCore.core_predict import PillInspector, InspectorConfig
    from MobilenetPatchCore.core_predict import PillYOLODetector
"""
from .yolo_detector import PillYOLODetector
from .inspector import PillInspector, InspectorConfig, make_square_crop
from .visualizer import (
    draw_pill_results,
    draw_summary,
    put_text_with_bg,
    put_text_top_right,
    put_text_top_left,
    COLOR_NORMAL,
    COLOR_ANOMALY,
)

__all__ = [
    # Main classes
    "PillInspector",
    "InspectorConfig",
    "PillYOLODetector",
    # Utilities
    "make_square_crop",
    # Visualization
    "draw_pill_results",
    "draw_summary",
    "put_text_with_bg",
    "put_text_top_right",
    "put_text_top_left",
    "COLOR_NORMAL",
    "COLOR_ANOMALY",
]

