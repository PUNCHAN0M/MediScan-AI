"""
MediScan-AI Configuration Package
=================================

Usage:
    from config.base import *           # Common settings (paths, YOLO, camera)
    from config.resnet import *         # ResNet PatchCore settings
"""

from config.base import (
    # Device
    DEVICE,
    # Image
    IMAGE_SIZE,
    IMAGE_EXTS,
    # Paths
    DATA_ROOT,
    BAD_DIR,
    SAVE_DIR,
    MODEL_OUTPUT_DIR,
    # YOLO Detection
    DETECTION_MODEL_PATH,
    DETECTION_CONF,
    DETECTION_IOU,
    # YOLO Segmentation
    SEGMENTATION_MODEL_PATH,
    SEGMENTATION_CONF,
    SEGMENTATION_IOU,
    SEGMENTATION_PAD,
    # Camera
    CAMERA_INDEX,
    FRAMES_BEFORE_SUMMARY,
    WINDOW_NAME,
    # Tracking
    TRACK_MAX_DISTANCE,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_AGE,
    # Training
    SEED,
    SELECTED_CLASSES,
    # Prediction
    COMPARE_CLASSES,
    DEFAULT_FALLBACK_THRESHOLD,
)

__all__ = [
    "DEVICE",
    "IMAGE_SIZE", "IMAGE_EXTS",
    "DATA_ROOT", "BAD_DIR", "SAVE_DIR", "MODEL_OUTPUT_DIR",
    "DETECTION_MODEL_PATH", "DETECTION_CONF", "DETECTION_IOU",
    "SEGMENTATION_MODEL_PATH", "SEGMENTATION_CONF", "SEGMENTATION_IOU", "SEGMENTATION_PAD",
    "CAMERA_INDEX", "FRAMES_BEFORE_SUMMARY", "WINDOW_NAME",
    "TRACK_MAX_DISTANCE", "TRACK_IOU_THRESHOLD", "TRACK_MAX_AGE",
    "SEED", "SELECTED_CLASSES",
    "COMPARE_CLASSES", "DEFAULT_FALLBACK_THRESHOLD",
]
