"""
MediScan-AI Configuration Package
=================================

Configuration modules for different PatchCore backbones.

Usage:
    from config.base import *           # Common settings
    from config.mobilenet import *      # MobileNet settings
    from config.resnet import *         # ResNet settings  
    from config.dinov2 import *         # DINOv2 settings
"""

from config.base import (
    # Paths
    DATA_ROOT,
    SAVE_DIR,
    SEGMENTATION_MODEL_PATH,
    DETECTION_MODEL_PATH,
    
    # Training
    SEED,
    IMAGE_EXTS,
    SELECTED_CLASSES,
    
    # Prediction
    CAMERA_INDEX,
    FRAMES_BEFORE_SUMMARY,
    WINDOW_NAME,
    COMPARE_CLASSES,
)

__all__ = [
    # Paths
    "DATA_ROOT",
    "SAVE_DIR",
    "SEGMENTATION_MODEL_PATH",
    "DETECTION_MODEL_PATH",
    
    # Training
    "SEED",
    "IMAGE_EXTS",
    "SELECTED_CLASSES",
    
    # Prediction
    "CAMERA_INDEX",
    "FRAMES_BEFORE_SUMMARY",
    "WINDOW_NAME",
    "COMPARE_CLASSES",
]
