"""
Base Configuration for MediScan-AI
==================================

การตั้งค่าพื้นฐานที่ใช้ร่วมกันทุก backbone models.
สำหรับการตั้งค่าเฉพาะแต่ละ model ดูที่:
- config/mobilenet.py
- config/resnet.py  
- config/dinov2.py
"""
from pathlib import Path


# =============================================================================
#                              PATH CONFIGURATION
# =============================================================================

# Data directories
DATA_ROOT = Path("D:\project\Medicine-AI\MediScan-AI\data")
SAVE_DIR = Path("./data/inspected")
BAD_DIR = "data_yolo/calibrate-bad"

# YOLO model paths
# SEGMENTATION_MODEL_PATH = "model/yolo12-seg.pt"
# DETECTION_MODEL_PATH = "model/yolo12-seg.pt"
DETECTION_MODEL_PATH = "model/SEGMENTATION/pill-detection-best-2.pt"
SEGMENTATION_MODEL_PATH = "model/SEGMENTATION/pill-detection-best-2.pt"

# =============================================================================
#                           TRAINING CONFIGURATION
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Classes to train (empty list = train all available classes)
SELECTED_CLASSES = [
    "black_sphere",

]


# =============================================================================
#                          PREDICTION CONFIGURATION
# =============================================================================

# Camera settings
CAMERA_INDEX = 1
FRAMES_BEFORE_SUMMARY = 3
WINDOW_NAME = "Pill Inspector"

# Classes to compare during prediction
COMPARE_CLASSES = [
    "black_sphere",
]


# =============================================================================
#                              DEFAULT THRESHOLDS
# =============================================================================

# Default fallback threshold if calibration fails
DEFAULT_FALLBACK_THRESHOLD = 0.50
