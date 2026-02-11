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
DATA_ROOT = Path("/home/punchan0m/project/MediScan-AI/MediScan-AI/data")
SAVE_DIR = Path("./data/inspected")

# YOLO model paths
# SEGMENTATION_MODEL_PATH = "model/yolo12-seg.pt"
# DETECTION_MODEL_PATH = "model/yolo12-seg.pt"
DETECTION_MODEL_PATH = "model/pill-detection.pt"
SEGMENTATION_MODEL_PATH = "model/yolo12-seg.pt"

# =============================================================================
#                           TRAINING CONFIGURATION
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Classes to train (empty list = train all available classes)
SELECTED_CLASSES = [
    "yellow_cap",
    "red_cap",
    "circle_yellow",
    "circle_white",
    # "vitaminc",
    # "paracap",
    # "white",
    # "white_medium",
    # "white_small",
    # "yellow",
    # "sara",
    # "oval",
]


# =============================================================================
#                          PREDICTION CONFIGURATION
# =============================================================================

# Camera settings
CAMERA_INDEX = 0
FRAMES_BEFORE_SUMMARY = 3
WINDOW_NAME = "Pill Inspector"

# Classes to compare during prediction
COMPARE_CLASSES = [
    # "yellow_cap",
    # "red_cap",
    # "circle_yellow",
    "circle_white",
]


# =============================================================================
#                              DEFAULT THRESHOLDS
# =============================================================================

# Default fallback threshold if calibration fails
DEFAULT_FALLBACK_THRESHOLD = 0.50
