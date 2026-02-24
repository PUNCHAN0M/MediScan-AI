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

# YOLO model paths
# SEGMENTATION_MODEL_PATH = "model/yolo12-seg.pt"
# DETECTION_MODEL_PATH = "model/yolo12-seg.pt"
DETECTION_MODEL_PATH = "model/pill-detection-best-2.pt"
SEGMENTATION_MODEL_PATH = "model/pill-detection-best-2.pt"

# =============================================================================
#                           TRAINING CONFIGURATION
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Classes to train (empty list = train all available classes)
SELECTED_CLASSES = [
    # "brown_cap_test",
    # "brown_cap",
    "yellow_circle",
    # "white_oval",
    # "circle_gray",
    # "cream",
    # "oval",
    # "red_fray_cap",
    # "white_half_cap",
    # "yellow_cap",
    # "white_front"

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
    # "brown_cap_test",
    # "brown_cap",
    "yellow_circle",
    # "white_oval",
    # "circle_gray",
    # "cream",
    # "oval",
    # "red_fray_cap",
    # "white_half_cap",
    # "yellow_cap",
    # "circle_yellow",
    # "white_front",
]


# =============================================================================
#                              DEFAULT THRESHOLDS
# =============================================================================

# Default fallback threshold if calibration fails
DEFAULT_FALLBACK_THRESHOLD = 0.50
