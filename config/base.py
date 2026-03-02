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
DATA_ROOT = Path("D:\project\Medicine-AI\MediScan-AI\data_train_defection")
SAVE_DIR = Path("./data/inspected")
BAD_DIR = "data_yolo\calibrate-bad/"

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
    # "Androxsil",
    # "ANTA1mg",
    # "Baclofen",
    # "Bestatin20",
    # "CANDY",
    # "Clozapine25mg",
    # "DiclofenacSodium50mg",
    # "DILIZEM",
    # "Exelon45mg",
    # "Fahtaliejone",
    # "Flunarizine5mg",
    # "Flupentixol05mg",
    # "Glipizide",
    # "green_circle",
    # "ISOTRATE",
    # "JANUMET_back",
    # "JANUMET_front",
    # "Lanzaar50",
    # "line",
    # "LOSARTAN_POTASSIUM50mg",
    # "NOXA20",
    # "panadol",
    # "paracap",
    # "pink_circle",
    # "Pregabalin25mg",
    # "Quetiepine25mg",
    # "sara",
    # "SITAGTIN-100",
    # "tiffy_back",
    # "tiffy_front",
    # "TOP",
    # "Trazodel",
    # "TURMERICCAPSULE",
    # "ULTRACET",
    # "vitaminc",
    # "white_oval",
    # "white_smaill_circle",
    # "XELJANZ5mg",
    # "yellow_circle",
    # "ZYMRON15",
]


# =============================================================================
#                              DEFAULT THRESHOLDS
# =============================================================================

# Default fallback threshold if calibration fails
DEFAULT_FALLBACK_THRESHOLD = 0.50
