"""
Configuration file for MediScan-AI PatchCore parameters.
เปลี่ยนค่า parameters ที่นี่ที่เดียว จะมีผลทั้ง training และ prediction.
"""
from pathlib import Path

# =========================================================
#               PATCHCORE PARAMETERS
# =========================================================
IMG_SIZE = 256          # Image size for feature extraction
GRID_SIZE = 20          # Grid size for patch extraction (patches = GRID_SIZE × GRID_SIZE)
CORESET_RATIO = 0.12    # Ratio of patches to keep in memory bank (0.0-1.0)
K_NEAREST = 11        # Number of nearest neighbors for anomaly scoring
FALLBACK_THRESHOLD = 0.50  # Default threshold if calibration fails

# =========================================================
#               PATH CONFIGURATION
# =========================================================
DATA_ROOT = Path("/home/punchan0m/project/MediScan-AI/MediScan-AI/data")
MODEL_OUTPUT_DIR = Path("./model/patchcore")
SAVE_DIR = Path("./data/inspected")

# =========================================================
#               YOLO MODELS
# =========================================================
SEGMENTATION_MODEL_PATH = "model/yolo12-seg.pt"
DETECTION_MODEL_PATH = "model/best(2).pt"

# =========================================================
#               TRAINING CONFIGURATION
# =========================================================
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Selected classes to train (empty list = train all)
SELECTED_CLASSES = [
    'vitaminc',
    # 'black_cap',
    # 'red_cap',
    # 'oval',
    # "paracap",
    # 'white',
    # 'white_medium',
    # 'white_small',
    # "yellow",
    # 'yellow_circle',
]

# =========================================================
#               PREDICTION CONFIGURATION
# =========================================================
CAMERA_INDEX = 0
FRAMES_BEFORE_SUMMARY = 3
WINDOW_NAME = "Pill Inspector"

# Classes to compare during prediction
COMPARE_CLASSES = [
    "vitaminc",
    # "white",
    # 'yellow',
    # "yellow_circle",
    # "paracap",
    # 'black_cap',
    # 'red_cap',
    # 'oval',
    # 'white_medium',
]

# =========================================================
#               TUNING GUIDE
# =========================================================
"""
แนะนำการปรับค่า parameters:

1. GRID_SIZE (ความละเอียดของ patches):
   - เพิ่ม → patches มากขึ้น, ละเอียดมากขึ้น, training ช้าขึ้น
   - ลด → patches น้อยลง, เร็วขึ้น, อาจพลาด anomaly เล็กๆ
   - แนะนำ: 16-28

2. CORESET_RATIO (ขนาด memory bank):
   - เพิ่ม → memory bank ใหญ่ขึ้น, cover variations มากขึ้น
   - ลด → memory bank เล็กลง, เร็วขึ้น, อาจ false positive
   - แนะนำ: 0.10-0.25

3. K_NEAREST (จำนวน neighbors):
   - เพิ่ม → score เรียบขึ้น, less sensitive
   - ลด → more sensitive, อาจ false positive
   - แนะนำ: 15-30

4. FALLBACK_THRESHOLD (เกณฑ์ anomaly):
   - เพิ่ม → ต้องแตกต่างมากกว่าถึงจะเป็น anomaly (conservative)
   - ลด → sensitive มากขึ้น, อาจ false positive
   - แนะนำ: 0.30-0.60

สำหรับ conservative mode (ตรวจแค่ anomaly ชัดเจน):
   GRID_SIZE = 28
   CORESET_RATIO = 0.18
   K_NEAREST = 25
   FALLBACK_THRESHOLD = 0.50

สำหรับ sensitive mode (ตรวจหา anomaly เล็กๆ):
   GRID_SIZE = 24
   CORESET_RATIO = 0.12
   K_NEAREST = 15
   FALLBACK_THRESHOLD = 0.35

สำหรับ balanced mode (สมดุล):
   GRID_SIZE = 20
   CORESET_RATIO = 0.15
   K_NEAREST = 19
   FALLBACK_THRESHOLD = 0.40
"""
