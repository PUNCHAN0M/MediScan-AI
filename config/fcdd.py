"""
FCDD Configuration
==================
Fully Convolutional Data Description สำหรับ pill anomaly detection.

Pipeline:
  1. Train FCDD ด้วย good images only
  2. Inference → anomaly heatmap per pill
  3. Threshold heatmap → binary mask
  4. Connected components → ตรวจจับบริเวณ anomaly
  5. คำนวณ center ของแต่ละเม็ดยา
  6. นับจำนวน pill & anomaly
"""
from pathlib import Path

# =============================================================================
#                         DATA PATHS
# =============================================================================
PILL_NAME = "white_front"
GOOD_DATA_DIR = Path(f"data_yolo/{PILL_NAME}_cropped")
BAD_DATA_DIR = Path("data_yolo/bad")
TEST_DATA_DIR = Path("data_yolo/test")
RESULT_DIR = Path(f"result_fcdd/{PILL_NAME}")

# =============================================================================
#                         MODEL PATHS
# =============================================================================
BACKBONE_PATH = Path("model/backbone_mbn_pill.pth")
SEGMENTATION_MODEL_PATH = Path("model/pill-detection-best-2.pt")
FCDD_MODEL_DIR = Path("model/fcdd")
FCDD_MODEL_PATH = FCDD_MODEL_DIR / f"fcdd_{PILL_NAME}_model.pth"

# =============================================================================
#                         IMAGE SETTINGS
# =============================================================================
IMG_SIZE = 256
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# =============================================================================
#                         TRAINING SETTINGS
# =============================================================================
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PSEUDO_ANOMALY_RATIO = 0.5       # สัดส่วน pseudo-anomaly ต่อ batch
NUM_WORKERS = 0
SEED = 42

# Backbone layers to extract features from (MobileNetV3-Large)
HOOK_INDICES = [3, 6, 9, 12, 15]

# =============================================================================
#                         INFERENCE SETTINGS
# =============================================================================
ANOMALY_THRESHOLD = 0.5           # threshold สำหรับ heatmap (0-1)
MIN_COMPONENT_AREA = 50           # พื้นที่ minimum ของ connected component (pixels)
YOLO_CONF = 0.5                   # YOLO detection confidence
YOLO_IMG_SIZE = 640               # YOLO input size
CENTER_DOT_RADIUS = 8             # ขนาดจุด center
FONT_SCALE = 0.8
FONT_THICKNESS = 2
