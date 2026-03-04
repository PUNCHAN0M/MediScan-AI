"""
Base Configuration for MediScan-AI
==================================
การตั้งค่าพื้นฐานที่ใช้ร่วมกัน (paths, camera, YOLO, training data).
การตั้งค่า ResNet50 PatchCore ดูที่ config/resnet.py
"""
from pathlib import Path
import torch


# =============================================================================
#                              DEVICE
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
#                              IMAGE
# =============================================================================
IMAGE_SIZE = 256                                    # ขนาดรูปภาพมาตรฐาน (ใช้ทั้ง train + predict)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# =============================================================================
#                              PATHS
# =============================================================================
DATA_ROOT            = Path(r"D:\project\Medicine-AI\MediScan-AI\data_train_defection")
BAD_DIR              = "data_bad/"
SAVE_DIR             = Path("./data/inspected")
MODEL_OUTPUT_DIR     = Path("./model/patchcore_resnet")


# =============================================================================
#                          YOLO DETECTION
# =============================================================================
DETECTION_MODEL_PATH     = "model/SEGMENTATION/pill-detection-best-2.onnx"
DETECTION_IMG_SIZE       = 512
DETECTION_CONF           = 0.5
DETECTION_IOU            = 0.6


# =============================================================================
#                        YOLO SEGMENTATION
# =============================================================================
SEGMENTATION_MODEL_PATH  = "model/SEGMENTATION/pill-detection-best-2.onnx"
SEGMENTATION_IMG_SIZE    = 512
SEGMENTATION_CONF        = 0.5
SEGMENTATION_IOU         = 0.6
SEGMENTATION_PAD         = 5              # padding (px) รอบ bbox เมื่อ crop


# =============================================================================
#                          CAMERA / REALTIME
# =============================================================================
CAMERA_INDEX             = 0
FRAMES_BEFORE_SUMMARY    = 3
WINDOW_NAME              = "Pill Inspector"


# =============================================================================
#                            TRACKING
# =============================================================================
TRACK_MAX_DISTANCE       = 80.0
TRACK_IOU_THRESHOLD      = 0.80
TRACK_MAX_AGE            = 10


# =============================================================================
#                          TRAINING
# =============================================================================
SEED                     = 42
SELECTED_CLASSES: list[str] = ["Androxsil"]          # ว่าง = train ทุก class


# =============================================================================
#                          PREDICTION
# =============================================================================
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
    "paracap",
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
#                        DEFAULT THRESHOLD
# =============================================================================
DEFAULT_FALLBACK_THRESHOLD = 0.50

