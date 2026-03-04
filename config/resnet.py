#config\resnet.py
"""
ResNet50 PatchCore Configuration
================================

Backbone : ResNet50 (pretrained / fine-tuned)
Layers   : layer1 (256) + layer2 (512) + layer3 (1024) → 1792-dim patches
Detection: YOLOv12-seg (.pt / .onnx)

Performance:
- Speed:   Fast  (ResNet50 + FAISS kNN, no backprop)
- Texture: ★★★★
- Color:   ★★★★★
- Shape:   ★★★★
"""
from config.base import DEFAULT_FALLBACK_THRESHOLD, IMAGE_SIZE


# =============================================================================
#  BACKBONE
# =============================================================================
BACKBONE = "model/backbone/resnet_last.pth"
LAYERS = ("layer1", "layer2", "layer3")
LAYER_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024}


# =============================================================================
#  IMAGE / PATCH  (IMAGE_SIZE มาจาก base.py = 256)
# =============================================================================
IMG_SIZE = IMAGE_SIZE                   # ใช้ขนาดเดียวกับ base
GRID_SIZE = 16                          # patches per side → 16×16 = 256 patches


# =============================================================================
#  PATCHCORE — MEMORY BANK
# =============================================================================
CORESET_RATIO = 0.25                    # fraction ที่เก็บจาก raw patches
CORESET_MIN_KEEP = 1_000               # จำนวน patch ขั้นต่ำที่เก็บ
CORESET_MAX_KEEP = 20_000              # จำนวน patch สูงสุดที่เก็บ


# =============================================================================
#  PATCHCORE — SCORING
# =============================================================================
K_NEAREST = 3                           # kNN neighbours
SCORE_METHOD = "max"                    # "max" | "top1_mean" | "top5_mean" | "top10_mean"
THRESHOLD_MULTIPLIER = 1.0              # คูณ threshold ตอน inference (>1 = ผ่อนปรน)
FALLBACK_THRESHOLD = DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#  COLOR FEATURES  (ต่อท้าย CNN features)
# =============================================================================
USE_COLOR_FEATURES = True               # +6 dims (RGB mean/std)
USE_HSV = True                          # +6 dims (HSV mean/std)
COLOR_WEIGHT = 1.5                      # น้ำหนักของ color features


# =============================================================================
#  FINE-TUNE BACKBONE
# =============================================================================
N_STEPS = 351                           # training steps สำหรับ fine-tune