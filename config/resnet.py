"""
ResNet50 PatchCore Configuration
================================

Backbone : ResNet50 (pretrained, frozen)
Layers   : layer2 (512 ch) + layer3 (1024 ch) → concat → 1536-dim patches
Detection: YOLOv12-seg (.pt / .onnx)

Best for:
- Color anomaly detection (white vs black pills)
- Small coloured defects (black spots, discolouration)
- Pills with colour-based quality criteria

Performance:
- Speed:   Fast  (ResNet50 + FAISS kNN, no backprop)
- Texture: ★★★★
- Color:   ★★★★★
- Shape:   ★★★★
"""
from pathlib import Path
from config.base import DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#  BACKBONE
# =============================================================================
BACKBONE = "model/backbone/resnet_backbone_20260226_234833.pth"  #"None"
LAYERS = ["layer1", "layer2", "layer3"]          # 512 + 1024 = 1536 dim/patch


# =============================================================================
#  IMAGE / PATCH
# =============================================================================
IMG_SIZE = 256                          # input resolution
GRID_SIZE = 16                          # patches per side  (16 → 256 patches)


# =============================================================================
#  PATCHCORE
# =============================================================================
CORESET_RATIO = 0.25                    # fraction of patches to keep
K_NEAREST = 3                           # kNN neighbours for scoring


# =============================================================================
#  THRESHOLD
# =============================================================================
FALLBACK_THRESHOLD = DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#  SCORING
# =============================================================================
SCORE_METHOD = "max"              # "max" (sensitive) | "top5_mean" (balanced)
# Multiplier applied to the calibrated threshold at inference.
# 1.0 = trust calibration exactly.  > 1.0 = more lenient.  Tune without retraining.
THRESHOLD_MULTIPLIER = 1


# =============================================================================
#  COLOR FEATURES  (optional, appended to CNN features)
# =============================================================================
USE_COLOR_FEATURES = True               # +6 dims (RGB mean/std)
USE_HSV = True                          # +6 dims (HSV mean/std)
COLOR_WEIGHT = 1.5                      # keep colour influence mild (was 1.0)


# =============================================================================
#  MODEL OUTPUT
# =============================================================================
MODEL_OUTPUT_DIR = Path("./model/patchcore_resnet")
