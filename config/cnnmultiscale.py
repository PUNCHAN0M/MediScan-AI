"""
CNN Multi-Scale PatchCore Configuration
========================================

การตั้งค่าสำหรับ Modified ResNet34 + Multi-Scale PatchCore
ออกแบบเฉพาะสำหรับตรวจจับ defect เล็กมาก (2-5 px cracks)

🎯 Best for:
- Tiny crack detection (2-5 px)
- Micro defects on pill surfaces
- High-resolution anomaly detection
- Surface scratches, hairline cracks

📊 Performance:
- Speed: ⭐⭐⭐ (Moderate — multi-scale processing)
- Tiny defect: ⭐⭐⭐⭐⭐ (Best — designed for this)
- Texture: ⭐⭐⭐⭐⭐
- Color: ⭐⭐⭐⭐
- Shape: ⭐⭐⭐⭐

🔥 Key Innovations:
1. Modified ResNet34 — no maxpool, stride=1 conv1 (4x more resolution)
2. Separate PatchCore memory per scale (layer1, layer2, layer3)
3. L2 normalize per scale before scoring
4. Score fusion: max(score_s1, score_s2, score_s3)
5. Multi-resolution input (512 + 768)
6. CLAHE preprocessing for contrast boost
7. Adaptive threshold (mean + k*std)
8. SE attention blocks for defect focus
"""
from pathlib import Path
from config.base import DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                     CNN MULTI-SCALE PATCHCORE PARAMETERS
# =============================================================================

# Image preprocessing — ใช้ขนาดใหญ่เพื่อรักษา detail
IMG_SIZE = 512                  # Primary input size (high resolution)
IMG_SIZE_SECONDARY = 768        # Secondary input for multi-resolution (optional)
ENABLE_MULTI_RESOLUTION = True  # Enable dual-resolution input

# Patch extraction — grid ใหญ่เพื่อตรวจ defect เล็ก
GRID_SIZE = 32                  # 32×32 = 1024 patches (fine-grained)
                                # ละเอียดกว่า MobileNet (20) และ ResNet (28)

# Memory bank
CORESET_RATIO = 0.15            # สูงกว่าปกติเพราะ multi-scale ต้องการ coverage มากขึ้น

# Anomaly scoring
K_NEAREST = 9                   # ใช้ K น้อยลงเพื่อ sensitivity สูงกว่า
                                # สำหรับ defect เล็กที่มี neighbor น้อย

# Threshold
FALLBACK_THRESHOLD = DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                        BACKBONE CONFIGURATION
# =============================================================================

# Backbone selection: "resnet34" or "resnet50"
BACKBONE = "resnet34"           # ResNet34 = balanced speed/quality
                                # ResNet50 = better features, slower

# Modified backbone settings (preserve spatial resolution)
REMOVE_MAXPOOL = True           # Remove initial maxpool layer
STRIDE1_CONV1 = True            # Change conv1 stride=2 → stride=1
USE_DILATED_LAYER3 = True       # Use dilation=2 in layer3 instead of stride=2

# Selected layers for feature extraction
# layer1: high resolution, receptive field ~11-15px → detects 2-5px cracks
# layer2: medium resolution, captures texture patterns
# layer3: semantic, captures shape context (with dilation to keep resolution)
# layer4: NOT used — receptive field too large, smooths out tiny defects
SELECTED_LAYERS = ["layer1", "layer2", "layer3"]


# =============================================================================
#                        MULTI-SCALE FUSION
# =============================================================================

# Score fusion strategy
# "max"    — max(score_s1, score_s2, score_s3) → เหมาะกับ defect เล็กที่โผล่แค่ scale เดียว
# "mean"   — mean(score_s1, score_s2, score_s3) → เสถียรกว่า แต่ sensitive น้อยลง
# "weighted" — w1*s1 + w2*s2 + w3*s3 → custom weights
SCORE_FUSION = "max"

# Weights for "weighted" fusion (layer1, layer2, layer3)
# layer1 สูงสุดเพราะ defect เล็กโผล่ที่ scale นี้
SCALE_WEIGHTS = [0.5, 0.3, 0.2]

# Use separate memory bank per scale (recommended)
SEPARATE_MEMORY_PER_SCALE = True

# Use 1x1 conv fusion after concat (learnable weights)
USE_CONV_FUSION = False         # True = learnable fusion, requires training
                                # False = manual fusion (recommended for PatchCore)


# =============================================================================
#                        PREPROCESSING (CONTRAST BOOST)
# =============================================================================

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
USE_CLAHE = True                # Boost micro-crack contrast
CLAHE_CLIP_LIMIT = 2.0          # Clip limit (1.0-4.0, higher = more contrast)
CLAHE_TILE_SIZE = 8             # Tile grid size (8 = good for pill surface)

# Laplacian edge boost
USE_LAPLACIAN_BOOST = False     # Add Laplacian edge channel
LAPLACIAN_WEIGHT = 0.3          # Weight of edge channel (0.0-1.0)


# =============================================================================
#                        ATTENTION MECHANISM
# =============================================================================

# SE (Squeeze-and-Excitation) attention
USE_SE_ATTENTION = True         # Channel attention for defect focus
SE_REDUCTION = 16               # SE reduction ratio


# =============================================================================
#                        ADAPTIVE THRESHOLD
# =============================================================================

# Adaptive threshold method
# "sigma"      — mean + k*std
# "percentile" — percentile-based
# "f1"         — optimize F1 score (requires anomaly test data)
THRESHOLD_METHOD = "sigma"

# Sigma method parameters
THRESHOLD_SIGMA = 3.0           # k value for mean + k*std
                                # 2.5 = more sensitive, 3.5 = more conservative

# Percentile method parameter
THRESHOLD_PERCENTILE = 99.5     # High percentile for low-variance surfaces


# =============================================================================
#                        COLOR FEATURES (OPTIONAL)
# =============================================================================

USE_COLOR_FEATURES = True       # Add RGB mean/std per patch
USE_HSV = True                  # Add HSV mean/std per patch
COLOR_WEIGHT = 0.5              # Lower than ResNet — focus on structure not color


# =============================================================================
#                              MODEL OUTPUT PATH
# =============================================================================

MODEL_OUTPUT_DIR = Path("./model/patchcore_cnn_multiscale")


# =============================================================================
#                              TUNING GUIDE
# =============================================================================
"""
🔧 Parameter Tuning Guide for CNN Multi-Scale (Tiny Defect):

1. IMG_SIZE (Resolution — most critical):
   - 512: Balanced (default, recommended)
   - 768: With multi-resolution enabled (better for 2px defects)
   - 256: Fast mode, will miss 2px defects

2. GRID_SIZE (Patch density):
   - 28-32: Good for 3-5px defects (recommended)
   - 32-40: Best for 2-3px defects (slower)
   - 20-24: Fast, misses tiny defects

3. SCORE_FUSION (How to combine multi-scale scores):
   - "max": Most sensitive — defect in ANY scale triggers (recommended)
   - "mean": More stable — requires evidence across scales
   - "weighted": Custom — weight layer1 highest for tiny defects

4. BACKBONE:
   - "resnet34": Balanced speed/quality (recommended)
   - "resnet50": Better features, 2x slower

5. CLAHE settings:
   - CLIP_LIMIT 1.5-2.0: Gentle boost (smooth surfaces)
   - CLIP_LIMIT 2.5-4.0: Aggressive (textured surfaces)
   
6. THRESHOLD_SIGMA:
   - 2.0-2.5: Very sensitive (catches more, more false positives)
   - 3.0: Balanced (default)
   - 3.5-4.0: Conservative (only obvious defects)

📋 Preset Configurations:

Maximum Sensitivity (2px cracks):
    IMG_SIZE = 512
    IMG_SIZE_SECONDARY = 768
    ENABLE_MULTI_RESOLUTION = True
    GRID_SIZE = 36
    K_NEAREST = 7
    CORESET_RATIO = 0.18
    SCORE_FUSION = "max"
    USE_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.5
    THRESHOLD_SIGMA = 2.5

Balanced Mode (Default):
    IMG_SIZE = 512
    ENABLE_MULTI_RESOLUTION = True
    GRID_SIZE = 32
    K_NEAREST = 9
    CORESET_RATIO = 0.15
    SCORE_FUSION = "max"
    USE_CLAHE = True
    THRESHOLD_SIGMA = 3.0

Fast Mode (5px+ defects):
    IMG_SIZE = 512
    ENABLE_MULTI_RESOLUTION = False
    GRID_SIZE = 24
    K_NEAREST = 11
    CORESET_RATIO = 0.12
    SCORE_FUSION = "max"
    USE_CLAHE = False
    THRESHOLD_SIGMA = 3.0
"""
