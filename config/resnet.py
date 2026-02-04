"""
ResNet18 PatchCore Configuration
================================

การตั้งค่าสำหรับ ResNet18 backbone พร้อม Color Features.

🎯 Best for:
- Color anomaly detection (white vs black pills)
- Small colored defects (black spots, discoloration)
- Pills with color-based quality criteria

📊 Performance:
- Speed: ⭐⭐⭐⭐ (Fast)
- Texture: ⭐⭐⭐⭐
- Color: ⭐⭐⭐⭐⭐ (Best)
- Shape: ⭐⭐⭐⭐
"""
from pathlib import Path
from config.base import DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                         RESNET PATCHCORE PARAMETERS
# =============================================================================

# Image preprocessing
IMG_SIZE = 256                  # Input image size (pixels)

# Patch extraction
GRID_SIZE = 28                  # Smaller patches for color detection
                                # More patches = better small defect detection

# Memory bank
CORESET_RATIO = 0.12            # Ratio of patches to keep (0.0-1.0)

# Anomaly scoring
K_NEAREST = 11                  # Number of nearest neighbors for scoring

# Threshold
FALLBACK_THRESHOLD = DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                              COLOR FEATURES
# =============================================================================

# Enable color feature extraction
USE_COLOR_FEATURES = True       # Add RGB mean/std per patch

# Enable HSV color space (recommended for color anomalies)
USE_HSV = True                  # Add HSV mean/std per patch

# Weight for color features (higher = more emphasis on color)
COLOR_WEIGHT = 1.0              # Range: 0.5-2.0


# =============================================================================
#                              MODEL OUTPUT PATH
# =============================================================================

MODEL_OUTPUT_DIR = Path("./model/patchcore_resnet")


# =============================================================================
#                              TUNING GUIDE
# =============================================================================
"""
🔧 Parameter Tuning Guide for ResNet (Color-Aware):

1. GRID_SIZE (Patch resolution):
   - 24-28: Good for small colored defects (recommended)
   - 28-32: Best for tiny spots, slower
   - 20-24: Faster, may miss small defects
   
2. COLOR_WEIGHT (Color importance):
   - 0.5-0.8: Texture more important than color
   - 1.0: Balanced (default)
   - 1.2-1.5: Color very important
   - 1.5-2.0: Color critical (black/white separation)

3. USE_HSV:
   - True: Better color separation (recommended)
   - False: Faster, RGB only

📋 Preset Configurations:

Color-Critical Mode (White vs Black pills):
    GRID_SIZE = 28
    CORESET_RATIO = 0.15
    K_NEAREST = 19
    COLOR_WEIGHT = 1.5
    USE_HSV = True
    
Balanced Color Mode (Default):
    GRID_SIZE = 28
    CORESET_RATIO = 0.12
    K_NEAREST = 11
    COLOR_WEIGHT = 1.0
    USE_HSV = True
    
Texture-Focused Mode:
    GRID_SIZE = 24
    CORESET_RATIO = 0.12
    K_NEAREST = 11
    COLOR_WEIGHT = 0.7
    USE_HSV = False

🎨 Color Anomaly Examples:
- Black spots on white pills → COLOR_WEIGHT = 1.5
- Discoloration → COLOR_WEIGHT = 1.2
- Faded colors → USE_HSV = True, COLOR_WEIGHT = 1.3
"""
