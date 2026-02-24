"""
MobileNet PatchCore Configuration
=================================

การตั้งค่าสำหรับ MobileNetV3 backbone.

🎯 Best for:
- Fast inference (lightweight model)
- Good texture detection
- Real-time applications

📊 Performance:
- Speed: ⭐⭐⭐⭐⭐ (Fastest)
- Texture: ⭐⭐⭐⭐
- Color: ⭐⭐⭐
- Shape: ⭐⭐⭐
"""
from pathlib import Path
from config.base import DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                         MOBILENET PATCHCORE PARAMETERS
# =============================================================================

# Image preprocessing
IMG_SIZE = 256                  # Input image size (pixels)

# Patch extraction
GRID_SIZE = 16                  # Grid size for patches (20×20 = 400 patches)

# Memory bank
CORESET_RATIO = 1            # Ratio of patches to keep (0.0-1.0)
                                # Higher = more coverage, slower inference

# Anomaly scoring
K_NEAREST = 3                  # Number of nearest neighbors for scoring
                                # Higher = smoother scores, less sensitive

# Threshold
FALLBACK_THRESHOLD = DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                              MODEL OUTPUT PATH
# =============================================================================

MODEL_OUTPUT_DIR = Path("./model/patchcore")


# =============================================================================
#                              TUNING GUIDE
# =============================================================================
"""
🔧 Parameter Tuning Guide for MobileNet:

1. GRID_SIZE (Patch resolution):
   - 16-20: Fast, good for large defects
   - 20-28: Balanced, recommended
   - 28-32: Slow, better for small defects
   
2. CORESET_RATIO (Memory bank size):
   - 0.08-0.12: Fast, may miss variations
   - 0.12-0.18: Balanced, recommended
   - 0.18-0.25: Thorough, slower
   
3. K_NEAREST (Scoring sensitivity):
   - 5-10: Very sensitive, may false positive
   - 11-19: Balanced, recommended
   - 20-30: Conservative, may miss small anomalies

4. FALLBACK_THRESHOLD (Anomaly threshold):
   - 0.30-0.40: Sensitive (catches more anomalies)
   - 0.40-0.50: Balanced
   - 0.50-0.60: Conservative (only obvious anomalies)

📋 Preset Configurations:

Fast Mode (Real-time):
    GRID_SIZE = 16
    CORESET_RATIO = 0.10
    K_NEAREST = 9
    
Balanced Mode (Default):
    GRID_SIZE = 20
    CORESET_RATIO = 0.12
    K_NEAREST = 11
    
Accurate Mode (Offline):
    GRID_SIZE = 24
    CORESET_RATIO = 0.18
    K_NEAREST = 15
"""
