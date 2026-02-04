"""
MobileNet + SIFE PatchCore Configuration
========================================

การตั้งค่าสำหรับ MobileNetV3 backbone + SIFE (Spatial Information Feature Enhancement).

🎯 Best for:
- Enhanced spatial awareness for defect localization
- Better small defect detection with position encoding
- Improved accuracy with minimal speed impact

📊 Performance:
- Speed: ⭐⭐⭐⭐ (Slightly slower than vanilla MobileNet)
- Texture: ⭐⭐⭐⭐⭐ (Enhanced with spatial info)
- Color: ⭐⭐⭐⭐ (Good)
- Shape: ⭐⭐⭐⭐⭐ (Best - spatial encoding helps)
- Small Defects: ⭐⭐⭐⭐⭐ (Position-aware detection)

🔬 SIFE (Spatial Information Feature Enhancement):
- Adds positional encoding to patch features
- Helps model learn WHERE defects typically occur
- Better at detecting edge/corner anomalies
"""
from pathlib import Path
from config.base import DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                         SIFE PATCHCORE PARAMETERS
# =============================================================================

# Image preprocessing
IMG_SIZE = 256                  # Input image size (pixels)

# Patch extraction - SMALLER = detect smaller defects
GRID_SIZE = 14                  # 🔥 40×40 = 1600 patches (จับ defect จิ๋วได้ดี)
                                # ยิ่ง grid เยอะ ยิ่งเห็น defect เล็กๆ

# Memory bank
CORESET_RATIO = 0.25            # 🔥 เพิ่มเป็น 0.25 (ครอบคลุม variation มากขึ้น)

# Anomaly scoring - LOWER = more sensitive
K_NEAREST = 3                   # 🔥 k=3 โหดสุดสำหรับ small defect
                                # k ยิ่งน้อย ยิ่ง sensitive (1-3 แนะนำ)

# Threshold - LOWER = catch more defects  
FALLBACK_THRESHOLD = 0.20       # 🔥 ลดเหลือ 0.20 (sensitive มากขึ้น)


# =============================================================================
#                         SIFE FEATURE SETTINGS
# =============================================================================

# Enable SIFE (Spatial Information Feature Enhancement)
USE_SIFE = True                 # Add spatial/positional encoding to features

# Position encoding dimension
SIFE_DIM = 64                   # Dimension of spatial features (8-64)
                                # Higher = more spatial detail, larger features

# Position encoding type
# "sinusoidal" - Sin/cos positional encoding (like Transformer)
# "learned" - Learnable position embeddings
# "coordinate" - Direct normalized coordinates (x, y)
SIFE_ENCODING_TYPE = "sinusoidal"

# 🔥 CNN vs SIFE Weighting (สำคัญมาก!)
# ปัญหา: CNN features dim เยอะ (~960), SIFE dim น้อย (~64)
# L2 normalize หลัง concat ทำให้ SIFE signal อ่อน
# แก้ไขโดยปรับ weight ก่อน concat
CNN_WEIGHT = 0.7                # 🔥 ลด CNN influence
SIFE_WEIGHT = 1.5               # 🔥 เพิ่ม SIFE influence
                                # Higher = more emphasis on position

# Include distance from center
USE_CENTER_DISTANCE = True      # Add distance from image center

# Include local gradient info
USE_LOCAL_GRADIENT = True       # Add gradient magnitude per patch

# 🔥 NEW: Laplacian variance for crack/scratch detection
USE_LAPLACIAN_VARIANCE = True   # Add Laplacian variance per patch
LAPLACIAN_WEIGHT = 1.2          # Weight for Laplacian features


# =============================================================================
#                    🔥 MULTI-SCALE DETECTION (NEW!)
# =============================================================================
# เปิดใช้ Multi-scale เพื่อ detect defect หลายขนาดพร้อมกัน

USE_MULTI_SCALE = True          # 🔥 Extract features at multiple grid sizes
MULTI_SCALE_GRIDS = [16, 32, 48, 64]  # Grid sizes: coarse → fine
                                # 16: large defects, 32: medium, 48: tiny defects


# =============================================================================
#                    🔥 EDGE ENHANCEMENT (NEW!)
# =============================================================================
# เพิ่มความสามารถในการ detect defect ที่ขอบ

USE_EDGE_ENHANCEMENT = True     # Enhance edge/border detection
EDGE_KERNEL_SIZE = 3            # Sobel kernel size (3, 5, 7)
EDGE_WEIGHT = 1.8               # 🔥 เพิ่มจาก 1.5 → 1.8 สำหรับ crack/scratch


# =============================================================================
#                         COLOR FEATURES (Optional)
# =============================================================================

# Can combine with color features for maximum detection
USE_COLOR_FEATURES = False      # Add RGB mean/std per patch
USE_HSV = False                 # Add HSV mean/std per patch
COLOR_WEIGHT = 1.0              # Weight for color features


# =============================================================================
#                    🔥 SCORING WEIGHTS (NEW!)
# =============================================================================
# เน้น top-k scoring สำหรับ tiny defect detection

SCORE_WEIGHT_MAX = 0.3          # Weight for max score
SCORE_WEIGHT_TOP_K = 0.5        # 🔥 เน้น top-k มากสุด (0.5)
SCORE_WEIGHT_PERCENTILE = 0.2   # Weight for percentile score
TOP_K_PERCENT = 0.05            # Top 5% patches for top-k mean


# =============================================================================
#                              MODEL OUTPUT PATH
# =============================================================================

MODEL_OUTPUT_DIR = Path("./model/patchcore_sife")


# =============================================================================
#                              TUNING GUIDE
# =============================================================================
"""
🔧 Parameter Tuning Guide for SIFE:

1. SIFE_DIM (Spatial feature dimension):
   - 8-16: Minimal spatial info, fast
   - 32: Balanced (default, recommended)
   - 64: Maximum spatial detail, slower
   
2. SIFE_ENCODING_TYPE:
   - "sinusoidal": Best for translation-invariant tasks
   - "coordinate": Best for fixed-position defects
   - "learned": Best when you have lots of training data

3. SIFE_WEIGHT:
   - 0.5-0.8: Spatial info less important
   - 1.0: Balanced (default)
   - 1.2-1.5: Position very important (edge defects)
   - 2.0: Position critical

4. USE_CENTER_DISTANCE:
   - True: Better for center vs edge anomaly detection
   - False: Faster, less position bias

5. USE_LOCAL_GRADIENT:
   - True: Better edge/scratch detection
   - False: Faster, focus on texture only

📋 Preset Configurations:

Balanced Mode (Default):
    SIFE_DIM = 32
    SIFE_ENCODING_TYPE = "sinusoidal"
    SIFE_WEIGHT = 1.0
    USE_CENTER_DISTANCE = True
    USE_LOCAL_GRADIENT = True
    
Fast Mode:
    SIFE_DIM = 16
    SIFE_ENCODING_TYPE = "coordinate"
    SIFE_WEIGHT = 0.8
    USE_CENTER_DISTANCE = False
    USE_LOCAL_GRADIENT = False
    
Maximum Accuracy Mode:
    SIFE_DIM = 64
    SIFE_ENCODING_TYPE = "sinusoidal"
    SIFE_WEIGHT = 1.2
    USE_CENTER_DISTANCE = True
    USE_LOCAL_GRADIENT = True
    USE_COLOR_FEATURES = True
    USE_HSV = True

Edge Defect Mode (scratches, cracks at borders):
    SIFE_DIM = 48
    SIFE_WEIGHT = 1.5
    USE_CENTER_DISTANCE = True
    USE_LOCAL_GRADIENT = True

🎯 When to use SIFE:
- Defects tend to occur in specific regions (edges, corners)
- Need better localization accuracy
- Small defects that vanilla PatchCore misses
- Training data has position-correlated defects
"""
