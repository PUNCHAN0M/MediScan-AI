"""
WideResNet50 PatchCore Configuration (Anomaly Detection)
=========================================================

การตั้งค่าสำหรับ WideResNet50 backbone สำหรับ anomaly detection.

🎯 Best for:
- ตรวจจับรอยแตก รอยขีด ยาบิ่น (defect detection)
- ใช้แค่ภาพยาดี train ไม่ต้องการ data ยาเสีย
- Intermediate layer features จับ texture/detail ได้ดีกว่า MobileNet
- Multi-image confirmation (3 ภาพ voting)

📊 Performance:
- Speed: ⭐⭐⭐ (ช้ากว่า MobileNet แต่แม่นกว่า)
- Texture: ⭐⭐⭐⭐⭐ (ดีเยี่ยม - layer2+layer3 จับ mid-level features)
- Crack/Scratch: ⭐⭐⭐⭐⭐ (ดีกว่า MobileNet มาก)
- Color: ⭐⭐⭐⭐ (Good)
- Shape: ⭐⭐⭐⭐⭐ (Best)

🔬 WideResNet50 vs MobileNet:
- WideResNet50 มี intermediate layers ที่กว้างกว่า (512+1024 channels)
- จับ texture variations ได้ดีกว่า → เห็นรอยแตกเล็กๆ
- Greedy coreset subsampling → memory bank ครอบคลุมมากกว่า random
"""
from pathlib import Path
from config.base import DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                     WIDERESNET PATCHCORE PARAMETERS
# =============================================================================

# Image preprocessing
IMG_SIZE = 256                  # Input image size (pixels)

# Patch extraction
GRID_SIZE = 10                  # 28×28 = 784 patches (จับ defect เล็กๆ ได้ดี)
                                # WideResNet feature map ใหญ่กว่า MobileNet
                                # ยิ่ง grid เยอะ ยิ่งเห็น defect เล็กๆ

# Memory bank
CORESET_RATIO = 0.20            # เพิ่มเป็น 0.20 (ครอบคลุม variation มากขึ้น)
USE_GREEDY_CORESET = True       # ใช้ greedy coreset (แม่นกว่า random)

# Anomaly scoring
K_NEAREST = 3                   # k=3 sensitive สำหรับ small defect

# Threshold
FALLBACK_THRESHOLD = 0.25       # sensitive มากขึ้น


# =============================================================================
#                     BACKBONE LAYER SELECTION
# =============================================================================

# WideResNet50 layers ที่จะใช้สกัด features
# layer2: mid-level (texture, edges) - 512 channels
# layer3: high-level (shape, structure) - 1024 channels
# รวม = 1536 channels (concatenate หลัง pool)
SELECTED_LAYERS = ["layer2", "layer3"]

# ถ้าต้องการ finer detail เพิ่ม layer1 ได้:
# SELECTED_LAYERS = ["layer1", "layer2", "layer3"]  # 256+512+1024 = 1792 channels


# =============================================================================
#                     SIFE (Spatial Information) SETTINGS
# =============================================================================

USE_SIFE = False                 # เปิดใช้ positional encoding
SIFE_DIM = 32                   # Dimension ของ spatial features
SIFE_ENCODING_TYPE = "sinusoidal"
SIFE_WEIGHT = 1.5               # เพิ่ม spatial signal

CNN_WEIGHT = 0.7                # ลด CNN dominance เพื่อให้ SIFE มีผลมากขึ้น

USE_CENTER_DISTANCE = True      # ระยะจากจุดกลาง
USE_LOCAL_GRADIENT = True       # gradient magnitude per patch


# =============================================================================
#                     LAPLACIAN VARIANCE (CRACK DETECTION)
# =============================================================================

USE_LAPLACIAN_VARIANCE = True   # Laplacian variance จับรอยแตก/ขีด
LAPLACIAN_WEIGHT = 1.5          # เพิ่ม weight สำหรับ crack detection


# =============================================================================
#                     EDGE ENHANCEMENT
# =============================================================================

USE_EDGE_ENHANCEMENT = True     # เน้นขอบ/รอยแตก
EDGE_WEIGHT = 2.0               # weight สูงขึ้น → sensitive to cracks


# =============================================================================
#                     MULTI-SCALE DETECTION
# =============================================================================

USE_MULTI_SCALE = True          # Multi-scale feature extraction
MULTI_SCALE_GRIDS = [14, 28, 42]  # Grid sizes: coarse → fine


# =============================================================================
#                     COLOR FEATURES
# =============================================================================

USE_COLOR_FEATURES = False      # ปิดไว้ ไม่จำเป็นสำหรับ defect detection
USE_HSV = False
COLOR_WEIGHT = 1.0


# =============================================================================
#                     SCORING WEIGHTS
# =============================================================================

SCORE_WEIGHT_MAX = 0.3          # Max score weight
SCORE_WEIGHT_TOP_K = 0.5        # Top-k mean weight (เน้นจับ defect patches)
SCORE_WEIGHT_PERCENTILE = 0.2   # Percentile weight
TOP_K_PERCENT = 0.05            # Top 5% patches


# =============================================================================
#                     MULTI-IMAGE CONFIRMATION
# =============================================================================

# ใช้กี่ภาพ confirm
CONFIRM_IMAGES = 3              # จำนวนภาพที่ใช้ vote
# Majority vote: ถ้า >= CONFIRM_THRESHOLD ภาพบอก ANOMALY → ยาเสีย
CONFIRM_THRESHOLD = 2           # 2 ใน 3 ภาพบอกเสีย = เสีย


# =============================================================================
#                     MODEL OUTPUT PATH
# =============================================================================

MODEL_OUTPUT_DIR = Path("./model/patchcore_wideresnet")


# =============================================================================
#                     TUNING GUIDE
# =============================================================================
"""
🔧 Parameter Tuning Guide for WideResNet50:

1. GRID_SIZE (Patch resolution):
   - 14-20: Fast, good for large defects
   - 20-28: Balanced, recommended
   - 28-42: Slow, better for tiny defects (cracks, scratches)

2. CORESET_RATIO (Memory bank size):
   - 0.10-0.15: Fast, may miss variations
   - 0.15-0.25: Balanced, recommended
   - 0.25-0.40: Thorough, slower (use with greedy coreset)

3. USE_GREEDY_CORESET:
   - True: Better coverage, slower build time
   - False: Random sampling, faster build time

4. K_NEAREST (Scoring sensitivity):
   - 1-3: Very sensitive, may false positive
   - 3-5: Balanced, recommended
   - 5-11: Conservative, may miss small anomalies

5. SELECTED_LAYERS:
   - ["layer2", "layer3"]: Best for defect detection (default)
   - ["layer1", "layer2", "layer3"]: More detail, slower
   - ["layer3"]: Fastest, misses fine detail

6. LAPLACIAN_WEIGHT:
   - 1.0: Standard
   - 1.5: เน้นรอยแตก (recommended)
   - 2.0+: Very sensitive to surface texture changes

📋 Preset Configurations:

Balanced Mode (Default):
    GRID_SIZE = 20
    K_NEAREST = 3
    USE_GREEDY_CORESET = True
    
Fast Mode:
    GRID_SIZE = 14
    K_NEAREST = 5
    USE_GREEDY_CORESET = False
    USE_MULTI_SCALE = False

Maximum Sensitivity (crack detection):
    GRID_SIZE = 42
    K_NEAREST = 1
    USE_GREEDY_CORESET = True
    LAPLACIAN_WEIGHT = 2.0
    EDGE_WEIGHT = 2.5
"""
