"""
DINOv2 PatchCore Configuration
==============================

การตั้งค่าสำหรับ DINOv2 (Vision Transformer) backbone.

🎯 Best for:
- Complex texture + shape anomalies
- High-quality offline inspection
- Research and development

📊 Performance:
- Speed: ⭐⭐⭐ (Slower, but powerful)
- Texture: ⭐⭐⭐⭐⭐ (Best)
- Color: ⭐⭐⭐⭐
- Shape: ⭐⭐⭐⭐⭐ (Best)
"""
from pathlib import Path
from config.base import DEFAULT_FALLBACK_THRESHOLD


# =============================================================================
#                         DINOV2 PATCHCORE PARAMETERS
# =============================================================================

# Image preprocessing
# NOTE: DINOv2 requires image size divisible by patch_size (14)
IMG_SIZE = 252                  # 252 = 14 × 18 (divisible by 14)

# Patch extraction
GRID_SIZE = 18                  # 18×18 = 324 patches
                                # DINOv2 naturally uses 14×14 patch tokens

# Memory bank
CORESET_RATIO = 0.18            # Higher ratio for ViT features

# Anomaly scoring
K_NEAREST = 19                  # More neighbors for stable scoring

# Threshold
FALLBACK_THRESHOLD = 0.40       # DINOv2 typically has different score range


# =============================================================================
#                           DINOV2 BACKBONE SETTINGS
# =============================================================================

# Model size selection
# "small"  → vit_small_patch14 (fast, good quality)
# "base"   → vit_base_patch14 (balanced, recommended)
# "large"  → vit_large_patch14 (best quality, slow)
# "giant"  → vit_giant_patch14 (research only, very slow)
BACKBONE_SIZE = "base"

# Multi-scale feature extraction
# Extracts features from multiple transformer layers
MULTI_SCALE = True              # Recommended for better results


# =============================================================================
#                              MODEL OUTPUT PATH
# =============================================================================

MODEL_OUTPUT_DIR = Path("./model/patchcore_dinov2")


# =============================================================================
#                              TUNING GUIDE
# =============================================================================
"""
🔧 Parameter Tuning Guide for DINOv2:

1. BACKBONE_SIZE (Model complexity):
   - "small": ~22M params, fastest
   - "base": ~86M params, balanced (recommended)
   - "large": ~307M params, best quality
   - "giant": ~1.1B params, research only
   
2. IMG_SIZE (Must be divisible by 14):
   - 224: 16×16 = 256 tokens (fast)
   - 252: 18×18 = 324 tokens (balanced)
   - 280: 20×20 = 400 tokens (detailed)
   - 336: 24×24 = 576 tokens (high-res)

3. MULTI_SCALE:
   - True: Better features, slower (recommended)
   - False: Faster, single layer only

4. CORESET_RATIO:
   - DINOv2 features are more discriminative
   - Higher ratio (0.15-0.25) recommended

📋 Preset Configurations:

Fast Mode (Real-time capable):
    BACKBONE_SIZE = "small"
    IMG_SIZE = 224
    GRID_SIZE = 16
    CORESET_RATIO = 0.12
    MULTI_SCALE = False
    
Balanced Mode (Default):
    BACKBONE_SIZE = "base"
    IMG_SIZE = 252
    GRID_SIZE = 18
    CORESET_RATIO = 0.18
    MULTI_SCALE = True
    
High-Quality Mode (Offline):
    BACKBONE_SIZE = "large"
    IMG_SIZE = 280
    GRID_SIZE = 20
    CORESET_RATIO = 0.22
    MULTI_SCALE = True

Research Mode (Best possible):
    BACKBONE_SIZE = "large"
    IMG_SIZE = 336
    GRID_SIZE = 24
    CORESET_RATIO = 0.25
    MULTI_SCALE = True

🔬 DINOv2 Advantages:
- Self-supervised pretraining (no labels needed)
- Excellent semantic understanding
- Works well with limited training data
- Better generalization to unseen defects
"""
