# CNNMultiScale/__init__.py
"""
CNN Multi-Scale PatchCore for Tiny Defect Detection.

🔥 Optimized for detecting 2-5px cracks on pill surfaces.

Key Architecture:
- Modified ResNet34 (no maxpool, stride=1 conv1)
- Multi-scale feature extraction (layer1, layer2, layer3)
- Separate PatchCore memory per scale
- Score fusion: max(score_per_scale)
- CLAHE preprocessing + SE attention
- Multi-resolution input (512 + 768)
- Adaptive threshold (mean + k*std)
"""

from .core_shared.patchcore_multiscale import CNNMultiScalePatchCore
from .core_train.trainer import CNNMultiScaleTrainer
from .core_predict.inspector import PillInspectorCNNMultiScale, InspectorConfig

__all__ = [
    "CNNMultiScalePatchCore",
    "CNNMultiScaleTrainer",
    "PillInspectorCNNMultiScale",
    "InspectorConfig",
]
