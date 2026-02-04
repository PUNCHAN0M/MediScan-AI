# ResnetPatchCore/__init__.py
"""
ResNet18-based PatchCore for Anomaly Detection.

ResNet18 advantages:
- Better color awareness than MobileNet/DINOv2
- Shallower features preserve color information
- Fast and lightweight
- Combined with explicit color features for pill inspection
"""

from .core_shared import ResNetPatchCore
from .core_train import ResNetPatchCoreTrainer
from .core_predict import PillInspectorResNet, InspectorConfig

__all__ = ["ResNetPatchCore", "ResNetPatchCoreTrainer", "PillInspectorResNet", "InspectorConfig"]
