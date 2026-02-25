# MobilenetPatchCore/__init__.py
"""
MobileNetV3-based PatchCore for Anomaly Detection.

Original implementation using MobileNetV3-Large backbone.
Lighter and faster than DINOv2, but less accurate for color/texture.
"""

from .core_shared import PatchCore
from .core_train import PatchCoreTrainer
from .core_predict import PillInspector, InspectorConfig

__all__ = ["PatchCore", "PatchCoreTrainer", "PillInspector", "InspectorConfig"]
