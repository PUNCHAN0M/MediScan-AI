# DINOv2PatchCore/__init__.py
"""
DINOv2-based PatchCore for Anomaly Detection.

Uses DINOv2 (ViT) backbone instead of MobileNetV3 for better feature extraction,
especially for texture and color differentiation.
"""

from .core_dinov2 import DINOv2PatchCore
from .trainer_dinov2 import DINOv2PatchCoreTrainer
from .inspector_dinov2 import DINOv2PillInspector, DINOv2InspectorConfig

__all__ = ["DINOv2PatchCore", "DINOv2PatchCoreTrainer", "DINOv2PillInspector", "DINOv2InspectorConfig"]
