"""
MobilenetSIFE - MobileNet + PatchCore + SIFE Package
=====================================================

PatchCore anomaly detection with MobileNetV3 backbone 
enhanced with SIFE (Spatial Information Feature Enhancement).

Usage:
    from MobilenetSIFE.core_shared import PatchCoreSIFE
    from MobilenetSIFE.core_train import PatchCoreSIFETrainer
    from MobilenetSIFE.core_predict import PillInspectorSIFE
"""

from .core_shared import PatchCoreSIFE
from .core_train import PatchCoreSIFETrainer
from .core_predict import PillInspectorSIFE, InspectorConfig

__all__ = [
    "PatchCoreSIFE",
    "PatchCoreSIFETrainer", 
    "PillInspectorSIFE",
    "InspectorConfig",
]
