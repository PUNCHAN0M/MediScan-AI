"""
WideResnetAnomalyCore - WideResNet50 PatchCore Package
=======================================================

PatchCore anomaly detection with WideResNet50 backbone
for pill defect detection (cracks, scratches, chips).

🔥 Key advantages over MobilenetSIFE:
- WideResNet50 captures finer texture/detail features
- Greedy coreset subsampling for better memory bank coverage
- Laplacian variance for crack/scratch detection
- Multi-image confirmation (3-image voting)

Usage:
    from WideResnetAnomalyCore.core_shared import PatchCoreWideResNet
    from WideResnetAnomalyCore.core_train import PatchCoreWideResNetTrainer
    from WideResnetAnomalyCore.core_predict import PillInspectorWideResNet, InspectorConfig
"""

from .core_shared import PatchCoreWideResNet
from .core_train import PatchCoreWideResNetTrainer
from .core_predict import PillInspectorWideResNet, InspectorConfig

__all__ = [
    "PatchCoreWideResNet",
    "PatchCoreWideResNetTrainer",
    "PillInspectorWideResNet",
    "InspectorConfig",
]
