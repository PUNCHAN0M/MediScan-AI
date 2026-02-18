"""
mobile_sife_cuda — MobileNet + PatchCore + SIFE (CUDA-Optimized)
================================================================

High-performance anomaly detection with:
- CUDA acceleration (AMP, GPU tensors, batched inference)
- ThreadPoolExecutor for parallel I/O
- FAISS GPU index support
- MobileNetV3 backbone + SIFE spatial encoding

Usage:
    from mobile_sife_cuda import PatchCoreSIFE, PatchCoreSIFETrainer, PillInspectorSIFE
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
