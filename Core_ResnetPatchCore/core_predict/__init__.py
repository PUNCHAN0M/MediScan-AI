"""Backward-compatibility shim — forwards to pipeline.infer."""
from Core_ResnetPatchCore.pipeline.infer import PillInspector, InspectorConfig

# Aliases so run_predict.py  `PillInspectorSIFE`  and old code work unchanged
PillInspectorSIFE = PillInspector
PillInspectorResNet = PillInspector

__all__ = ["PillInspector", "PillInspectorSIFE", "PillInspectorResNet",
           "InspectorConfig"]
