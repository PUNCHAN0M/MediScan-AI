# Core_ResnetPatchCore\core_predict\inspector.py
"""Backward-compatibility shim — ``run_predict.py`` imports from here."""
from Core_ResnetPatchCore.pipeline.infer import PillInspector, InspectorConfig

# run_predict.py looks for PillInspectorSIFE
PillInspectorSIFE = PillInspector
PillInspectorResNet = PillInspector

__all__ = ["PillInspector", "PillInspectorSIFE", "PillInspectorResNet",
           "InspectorConfig"]
