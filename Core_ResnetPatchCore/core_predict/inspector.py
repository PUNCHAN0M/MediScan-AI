"""Backward-compatibility shim — ``run_predict.py`` imports from here."""
from ResnetPatchCore.pipeline.infer import PillInspector, InspectorConfig

# run_predict.py looks for PillInspectorSIFE
PillInspectorSIFE = PillInspector
PillInspectorResNet = PillInspector

__all__ = ["PillInspector", "PillInspectorSIFE", "PillInspectorResNet",
           "InspectorConfig"]
