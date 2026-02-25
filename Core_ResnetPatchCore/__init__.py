"""
Core_ResnetPatchCore — Production PatchCore for Pill Inspection
==========================================================

Backbone : ResNet50  (pretrained, frozen)
Layers   : layer2 (512 ch) + layer3 (1024 ch)  → concat → 1536-dim patches
Scoring  : FAISS kNN (cosine similarity)
Detection: YOLOv12-seg  (.pt + .onnx)

Modules
-------
``segmentation``  — YOLOSegmentor  (detect + mask-crop pills)
``patchcore``     — feature extractor, memory bank, scorer
``pipeline``      — train / infer / visualizer

Quick start
-----------
::

    # training
    python run_train.py --model=resnet

    # realtime camera
    python run_realtime.py --model=resnet

    # folder prediction
    python run_predict.py --model=resnet --input=data_yolo/test
"""
from Core_ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor
from Core_ResnetPatchCore.patchcore.memory_bank import MemoryBank, CoresetSampler
from Core_ResnetPatchCore.patchcore.scorer import PatchCoreScorer
from Core_ResnetPatchCore.segmentation.yolo_infer import YOLOSegmentor, PillDetection
from Core_ResnetPatchCore.pipeline.infer import PillInspector, InspectorConfig
from Core_ResnetPatchCore.pipeline.train import TrainPipeline

__all__ = [
    # patchcore
    "ResNet50FeatureExtractor",
    "MemoryBank",
    "CoresetSampler",
    "PatchCoreScorer",
    # segmentation
    "YOLOSegmentor",
    "PillDetection",
    # pipeline
    "PillInspector",
    "InspectorConfig",
    "TrainPipeline",
]
