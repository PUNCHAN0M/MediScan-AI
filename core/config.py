# core/config.py
"""
Unified Configuration for MediScan-AI
======================================
All settings in one place. No model logic. No I/O.
Layers above import from here — never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


# ─────────────────────────────────────────────────────────
#  Base config (frozen — immutable after creation)
# ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PathConfig:
    """All filesystem paths."""
    data_root: Path            = Path("./data_train_defection")
    bad_dir: Path              = Path("./data_bad")
    save_dir: Path             = Path("./data/inspected")
    model_output_dir: Path     = Path("./weights/patchcore_resnet")
    segmentation_model: str    = "weights/detection/pill-detection-best-2.onnx"
    backbone: str              = "weights/backbone/resnet_last.pth"


@dataclass(frozen=True)
class ImageConfig:
    """Image processing constants."""
    image_size: int            = 256
    image_exts: frozenset      = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass(frozen=True)
class YOLOConfig:
    """YOLO detection / segmentation settings."""
    img_size: int              = 1280
    conf: float                = 0.25
    iou: float                 = 0.6
    pad: int                   = 5


@dataclass(frozen=True)
class TrackingConfig:
    """Centroid + IoU tracker hyper-parameters."""
    max_distance: float        = 80.0
    iou_threshold: float       = 0.80
    max_age: int               = 10


@dataclass(frozen=True)
class BackboneConfig:
    """ResNet50 PatchCore backbone settings."""
    layers: tuple              = ("layer1", "layer2", "layer3")
    layer_channels: dict       = field(default_factory=lambda: {"layer1": 256, "layer2": 512, "layer3": 1024})
    grid_size: int             = 16
    coreset_ratio: float       = 0.25
    coreset_min_keep: int      = 1_000
    coreset_max_keep: int      = 20_000
    k_nearest: int             = 3
    score_method: str          = "max"
    threshold_multiplier: float = 1.0
    fallback_threshold: float  = 0.50
    use_color_features: bool   = True
    use_hsv: bool              = True
    color_weight: float        = 1.0
    n_finetune_steps: int      = 601


@dataclass(frozen=True)
class CameraConfig:
    """Camera / realtime settings."""
    camera_index: int          = 0
    ema_alpha: float           = 0.3
    window_name: str           = "Pill Inspector"


@dataclass(frozen=True)
class TrainConfig:
    """Training global settings."""
    seed: int                  = 42
    batch_size: int            = 32
    selected_classes: tuple    = ("Androxsil",)


# ─────────────────────────────────────────────────────────
#  Master config (combine all)
# ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    """
    Master configuration. Create once, inject everywhere.

    Usage::

        cfg = Config()
        cfg = Config(train=TrainConfig(seed=123))
    """
    paths: PathConfig          = field(default_factory=PathConfig)
    image: ImageConfig         = field(default_factory=ImageConfig)
    yolo: YOLOConfig           = field(default_factory=YOLOConfig)
    tracking: TrackingConfig   = field(default_factory=TrackingConfig)
    backbone: BackboneConfig   = field(default_factory=BackboneConfig)
    camera: CameraConfig       = field(default_factory=CameraConfig)
    train: TrainConfig         = field(default_factory=TrainConfig)
