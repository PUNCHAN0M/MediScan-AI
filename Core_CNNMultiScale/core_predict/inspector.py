# CNNMultiScale/core_predict/inspector.py
"""
PillInspectorCNNMultiScale - Main inspection class for tiny defect detection.

🔥 Optimized for 2-5px crack detection using:
- Modified ResNet34 (no maxpool, stride=1 conv1)
- Separate PatchCore memory per scale
- Score fusion: max(score_per_scale)
- CLAHE preprocessing
- Multi-resolution input
- Adaptive thresholding

Single Responsibility:
- Detect pills using YOLO
- Classify anomalies using CNN Multi-Scale PatchCore
- Accumulate votes across frames
- Summarize with majority vote
"""
from __future__ import annotations

import cv2
import numpy as np
import faiss
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Any

from CNNMultiScale.core_shared.patchcore_multiscale import CNNMultiScalePatchCore
from CNNMultiScale.core_predict.visualizer import (
    draw_pill_results,
    draw_summary,
    draw_heatmap_overlay,
)

# Import YOLO detector from MobilenetPatchCore (shared)
from MobilenetPatchCore.core_predict.yolo_detector import PillYOLODetector


# =========================================================
# CONFIGURATION
# =========================================================
@dataclass
class InspectorConfig:
    """Configuration for PillInspectorCNNMultiScale."""

    compare_classes: List[str] = field(default_factory=lambda: [
        "vitaminc_front", "vitaminc_back",
    ])

    model_dir: Path = field(default_factory=lambda: Path("./model/patchcore_cnn_multiscale"))
    yolo_model_path: str = "model/yolo12-seg.pt"
    yolo_det_model_path: Optional[str] = None
    device: Optional[str] = None

    # Input
    model_size: int = 512
    model_size_secondary: int = 768
    enable_multi_resolution: bool = True

    # Patch extraction
    grid_size: int = 32
    k_nearest: int = 9

    # Backbone
    backbone: str = "resnet34"
    remove_maxpool: bool = True
    stride1_conv1: bool = True
    use_dilated_layer3: bool = True
    selected_layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])

    # Fusion
    score_fusion: str = "max"
    scale_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    separate_memory_per_scale: bool = True

    # Preprocessing
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    use_laplacian_boost: bool = False
    laplacian_weight: float = 0.3

    # Attention
    use_se_attention: bool = True
    se_reduction: int = 16

    # Color
    use_color_features: bool = True
    use_hsv: bool = True
    color_weight: float = 0.5

    # YOLO
    img_size: int = 512
    conf: float = 0.5
    iou: float = 0.6
    pad: int = 5
    merge_dist_px: int = 60

    # Tracking
    track_max_distance: float = 80.0
    track_iou_threshold: float = 0.80
    track_max_age: int = 10

    crop_size: int = 512  # Larger crop for better detail

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(self.model_dir)


# =========================================================
# UTILITY
# =========================================================
def make_square_crop(crop: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Pad crop to square and resize."""
    if crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    max_side = max(h, w)
    square = np.zeros((max_side, max_side, 3), dtype=np.uint8)

    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = crop

    if max_side != target_size:
        square = cv2.resize(
            square, (target_size, target_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

    return square


# =========================================================
# INSPECTOR
# =========================================================
class PillInspectorCNNMultiScale:
    """
    Pill anomaly inspector using CNN Multi-Scale PatchCore.
    
    🔥 Designed for detecting tiny defects (2-5px cracks):
    - Modified ResNet34 preserves 4x more spatial resolution
    - Separate memory bank per scale → defect in ANY scale triggers alarm
    - CLAHE boosts micro-crack contrast
    - Multi-resolution captures defects at multiple sizes
    - Adaptive threshold handles low-variance surfaces
    
    Usage:
        inspector = PillInspectorCNNMultiScale()
        for frame in frames:
            preview = inspector.classify_anomaly(frame)
        result = inspector.summarize()
    """

    def __init__(self, config: Optional[InspectorConfig] = None):
        self.config = config or InspectorConfig()
        self._init_models()
        self._init_state()

    def _init_models(self) -> None:
        cfg = self.config

        # CNN Multi-Scale PatchCore
        self._patchcore = CNNMultiScalePatchCore(
            model_size=cfg.model_size,
            model_size_secondary=cfg.model_size_secondary,
            enable_multi_resolution=cfg.enable_multi_resolution,
            grid_size=cfg.grid_size,
            k_nearest=cfg.k_nearest,
            device=cfg.device,
            backbone=cfg.backbone,
            remove_maxpool=cfg.remove_maxpool,
            stride1_conv1=cfg.stride1_conv1,
            use_dilated_layer3=cfg.use_dilated_layer3,
            selected_layers=cfg.selected_layers,
            score_fusion=cfg.score_fusion,
            scale_weights=cfg.scale_weights,
            separate_memory_per_scale=cfg.separate_memory_per_scale,
            use_clahe=cfg.use_clahe,
            clahe_clip_limit=cfg.clahe_clip_limit,
            clahe_tile_size=cfg.clahe_tile_size,
            use_laplacian_boost=cfg.use_laplacian_boost,
            laplacian_weight=cfg.laplacian_weight,
            use_se_attention=cfg.use_se_attention,
            se_reduction=cfg.se_reduction,
            use_color_features=cfg.use_color_features,
            use_hsv=cfg.use_hsv,
            color_weight=cfg.color_weight,
        )

        # Model storage
        self._model_dir = cfg.model_dir
        self._model_data: Dict[str, Any] = {}
        self._subclass_map: Dict[str, List[str]] = {}

        # Per-subclass, per-scale FAISS indices
        # Structure: {subclass_name: {layer_name: faiss.Index}}
        self._scale_indices: Dict[str, Dict[str, faiss.Index]] = {}
        self._thresholds: Dict[str, float] = {}

        # YOLO detector
        self._detector = PillYOLODetector(
            seg_model_path=cfg.yolo_model_path,
            det_model_path=cfg.yolo_det_model_path,
            img_size=cfg.img_size,
            conf=cfg.conf,
            iou=cfg.iou,
            pad=cfg.pad,
            merge_dist_px=cfg.merge_dist_px,
            enable_tracking=True,
            track_max_distance=cfg.track_max_distance,
            track_iou_threshold=cfg.track_iou_threshold,
            track_max_age=cfg.track_max_age,
            start_with_detection=cfg.yolo_det_model_path is not None,
        )

    def _init_state(self) -> None:
        self._votes: Dict[int, Dict[str, Any]] = {}
        self._last_frame: Optional[np.ndarray] = None
        self._last_results: List[Dict[str, Any]] = []
        self._last_crops: Dict[int, np.ndarray] = {}
        self._first_detection_done: bool = False

    def _get_subclasses(self, parent_class: str) -> List[str]:
        """Get all subclasses for a parent class (lazy load)."""
        if parent_class in self._subclass_map:
            return self._subclass_map[parent_class]

        parent_dir = self._model_dir / parent_class
        if parent_dir.is_dir():
            subclasses = [f.stem for f in parent_dir.glob("*.pth")]
            self._subclass_map[parent_class] = subclasses
        else:
            pth_path = self._model_dir / f"{parent_class}.pth"
            if pth_path.exists():
                self._subclass_map[parent_class] = [parent_class]
            else:
                self._subclass_map[parent_class] = []

        return self._subclass_map[parent_class]

    def _ensure_index(self, subclass_name: str, parent_class: Optional[str] = None) -> bool:
        """Lazy-load per-scale FAISS indices for a subclass."""
        if subclass_name in self._scale_indices:
            return True

        # Find pth file
        if subclass_name not in self._model_data:
            if parent_class:
                pth_path = self._model_dir / parent_class / f"{subclass_name}.pth"
            else:
                parts = subclass_name.rsplit("_", 1)
                if len(parts) == 2:
                    pth_path = self._model_dir / parts[0] / f"{subclass_name}.pth"
                else:
                    pth_path = self._model_dir / f"{subclass_name}.pth"

            if not pth_path.exists():
                pth_path = self._model_dir / f"{subclass_name}.pth"

            if not pth_path.exists():
                print(f"[Warning] Model not found: {subclass_name}")
                return False

            try:
                self._model_data[subclass_name] = torch.load(
                    str(pth_path),
                    map_location=self.config.device,
                    weights_only=False,
                )
                print(f"[Loaded] {subclass_name} from {pth_path}")
            except Exception as e:
                print(f"[Error] Failed to load {pth_path}: {e}")
                return False

        data = self._model_data[subclass_name]

        # Build per-scale FAISS indices
        scale_banks = data.get("scale_memory_banks", {})
        if not scale_banks:
            # Fallback: try single memory_bank (compatibility)
            single_bank = data.get("memory_bank")
            if single_bank is not None:
                bank_np = single_bank.cpu().numpy().astype(np.float32)
                index = faiss.IndexFlatIP(bank_np.shape[1])
                faiss.normalize_L2(bank_np)
                index.add(bank_np)
                # Use same index for all scales (fallback mode)
                self._scale_indices[subclass_name] = {
                    layer: index for layer in self._patchcore.selected_layers
                }
                self._thresholds[subclass_name] = float(data.get("threshold", 0.50))
                return True
            return False

        indices = {}
        for layer_name, bank_tensor in scale_banks.items():
            bank_np = bank_tensor.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(bank_np)
            index = faiss.IndexFlatIP(bank_np.shape[1])
            index.add(bank_np)
            indices[layer_name] = index

        self._scale_indices[subclass_name] = indices
        self._thresholds[subclass_name] = float(data.get("threshold", 0.50))
        return True

    def _classify_single(
        self, crop: np.ndarray, classes: Sequence[str]
    ) -> Tuple[str, Dict[str, float], List[str], Dict[str, Dict[str, float]]]:
        """
        Classify a single crop using multi-scale PatchCore.
        
        Returns:
            (status, class_scores, normal_from, per_scale_details)
        """
        features = self._patchcore.extract_from_numpy(
            crop, multi_resolution=self.config.enable_multi_resolution
        )
        if features is None:
            return "NORMAL", {}, [], {}

        scores: Dict[str, float] = {}
        normal_from: List[str] = []
        per_scale_details: Dict[str, Dict[str, float]] = {}

        for parent_cls in classes:
            subclasses = self._get_subclasses(parent_cls)
            if not subclasses:
                if self._ensure_index(parent_cls):
                    subclasses = [parent_cls]
                else:
                    continue

            for subcls in subclasses:
                if not self._ensure_index(subcls, parent_cls):
                    continue

                # Get per-scale scores
                per_scale = self._patchcore.get_per_scale_anomaly_scores(
                    features, self._scale_indices[subcls]
                )
                fused_score = self._patchcore.fuse_scores(per_scale)

                scores[subcls] = fused_score
                per_scale_details[subcls] = per_scale

                if fused_score <= self._thresholds[subcls]:
                    normal_from.append(subcls)

        status = "NORMAL" if normal_from else "ANOMALY"
        return status, scores, normal_from, per_scale_details

    def reset(self) -> None:
        """Clear all votes and tracking."""
        self._init_state()
        self._detector.reset_tracking()

    def classify_anomaly(
        self,
        frame: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Process one frame and accumulate votes.
        
        Returns: Preview image with bboxes and per-scale scores
        """
        classes = list(class_names or self.config.compare_classes)
        self._last_frame = frame.copy()

        if not self._first_detection_done:
            print(f"[MODE] Using {self._detector.current_mode} mode for initial crop")
        elif self._first_detection_done and self._detector.use_detection and self._detector.has_separate_det_model:
            self._detector.switch_to_segmentation()

        crops, infos = self._detector.detect_and_crop(frame)

        if crops and not self._first_detection_done:
            self._first_detection_done = True
            print("[MODE] First detection completed, will switch to SEGMENTATION on next frame")

        frame_results: List[Dict[str, Any]] = []
        frame_crops: Dict[int, np.ndarray] = {}

        for crop, info in zip(crops, infos):
            tid = int(info.get("track_id", -1))
            conf = float(info.get("conf", 0.0))
            bbox = tuple(map(int, info.get("padded_region", (0, 0, 0, 0))))

            square = make_square_crop(crop, self.config.crop_size)

            status, scores, normal_from, per_scale = self._classify_single(square, classes)

            if tid >= 0:
                frame_crops[tid] = square
            frame_results.append({
                "id": tid,
                "bbox": bbox,
                "conf": conf,
                "status": status,
                "class_scores": scores,
                "normal_from": normal_from,
                "per_scale_scores": per_scale,
            })

        # Dedup: keep highest confidence per track_id
        best: Dict[int, Dict[str, Any]] = {}
        for r in frame_results:
            tid = r["id"]
            if tid < 0:
                continue
            if tid not in best or r["conf"] > best[tid]["conf"]:
                best[tid] = r

        # Accumulate votes
        for tid, r in best.items():
            if tid not in self._votes:
                self._votes[tid] = {
                    "bbox": r["bbox"],
                    "NORMAL": 0,
                    "ANOMALY": 0,
                }
            self._votes[tid]["bbox"] = r["bbox"]
            self._votes[tid][r["status"]] += 1

        self._last_results = list(best.values())
        self._last_crops = {tid: frame_crops[tid] for tid in best if tid in frame_crops}

        return draw_pill_results(self._last_frame, self._last_results)

    def summarize(self) -> Dict[str, Any]:
        """
        Compute final status using majority vote.
        
        Returns: {"image": np.ndarray, "good": int, "bad": int, "items": [...]}
        """
        if self._last_frame is None:
            return {"image": None, "good": 0, "bad": 0, "items": []}

        items: List[Dict[str, Any]] = []
        good, bad = 0, 0

        for tid in sorted(self._votes.keys()):
            v = self._votes[tid]
            normal_n = v.get("NORMAL", 0)
            anomaly_n = v.get("ANOMALY", 0)
            status = "NORMAL" if normal_n > anomaly_n else "ANOMALY"

            if status == "NORMAL":
                good += 1
            else:
                bad += 1

            items.append({
                "id": tid,
                "bbox": v["bbox"],
                "status": status,
                "votes": {"NORMAL": normal_n, "ANOMALY": anomaly_n},
            })

        vis = draw_summary(self._last_frame, items)

        return {
            "image": vis,
            "good": good,
            "bad": bad,
            "total": good + bad,
            "items": items,
        }

    @property
    def last_crops(self) -> Dict[int, np.ndarray]:
        return dict(self._last_crops)

    @property
    def anomaly_counts(self) -> Dict[int, int]:
        return {tid: v.get("ANOMALY", 0) for tid, v in self._votes.items()}

    @property
    def detector(self) -> PillYOLODetector:
        return self._detector
