"""
Optimized Inference Pipeline (Parent-Level Memory Only)
========================================================
• ใช้ 1 memory bank ต่อ 1 parent class
• ไม่มี subclass loop
• Faster FAISS scoring (2–5x)
"""

from __future__ import annotations
import cv2
import numpy as np
import faiss
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Core_ResnetPatchCore.segmentation.yolo_tracking import YOLOTracking
from Core_ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor
from Core_ResnetPatchCore.patchcore.memory_bank import MemoryBank
from Core_ResnetPatchCore.patchcore.scorer import PatchCoreScorer
from Core_ResnetPatchCore.pipeline.visualizer import draw_pill_results, draw_summary
from config.base import (
    SEGMENTATION_MODEL_PATH,
    SEGMENTATION_CONF,
    SEGMENTATION_IOU,
    SEGMENTATION_PAD,
    MODEL_OUTPUT_DIR,
    IMAGE_SIZE,
    TRACK_MAX_DISTANCE,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_AGE,
    FRAMES_BEFORE_SUMMARY,
    DEVICE as BASE_DEVICE,
)
from config.resnet import (
    IMG_SIZE,
    GRID_SIZE,
    K_NEAREST,
    SCORE_METHOD,
    THRESHOLD_MULTIPLIER,
    USE_COLOR_FEATURES,
    USE_HSV,
    COLOR_WEIGHT,
    BACKBONE,
)


# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════

@dataclass
class InspectorConfig:
    compare_classes: List[str] = field(default_factory=list)
    model_dir: Path = field(default_factory=lambda: Path(MODEL_OUTPUT_DIR))
    yolo_model_path: str = str(SEGMENTATION_MODEL_PATH)

    img_size: int = 640
    conf: float = SEGMENTATION_CONF
    iou: float = SEGMENTATION_IOU
    pad: int = SEGMENTATION_PAD

    model_size: int = IMG_SIZE
    grid_size: int = GRID_SIZE

    use_color_features: bool = USE_COLOR_FEATURES
    use_hsv: bool = USE_HSV
    color_weight: float = COLOR_WEIGHT

    k_nearest: int = K_NEAREST
    score_method: str = SCORE_METHOD
    threshold_multiplier: float = THRESHOLD_MULTIPLIER

    backbone_path: Optional[str] = (
        BACKBONE if BACKBONE and str(BACKBONE).endswith(".pth") else None
    )

    crop_size: int = IMAGE_SIZE
    bg_value: int = 0

    track_max_distance: float = TRACK_MAX_DISTANCE
    track_iou_threshold: float = TRACK_IOU_THRESHOLD
    track_max_age: int = TRACK_MAX_AGE

    frames_before_summary: int = FRAMES_BEFORE_SUMMARY
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = BASE_DEVICE
        self.model_dir = Path(self.model_dir)


# ═════════════════════════════════════════════════════════════
# MAIN INSPECTOR
# ═════════════════════════════════════════════════════════════

class PillInspector:

    def __init__(self, config: Optional[InspectorConfig] = None):
        self.config = config or InspectorConfig()
        self._init_components()
        self._init_state()

    # ──────────────────────────────
    # Init Components
    # ──────────────────────────────
    def _init_components(self):

        cfg = self.config

        self._segmentor = YOLOTracking(
            model_path=cfg.yolo_model_path,
            img_size=cfg.img_size,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            target_size=cfg.crop_size,
            pad=cfg.pad,
            bg_value=cfg.bg_value,
            enable_tracking=True,
            track_max_distance=cfg.track_max_distance,
            track_iou_threshold=cfg.track_iou_threshold,
            track_max_age=cfg.track_max_age,
        )

        self._extractor = ResNet50FeatureExtractor(
            img_size=cfg.model_size,
            grid_size=cfg.grid_size,
            device=cfg.device,
            use_color_features=cfg.use_color_features,
            use_hsv=cfg.use_hsv,
            color_weight=cfg.color_weight,
            backbone_path=cfg.backbone_path,
        )

        self._scorer = PatchCoreScorer(k_nearest=cfg.k_nearest, assume_normalized=False)

        self._indices: Dict[str, faiss.Index] = {}
        self._thresholds: Dict[str, float] = {}

    def _init_state(self):
        self._votes: Dict[int, Dict[str, Any]] = {}
        self._last_frame: Optional[np.ndarray] = None
        self._last_results: List[Dict[str, Any]] = []
        self._last_crops: Dict[int, np.ndarray] = {}

    # ──────────────────────────────
    # Memory Loading (Parent Only)
    # ──────────────────────────────
    def _ensure_index(self, class_name: str) -> bool:

        if class_name in self._indices:
            return True

        pth = self.config.model_dir / f"{class_name}.pth"
        if not pth.exists():
            print(f"[Missing Model] {pth}")
            return False

        try:
            bank, meta = MemoryBank.load(pth)
            idx = self._scorer.build_index(bank)

            self._indices[class_name] = idx
            raw_thr = float(meta.get("threshold", 0.5))
            self._thresholds[class_name] = raw_thr * self.config.threshold_multiplier

            print(
                f"[Loaded] {class_name} | "
                f"dim={bank.shape[1]} | "
                f"thr={raw_thr:.4f} x {self.config.threshold_multiplier} "
                f"= {self._thresholds[class_name]:.4f}"
            )
            return True

        except Exception as e:
            print(f"[Load Error] {class_name}: {e}")
            return False

    # ──────────────────────────────
    # Classification
    # ──────────────────────────────
    def _classify_pill(
        self,
        feats: np.ndarray,
        classes: List[str],
    ) -> Tuple[str, Dict[str, float], List[str]]:

        if feats is None or feats.shape[0] == 0:
            return "NORMAL", {}, []

        scores: Dict[str, float] = {}
        normal_from: List[str] = []

        for class_name in classes:

            if not self._ensure_index(class_name):
                continue

            score = self._scorer.score_pill(
                feats,
                self._indices[class_name],
                method=self.config.score_method,
            )

            scores[class_name] = score

            if score <= self._thresholds[class_name]:
                normal_from.append(class_name)

        status = "NORMAL" if normal_from else "ANOMALY"
        return status, scores, normal_from

    # ═══════════════════════════════
    # REALTIME
    # ═══════════════════════════════
    def classify_anomaly(
        self,
        frame: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:

        classes = list(class_names or self.config.compare_classes)
        self._last_frame = frame.copy()

        crops, infos = self._segmentor.detect_and_crop(frame)

        batch_feats = self._extractor.extract_batch(crops) if crops else []

        frame_results = []
        frame_crops = {}

        for i, (crop, info) in enumerate(zip(crops, infos)):

            feats = batch_feats[i]
            status, scores, normal_from = self._classify_pill(feats, classes)

            result = {
                "id": int(info.get("track_id", -1)),
                "bbox": info["bbox"],
                "conf": info["conf"],
                "status": status,
                "class_scores": scores,
                "normal_from": normal_from,
            }

            frame_results.append(result)

            tid = result["id"]
            if tid >= 0:
                frame_crops[tid] = crop

        self._last_results = frame_results
        self._last_crops = frame_crops

        return draw_pill_results(self._last_frame, frame_results)

    # ═══════════════════════════════
    # SUMMARY
    # ═══════════════════════════════
    def summarize(self) -> Dict[str, Any]:

        if self._last_frame is None:
            return {"image": None, "count": 0, "good_count": 0, "bad_count": 0}

        items = []

        for r in self._last_results:
            items.append({
                "id": r["id"],
                "bbox": r["bbox"],
                "status": r["status"],
            })

        vis = draw_summary(self._last_frame, items)

        good = sum(1 for it in items if it["status"] == "NORMAL")
        bad = sum(1 for it in items if it["status"] != "NORMAL")

        return {
            "image": vis,
            "count": len(items),
            "good_count": good,
            "bad_count": bad,
        }

    # ──────────────────────────────
    # Reset
    # ──────────────────────────────
    def reset(self):
        self._init_state()
        self._segmentor.reset_tracking()

    # ──────────────────────────────
    # Properties
    # ──────────────────────────────
    @property
    def last_crops(self):
        return dict(self._last_crops)

    @property
    def detector(self):
        return self._segmentor