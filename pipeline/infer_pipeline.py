# pipeline/infer_pipeline.py
"""
Inference Pipeline — realtime pill inspection.
================================================
Layer 4 — the ONLY place with frame loops / state.

Performance notes:
    - Eager pre-loading of all class indices at init
    - Shared GPU resources across all FAISS indices
    - Early-exit subclass scoring (stop once NORMAL found)
    - Deferred frame copy (only when needed for drawing)
"""
from __future__ import annotations

import numpy as np
import faiss
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.config import Config
from detection.yolo_detector import YOLODetector
from modules.feature_extractor import ResNet50FeatureExtractor
from modules.memory_bank import MemoryBank
from modules.scorer import PatchCoreScorer
from visualization.visualizer import draw_pill_results, draw_summary


# ─────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────
@dataclass
class InspectorConfig:
    """All knobs for PillInspector in one place."""
    compare_classes: List[str]       = field(default_factory=list)
    model_dir: Path                  = field(default_factory=lambda: Path("./weights/patchcore_resnet"))
    yolo_model_path: str             = "weights/detection/pill-detection-best-2.onnx"

    # YOLO
    img_size: int                    = 640
    conf: float                      = 0.5
    iou: float                       = 0.6
    pad: int                         = 0

    # backbone
    model_size: int                  = 256
    grid_size: int                   = 16
    use_color_features: bool         = True
    use_hsv: bool                    = True
    color_weight: float              = 1.0
    backbone_path: Optional[str]     = None

    # scorer
    k_nearest: int                   = 3
    score_method: str                = "max"
    threshold_multiplier: float      = 1.0

    # crop
    crop_size: int                   = 256
    bg_value: int                    = 0

    # tracking
    track_max_distance: float        = 80.0
    track_iou_threshold: float       = 0.80
    track_max_age: int               = 10

    # pipeline
    ema_alpha: float                 = 0.3
    device: Optional[str]            = None

    # performance
    use_half: bool                   = False  # FP32 for accuracy (match Old Version)
    early_exit_normal: bool          = True   # stop scoring once NORMAL found

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)

    @classmethod
    def from_config(cls, cfg: Config, compare_classes: Optional[List[str]] = None) -> "InspectorConfig":
        """Build InspectorConfig from the unified Config object."""
        backbone_path = None
        if cfg.paths.backbone and str(cfg.paths.backbone).endswith(".pth"):
            backbone_path = str(cfg.paths.backbone)

        return cls(
            compare_classes=compare_classes or [],
            model_dir=cfg.paths.model_output_dir,
            yolo_model_path=str(cfg.paths.segmentation_model),
            img_size=cfg.yolo.img_size,
            conf=cfg.yolo.conf,
            iou=cfg.yolo.iou,
            pad=cfg.yolo.pad,
            model_size=cfg.image.image_size,
            grid_size=cfg.backbone.grid_size,
            use_color_features=cfg.backbone.use_color_features,
            use_hsv=cfg.backbone.use_hsv,
            color_weight=cfg.backbone.color_weight,
            backbone_path=backbone_path,
            k_nearest=cfg.backbone.k_nearest,
            score_method=cfg.backbone.score_method,
            threshold_multiplier=cfg.backbone.threshold_multiplier,
            crop_size=cfg.image.image_size,
            track_max_distance=cfg.tracking.max_distance,
            track_iou_threshold=cfg.tracking.iou_threshold,
            track_max_age=cfg.tracking.max_age,
            ema_alpha=cfg.camera.ema_alpha,
        )

    @classmethod
    def from_settings(cls, settings: dict, compare_classes: Optional[List[str]] = None) -> "InspectorConfig":
        """
        Build InspectorConfig from app_settings.json dict.

        Mirrors ``from_config`` but reads the key names used by
        ``SettingsManager`` so the desktop app produces **identical**
        results to ``main.py predict``.
        """
        backbone_path = settings.get("backbone_model") or None
        if backbone_path and not str(backbone_path).endswith(".pth"):
            backbone_path = None

        return cls(
            compare_classes=compare_classes or [],
            model_dir=Path(settings.get("model_dir", "./weights/patchcore_resnet")),
            yolo_model_path=str(settings.get("segmentation_model", "weights/detection/pill-detection-best-2.onnx")),
            img_size=int(settings.get("seg_img_size", 512)),
            conf=float(settings.get("seg_conf", 0.5)),
            iou=float(settings.get("seg_iou", 0.6)),
            pad=int(settings.get("seg_pad", 0)),
            model_size=int(settings.get("img_size", 256)),
            grid_size=int(settings.get("grid_size", 16)),
            use_color_features=bool(settings.get("use_color", True)),
            use_hsv=bool(settings.get("use_hsv", True)),
            color_weight=float(settings.get("color_weight", 1.0)),
            backbone_path=str(backbone_path) if backbone_path else None,
            k_nearest=int(settings.get("k_nearest", 3)),
            threshold_multiplier=float(settings.get("threshold_mult", 1.0)),
            crop_size=int(settings.get("img_size", 256)),
            track_max_distance=float(settings.get("track_max_dist", 80.0)),
            track_iou_threshold=float(settings.get("track_iou_thr", 0.80)),
            track_max_age=int(settings.get("track_max_age", 10)),
        )


# ─────────────────────────────────────────────────────────
#  Main Inspector
# ─────────────────────────────────────────────────────────
class PillInspector:
    """
    Realtime pill anomaly inspector.

    ``classify_anomaly(frame)`` → annotated frame.
    ``summarize()`` → dict with counts + summary image.
    """

    def __init__(self, config: Optional[InspectorConfig] = None):
        self.config = config or InspectorConfig()
        self._init_components()
        self._init_state()

    # ───────── Components ─────────
    def _init_components(self):
        cfg = self.config

        self._segmentor = YOLODetector(
            model_path=cfg.yolo_model_path,
            img_size=cfg.img_size,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device or "cuda",
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
            use_half=cfg.use_half,
            use_color_features=cfg.use_color_features,
            use_hsv=cfg.use_hsv,
            color_weight=cfg.color_weight,
            backbone_path=cfg.backbone_path,
        )

        self._scorer = PatchCoreScorer(k_nearest=cfg.k_nearest, assume_normalized=False)

        # parent_class → { subclass_name → faiss.Index }
        self._sub_indices: Dict[str, Dict[str, faiss.Index]] = {}
        # parent_class → { subclass_name → threshold }
        self._sub_thresholds: Dict[str, Dict[str, float]] = {}

        # ── Eager pre-load all configured classes ──
        for class_name in cfg.compare_classes:
            self._ensure_index(class_name)

    def _init_state(self):
        self._last_frame: Optional[np.ndarray] = None
        self._last_results: List[Dict[str, Any]] = []
        self._last_crops: Dict[int, np.ndarray] = {}

    # ───────── Memory Loading ─────────
    def _ensure_index(self, class_name: str) -> bool:
        """
        Load ``class_name.pth`` (multi-subclass format).
        Builds a separate FAISS index + threshold per subclass.
        """
        if class_name in self._sub_indices:
            return True

        pth = self.config.model_dir / f"{class_name}.pth"
        if not pth.exists():
            print(f"[Missing Model] {pth}")
            return False

        try:
            subclasses, shared_meta = MemoryBank.load_multi(pth)

            indices: Dict[str, faiss.Index] = {}
            thresholds: Dict[str, float] = {}

            for sub_name, (bank, sub_meta) in subclasses.items():
                idx = self._scorer.build_index(bank)
                indices[sub_name] = idx

                raw_thr = float(sub_meta.get("threshold", 0.5))
                thr = raw_thr * self.config.threshold_multiplier
                thresholds[sub_name] = thr

                print(
                    f"[Loaded] {class_name}/{sub_name} | "
                    f"dim={bank.shape[1]} patches={bank.shape[0]} | "
                    f"thr={raw_thr:.4f} x {self.config.threshold_multiplier} "
                    f"= {thr:.4f}"
                )

            self._sub_indices[class_name] = indices
            self._sub_thresholds[class_name] = thresholds
            return True
        except Exception as e:
            print(f"[Load Error] {class_name}: {e}")
            return False

    # ───────── Classification ─────────
    def _classify_pill(
        self,
        feats: np.ndarray,
        classes: List[str],
    ) -> Tuple[str, Dict[str, float], List[str]]:
        """
        Score pill features against every subclass of every requested
        parent class.  Pill is NORMAL if **any** subclass score ≤ threshold.

        With ``early_exit_normal=True``, stops scoring once NORMAL is found
        for faster inference (skips remaining subclasses).

        Returns
        -------
        status : "NORMAL" | "ANOMALY"
        scores : { "parent/sub" : float }
        normal_from : [ "parent/sub", ... ]
        """
        if feats is None or feats.shape[0] == 0:
            return "NORMAL", {}, []

        scores: Dict[str, float] = {}
        normal_from: List[str] = []
        early_exit = self.config.early_exit_normal

        for class_name in classes:
            if not self._ensure_index(class_name):
                continue

            sub_indices = self._sub_indices.get(class_name, {})
            sub_thresholds = self._sub_thresholds.get(class_name, {})

            for sub_name, idx in sub_indices.items():
                key = f"{class_name}/{sub_name}"
                score = self._scorer.score_pill(
                    feats, idx,
                    method=self.config.score_method,
                )
                scores[key] = score
                if score <= sub_thresholds.get(sub_name, 0.5):
                    normal_from.append(key)
                    if early_exit:
                        # Found normal — skip remaining subclasses
                        return "NORMAL", scores, normal_from

        status = "NORMAL" if normal_from else "ANOMALY"
        return status, scores, normal_from

    # ═══════════════════════════════
    # SPLIT API (for threaded pipelines)
    # ═══════════════════════════════
    def detect_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Phase 1 — YOLO detection only (fast, ~20-50 ms after warmup).

        Returns (crops, infos).  Tracking state is updated here.
        **Call from the display thread** for instant bbox response.
        """
        crops, infos = self._segmentor.detect_and_crop(frame)
        self._last_frame = frame
        return crops, infos

    def score_pills(
        self,
        crops: List[np.ndarray],
        infos: List[Dict[str, Any]],
        class_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Phase 2 — feature extraction + anomaly scoring (slow).

        Returns a list of result dicts (one per pill).
        **Call from a background thread** — only touches ``_extractor``
        and ``_scorer`` (read-only indices), no shared mutable state.
        """
        classes = list(class_names or self.config.compare_classes)
        batch_feats = self._extractor.extract_batch(crops) if crops else []

        results: List[Dict[str, Any]] = []
        for i, (crop, info) in enumerate(zip(crops, infos)):
            feats = batch_feats[i] if i < len(batch_feats) else None
            status, scores, normal_from = self._classify_pill(feats, classes)
            results.append({
                "id": int(info.get("track_id", -1)),
                "bbox": info["bbox"],
                "conf": info["conf"],
                "status": status,
                "class_scores": scores,
                "normal_from": normal_from,
            })
        return results

    def warmup(self) -> None:
        """Run a dummy YOLO inference to trigger ONNX compilation."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._segmentor.detect(dummy)

    # ═══════════════════════════════
    # REALTIME (combined — kept for backward compat)
    # ═══════════════════════════════
    def classify_anomaly(
        self,
        frame: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        classes = list(class_names or self.config.compare_classes)

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

        # Defer frame copy to drawing (only copy once)
        self._last_frame = frame.copy()
        self._last_results = frame_results
        self._last_crops = frame_crops
        return draw_pill_results(self._last_frame, frame_results)

    # ═══════════════════════════════
    # SUMMARY
    # ═══════════════════════════════
    def summarize(self) -> Dict[str, Any]:
        if self._last_frame is None:
            return {"image": None, "count": 0, "good_count": 0, "bad_count": 0}

        items = [
            {"id": r["id"], "bbox": r["bbox"], "status": r["status"]}
            for r in self._last_results
        ]
        vis = draw_summary(self._last_frame, items)

        good = sum(1 for it in items if it["status"] == "NORMAL")
        bad = sum(1 for it in items if it["status"] != "NORMAL")

        return {
            "image": vis,
            "count": len(items),
            "good_count": good,
            "bad_count": bad,
        }

    # ───────── Reset ─────────
    def reset(self):
        self._init_state()
        self._segmentor.reset_tracking()

    @property
    def last_crops(self):
        return dict(self._last_crops)

    @property
    def detector(self):
        return self._segmentor
