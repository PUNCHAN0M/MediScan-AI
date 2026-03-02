"""
Inference Pipeline
==================

Folder-based and realtime pill inspection.

Classes
-------
``InspectorConfig``
    Dataclass with all tuneable knobs.
``PillInspector``
    Main façade — detect → crop → extract → score → classify.

Output format
-------------
::

    {
        "count": 50,
        "bad_count": 3,
        "bad_pills":  [{"bbox": [x1,y1,x2,y2], "score": 0.87, …}, …],
        "good_count": 47,
        "good_pills": [{"bbox": [x1,y1,x2,y2], "score": 0.12, …}, …],
    }

Realtime flow
-------------
::

    frame → classify_anomaly() → preview image  (repeat N frames)
    → summarize() → JSON result with majority vote
"""
from __future__ import annotations

import cv2
import numpy as np
import faiss
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Core_ResnetPatchCore.segmentation.yolo_infer import YOLOSegmentor
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


# ─────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────
@dataclass
class InspectorConfig:
    """All tuneable knobs for PillInspector — defaults from config/base.py + config/resnet.py."""

    # classes to compare each pill against
    compare_classes: List[str] = field(default_factory=list)
    model_dir: Path = field(default_factory=lambda: Path(MODEL_OUTPUT_DIR))

    # ── YOLO Segmentation ──
    yolo_model_path: str = str(SEGMENTATION_MODEL_PATH)
    img_size: int = 512
    conf: float = SEGMENTATION_CONF
    iou: float = SEGMENTATION_IOU
    pad: int = SEGMENTATION_PAD

    # ── Feature Extractor ──
    model_size: int = IMG_SIZE
    grid_size: int = GRID_SIZE
    use_color_features: bool = USE_COLOR_FEATURES
    use_hsv: bool = USE_HSV
    color_weight: float = COLOR_WEIGHT

    # ── Scoring ──
    k_nearest: int = K_NEAREST
    score_method: str = SCORE_METHOD
    threshold_multiplier: float = THRESHOLD_MULTIPLIER

    # ── Backbone ──
    backbone_path: Optional[str] = BACKBONE if BACKBONE and str(BACKBONE).endswith(".pth") else None

    # ── Crop ──
    crop_size: int = IMAGE_SIZE
    bg_value: int = 0

    # ── Tracking (realtime) ──
    track_max_distance: float = TRACK_MAX_DISTANCE
    track_iou_threshold: float = TRACK_IOU_THRESHOLD
    track_max_age: int = TRACK_MAX_AGE
    merge_dist_px: int = 60

    # ── Voting ──
    frames_before_summary: int = FRAMES_BEFORE_SUMMARY

    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = BASE_DEVICE
        self.model_dir = Path(self.model_dir)


# ─────────────────────────────────────────────────────────
#  PillInspector
# ─────────────────────────────────────────────────────────
class PillInspector:
    """
    Main pill anomaly inspector.

    Supports **both** camera (frame-by-frame) and folder-based prediction.

    Multi-class comparison
    ~~~~~~~~~~~~~~~~~~~~~~
    Each pill is compared against every class in ``compare_classes``.
    If **any** class scores ≤ its threshold → pill is **NORMAL**.
    """

    def __init__(self, config: Optional[InspectorConfig] = None):
        self.config = config or InspectorConfig()
        self._init_components()
        self._init_state()

    # ─────────────────── init ───────────────────
    def _init_components(self) -> None:
        cfg = self.config

        # YOLO segmentor (supports .pt + .onnx)
        self._segmentor = YOLOSegmentor(
            model_path=cfg.yolo_model_path,
            img_size=cfg.img_size,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            enable_tracking=True,
            track_max_distance=cfg.track_max_distance,
            track_iou_threshold=cfg.track_iou_threshold,
            track_max_age=cfg.track_max_age,
        )

        # ResNet50 feature extractor
        self._extractor = ResNet50FeatureExtractor(
            img_size=cfg.model_size,
            grid_size=cfg.grid_size,
            device=cfg.device,
            use_color_features=cfg.use_color_features,
            use_hsv=cfg.use_hsv,
            color_weight=cfg.color_weight,
            backbone_path=cfg.backbone_path,
        )

        # scorer
        self._scorer = PatchCoreScorer(k_nearest=cfg.k_nearest)

        # lazy-loaded per-class FAISS indices
        self._indices: Dict[str, faiss.Index] = {}
        self._thresholds: Dict[str, float] = {}
        self._subclass_map: Dict[str, List[str]] = {}

    def _init_state(self) -> None:
        """Clear per-session state (votes, frame buffer)."""
        self._votes: Dict[int, Dict[str, Any]] = {}
        self._last_frame: Optional[np.ndarray] = None
        self._last_results: List[Dict[str, Any]] = []
        self._last_crops: Dict[int, np.ndarray] = {}

    # ─────────────────── lazy model loading ───────────────────
    def _get_subclasses(self, parent: str) -> List[str]:
        if parent in self._subclass_map:
            return self._subclass_map[parent]

        parent_dir = self.config.model_dir / parent
        if parent_dir.is_dir():
            subs = [f.stem for f in parent_dir.glob("*.pth")]
        else:
            pth = self.config.model_dir / f"{parent}.pth"
            subs = [parent] if pth.exists() else []

        self._subclass_map[parent] = subs
        return subs

    def _ensure_index(self, name: str, parent: Optional[str] = None) -> bool:
        if name in self._indices:
            return True

        if parent:
            pth = self.config.model_dir / parent / f"{name}.pth"
        else:
            pth = self.config.model_dir / f"{name}.pth"

        if not pth.exists():
            return False

        try:
            bank, meta = MemoryBank.load(pth)
            idx = self._scorer.build_index(bank)
            self._indices[name] = idx
            raw_thr = float(meta.get("threshold", 0.50))
            self._thresholds[name] = raw_thr * self.config.threshold_multiplier
            print(f"[Loaded] {name}: dim={bank.shape[1]:,}  "
                  f"thr={raw_thr:.4f} × {self.config.threshold_multiplier} "
                  f"= {self._thresholds[name]:.4f}")
            return True
        except Exception as exc:
            print(f"[Error] {name}: {exc}")
            return False

    # ─────────────────── classify one pill ───────────────────
    def _classify_pill(
        self,
        feats: np.ndarray,
        classes: List[str],
    ) -> Tuple[str, Dict[str, float], List[str]]:
        """
        Score one pill's features against all compare classes.

        Returns ``(status, class_scores, normal_from)``.
        """
        if feats is None or feats.shape[0] == 0:
            return "NORMAL", {}, []

        scores: Dict[str, float] = {}
        normal_from: List[str] = []

        for parent in classes:
            subs = self._get_subclasses(parent)
            if not subs:
                if self._ensure_index(parent):
                    subs = [parent]
                else:
                    continue

            for sub in subs:
                if not self._ensure_index(sub, parent):
                    continue
                s = self._scorer.score_pill(
                    feats, self._indices[sub],
                    method=self.config.score_method,
                )
                scores[sub] = s
                if s <= self._thresholds[sub]:
                    normal_from.append(sub)

        status = "NORMAL" if normal_from else "ANOMALY"
        return status, scores, normal_from

    # ═════════════════════════════════════════════════════════
    #  REALTIME  (per-frame + vote accumulation)
    # ═════════════════════════════════════════════════════════
    def classify_anomaly(
        self,
        frame: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Process **one frame**: detect → crop → batch extract → score → vote.

        Returns annotated preview image.
        """
        classes = list(class_names or self.config.compare_classes)
        self._last_frame = frame.copy()

        # 1. YOLO detect + crop
        crops, infos = self._segmentor.detect_and_crop(
            frame,
            target_size=self.config.crop_size,
            bg_value=self.config.bg_value,
            pad=self.config.pad,
        )

        # 2. Batch feature extraction  ← 🔥 speed trick
        batch_feats = self._extractor.extract_batch(crops) if crops else []

        # 3. Per-pill classification
        frame_results: List[Dict[str, Any]] = []
        frame_crops: Dict[int, np.ndarray] = {}

        for i, (crop, info) in enumerate(zip(crops, infos)):
            tid = int(info.get("track_id", -1))
            feats = batch_feats[i]

            status, scores, normal_from = self._classify_pill(feats, classes)

            frame_results.append({
                "id": tid,
                "bbox": info["bbox"],
                "conf": info["conf"],
                "status": status,
                "class_scores": scores,
                "normal_from": normal_from,
            })

            if tid >= 0:
                frame_crops[tid] = crop

        # 4. De-dup: keep highest conf per track_id
        best: Dict[int, Dict] = {}
        for r in frame_results:
            tid = r["id"]
            if tid < 0:
                continue
            if tid not in best or r["conf"] > best[tid]["conf"]:
                best[tid] = r

        # 5. Vote accumulation
        for tid, r in best.items():
            if tid not in self._votes:
                self._votes[tid] = {
                    "bbox": r["bbox"],
                    "NORMAL": 0,
                    "ANOMALY": 0,
                    "best_scores": {},
                }
            self._votes[tid]["bbox"] = r["bbox"]
            self._votes[tid][r["status"]] += 1
            self._votes[tid]["best_scores"] = r.get("class_scores", {})

        self._last_results = list(best.values())
        self._last_crops = {
            tid: frame_crops[tid]
            for tid in best if tid in frame_crops
        }

        return draw_pill_results(self._last_frame, self._last_results)

    # ═════════════════════════════════════════════════════════
    #  SINGLE-IMAGE prediction  → JSON
    # ═════════════════════════════════════════════════════════
    def predict_image(
        self,
        image: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict anomalies in **one image** (folder-based mode).

        Returns
        -------
        ::

            {
                "count":      int,
                "bad_count":  int,
                "bad_pills":  [{"bbox": [...], "score": float}, …],
                "good_count": int,
                "good_pills": [{"bbox": [...], "score": float}, …],
            }
        """
        classes = list(class_names or self.config.compare_classes)

        crops, infos = self._segmentor.detect_and_crop(
            image,
            target_size=self.config.crop_size,
            bg_value=self.config.bg_value,
            pad=self.config.pad,
        )

        batch_feats = self._extractor.extract_batch(crops) if crops else []

        good_pills: List[Dict] = []
        bad_pills: List[Dict] = []

        for i, (_, info) in enumerate(zip(crops, infos)):
            feats = batch_feats[i]
            status, all_scores, normal_from = self._classify_pill(
                feats, classes)

            pill = {
                "bbox": list(info["bbox"]),
                "score": round(
                    min(all_scores.values()) if all_scores else 0.0, 4),
                "class_scores": {k: round(v, 4) for k, v in all_scores.items()},
            }
            if normal_from:
                pill["matched_class"] = normal_from
                good_pills.append(pill)
            else:
                bad_pills.append(pill)

        return {
            "count": len(crops),
            "bad_count": len(bad_pills),
            "bad_pills": bad_pills,
            "good_count": len(good_pills),
            "good_pills": good_pills,
        }

    # ═════════════════════════════════════════════════════════
    #  SUMMARY  (majority vote across frames)
    # ═════════════════════════════════════════════════════════
    def summarize(self) -> Dict[str, Any]:
        """
        Majority-vote summary over all accumulated frames.

        Returns dict with ``image``, ``count``, ``good_count / bad_count``,
        ``good_pills / bad_pills``.
        """
        if self._last_frame is None:
            return {
                "image": None, "count": 0,
                "good_count": 0, "bad_count": 0,
                "good_pills": [], "bad_pills": [],
            }

        good_pills: List[Dict] = []
        bad_pills: List[Dict] = []
        items: List[Dict] = []

        for tid in sorted(self._votes):
            v = self._votes[tid]
            n_ok = v.get("NORMAL", 0)
            n_bad = v.get("ANOMALY", 0)
            status = "NORMAL" if n_ok > n_bad else "ANOMALY"

            best = v.get("best_scores", {})
            score = round(min(best.values()) if best else 0.0, 4)

            pill = {
                "bbox": list(v["bbox"]),
                "score": score,
                "votes": {"NORMAL": n_ok, "ANOMALY": n_bad},
            }

            if status == "NORMAL":
                good_pills.append(pill)
            else:
                bad_pills.append(pill)

            items.append({
                "id": tid,
                "bbox": v["bbox"],
                "status": status,
            })

        vis = draw_summary(self._last_frame, items)

        return {
            "image": vis,
            "count": len(good_pills) + len(bad_pills),
            "good_count": len(good_pills),
            "bad_count": len(bad_pills),
            "good_pills": good_pills,
            "bad_pills": bad_pills,
        }

    # ─────────────────── reset ───────────────────
    def reset(self) -> None:
        """Clear votes, tracking, and frame state."""
        self._init_state()
        self._segmentor.reset_tracking()

    # ─────────────────── properties ───────────────────
    @property
    def last_crops(self) -> Dict[int, np.ndarray]:
        return dict(self._last_crops)

    @property
    def anomaly_counts(self) -> Dict[int, int]:
        return {tid: v.get("ANOMALY", 0) for tid, v in self._votes.items()}

    @property
    def detector(self) -> YOLOSegmentor:
        return self._segmentor
