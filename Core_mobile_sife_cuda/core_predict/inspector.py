# core_predict/inspector.py
"""
PillInspectorSIFE - Pill anomaly inspector with CUDA + batch optimization.

CUDA Optimized:
- Batch feature extraction: all crops → one backbone forward pass
- ThreadPoolExecutor for parallel crop preparation
- Persistent thread pool (no per-frame overhead)
- Lazy model loading with caching

Responsibilities:
- Detect pills using YOLO
- Classify anomalies using PatchCore + SIFE (batched)
- Accumulate votes across frames
- Summarize with majority vote
"""
from __future__ import annotations

import cv2
import faiss
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core_shared.patchcore_sife import PatchCoreSIFE
from .yolo_detector import PillYOLODetector
from .visualizer import draw_pill_results, draw_summary


# =========================================================
# CONFIGURATION
# =========================================================

@dataclass
class InspectorConfig:
    """Full configuration for PillInspectorSIFE."""

    compare_classes: List[str] = field(default_factory=lambda: ["vitaminc"])

    model_dir: Path = field(default_factory=lambda: Path("./model/patchcore_sife"))
    yolo_model_path: str = "model/yolo12-seg.pt"
    yolo_det_model_path: Optional[str] = None
    device: Optional[str] = None

    # PatchCore
    model_size: int = 256
    grid_size: int = 20
    k_nearest: int = 11

    # SIFE
    use_sife: bool = True
    sife_dim: int = 32
    sife_encoding_type: str = "sinusoidal"
    sife_weight: float = 1.0
    use_center_distance: bool = True
    use_local_gradient: bool = True

    # CNN/SIFE balance
    cnn_weight: float = 1.0

    # Laplacian
    use_laplacian_variance: bool = False
    laplacian_weight: float = 1.0

    # Color
    use_color_features: bool = False
    use_hsv: bool = False
    color_weight: float = 1.0

    # Multi-scale & Edge
    use_multi_scale: bool = False
    multi_scale_grids: List[int] = field(default_factory=lambda: [16, 32, 48])
    use_edge_enhancement: bool = False
    edge_weight: float = 1.5

    # Scoring
    use_detailed_scoring: bool = True
    score_weight_max: float = 0.3
    score_weight_top_k: float = 0.5
    score_weight_percentile: float = 0.2
    top_k_percent: float = 0.05

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

    crop_size: int = 256

    # CLAHE preprocessing (ต้องตรงกับ training data ที่ใช้ CLAHE ตอนถ่าย)
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # Fine-tuned backbone
    finetuned_backbone_path: Optional[Path] = None

    # Threading
    num_workers: int = 4

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(self.model_dir)


# =========================================================
# UTILITY
# =========================================================

def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE on V channel (HSV) — matches cropped_rm_bg.py preprocess_light()."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    v = clahe.apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def make_square_crop(crop: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Pad crop to square (black padding). Resize is left to self.transform
    (PIL BICUBIC) so that interpolation matches training exactly.
    Thread-safe (no shared state)."""
    if crop is None or crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    max_side = max(h, w)
    square = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    y_off = (max_side - h) // 2
    x_off = (max_side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop

    # NOTE: ไม่ resize ที่นี่ — ปล่อยให้ self.transform (PIL Resize BICUBIC)
    # ทำ resize เหมือนกับ training path พอดี
    # Training: Image.open (natural size e.g. 75x75) → PIL Resize(256, BICUBIC)
    # Predict:  square pad (natural size e.g. 75x75) → PIL Resize(256, BICUBIC)
    return square


# =========================================================
# INSPECTOR
# =========================================================

class PillInspectorSIFE:
    """
    Pill anomaly inspector with batch CUDA inference.
    
    Pipeline per frame:
        1. YOLO detect → crops
        2. ThreadPool: pad & resize crops (parallel CPU)
        3. Batch feature extraction (single GPU forward pass)
        4. Score each crop against memory banks
        5. Accumulate votes per track ID
    
    Usage:
        inspector = PillInspectorSIFE(config)
        for frame in frames:
            preview = inspector.classify_anomaly(frame)
        result = inspector.summarize()
    """

    def __init__(self, config: Optional[InspectorConfig] = None):
        self.config = config or InspectorConfig()
        self._init_models()
        self._init_state()

        # Persistent thread pool for crop preparation
        self._crop_pool = ThreadPoolExecutor(
            max_workers=self.config.num_workers,
            thread_name_prefix="crop_prep",
        )

    def __del__(self):
        """Shutdown thread pool on cleanup."""
        if hasattr(self, "_crop_pool"):
            self._crop_pool.shutdown(wait=False)

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------

    def _init_models(self) -> None:
        """Initialize PatchCore + YOLO with CUDA support."""
        cfg = self.config

        self._patchcore = PatchCoreSIFE(
            model_size=cfg.model_size,
            grid_size=cfg.grid_size,
            k_nearest=cfg.k_nearest,
            device=cfg.device,
            use_sife=cfg.use_sife,
            sife_dim=cfg.sife_dim,
            sife_encoding_type=cfg.sife_encoding_type,
            sife_weight=cfg.sife_weight,
            use_center_distance=cfg.use_center_distance,
            use_local_gradient=cfg.use_local_gradient,
            cnn_weight=cfg.cnn_weight,
            use_laplacian_variance=cfg.use_laplacian_variance,
            laplacian_weight=cfg.laplacian_weight,
            use_color_features=cfg.use_color_features,
            use_hsv=cfg.use_hsv,
            color_weight=cfg.color_weight,
            use_multi_scale=cfg.use_multi_scale,
            multi_scale_grids=cfg.multi_scale_grids,
            use_edge_enhancement=cfg.use_edge_enhancement,
            edge_weight=cfg.edge_weight,
            score_weight_max=cfg.score_weight_max,
            score_weight_top_k=cfg.score_weight_top_k,
            score_weight_percentile=cfg.score_weight_percentile,
            top_k_percent=cfg.top_k_percent,
            finetuned_backbone_path=cfg.finetuned_backbone_path,
        )

        # Model storage & FAISS indices
        self._model_dir = cfg.model_dir
        self._model_data: Dict[str, Any] = {}
        self._subclass_map: Dict[str, List[str]] = {}
        self._indices: Dict[str, faiss.Index] = {}
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
            device=cfg.device,
        )

    def _init_state(self) -> None:
        """Reset all frame-to-frame state."""
        self._votes: Dict[int, Dict[str, Any]] = {}
        self._last_frame: Optional[np.ndarray] = None
        self._last_results: List[Dict[str, Any]] = []
        self._last_crops: Dict[int, np.ndarray] = {}
        self._first_detection_done: bool = False

    # ---------------------------------------------------------
    # MODEL LOADING (LAZY + CACHED)
    # ---------------------------------------------------------

    def _get_subclasses(self, parent_class: str) -> List[str]:
        """Get subclass names for a parent class (lazy discovery)."""
        if parent_class in self._subclass_map:
            return self._subclass_map[parent_class]

        parent_dir = self._model_dir / parent_class
        if parent_dir.is_dir():
            self._subclass_map[parent_class] = [f.stem for f in parent_dir.glob("*.pth")]
        else:
            pth = self._model_dir / f"{parent_class}.pth"
            self._subclass_map[parent_class] = [parent_class] if pth.exists() else []

        return self._subclass_map[parent_class]

    def _ensure_index(self, subclass_name: str, parent_class: Optional[str] = None) -> bool:
        """Lazy-load model + FAISS index for a subclass."""
        if subclass_name in self._indices:
            return True

        # Resolve .pth path
        if subclass_name not in self._model_data:
            pth_path = self._resolve_pth_path(subclass_name, parent_class)
            if not pth_path or not pth_path.exists():
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

        # Build FAISS index
        data = self._model_data[subclass_name]
        bank = data["memory_bank"].cpu().numpy().astype(np.float32)

        index = faiss.IndexFlatIP(bank.shape[1])
        faiss.normalize_L2(bank)
        index.add(bank)

        self._indices[subclass_name] = index
        self._thresholds[subclass_name] = float(data.get("threshold", 0.35))
        return True

    def _resolve_pth_path(self, subclass_name: str, parent_class: Optional[str]) -> Optional[Path]:
        """Resolve .pth file path with fallback logic."""
        candidates = []

        if parent_class:
            candidates.append(self._model_dir / parent_class / f"{subclass_name}.pth")

        parts = subclass_name.rsplit("_", 1)
        if len(parts) == 2:
            candidates.append(self._model_dir / parts[0] / f"{subclass_name}.pth")

        candidates.append(self._model_dir / f"{subclass_name}.pth")

        for path in candidates:
            if path.exists():
                return path
        return None

    # ---------------------------------------------------------
    # CLASSIFICATION (BATCHED)
    # ---------------------------------------------------------

    def _score_features(
        self, feats: np.ndarray, subclass_name: str
    ) -> float:
        """Score pre-extracted features against a single subclass index."""
        if self.config.use_detailed_scoring:
            detailed = self._patchcore.get_anomaly_score_detailed(
                feats, self._indices[subclass_name]
            )
            return detailed["score"]
        return self._patchcore.get_max_anomaly_score(
            feats, self._indices[subclass_name]
        )

    def _classify_with_features(
        self, feats: Optional[np.ndarray], classes: Sequence[str]
    ) -> Tuple[str, Dict[str, float], List[str]]:
        """Classify pre-extracted features against all parent classes."""
        if feats is None or feats.shape[0] == 0:
            return "NORMAL", {}, []

        scores: Dict[str, float] = {}
        normal_from: List[str] = []

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

                score = self._score_features(feats, subcls)
                scores[subcls] = score
                if score <= self._thresholds[subcls]:
                    normal_from.append(subcls)

        status = "NORMAL" if normal_from else "ANOMALY"
        return status, scores, normal_from

    # ---------------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------------

    def classify_anomaly(
        self,
        frame: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Process one frame with batch CUDA inference.
        
        Pipeline:
            1. YOLO detection
            2. Parallel crop preparation (ThreadPool)
            3. Batch feature extraction (single GPU call)
            4. Score + accumulate votes
        
        Returns: Preview image with bounding boxes
        """
        classes = list(class_names or self.config.compare_classes)
        frame_copy = frame.copy()  # local copy — immune to reset() race
        self._last_frame = frame_copy

        # --- CLAHE preprocessing (ตรงกับ cropped_rm_bg.py ที่ใช้ตอนถ่ายrูป train) ---
        if self.config.use_clahe:
            frame = apply_clahe(
                frame,
                clip_limit=self.config.clahe_clip_limit,
                tile_size=self.config.clahe_tile_size,
            )

        # --- Mode switching ---
        if not self._first_detection_done:
            print(f"[MODE] Using {self._detector.current_mode} for initial crop")
        elif self._first_detection_done and self._detector.use_detection and self._detector.has_separate_det_model:
            self._detector.switch_to_segmentation()

        # --- 1. YOLO detect & crop ---
        crops, infos = self._detector.detect_and_crop(frame)

        if not crops:
            self._last_results = []
            self._last_crops = {}
            return draw_pill_results(frame_copy, [])

        if not self._first_detection_done:
            self._first_detection_done = True
            print("[MODE] First detection done, will switch to SEGMENTATION next frame")

        # --- 2. Parallel crop preparation (CPU threads) ---
        crop_size = self.config.crop_size
        futures = [
            self._crop_pool.submit(make_square_crop, c, crop_size)
            for c in crops
        ]
        squares = [f.result() for f in futures]

        # --- 3. Batch feature extraction (single GPU forward pass) ---
        features_list = self._patchcore.extract_from_numpy_batch(squares)

        # --- 4. Classify each crop ---
        frame_results: List[Dict[str, Any]] = []
        frame_crops: Dict[int, np.ndarray] = {}

        for i, (feat, info) in enumerate(zip(features_list, infos)):
            tid = int(info.get("track_id", -1))
            conf = float(info.get("conf", 0.0))
            bbox = tuple(map(int, info.get("padded_region", (0, 0, 0, 0))))

            status, scores, normal_from = self._classify_with_features(feat, classes)

            if tid >= 0:
                frame_crops[tid] = squares[i]

            frame_results.append({
                "id": tid,
                "bbox": bbox,
                "conf": conf,
                "status": status,
                "class_scores": scores,
                "normal_from": normal_from,
            })

        # --- 5. Dedup: keep highest confidence per track_id ---
        best: Dict[int, Dict[str, Any]] = {}
        for r in frame_results:
            tid = r["id"]
            if tid < 0:
                continue
            if tid not in best or r["conf"] > best[tid]["conf"]:
                best[tid] = r

        # --- 6. Accumulate votes ---
        for tid, r in best.items():
            if tid not in self._votes:
                self._votes[tid] = {"bbox": r["bbox"], "NORMAL": 0, "ANOMALY": 0}
            self._votes[tid]["bbox"] = r["bbox"]
            self._votes[tid][r["status"]] += 1

        self._last_results = list(best.values())
        self._last_crops = {tid: frame_crops[tid] for tid in best if tid in frame_crops}

        return draw_pill_results(frame_copy, self._last_results)

    # ---------------------------------------------------------
    # SUMMARIZE & RESET
    # ---------------------------------------------------------

    def summarize(self) -> Dict[str, Any]:
        """
        Compute final status via majority vote across accumulated frames.
        
        Returns:
            {"image": np.ndarray, "good": int, "bad": int, "total": int, "items": [...]}
        """
        if self._last_frame is None:
            return {"image": None, "good": 0, "bad": 0, "total": 0, "items": []}

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
        return {"image": vis, "good": good, "bad": bad, "total": good + bad, "items": items}

    def reset(self) -> None:
        """Clear all votes and tracking state."""
        self._init_state()
        self._detector.reset_tracking()

    # ---------------------------------------------------------
    # PROPERTIES
    # ---------------------------------------------------------

    @property
    def last_crops(self) -> Dict[int, np.ndarray]:
        return dict(self._last_crops)

    @property
    def anomaly_counts(self) -> Dict[int, int]:
        return {tid: v.get("ANOMALY", 0) for tid, v in self._votes.items()}

    @property
    def detector(self) -> PillYOLODetector:
        return self._detector
