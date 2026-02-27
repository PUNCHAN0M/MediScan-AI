# core_predict/inspector.py
"""
PillInspector - Main inspection class with vote accumulation.

Single Responsibility:
- Detect pills using YOLO
- Classify anomalies using PatchCore
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

from MobilenetPatchCore.core_shared.patchcore import PatchCore
from MobilenetPatchCore.core_predict.yolo_detector import PillYOLODetector
from MobilenetPatchCore.core_predict.visualizer import draw_pill_results, draw_summary


# =========================================================
# CONFIGURATION
# =========================================================
@dataclass
class InspectorConfig:
    """Configuration for PillInspector."""
    
    compare_classes: List[str] = field(default_factory=lambda: [
        "vitaminc_front", "vitaminc_back",
        "white_front", "white_back",
    ])
    
    model_dir: Path = field(default_factory=lambda: Path("./model/patchcore"))  # โฟลเดอร์ .pth แยกแต่ละ class
    yolo_model_path: str = "model/yolo12-seg.pt"
    yolo_det_model_path: Optional[str] = None  # None = ไม่ใช้ separate detection model
    device: Optional[str] = None
    
    # PatchCore
    model_size: int = 256
    grid_size: int = 20
    k_nearest: int = 19
    
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

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(self.model_dir)


# =========================================================
# UTILITY
# =========================================================
def make_square_crop(crop: np.ndarray, target_size: int = 224) -> np.ndarray:
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
        square = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    return square


# =========================================================
# INSPECTOR
# =========================================================
class PillInspector:
    """
    Pill anomaly inspector with frame-based vote accumulation.
    
    Usage:
        inspector = PillInspector()
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
        
        # PatchCore
        self._patchcore = PatchCore(
            model_size=cfg.model_size,
            grid_size=cfg.grid_size,
            k_nearest=cfg.k_nearest,
            device=cfg.device,
        )
        
        # Load individual class models (lazy loading)
        self._model_dir = cfg.model_dir
        self._model_data: Dict[str, Any] = {}  # จะโหลดเมื่อต้องการใช้
        self._subclass_map: Dict[str, List[str]] = {}  # parent -> [subclasses]
        
        # FAISS indices (lazy loaded)
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
            # โฟลเดอร์ parent มี subclass .pth files
            subclasses = [f.stem for f in parent_dir.glob("*.pth")]
            self._subclass_map[parent_class] = subclasses
        else:
            # ไม่มีโฟลเดอร์ parent, ลองหา .pth ตรงๆ (fallback)
            pth_path = self._model_dir / f"{parent_class}.pth"
            if pth_path.exists():
                self._subclass_map[parent_class] = [parent_class]
            else:
                self._subclass_map[parent_class] = []
        
        return self._subclass_map[parent_class]
    
    def _ensure_index(self, subclass_name: str, parent_class: Optional[str] = None) -> bool:
        """Lazy-load FAISS index for a subclass."""
        if subclass_name in self._indices:
            return True
        
        # โหลด model data ถ้ายังไม่ได้โหลด
        if subclass_name not in self._model_data:
            # ลองหาใน parent folder ก่อน
            if parent_class:
                pth_path = self._model_dir / parent_class / f"{subclass_name}.pth"
            else:
                # ลองหาจากชื่อ subclass (เช่น vitaminc_front -> vitaminc folder)
                parts = subclass_name.rsplit("_", 1)
                if len(parts) == 2:
                    pth_path = self._model_dir / parts[0] / f"{subclass_name}.pth"
                else:
                    pth_path = self._model_dir / f"{subclass_name}.pth"
            
            if not pth_path.exists():
                # Fallback: ลองหา .pth ตรงๆ
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
        bank = data["memory_bank"].cpu().numpy().astype(np.float32)
        
        index = faiss.IndexFlatIP(bank.shape[1])
        faiss.normalize_L2(bank)
        index.add(bank)
        
        self._indices[subclass_name] = index
        self._thresholds[subclass_name] = float(data.get("threshold", 0.35))
        return True
    
    def _classify_single(
        self, crop: np.ndarray, classes: Sequence[str]
    ) -> Tuple[str, Dict[str, float], List[str]]:
        """Classify a single crop against parent classes (expands to subclasses)."""
        feats = self._patchcore.extract_from_numpy(crop)
        if feats is None or feats.shape[0] == 0:
            return "NORMAL", {}, []
        
        scores: Dict[str, float] = {}
        normal_from: List[str] = []
        
        # Expand parent classes to subclasses
        for parent_cls in classes:
            subclasses = self._get_subclasses(parent_cls)
            if not subclasses:
                # Fallback: try as direct subclass name
                if self._ensure_index(parent_cls):
                    subclasses = [parent_cls]
                else:
                    continue
            
            for subcls in subclasses:
                if not self._ensure_index(subcls, parent_cls):
                    continue
                score = float(self._patchcore.get_max_anomaly_score(feats, self._indices[subcls]))
                scores[subcls] = score
                if score <= self._thresholds[subcls]:
                    normal_from.append(subcls)
        
        status = "NORMAL" if normal_from else "ANOMALY"
        return status, scores, normal_from
    
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
        
        Returns: Preview image with bboxes
        """
        classes = list(class_names or self.config.compare_classes)
        self._last_frame = frame.copy()
        
        # Switch to segmentation mode after first detection
        if not self._first_detection_done:
            print(f"[MODE] Using {self._detector.current_mode} mode for initial crop")
        elif self._first_detection_done and self._detector.use_detection and self._detector.has_separate_det_model:
            self._detector.switch_to_segmentation()
        
        crops, infos = self._detector.detect_and_crop(frame)
        
        # Mark first detection as done if we got any crops
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

            status, scores, normal_from = self._classify_single(square, classes)
            
            if tid >= 0:
                frame_crops[tid] = square
            frame_results.append({
                "id": tid,
                "bbox": bbox,
                "conf": conf,
                "status": status,
                "class_scores": scores,
                "normal_from": normal_from,
            })

        # Dedup: keep highest confidence per track_id
        best: Dict[int, Dict[str, Any]] = {}
        for r in frame_results:
            tid = r["id"]
            if tid < 0:
                continue
            if tid not in best or r["conf"] > best[tid]["conf"]:
                print(f"Best for track_id {tid}: conf={r['conf']:.4f}, status={r['status']}")
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
