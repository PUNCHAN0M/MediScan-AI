"""
YOLO Segmentation Module
========================
Pill detection + instance segmentation using YOLOv12-seg.

Supports:
    .pt   — PyTorch (ultralytics native)
    .onnx — ONNX Runtime (ultralytics auto-handles)

Classes:
    PillDetection  — single detection result (bbox + mask + conf)
    YOLOSegmentor  — detector + mask cropper + centroid tracker
"""
from __future__ import annotations

import cv2
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────
@dataclass
class PillDetection:
    """Single detected pill."""
    bbox: Tuple[int, int, int, int]       # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None     # binary mask (H, W) uint8, same as frame
    conf: float = 0.0
    class_id: int = 0


# ─────────────────────────────────────────────────────────
#  YOLOSegmentor
# ─────────────────────────────────────────────────────────
class YOLOSegmentor:
    """
    YOLO-based pill detection + instance segmentation.

    Pipeline:
        frame → YOLO → bboxes + masks → per-pill crop → (optional) tracking

    Supports .pt and .onnx through *ultralytics*.
    For .onnx the package uses ``onnxruntime`` under the hood —
    make sure it is installed (``pip install onnxruntime-gpu``).
    """

    SUPPORTED_EXTS = {".pt", ".onnx", ".engine", ".torchscript"}

    def __init__(
        self,
        model_path: str,
        img_size: int = 512,
        conf: float = 0.5,
        iou: float = 0.6,
        device: str = "cuda",
        # tracking
        enable_tracking: bool = False,
        track_max_distance: float = 80.0,
        track_iou_threshold: float = 0.80,
        track_max_age: int = 10,
    ):
        suffix = Path(model_path).suffix.lower()
        if suffix not in self.SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported YOLO format '{suffix}'. "
                f"Use one of {self.SUPPORTED_EXTS}"
            )

        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.device = device

        # tracking state
        self._tracking_enabled = enable_tracking
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 1
        self._max_dist = track_max_distance
        self._iou_thr = track_iou_threshold
        self._max_age = track_max_age

        tag = "ONNX" if suffix == ".onnx" else "PyTorch"
        print(f"[YOLOSegmentor] {model_path} ({tag}) | "
              f"img={img_size} conf={conf} iou={iou}")

    # ─────────────────────────────────────────────────────
    #  Detection
    # ─────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[PillDetection]:
        """
        Run YOLO inference on a single frame.

        Returns list of ``PillDetection`` (bbox + mask + conf).
        """
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections: List[PillDetection] = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            masks_data = None
            if r.masks is not None:
                masks_data = r.masks.data.cpu().numpy()

            fh, fw = frame.shape[:2]

            for i, (box, c, cid) in enumerate(zip(boxes, confs, cls_ids)):
                x1, y1, x2, y2 = box
                mask = None
                if masks_data is not None and i < len(masks_data):
                    m = masks_data[i]
                    mask = cv2.resize(m, (fw, fh))
                    mask = (mask > 0.5).astype(np.uint8)

                detections.append(PillDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    mask=mask,
                    conf=float(c),
                    class_id=int(cid),
                ))

        return detections

    # ─────────────────────────────────────────────────────
    #  Crop one pill
    # ─────────────────────────────────────────────────────
    @staticmethod
    def crop_pill(
        frame: np.ndarray,
        det: PillDetection,
        target_size: int = 256,
        bg_value: int = 0,
        pad: int = 5,
    ) -> np.ndarray:
        """
        Mask-crop a single pill.

        Steps:
            1. ``pill = frame * mask`` (background → *bg_value*)
            2. crop by bbox (with *pad*)
            3. pad to square
            4. resize to *target_size*

        Returns:
            (target_size, target_size, 3) uint8
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox

        # pad bbox
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(fw, x2 + pad)
        y2 = min(fh, y2 + pad)

        # apply mask with pill_collect.py style preprocessing + alpha blend
        if det.mask is not None:
            m_u8 = (det.mask * 255).astype(np.uint8) if det.mask.max() <= 1 else det.mask.astype(np.uint8)
            _morph_k = np.ones((5, 5), np.uint8)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, _morph_k)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, _morph_k)
            m_u8 = cv2.GaussianBlur(m_u8, (7, 7), 0)
            _, m_u8 = cv2.threshold(m_u8, 128, 255, cv2.THRESH_BINARY)
            alpha = m_u8.astype(np.float32) / 255.0
            masked = (frame.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
        else:
            masked = frame

        # bbox crop
        crop = masked[y1:y2, x1:x2]
        if crop.size == 0:
            return np.full((target_size, target_size, 3), bg_value, np.uint8)

        # square pad
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        square = np.full((side, side, 3), bg_value, np.uint8)
        sy, sx = (side - ch) // 2, (side - cw) // 2
        square[sy:sy + ch, sx:sx + cw] = crop

        # resize
        if side != target_size:
            square = cv2.resize(
                square, (target_size, target_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        return square

    # ─────────────────────────────────────────────────────
    #  Detect + Crop  (convenience)
    # ─────────────────────────────────────────────────────
    def detect_and_crop(
        self,
        frame: np.ndarray,
        target_size: int = 256,
        bg_value: int = 0,
        pad: int = 5,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Detect pills → crop each one → (optionally) track.

        Returns:
            crops : list of ``(target_size, target_size, 3)`` images
            infos : list of dicts with keys
                    ``bbox, padded_region, conf, class_id, center, track_id``
        """
        detections = self.detect(frame)

        crops: List[np.ndarray] = []
        infos: List[Dict[str, Any]] = []

        for det in detections:
            crop = self.crop_pill(frame, det, target_size, bg_value, pad)

            x1, y1, x2, y2 = det.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            infos.append({
                "bbox": det.bbox,
                "padded_region": det.bbox,
                "conf": det.conf,
                "class_id": det.class_id,
                "center": (cx, cy),
                "track_id": -1,
            })
            crops.append(crop)

        if self._tracking_enabled:
            self._assign_tracks(infos)

        return crops, infos

    # ─────────────────────────────────────────────────────
    #  Centroid + IoU Tracker
    # ─────────────────────────────────────────────────────
    def _assign_tracks(self, infos: List[Dict[str, Any]]) -> None:
        """
        Greedy centroid + IoU tracker.

        Priority:
            0 — IoU ≥ threshold  (shape overlap)
            1 — distance ≤ max   (position proximity)
        Sort candidates by ``(priority, distance, -iou)`` then greedily assign.
        """
        if not infos:
            self._age_tracks()
            return

        # build candidates
        candidates: list = []
        for di, info in enumerate(infos):
            for tid, track in self._tracks.items():
                dist = math.hypot(
                    info["center"][0] - track["center"][0],
                    info["center"][1] - track["center"][1],
                )
                iou = self._iou(info["bbox"], track["bbox"])

                if iou >= self._iou_thr:
                    candidates.append((0, dist, -iou, di, tid))
                elif dist <= self._max_dist:
                    candidates.append((1, dist, -iou, di, tid))

        candidates.sort()

        used_det: set = set()
        used_trk: set = set()

        for _, _, _, di, tid in candidates:
            if di in used_det or tid in used_trk:
                continue
            # match
            infos[di]["track_id"] = tid
            self._tracks[tid]["bbox"] = infos[di]["bbox"]
            self._tracks[tid]["center"] = infos[di]["center"]
            self._tracks[tid]["age"] = 0
            used_det.add(di)
            used_trk.add(tid)

        # new tracks for unmatched detections
        for di in range(len(infos)):
            if di not in used_det:
                tid = self._next_id
                self._next_id += 1
                infos[di]["track_id"] = tid
                self._tracks[tid] = {
                    "bbox": infos[di]["bbox"],
                    "center": infos[di]["center"],
                    "age": 0,
                }

        # age unmatched tracks
        self._age_tracks(used_trk)

    def _age_tracks(self, keep: Optional[set] = None) -> None:
        keep = keep or set()
        for tid in list(self._tracks):
            if tid not in keep:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self._max_age:
                    del self._tracks[tid]

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    def reset_tracking(self) -> None:
        """Clear all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_id = 1
