# detection/yolo_detector.py
"""
YOLO Segmentation Detector + Pill Cropper
==========================================
Wraps ultralytics YOLO → detect pills → preprocess masks → crop.
Supports: .pt (FP16 auto) / .onnx / .engine (TensorRT) / .torchscript
No pipeline logic. No tracking (see tracker.py).

Performance notes:
    - FP16 auto for .pt on CUDA
    - Region-only mask crop (no full-frame alpha blend)
    - Vectorised filter (conf + class + area in one pass)
    - INTER_LINEAR for crop resize (~3× faster than LANCZOS4)
    - Minimal allocations in crop_pill hot path
"""
from __future__ import annotations

import cv2
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from detection.tracker import CentroidIoUTracker


# ─────────────────────────────────────────────────────────
#  Data class
# ─────────────────────────────────────────────────────────
@dataclass
class PillDetection:
    """Single detected pill instance."""
    bbox:     Tuple[int, int, int, int]
    mask:     Optional[np.ndarray] = None        # uint8 0/255
    conf:     float = 0.0
    class_id: int   = 0


# ─────────────────────────────────────────────────────────
#  YOLO Detector
# ─────────────────────────────────────────────────────────
class YOLODetector:
    """
    YOLO-seg detector + mask preprocess + crop.

    Optimizations:
        - FP16 auto for ``.pt`` on CUDA  (``model.model.half()``)
        - TensorRT ``.engine`` handled without ``.to(device)``
        - Region-only mask crop (no full-frame alpha blend)
        - Optional class_id filter
        - Vectorised multi-criterion filtering
        - INTER_LINEAR resize in crop (faster than LANCZOS4)

    Optionally delegates tracking to ``CentroidIoUTracker``.
    """

    SUPPORTED_EXTS = {".pt", ".onnx", ".engine", ".torchscript"}

    def __init__(
        self,
        model_path: str,
        mode: str = "segment",
        img_size: int = 512,
        conf: float = 0.5,
        iou: float = 0.6,
        device: str = "cuda",
        # crop
        target_size: int = 256,
        pad: int = 5,
        bg_value: int = 0,
        retina_masks: bool = True,
        # filter
        class_filter: Optional[int] = None,
        min_box_area_frac: float = 0.001,
        max_box_area_frac: float = 0.90,
        # tracking
        enable_tracking: bool = False,
        track_max_distance: float = 80.0,
        track_iou_threshold: float = 0.80,
        track_max_age: int = 10,
    ) -> None:
        suffix = Path(model_path).suffix.lower()
        if suffix not in self.SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported YOLO format '{suffix}'. "
                f"Use one of {self.SUPPORTED_EXTS}"
            )

        from ultralytics import YOLO

        self._is_onnx   = suffix == ".onnx"
        self._is_engine  = suffix == ".engine"
        self.mode        = mode

        # ── Load model per format ──
        if self._is_onnx:
            self.model = YOLO(str(model_path), task=mode)
        elif self._is_engine:
            # TensorRT: don't call .to(device), runtime picks device
            self.model = YOLO(str(model_path))
        else:
            self.model = YOLO(str(model_path)).to(device)

        # ── FP16 auto: only .pt on CUDA ──
        self.use_half = (
            device == "cuda"
            and not self._is_onnx
            and not self._is_engine
        )
        if self.use_half:
            self.model.model.half()

        self.model.overrides["conf"] = conf
        self.model.overrides["iou"] = iou

        self.img_size          = img_size
        self.conf              = conf
        self.iou               = iou
        self.device            = device
        self.target_size       = target_size
        self.pad               = pad
        self.bg_value          = bg_value
        self.retina_masks      = retina_masks
        self.class_filter      = class_filter
        self.min_box_area_frac = min_box_area_frac
        self.max_box_area_frac = max_box_area_frac

        self._morph_k = np.ones((5, 5), np.uint8)

        # ── CUDA optimizations ──
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        # ── optional tracker ──
        self._tracker: Optional[CentroidIoUTracker] = None
        if enable_tracking:
            self._tracker = CentroidIoUTracker(
                max_distance=track_max_distance,
                iou_threshold=track_iou_threshold,
                max_age=track_max_age,
            )

        self._debug_frames_left: int = 3

        tag = "ONNX" if self._is_onnx else ("TensorRT" if self._is_engine else "PyTorch")
        fp16 = " FP16" if self.use_half else ""
        print(
            f"[YOLODetector] {model_path} ({tag}{fp16}) | "
            f"img={img_size} conf={conf} iou={iou} "
            f"crop={target_size} pad={pad} retina={retina_masks}"
        )

    # ═════════════════════════════════════════════════════════
    #  1. DETECT
    # ═════════════════════════════════════════════════════════
    def detect(self, frame: np.ndarray) -> List[PillDetection]:
        """Run YOLO-seg on a single BGR frame. Returns filtered detections."""
        # ONNX / TensorRT: let runtime pick device
        infer_device = None if (self._is_onnx or self._is_engine) else self.device

        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=infer_device,
            half=self.use_half,
            retina_masks=self.retina_masks if self.mode == "segment" else False,
            verbose=False,
            task=self.mode if self._is_onnx else None,
        )

        fh, fw = frame.shape[:2]
        frame_area = fh * fw
        detections: List[PillDetection] = []

        for r in results:
            if r.boxes is None:
                continue

            # ── Single .cpu().numpy() call per tensor ──
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(np.int32)

            masks_data: Optional[np.ndarray] = None
            if self.mode == "segment" and r.masks is not None:
                masks_data = r.masks.data.cpu().numpy()

            # debug first N frames
            if self._debug_frames_left > 0:
                self._debug_frames_left -= 1
                if len(confs):
                    print(
                        f"[YOLO DEBUG] raw={len(confs)} "
                        f"conf min={confs.min():.3f} max={confs.max():.3f} "
                        f"mean={confs.mean():.3f}  (thr={self.conf})"
                    )
                else:
                    print("[YOLO DEBUG] raw=0")

            # ── Vectorised multi-criterion filter (one pass) ──
            if len(confs) == 0:
                continue

            keep = confs >= self.conf

            if self.class_filter is not None:
                keep &= cls_ids == self.class_filter

            w = (boxes[:, 2] - boxes[:, 0]).clip(0)
            h = (boxes[:, 3] - boxes[:, 1]).clip(0)
            frac = (w * h).astype(np.float32) / frame_area
            keep &= (frac >= self.min_box_area_frac) & (frac <= self.max_box_area_frac)

            # Apply combined mask
            boxes = boxes[keep]
            confs = confs[keep]
            cls_ids = cls_ids[keep]
            if masks_data is not None:
                masks_data = masks_data[keep]

            for i, (box, c, cid) in enumerate(zip(boxes, confs, cls_ids)):
                x1, y1, x2, y2 = box
                mask = None
                if masks_data is not None and i < len(masks_data):
                    m = masks_data[i]
                    mask = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_LINEAR)
                    mask = (mask > 0.5).astype(np.uint8) * 255

                detections.append(PillDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    mask=mask,
                    conf=float(c),
                    class_id=int(cid),
                ))

        return detections

    # ═════════════════════════════════════════════════════════
    #  2. PREPROCESS MASK
    # ═════════════════════════════════════════════════════════
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean a raw binary mask via morphology. Input/output: uint8 0/255."""
        m = mask
        if m.max() <= 1:
            m = (m * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._morph_k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._morph_k)
        m = cv2.GaussianBlur(m, (7, 7), 0)
        _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
        return m

    # ═════════════════════════════════════════════════════════
    #  3. CROP ONE PILL  (region-only mask — fast)
    # ═════════════════════════════════════════════════════════
    def crop_pill(self, frame: np.ndarray, det: PillDetection) -> np.ndarray:
        """
        Crop + preprocess a single pill.

        Mask is applied ONLY on the crop region (not full frame)
        for significantly lower memory and compute cost.
        Uses INTER_LINEAR for ~3× faster resize than LANCZOS4.
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox
        pad, ts, bg = self.pad, self.target_size, self.bg_value

        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(fw, x2 + pad)
        y2p = min(fh, y2 + pad)

        # bbox crop first (small region) — view, no copy yet
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return np.full((ts, ts, 3), bg, np.uint8)

        # apply mask ONLY on the crop region
        if det.mask is not None:
            mask_crop = det.mask[y1p:y2p, x1p:x2p]
            clean = self.preprocess_mask(mask_crop)
            crop = crop.copy()          # copy only when mask present
            crop[clean == 0] = bg

        # square pad (centred)
        ch, cw = crop.shape[:2]
        side = max(ch, cw)

        if side == ts and ch == cw:
            # Perfect fit — no pad or resize needed
            return crop if det.mask is not None else crop.copy()

        square = np.empty((side, side, 3), np.uint8)
        square[:] = bg
        sy, sx = (side - ch) // 2, (side - cw) // 2
        square[sy:sy + ch, sx:sx + cw] = crop

        if side != ts:
            square = cv2.resize(square, (ts, ts), interpolation=cv2.INTER_LINEAR)
        return square

    # ═════════════════════════════════════════════════════════
    #  4. DETECT + CROP
    # ═════════════════════════════════════════════════════════
    def detect_and_crop(
        self,
        frame: np.ndarray,
        target_size: Optional[int] = None,
        bg_value: Optional[int] = None,
        pad: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Detect → crop → (optionally) track.

        Returns (crops, infos) where each info dict has:
            bbox, padded_region, conf, class_id, center, track_id
        """
        _orig = (self.target_size, self.bg_value, self.pad)
        if target_size is not None:
            self.target_size = target_size
        if bg_value is not None:
            self.bg_value = bg_value
        if pad is not None:
            self.pad = pad

        detections = self.detect(frame)

        crops: List[np.ndarray] = []
        infos: List[Dict[str, Any]] = []

        for det in detections:
            crop = self.crop_pill(frame, det)
            x1, y1, x2, y2 = det.bbox
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

            infos.append({
                "bbox": det.bbox,
                "padded_region": det.bbox,
                "conf": det.conf,
                "class_id": det.class_id,
                "center": (cx, cy),
                "track_id": -1,
            })
            crops.append(crop)

        if self._tracker is not None:
            self._tracker.update(infos)

        self.target_size, self.bg_value, self.pad = _orig
        return crops, infos

    # ═════════════════════════════════════════════════════════
    #  5. PROCESS FRAME (convenience for realtime / UI)
    # ═════════════════════════════════════════════════════════
    def process_frame(
        self,
        frame: np.ndarray,
        *,
        draw: bool = True,
        grid: bool = False,
        grid_max_cols: int = 5,
        grid_cell_size: int = 128,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Optional[np.ndarray]]:
        crops, infos = self.detect_and_crop(frame)

        preview: Optional[np.ndarray] = None
        if draw:
            preview = frame.copy()
            for info in infos:
                x1, y1, x2, y2 = info["bbox"]
                tid, conf = info["track_id"], info["conf"]
                colour = (0, 255, 0) if tid >= 0 else (0, 255, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), colour, 2)
                label = f"ID:{tid} {conf:.2f}" if tid >= 0 else f"{conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(preview, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(preview, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if grid and crops:
            grid_img = self.build_grid(crops, infos, grid_cell_size, grid_max_cols)
            if preview is not None and grid_img is not None:
                gh, gw = grid_img.shape[:2]
                ph = preview.shape[0]
                if gh != ph:
                    scale = ph / gh
                    grid_img = cv2.resize(grid_img, (int(gw * scale), ph),
                                          interpolation=cv2.INTER_LINEAR)
                preview = np.hstack([preview, grid_img])

        return crops, infos, preview

    # ═════════════════════════════════════════════════════════
    #  6. GRID PREVIEW
    # ═════════════════════════════════════════════════════════
    @staticmethod
    def build_grid(
        crops: List[np.ndarray],
        infos: Optional[List[Dict[str, Any]]] = None,
        cell_size: int = 128,
        max_cols: int = 5,
    ) -> Optional[np.ndarray]:
        if not crops:
            return None

        n = len(crops)
        cols = min(n, max_cols)
        rows = (n + cols - 1) // cols
        lh = 20

        grid = np.zeros((rows * (cell_size + lh), cols * cell_size, 3), dtype=np.uint8)

        for idx, crop in enumerate(crops):
            r, c = idx // cols, idx % cols
            y, x = r * (cell_size + lh), c * cell_size

            cell = crop[:, :, :3] if crop.ndim == 3 and crop.shape[2] == 4 else crop
            cell = cv2.resize(cell, (cell_size, cell_size), interpolation=cv2.INTER_LANCZOS4)
            grid[y + lh:y + lh + cell_size, x:x + cell_size] = cell

            if infos and idx < len(infos):
                tid, conf = infos[idx].get("track_id", -1), infos[idx].get("conf", 0.0)
                txt = f"ID:{tid} {conf:.2f}" if tid >= 0 else f"#{idx} {conf:.2f}"
            else:
                txt = f"#{idx}"
            cv2.putText(grid, txt, (x + 4, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        return grid

    # ───────── Public helpers ─────────
    def reset_tracking(self) -> None:
        if self._tracker:
            self._tracker.reset()

    @property
    def tracks(self) -> Dict[int, Dict[str, Any]]:
        return self._tracker.tracks if self._tracker else {}

    @property
    def is_onnx(self) -> bool:
        return self._is_onnx
