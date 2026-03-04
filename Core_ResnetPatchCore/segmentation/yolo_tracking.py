"""
YOLOTracking — Unified YOLO Segmentation + Tracking + Preprocessing
====================================================================

Single class that handles **everything** from raw frame → cleaned crops:

    frame → YOLO-seg inference (.pt / .onnx, GPU-first)
          → manual conf & area filter (ONNX-safe)
          → mask morphology + alpha-blend crop
          → square pad → resize to target_size
          → centroid + IoU tracking
          → (optional) grid preview for logging

Designed for:
    • ``Core_ResnetPatchCore`` — realtime & folder-based pill inspection
    • ``pill_collect.py``      — pill image collection / dataset building

Usage
-----
::

    from Core_ResnetPatchCore.segmentation.yolo_tracking import YOLOTracking

    yolo = YOLOTracking(
        model_path="model/SEGMENTATION/pill-detection-best-2.pt",
        conf=0.5, iou=0.6, device="cuda",
        enable_tracking=True,
    )

    # Per-frame (realtime)
    crops, infos, preview = yolo.process_frame(frame, draw=True, grid=True)

    # Single image (batch / predict)
    crops, infos = yolo.detect_and_crop(image)
"""
from __future__ import annotations

import cv2
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────
#  Data class
# ─────────────────────────────────────────────────────────
@dataclass
class PillDetection:
    """Single detected pill instance."""
    bbox:     Tuple[int, int, int, int]          # x1, y1, x2, y2
    mask:     Optional[np.ndarray] = None        # binary mask (H, W) uint8
    conf:     float = 0.0
    class_id: int   = 0


# ═════════════════════════════════════════════════════════════
#  YOLOTracking
# ═════════════════════════════════════════════════════════════
class YOLOTracking:
    """
    All-in-one YOLO segmentation + preprocess + tracking.

    Parameters
    ----------
    model_path : str
        Path to ``.pt`` or ``.onnx`` YOLO segmentation model.
    img_size : int
        Inference resolution (longer side).  Default ``512``.
    conf : float
        Confidence threshold.  Applied *after* inference for ONNX safety.
    iou : float
        NMS IoU threshold.
    device : str
        ``"cuda"`` or ``"cpu"``.  For ``.onnx`` the ONNX EP is chosen
        automatically by ``onnxruntime``; ``device`` is **still** passed to
        ``ultralytics`` so that ``.pt`` models run on the right device.
    target_size : int
        Output crop size (``target_size × target_size``).
    pad : int
        Extra pixels around bbox when cropping.
    bg_value : int
        Background fill value for the square crop (0 = black).
    retina_masks : bool
        Use high-resolution (retina) YOLO masks — slower but more accurate
        contour.  Set ``False`` for maximum speed.
    min_box_area_frac / max_box_area_frac : float
        Reject detections whose area (as a fraction of the full frame)
        falls outside ``[min, max]``.
    enable_tracking : bool
        Centroid + IoU tracker across consecutive frames.
    track_max_distance / track_iou_threshold / track_max_age
        Tracker hyper-parameters.
    """

    SUPPORTED_EXTS = {".pt", ".onnx", ".engine", ".torchscript"}

    # ─────────────────── init ───────────────────────────
    def __init__(
        self,
        model_path: str,
        img_size: int = 512,
        conf: float = 0.5,
        iou: float = 0.6,
        device: str = "cuda",
        # crop / preprocess
        target_size: int = 256,
        pad: int = 5,
        bg_value: int = 0,
        retina_masks: bool = True,
        # area filter
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

        self._is_onnx = suffix == ".onnx"
        if self._is_onnx:
            self.model = YOLO(str(model_path), task="segment")
        else:
            self.model = YOLO(str(model_path)).to(device)

        # Force overrides (best-effort — ONNX may ignore these)
        self.model.overrides["conf"] = conf
        self.model.overrides["iou"]  = iou

        self.img_size          = img_size
        self.conf              = conf
        self.iou               = iou
        self.device            = device
        self.target_size       = target_size
        self.pad               = pad
        self.bg_value          = bg_value
        self.retina_masks      = retina_masks
        self.min_box_area_frac = min_box_area_frac
        self.max_box_area_frac = max_box_area_frac

        # ── morphology kernel (shared, allocated once) ──
        self._morph_k = np.ones((5, 5), np.uint8)

        # ── tracking state ──
        self._tracking_enabled = enable_tracking
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 1
        self._max_dist  = track_max_distance
        self._iou_thr   = track_iou_threshold
        self._max_age   = track_max_age

        # ── debug (first N frames) ──
        self._debug_frames_left: int = 3

        tag = "ONNX" if self._is_onnx else "PyTorch"
        print(
            f"[YOLOTracking] {model_path} ({tag}) | "
            f"img={img_size} conf={conf} iou={iou} "
            f"crop={target_size} pad={pad} retina={retina_masks} "
            f"area=[{min_box_area_frac:.3f}, {max_box_area_frac:.2f}]"
        )

    # ═════════════════════════════════════════════════════════
    #  1. DETECT  — raw YOLO → filtered PillDetection list
    # ═════════════════════════════════════════════════════════
    def detect(self, frame: np.ndarray) -> List[PillDetection]:
        """
        Run YOLO-seg on a single BGR frame.

        Returns a **filtered** list of ``PillDetection`` — confidence and
        box-area filters are applied here so callers never see noise.
        """
        # ONNX: send device=None so onnxruntime picks its own EP
        infer_device = None if self._is_onnx else self.device

        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=infer_device,
            retina_masks=self.retina_masks,
            verbose=False,
            task="segment" if self._is_onnx else None,
        )

        fh, fw = frame.shape[:2]
        frame_area = fh * fw
        detections: List[PillDetection] = []

        for r in results:
            if r.boxes is None:
                continue

            boxes   = r.boxes.xyxy.cpu().numpy().astype(int)
            confs   = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            masks_data: Optional[np.ndarray] = None
            if r.masks is not None:
                masks_data = r.masks.data.cpu().numpy()

            # ── DEBUG (first N frames) ──────────────────────────
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
            # ────────────────────────────────────────────────────

            # ── 1. Conf filter (ONNX-safe — always applied) ────
            keep = confs >= self.conf
            boxes     = boxes[keep]
            confs     = confs[keep]
            cls_ids   = cls_ids[keep]
            if masks_data is not None:
                masks_data = masks_data[keep]

            # ── 2. Box-area filter ─────────────────────────────
            if len(boxes):
                w = (boxes[:, 2] - boxes[:, 0]).clip(0)
                h = (boxes[:, 3] - boxes[:, 1]).clip(0)
                frac = (w * h).astype(float) / frame_area
                ok = (frac >= self.min_box_area_frac) & (frac <= self.max_box_area_frac)
                boxes     = boxes[ok]
                confs     = confs[ok]
                cls_ids   = cls_ids[ok]
                if masks_data is not None:
                    masks_data = masks_data[ok]
            # ────────────────────────────────────────────────────

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

    # ═════════════════════════════════════════════════════════
    #  2. PREPROCESS MASK  (pill_collect.py compatible)
    # ═════════════════════════════════════════════════════════
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean a raw binary mask via morphology + Gaussian blur + threshold.
        Same pipeline as ``pill_collect.py::PillProcessor._preprocess_mask``.

        Input/output: uint8, values 0 or 255.
        """
        m = mask
        if m.max() <= 1:
            m = (m * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._morph_k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  self._morph_k)
        m = cv2.GaussianBlur(m, (7, 7), 0)
        _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
        return m

    # ═════════════════════════════════════════════════════════
    #  3. CROP ONE PILL  (mask-crop → square pad → resize)
    # ═════════════════════════════════════════════════════════
    def crop_pill(
        self,
        frame: np.ndarray,
        det: PillDetection,
    ) -> np.ndarray:
        """
        Crop + preprocess a single pill.

        Steps (same logic as ``pill_collect.py``):
            1. ``mask → morphology → clean``
            2. ``pill = frame × alpha``  (background → bg_value)
            3. bbox crop with ``pad``
            4. square pad (centred)
            5. resize → ``target_size × target_size``

        Returns
        -------
        np.ndarray
            ``(target_size, target_size, 3)`` uint8 BGR.
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox
        pad = self.pad
        ts  = self.target_size
        bg  = self.bg_value

        # pad bbox
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(fw, x2 + pad)
        y2 = min(fh, y2 + pad)

        # mask → alpha blend
        if det.mask is not None:
            clean = self.preprocess_mask(det.mask)
            alpha = clean.astype(np.float32) / 255.0
            masked = (frame.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
        else:
            masked = frame

        # bbox crop
        crop = masked[y1:y2, x1:x2]
        if crop.size == 0:
            return np.full((ts, ts, 3), bg, np.uint8)

        # square pad (centred, black)
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        square = np.full((side, side, 3), bg, np.uint8)
        sy = (side - ch) // 2
        sx = (side - cw) // 2
        square[sy:sy + ch, sx:sx + cw] = crop

        # resize
        if side != ts:
            square = cv2.resize(
                square, (ts, ts),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return square

    # ═════════════════════════════════════════════════════════
    #  4. DETECT + CROP  (convenience — one call)
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

        Returns
        -------
        crops : list of ``(H, W, 3)`` uint8
        infos : list of dicts::

            {
                "bbox":          (x1, y1, x2, y2),
                "padded_region": (x1, y1, x2, y2),
                "conf":          float,
                "class_id":      int,
                "center":        (cx, cy),
                "track_id":      int,     # -1 if tracking disabled
            }
        """
        # Allow per-call overrides (original values restored after)
        _orig_ts  = self.target_size
        _orig_bg  = self.bg_value
        _orig_pad = self.pad
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
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            infos.append({
                "bbox":          det.bbox,
                "padded_region": det.bbox,
                "conf":          det.conf,
                "class_id":      det.class_id,
                "center":        (cx, cy),
                "track_id":      -1,
            })
            crops.append(crop)

        if self._tracking_enabled:
            self._assign_tracks(infos)

        # restore
        self.target_size = _orig_ts
        self.bg_value    = _orig_bg
        self.pad         = _orig_pad

        return crops, infos

    # ═════════════════════════════════════════════════════════
    #  5. PROCESS FRAME  (full pipeline — for realtime & pill_collect)
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
        """
        End-to-end single-frame pipeline.

        Parameters
        ----------
        frame : np.ndarray
            BGR input image.
        draw : bool
            Annotate ``frame`` with bboxes + conf labels.
        grid : bool
            Build a crop preview grid (for logging / UI).
        grid_max_cols / grid_cell_size
            Grid layout parameters.

        Returns
        -------
        crops   : list of ``(target_size, target_size, 3)`` uint8
        infos   : list of dicts  (see ``detect_and_crop``)
        preview : annotated frame (if ``draw=True``) **or** ``None``
        """
        crops, infos = self.detect_and_crop(frame)

        preview: Optional[np.ndarray] = None
        if draw:
            preview = frame.copy()
            for info in infos:
                x1, y1, x2, y2 = info["bbox"]
                tid  = info["track_id"]
                conf = info["conf"]

                # colour: green=tracked, yellow=untracked
                colour = (0, 255, 0) if tid >= 0 else (0, 255, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), colour, 2)

                # label
                label = f"{conf:.2f}"
                if tid >= 0:
                    label = f"ID:{tid} {label}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(preview, (x1, y1 - th - 6),
                              (x1 + tw + 4, y1), colour, -1)
                cv2.putText(preview, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)

        # optional grid
        if grid and crops:
            grid_img = self.build_grid(
                crops, infos,
                cell_size=grid_cell_size,
                max_cols=grid_max_cols,
            )
            if preview is not None and grid_img is not None:
                # side-by-side: resize grid height to match preview
                gh, gw = grid_img.shape[:2]
                ph = preview.shape[0]
                if gh != ph:
                    scale = ph / gh
                    grid_img = cv2.resize(
                        grid_img,
                        (int(gw * scale), ph),
                        interpolation=cv2.INTER_LINEAR,
                    )
                preview = np.hstack([preview, grid_img])

        return crops, infos, preview

    # ═════════════════════════════════════════════════════════
    #  6. GRID PREVIEW  (for logging / debugging)
    # ═════════════════════════════════════════════════════════
    @staticmethod
    def build_grid(
        crops: List[np.ndarray],
        infos: Optional[List[Dict[str, Any]]] = None,
        cell_size: int = 128,
        max_cols: int = 5,
    ) -> Optional[np.ndarray]:
        """
        Arrange crop images in a labelled grid.

        Parameters
        ----------
        crops : list of images
        infos : if given, ID + conf labels are drawn in each cell
        cell_size : pixel size per cell
        max_cols : columns before wrapping

        Returns
        -------
        np.ndarray or None
        """
        if not crops:
            return None

        n    = len(crops)
        cols = min(n, max_cols)
        rows = (n + cols - 1) // cols
        lh   = 20                       # label bar height

        grid = np.zeros(
            (rows * (cell_size + lh), cols * cell_size, 3), dtype=np.uint8)

        for idx, crop in enumerate(crops):
            r = idx // cols
            c = idx % cols
            y = r * (cell_size + lh)
            x = c * cell_size

            cell = crop[:, :, :3] if crop.ndim == 3 and crop.shape[2] == 4 else crop
            cell = cv2.resize(cell, (cell_size, cell_size),
                              interpolation=cv2.INTER_LANCZOS4)
            grid[y + lh : y + lh + cell_size, x : x + cell_size] = cell

            # label
            if infos and idx < len(infos):
                tid  = infos[idx].get("track_id", -1)
                conf = infos[idx].get("conf", 0.0)
                if tid >= 0:
                    txt = f"ID:{tid} {conf:.2f}"
                else:
                    txt = f"#{idx} {conf:.2f}"
            else:
                txt = f"#{idx}"
            cv2.putText(grid, txt, (x + 4, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (200, 200, 200), 1, cv2.LINE_AA)

        return grid

    # ═════════════════════════════════════════════════════════
    #  7. TRACKING  (centroid + IoU greedy matcher)
    # ═════════════════════════════════════════════════════════
    def _assign_tracks(self, infos: List[Dict[str, Any]]) -> None:
        """
        Greedy centroid + IoU tracker.

        Priority:
            0 — IoU ≥ threshold  (shape overlap)
            1 — distance ≤ max   (position proximity)
        Sort by ``(priority, distance, -iou)`` then greedily assign.
        """
        if not infos:
            self._age_tracks()
            return

        candidates: list = []
        for di, info in enumerate(infos):
            for tid, track in self._tracks.items():
                dist = math.hypot(
                    info["center"][0] - track["center"][0],
                    info["center"][1] - track["center"][1],
                )
                iou = self._iou_score(info["bbox"], track["bbox"])

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
            infos[di]["track_id"] = tid
            self._tracks[tid]["bbox"]   = infos[di]["bbox"]
            self._tracks[tid]["center"] = infos[di]["center"]
            self._tracks[tid]["age"]    = 0
            used_det.add(di)
            used_trk.add(tid)

        # new tracks for unmatched detections
        for di in range(len(infos)):
            if di not in used_det:
                tid = self._next_id
                self._next_id += 1
                infos[di]["track_id"] = tid
                self._tracks[tid] = {
                    "bbox":   infos[di]["bbox"],
                    "center": infos[di]["center"],
                    "age":    0,
                }

        self._age_tracks(used_trk)

    def _age_tracks(self, keep: Optional[set] = None) -> None:
        keep = keep or set()
        for tid in list(self._tracks):
            if tid not in keep:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self._max_age:
                    del self._tracks[tid]

    @staticmethod
    def _iou_score(a: tuple, b: tuple) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter)

    # ─────────────────── public helpers ─────────────────────
    def reset_tracking(self) -> None:
        """Clear all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def tracks(self) -> Dict[int, Dict[str, Any]]:
        """Read-only snapshot of current tracks."""
        return dict(self._tracks)

    @property
    def is_onnx(self) -> bool:
        return self._is_onnx
