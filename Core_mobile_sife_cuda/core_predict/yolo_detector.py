# core_predict/yolo_detector.py
"""
YOLO Pill Detector with Optimized Tracking.

Supports two model backends:
  - .pt  → Ultralytics YOLO (auto-selects CUDA via ultralytics)
  - .onnx → ONNX Runtime with CUDAExecutionProvider fallback to CPU

CUDA Optimized:
- Vectorized distance/IoU computation with numpy broadcasting
- Efficient NMS with pre-allocated arrays
- ONNX Runtime CUDAExecutionProvider for GPU-accelerated inference
- Ultralytics YOLO uses CUDA automatically

Responsibilities:
- Detect pills using YOLO segmentation/detection
- Track pills across frames (center distance + IoU matching)
- Crop pill regions with padding
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


# =========================================================
# TRACKING DATA
# =========================================================

@dataclass
class _Track:
    """State for a single tracked pill."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    age: int = 0


def _center_xy(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# =========================================================
# VECTORIZED GEOMETRY
# =========================================================

def _pairwise_distances(
    det_centers: np.ndarray, track_centers: np.ndarray
) -> np.ndarray:
    """
    Vectorized pairwise Euclidean distance.
    
    Args:
        det_centers: (N, 2) detection centers
        track_centers: (M, 2) track centers
    Returns:
        (N, M) distance matrix
    """
    diff = det_centers[:, None, :] - track_centers[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _pairwise_iou(
    det_boxes: np.ndarray, track_boxes: np.ndarray
) -> np.ndarray:
    """
    Vectorized pairwise IoU.
    
    Args:
        det_boxes: (N, 4) [x1,y1,x2,y2]
        track_boxes: (M, 4) [x1,y1,x2,y2]
    Returns:
        (N, M) IoU matrix
    """
    N = det_boxes.shape[0]
    M = track_boxes.shape[0]

    # Expand for broadcasting: (N, 1, 4) vs (1, M, 4)
    d = det_boxes[:, None, :]
    t = track_boxes[None, :, :]

    ix1 = np.maximum(d[:, :, 0], t[:, :, 0])
    iy1 = np.maximum(d[:, :, 1], t[:, :, 1])
    ix2 = np.minimum(d[:, :, 2], t[:, :, 2])
    iy2 = np.minimum(d[:, :, 3], t[:, :, 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)

    area_d = (d[:, :, 2] - d[:, :, 0]) * (d[:, :, 3] - d[:, :, 1])
    area_t = (t[:, :, 2] - t[:, :, 0]) * (t[:, :, 3] - t[:, :, 1])
    union = area_d + area_t - inter

    return inter / (union + 1e-6)


# =========================================================
# NMS
# =========================================================

def box_nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.7) -> np.ndarray:
    """Standard greedy box NMS."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)

        inds = np.where(iou <= thr)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def mask_nms(masks: np.ndarray, scores: np.ndarray, thr: float = 0.75) -> np.ndarray:
    """Mask-based NMS using boolean mask overlap."""
    masks = masks > 0.5
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        cur = masks[i]
        rest = masks[order[1:]]
        inter = (rest & cur).sum(axis=(1, 2))
        union = (rest | cur).sum(axis=(1, 2))
        iou = inter / (union + 1e-6)

        inds = np.where(iou <= thr)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


# =========================================================
# ONNX RUNTIME WRAPPER
# =========================================================

def _get_onnx_providers(device: Optional[str] = None) -> List[str]:
    """
    Select ONNX Runtime execution providers.

    Priority: CUDAExecutionProvider → CPUExecutionProvider.
    If *device* is explicitly ``"cpu"``, skip CUDA.
    """
    if device == "cpu" or not _ORT_AVAILABLE:
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    providers: List[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize + pad image to *new_shape* square while keeping aspect ratio.

    Returns:
        padded image, scale ratio, (pad_w, pad_h)
    """
    h, w = img.shape[:2]
    r = new_shape / max(h, w)
    new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))
    dw = (new_shape - new_unpad_w) / 2
    dh = (new_shape - new_unpad_h) / 2

    if (w, h) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (int(round(dw)), int(round(dh)))


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    y = np.empty_like(x)
    half_w = x[:, 2] / 2
    half_h = x[:, 3] / 2
    y[:, 0] = x[:, 0] - half_w
    y[:, 1] = x[:, 1] - half_h
    y[:, 2] = x[:, 0] + half_w
    y[:, 3] = x[:, 1] + half_h
    return y


def _nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float = 0.45,
) -> np.ndarray:
    """Greedy NMS on numpy arrays, returns kept indices."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return np.array(keep, dtype=int)


# --------------- result dataclasses (mimic ultralytics) ---------------

class _OnnxBoxes:
    """Lightweight wrapper that mirrors ``ultralytics.Results.boxes``."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: Optional[np.ndarray] = None,
    ):
        # Store as torch tensors so `.cpu().numpy()` works transparently
        self.xyxy = torch.from_numpy(xyxy.astype(np.float32))
        self.conf = torch.from_numpy(conf.astype(np.float32))
        if cls is not None:
            self.cls = torch.from_numpy(cls.astype(np.float32))
        else:
            self.cls = torch.zeros_like(self.conf)

    def __len__(self) -> int:
        return len(self.conf)


class _OnnxMasks:
    """Lightweight wrapper that mirrors ``ultralytics.Results.masks``."""

    def __init__(self, masks: np.ndarray):
        self.data = torch.from_numpy(masks.astype(np.float32))


class _OnnxResult:
    """Single-image result returned by :class:`OnnxYOLOModel`."""

    def __init__(
        self,
        boxes: Optional[_OnnxBoxes] = None,
        masks: Optional[_OnnxMasks] = None,
    ):
        self.boxes = boxes
        self.masks = masks


class OnnxYOLOModel:
    """
    Drop-in replacement for ``ultralytics.YOLO`` using ONNX Runtime.

    Supports both **detection** (.onnx with 1 output) and **segmentation**
    (.onnx with 2 outputs: detections + mask prototypes).

    ONNX output format (ultralytics export):
      Detection:
        output0  →  (1, 4 + num_classes, N)
      Segmentation:
        output0  →  (1, 4 + num_classes + 32, N)
        output1  →  (1, 32, mask_h, mask_w)   mask prototypes
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        default_imgsz: int = 640,
    ):
        if not _ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Install with: pip install onnxruntime-gpu  (or onnxruntime)"
            )

        self._path = model_path
        self._default_imgsz = default_imgsz
        providers = _get_onnx_providers(device)

        # --- Session options for speed ---
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Intra-op threads (per operator parallelism)
        sess_opts.intra_op_num_threads = 0          # 0 = let ORT decide
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers,
        )
        actual = self._session.get_providers()
        self._using_cuda = "CUDAExecutionProvider" in actual
        print(f"  [ONNX] Loaded: {model_path}")
        print(f"  [ONNX] Providers: {actual}")

        # Introspect model I/O — handle dynamic shapes
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        raw_shape = inp.shape  # e.g. ['batch', 3, 'height', 'width'] or [1, 3, 640, 640]

        self._channels = int(raw_shape[1]) if isinstance(raw_shape[1], int) else 3
        self._dynamic_hw = (
            not isinstance(raw_shape[2], int) or not isinstance(raw_shape[3], int)
        )

        if self._dynamic_hw:
            # Dynamic shape — use caller-provided imgsz each frame
            self._model_h = default_imgsz
            self._model_w = default_imgsz
            print(f"  [ONNX] Dynamic input shape detected — "
                  f"default size: {default_imgsz}×{default_imgsz}")
        else:
            self._model_h = int(raw_shape[2])
            self._model_w = int(raw_shape[3])

        # Detect FP16 model
        self._is_fp16 = (inp.type == "tensor(float16)")
        if self._is_fp16:
            print(f"  [ONNX] FP16 model detected — using half precision input")

        self._output_names = [o.name for o in self._session.get_outputs()]
        self._is_seg = len(self._output_names) >= 2  # seg model has ≥2 outputs

    # ------------------------------------------------------------------

    def __call__(
        self,
        frame: np.ndarray,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        retina_masks: bool = False,
        verbose: bool = False,
    ) -> List[_OnnxResult]:
        """Run inference on a single BGR frame. Returns ``[result]``."""

        orig_h, orig_w = frame.shape[:2]

        # Dynamic model → honour caller's imgsz; fixed model → use baked-in size
        if self._dynamic_hw:
            # Round to nearest multiple of 32 (YOLO stride requirement)
            img_size = max(32, (imgsz + 15) // 32 * 32)
        else:
            img_size = self._model_h

        # --- pre-process ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lb, ratio, (pad_w, pad_h) = _letterbox(img_rgb, img_size)
        blob = img_lb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        # FP16 model expects float16 input
        if self._is_fp16:
            blob = blob.astype(np.float16)

        # --- inference ---
        outputs = self._session.run(self._output_names, {self._input_name: blob})

        # --- post-process ---
        pred = outputs[0]  # (1, 4+nc[+32], N)
        pred = pred[0].T   # (N, 4+nc[+32])

        if self._is_seg:
            protos = outputs[1]  # (1, 32, mh, mw)
            return [self._postprocess_seg(
                pred, protos, ratio, pad_w, pad_h,
                orig_h, orig_w, img_size, conf, iou,
            )]
        else:
            return [self._postprocess_det(
                pred, ratio, pad_w, pad_h,
                orig_h, orig_w, img_size, conf, iou,
            )]

    # ------------------------------------------------------------------
    # Detection post-process
    # ------------------------------------------------------------------

    def _postprocess_det(
        self,
        pred: np.ndarray,        # (N, 4 + nc)
        ratio: float,
        pad_w: int, pad_h: int,
        orig_h: int, orig_w: int,
        img_size: int,
        conf_thr: float,
        iou_thr: float,
    ) -> _OnnxResult:
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]            # (N, nc)
        max_scores = class_scores.max(axis=1)  # (N,)
        class_ids = class_scores.argmax(axis=1).astype(np.float32)  # (N,)

        # confidence filter
        mask = max_scores >= conf_thr
        if not mask.any():
            return _OnnxResult(boxes=_OnnxBoxes(
                np.empty((0, 4)), np.empty((0,)), np.empty((0,)),
            ))

        boxes_xywh = boxes_xywh[mask]
        scores = max_scores[mask]
        cls_ids = class_ids[mask]
        boxes_xyxy = _xywh2xyxy(boxes_xywh)

        # NMS
        keep = _nms_numpy(boxes_xyxy, scores, iou_thr)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # rescale to original image
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / ratio
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        return _OnnxResult(boxes=_OnnxBoxes(boxes_xyxy, scores, cls_ids))

    # ------------------------------------------------------------------
    # Segmentation post-process
    # ------------------------------------------------------------------

    def _postprocess_seg(
        self,
        pred: np.ndarray,        # (N, 4 + nc + 32)
        protos: np.ndarray,      # (1, 32, mh, mw)
        ratio: float,
        pad_w: int, pad_h: int,
        orig_h: int, orig_w: int,
        img_size: int,
        conf_thr: float,
        iou_thr: float,
    ) -> _OnnxResult:
        boxes_xywh = pred[:, :4]
        nc = pred.shape[1] - 4 - 32  # num classes
        class_scores = pred[:, 4:4 + nc]
        mask_coeffs = pred[:, 4 + nc:]  # (N, 32)
        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1).astype(np.float32)  # (N,)

        # confidence filter
        mask = max_scores >= conf_thr
        if not mask.any():
            return _OnnxResult(
                boxes=_OnnxBoxes(np.empty((0, 4)), np.empty((0,)), np.empty((0,))),
                masks=None,
            )

        boxes_xywh = boxes_xywh[mask]
        scores = max_scores[mask]
        cls_ids = class_ids[mask]
        mask_coeffs = mask_coeffs[mask]
        boxes_xyxy = _xywh2xyxy(boxes_xywh)

        # NMS
        keep = _nms_numpy(boxes_xyxy, scores, iou_thr)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        # decode masks: coefficients @ prototypes → per-instance masks
        proto = protos[0]  # (32, mh, mw)
        mh, mw = proto.shape[1], proto.shape[2]
        # (K, 32) @ (32, mh*mw) → (K, mh*mw)
        masks_flat = mask_coeffs @ proto.reshape(32, -1)
        masks_sigmoid = 1.0 / (1.0 + np.exp(-masks_flat))  # sigmoid
        masks_full = masks_sigmoid.reshape(-1, mh, mw)     # (K, mh, mw)

        # crop masks to bbox region (in letterboxed space) for cleaner boundaries
        for i, box in enumerate(boxes_xyxy):
            # scale box to mask resolution
            mx1 = int(max(0, box[0] * mw / img_size))
            my1 = int(max(0, box[1] * mh / img_size))
            mx2 = int(min(mw, box[2] * mw / img_size))
            my2 = int(min(mh, box[3] * mh / img_size))
            crop_mask = np.zeros_like(masks_full[i])
            crop_mask[my1:my2, mx1:mx2] = masks_full[i, my1:my2, mx1:mx2]
            masks_full[i] = crop_mask

        # resize masks to original image size, removing letterbox padding
        masks_orig = np.zeros((len(masks_full), orig_h, orig_w), dtype=np.float32)
        for i, m in enumerate(masks_full):
            # resize to letterboxed image size
            m_resized = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            # remove padding
            unpad_h = int(round(orig_h * ratio))
            unpad_w = int(round(orig_w * ratio))
            m_cropped = m_resized[pad_h:pad_h + unpad_h, pad_w:pad_w + unpad_w]
            # resize to original
            masks_orig[i] = cv2.resize(m_cropped, (orig_w, orig_h),
                                       interpolation=cv2.INTER_LINEAR)

        # rescale boxes to original image
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / ratio
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        return _OnnxResult(
            boxes=_OnnxBoxes(boxes_xyxy, scores, cls_ids),
            masks=_OnnxMasks(masks_orig),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def using_cuda(self) -> bool:
        return self._using_cuda

    @property
    def is_segmentation(self) -> bool:
        return self._is_seg

    @property
    def is_dynamic(self) -> bool:
        return self._dynamic_hw

    @property
    def is_fp16(self) -> bool:
        return self._is_fp16

    def __repr__(self) -> str:
        mode = "seg" if self._is_seg else "det"
        prov = "CUDA" if self._using_cuda else "CPU"
        dyn = "dynamic" if self._dynamic_hw else f"{self._model_h}x{self._model_w}"
        prec = "fp16" if self._is_fp16 else "fp32"
        return f"OnnxYOLOModel({self._path}, {mode}, {prov}, {dyn}, {prec})"


# =========================================================
# MODEL LOADER HELPER
# =========================================================

def _load_yolo_model(
    model_path: str,
    device: Optional[str] = None,
    default_imgsz: int = 640,
) -> Any:
    """
    Load a YOLO model from *model_path*.

    - ``.pt``   → ``ultralytics.YOLO``
    - ``.onnx`` → ``OnnxYOLOModel`` with CUDA if available

    Args:
        model_path:    Path to model file (.pt or .onnx)
        device:        "cuda" / "cpu" / None (auto)
        default_imgsz: Fallback input size for dynamic-shape ONNX models
    """
    ext = Path(model_path).suffix.lower()
    if ext == ".onnx":
        return OnnxYOLOModel(model_path, device=device, default_imgsz=default_imgsz)
    else:
        # .pt / .torchscript / etc.  — ultralytics handles device internally
        return YOLO(model_path)

class PillYOLODetector:
    """
    YOLO pill detector with frame-to-frame tracking.

    Supports both ``.pt`` (ultralytics) and ``.onnx`` (ONNX Runtime) models.
    When ``.onnx`` is used, CUDAExecutionProvider is selected automatically
    if available, giving GPU-accelerated inference without ultralytics.

    Modes:
    - DETECTION: bbox-only cropping (faster, no masks)
    - SEGMENTATION: mask-based cropping (more accurate boundaries)

    Tracking uses vectorized center-distance + IoU matching.
    """

    def __init__(
        self,
        seg_model_path: str = "model/yolo26-segmentation.pt",
        det_model_path: str = None,
        img_size: int = 768,
        conf: float = 0.38,
        iou: float = 0.6,
        pad: int = 5,
        merge_dist_px: int = 60,
        enable_tracking: bool = True,
        track_max_distance: float = 80.0,
        track_iou_threshold: float = 0.80,
        track_max_age: int = 10,
        start_with_detection: bool = True,
        device: Optional[str] = None,
    ):
        # --- Resolve device for ONNX models ---
        if device is None and torch.cuda.is_available():
            device = "cuda"
        self._device = device
        print(f"[YOLODetector] Device preference: {device}")

        # --- Load models (.pt or .onnx) ---
        print("Loading YOLO models...")
        self.seg_model = _load_yolo_model(seg_model_path, device=device)
        self._seg_is_onnx = isinstance(self.seg_model, OnnxYOLOModel)
        print(f"  Loaded segmentation: {seg_model_path}"
              f" ({'ONNX' if self._seg_is_onnx else 'PT'})")

        self.det_model = None
        self.has_separate_det_model = False
        self._det_is_onnx = False

        if start_with_detection and det_model_path:
            try:
                self.det_model = _load_yolo_model(det_model_path, device=device)
                self._det_is_onnx = isinstance(self.det_model, OnnxYOLOModel)
                self.has_separate_det_model = True
                print(f"  Loaded detection: {det_model_path}"
                      f" ({'ONNX' if self._det_is_onnx else 'PT'})")
            except Exception as e:
                print(f"  Detection model failed: {e}, using segmentation for both")

        if self.det_model is None:
            self.det_model = self.seg_model
            self._det_is_onnx = self._seg_is_onnx

        self.use_detection = start_with_detection and self.has_separate_det_model
        self.model = self.det_model if self.use_detection else self.seg_model

        mode = "DETECTION" if self.use_detection else "SEGMENTATION"
        backend = "ONNX" if isinstance(self.model, OnnxYOLOModel) else "PT"
        print(f"  Starting in {mode} mode ({backend} backend)")

        # --- Config ---
        self.img_size = img_size
        self.conf = conf
        self.iou = iou
        self.pad = pad
        self.merge_dist_px = merge_dist_px

        # --- Tracking ---
        self.enable_tracking = enable_tracking
        self.track_max_distance = float(track_max_distance)
        self.track_iou_threshold = float(track_iou_threshold)
        self.track_max_age = int(track_max_age)
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    # =========================================================
    # TRACKING (VECTORIZED)
    # =========================================================

    def _assign_track_ids(self, infos: List[Dict]) -> List[Dict]:
        """Assign track IDs using vectorized distance + IoU matching."""
        if not self.enable_tracking or not infos:
            for info in infos:
                info.update({"track_id": -1, "center": (0.0, 0.0),
                             "match_distance": -1.0, "match_iou": 0.0})
            return infos

        # Prepare detection data
        det_bboxes = np.array([
            tuple(map(int, info["padded_region"])) for info in infos
        ], dtype=np.float64)
        det_centers = np.column_stack([
            (det_bboxes[:, 0] + det_bboxes[:, 2]) / 2,
            (det_bboxes[:, 1] + det_bboxes[:, 3]) / 2,
        ])

        N_det = len(infos)
        assignments: Dict[int, int] = {}
        match_meta: Dict[int, Dict[str, float]] = {}

        if self._tracks:
            track_ids = list(self._tracks.keys())
            track_items = [self._tracks[tid] for tid in track_ids]
            track_bboxes = np.array([t.bbox for t in track_items], dtype=np.float64)
            track_centers_arr = np.array([t.center for t in track_items], dtype=np.float64)

            # Vectorized distance + IoU
            dist_mat = _pairwise_distances(det_centers, track_centers_arr)
            iou_mat = _pairwise_iou(det_bboxes, track_bboxes)

            # Build candidates: (priority, dist, -iou, det_idx, track_list_idx)
            candidates = []
            for di in range(N_det):
                for ti in range(len(track_ids)):
                    iou_val = iou_mat[di, ti]
                    dist_val = dist_mat[di, ti]
                    if iou_val >= self.track_iou_threshold:
                        candidates.append((0, dist_val, -iou_val, di, ti))
                    elif dist_val <= self.track_max_distance:
                        candidates.append((1, dist_val, -iou_val, di, ti))

            candidates.sort(key=lambda t: (t[0], t[1], t[2]))

            assigned_det: set = set()
            assigned_trk: set = set()

            for _, dist_val, neg_iou, di, ti in candidates:
                if di in assigned_det or ti in assigned_trk:
                    continue
                assigned_det.add(di)
                assigned_trk.add(ti)
                tid = track_ids[ti]
                assignments[di] = tid
                match_meta[di] = {"distance": float(dist_val), "iou": float(-neg_iou)}

            # Age unmatched tracks
            for ti, tid in enumerate(track_ids):
                if ti not in assigned_trk:
                    self._tracks[tid].age += 1
                    if self._tracks[tid].age > self.track_max_age:
                        del self._tracks[tid]

        # Create new tracks for unmatched detections
        for di in range(N_det):
            if di not in assignments:
                new_id = self._next_id
                self._next_id += 1
                bbox = tuple(map(int, det_bboxes[di]))
                center = (float(det_centers[di, 0]), float(det_centers[di, 1]))
                self._tracks[new_id] = _Track(track_id=new_id, bbox=bbox, center=center)
                assignments[di] = new_id
                match_meta[di] = {"distance": -1.0, "iou": 0.0}

        # Update matched tracks
        for di, tid in assignments.items():
            bbox = tuple(map(int, det_bboxes[di]))
            center = (float(det_centers[di, 0]), float(det_centers[di, 1]))
            if tid in self._tracks:
                t = self._tracks[tid]
                t.bbox = bbox
                t.center = center
                t.age = 0
            else:
                self._tracks[tid] = _Track(track_id=tid, bbox=bbox, center=center)

        # Write results
        for di, info in enumerate(infos):
            tid = assignments.get(di, -1)
            cx, cy = float(det_centers[di, 0]), float(det_centers[di, 1])
            meta = match_meta.get(di, {"distance": -1.0, "iou": 0.0})
            info.update({
                "track_id": int(tid),
                "center": (cx, cy),
                "match_distance": meta["distance"],
                "match_iou": meta["iou"],
            })

        return infos

    # =========================================================
    # INSTANCE MERGING
    # =========================================================

    def _merge_instances(
        self,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge overlapping/close instances into single detections."""
        N = len(masks)
        used = np.zeros(N, dtype=bool)

        new_masks, new_boxes, new_scores = [], [], []
        centers = np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        ])

        for i in range(N):
            if used[i]:
                continue
            cur_mask = masks[i].copy()
            cur_box = list(boxes[i])
            cur_score = scores[i]
            used[i] = True

            for j in range(i + 1, N):
                if used[j]:
                    continue
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > self.merge_dist_px:
                    continue

                inter = (cur_mask & masks[j]).sum()
                union = (cur_mask | masks[j]).sum()
                if inter / (union + 1e-6) > 0.3:
                    cur_mask |= masks[j]
                    cur_box = [
                        min(cur_box[0], boxes[j][0]),
                        min(cur_box[1], boxes[j][1]),
                        max(cur_box[2], boxes[j][2]),
                        max(cur_box[3], boxes[j][3]),
                    ]
                    cur_score = max(cur_score, scores[j])
                    used[j] = True

            new_masks.append(cur_mask)
            new_boxes.append(cur_box)
            new_scores.append(cur_score)

        return np.array(new_masks), np.array(new_boxes), np.array(new_scores)

    # =========================================================
    # DETECTION MODES
    # =========================================================

    def _detect_bbox_mode(
        self, frame: np.ndarray, results
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Detection mode: crop via bounding boxes."""
        if results.boxes is None or len(results.boxes) == 0:
            return [], []

        h, w = frame.shape[:2]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        keep = box_nms(boxes, scores, thr=self.iou)
        boxes, scores = boxes[keep], scores[keep]

        cropped, infos = [], []
        pad = self.pad

        for box, conf in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            cropped.append(crop)
            infos.append({
                "bbox": (x1, y1, x2, y2),
                "padded_region": (x1, y1, x2, y2),
                "conf": float(conf),
                "mode": "detection",
            })

        infos = self._assign_track_ids(infos)
        return cropped, infos

    def _detect_seg_mode(
        self, frame: np.ndarray, results
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Segmentation mode: crop via masks."""
        if results.masks is None:
            return [], []

        h, w = frame.shape[:2]
        masks_raw = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        # Resize masks to frame size
        masks = np.array([
            cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            for m in masks_raw
        ], dtype=bool)

        # NMS pipeline
        keep = box_nms(boxes, scores, thr=self.iou)
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]

        keep = mask_nms(masks, scores)
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]

        # Merge close instances
        masks, boxes, scores = self._merge_instances(masks, boxes, scores)

        cropped, infos = [], []
        pad = self.pad
        _morph_k = np.ones((5, 5), np.uint8)

        for mask, box, conf in zip(masks, boxes, scores):
            # pill_collect.py style mask preprocessing
            m_u8 = mask.astype(np.uint8) * 255
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, _morph_k)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, _morph_k)
            m_u8 = cv2.GaussianBlur(m_u8, (7, 7), 0)
            _, m_u8 = cv2.threshold(m_u8, 128, 255, cv2.THRESH_BINARY)

            ys, xs = np.where(m_u8)
            if len(xs) == 0:
                continue

            x1 = max(0, int(xs.min()) - pad)
            y1 = max(0, int(ys.min()) - pad)
            x2 = min(w, int(xs.max()) + pad)
            y2 = min(h, int(ys.max()) + pad)

            # Alpha blend on black background (smooth edges)
            crop_img = frame[y1:y2, x1:x2].copy()
            crop_mask = m_u8[y1:y2, x1:x2]
            alpha = crop_mask.astype(np.float32) / 255.0
            crop = np.zeros_like(crop_img)
            for c in range(3):
                crop[:, :, c] = (crop_img[:, :, c] * alpha).astype(np.uint8)

            cropped.append(crop)
            infos.append({
                "bbox": box.astype(int).tolist(),
                "conf": float(conf),
                "padded_region": (x1, y1, x2, y2),
                "mode": "segmentation",
            })

        infos = self._assign_track_ids(infos)
        return cropped, infos

    # =========================================================
    # PUBLIC API
    # =========================================================

    def detect_and_crop(
        self, frame: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Detect pills and return cropped images with tracking info.
        
        Returns:
            crops: List of BGR numpy arrays
            infos: List of dicts with bbox, conf, track_id, etc.
        """
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            retina_masks=False,
            verbose=False,
        )[0]

        if self.use_detection:
            return self._detect_bbox_mode(frame, results)
        return self._detect_seg_mode(frame, results)

    def reset_tracking(self) -> None:
        """Clear all tracking state."""
        self._tracks.clear()
        self._next_id = 1

    def switch_to_segmentation(self) -> None:
        """Switch to mask-based segmentation mode."""
        if not self.use_detection:
            return
        self.use_detection = False
        self.model = self.seg_model
        backend = "ONNX" if self._seg_is_onnx else "PT"
        print(f"[MODE] Switched to SEGMENTATION ({backend})")

    def switch_to_detection(self) -> None:
        """Switch to bbox-based detection mode."""
        if self.use_detection or not self.has_separate_det_model:
            return
        self.use_detection = True
        self.model = self.det_model
        backend = "ONNX" if self._det_is_onnx else "PT"
        print(f"[MODE] Switched to DETECTION ({backend})")

    @property
    def current_mode(self) -> str:
        backend = "ONNX" if isinstance(self.model, OnnxYOLOModel) else "PT"
        mode = "DETECTION" if self.use_detection else "SEGMENTATION"
        return f"{mode} ({backend})"
