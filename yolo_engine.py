import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class YOLOEngine:
    """
    Ultra Performance YOLO Engine
    - Supports: .pt / .onnx / .engine
    - Fixed input size
    - FP16 auto (.pt + CUDA)
    - Optimized segmentation crop
    """

    SUPPORTED_EXTS = {".pt", ".onnx", ".engine"}

    def __init__(
        self,
        model_path: str,
        mode: str = "segment",
        device: str = "cuda",
        input_size: int = 640,
        conf: float = 0.25,
        iou: float = 0.6,
        class_id: int | None = 0,
        crop_size: int = 256,
        retina_masks: bool = False,
    ):

        suffix = Path(model_path).suffix.lower()
        if suffix not in self.SUPPORTED_EXTS:
            raise ValueError(f"Unsupported format: {suffix}")

        self.mode = mode
        self.device = device
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.class_filter = class_id
        self.crop_size = crop_size
        self.retina_masks = retina_masks

        self._is_onnx = suffix == ".onnx"
        self._is_engine = suffix == ".engine"

        # Load model
        if self._is_onnx:
            self.model = YOLO(model_path, task=mode)
        else:
            self.model = YOLO(model_path)

        # FP16 only for .pt on CUDA
        self.use_half = (
            device == "cuda"
            and not self._is_onnx
            and not self._is_engine
        )

        if not self._is_onnx and not self._is_engine:
            self.model.to(device)

        if self.use_half:
            self.model.model.half()

        # CUDA optimizations
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    # ─────────────────────────────────────────────

    def inference(self, frame: np.ndarray):

        infer_device = None if self._is_engine else self.device

        results = self.model(
            frame,
            imgsz=self.input_size,
            conf=self.conf,
            iou=self.iou,
            device=infer_device,
            half=self.use_half,
            retina_masks=self.retina_masks if self.mode == "segment" else False,
            verbose=False,
        )

        return self._postprocess(results)

    # ─────────────────────────────────────────────

    def _postprocess(self, results):

        outputs = []

        for r in results:

            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            masks = None
            if self.mode == "segment" and r.masks is not None:
                masks = r.masks.data.cpu().numpy()

            for i in range(len(boxes)):

                if self.class_filter is not None:
                    if cls_ids[i] != self.class_filter:
                        continue

                mask = None
                if masks is not None:
                    mask = (masks[i] > 0.5).astype(np.uint8) * 255

                outputs.append({
                    "bbox": tuple(boxes[i]),
                    "conf": float(confs[i]),
                    "class_id": int(cls_ids[i]),
                    "mask": mask,
                })

        return outputs

    # ─────────────────────────────────────────────
    # Center crop 256 (region-optimized)
    # ─────────────────────────────────────────────

    def crop_object(self, frame, det):

        x1, y1, x2, y2 = det["bbox"]
        fh, fw = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fw, x2)
        y2 = min(fh, y2)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros((self.crop_size, self.crop_size, 3), np.uint8)

        # Apply mask only to region
        if det["mask"] is not None:
            mask_crop = det["mask"][y1:y2, x1:x2]
            crop = crop.copy()
            crop[mask_crop == 0] = 0

        ch, cw = crop.shape[:2]
        side = max(ch, cw)

        square = np.zeros((side, side, 3), np.uint8)

        sy = (side - ch) // 2
        sx = (side - cw) // 2
        square[sy:sy + ch, sx:sx + cw] = crop

        return cv2.resize(square, (self.crop_size, self.crop_size))