"""
FCDD Inference Pipeline
=======================
Full inference pipeline:
  1. YOLO detect pills in image → bounding boxes
  2. Crop each pill → run FCDD → anomaly heatmap
  3. Threshold heatmap → binary mask
  4. Connected components → ตรวจจับบริเวณ anomaly
  5. คำนวณ center ของแต่ละเม็ดยา
  6. นับจำนวน pill & anomaly
  7. วาด center dots (green=normal, red=anomaly)
  8. Save result

Usage:
  python -m FCDD.fcdd_inference
  หรือ python run_predict_fcdd.py
"""
import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import fcdd as cfg
from FCDD.fcdd_model import FCDDNet


# =============================================================================
#  YOLO Pill Detector (lightweight wrapper)
# =============================================================================
class YOLOPillDetector:
    """Detect pills in full images using YOLO segmentation model."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.5,
        img_size: int = 640,
        pad: int = 5,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.img_size = img_size
        self.pad = pad
        print(f"[YOLO] Loaded: {model_path}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect pills and return bounding box info + segmentation mask.

        Returns:
            list of dicts with keys:
              bbox, conf, center, crop,
              pill_mask (H_crop, W_crop) float32 0/1  ← region ที่เป็น pill จริงๆ
        """
        h, w = frame.shape[:2]
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf,
            verbose=False,
        )[0]

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes  = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        # ---- Try to get segmentation masks from YOLO ----
        seg_masks = None
        if results.masks is not None:
            raw = results.masks.data.cpu().numpy()          # (N, H_mask, W_mask)
            seg_masks = []
            for m in raw:
                m_full = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                seg_masks.append((m_full > 0.5).astype(np.uint8))

        for i, (box, conf) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)

            # Add padding
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(w, x2 + self.pad)
            y2 = min(h, y2 + self.pad)

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # ---- Build masked crop (background → white like training data) ----
            if seg_masks is not None and i < len(seg_masks):
                mask_region = seg_masks[i][y1:y2, x1:x2]          # (crop_h, crop_w)
                crop_raw    = frame[y1:y2, x1:x2].copy()
                # Fill outside-pill area with white (255) to match white_back_cropped
                white_bg = np.full_like(crop_raw, 255)
                crop = np.where(mask_region[:, :, None] == 1, crop_raw, white_bg)
                pill_mask = mask_region.astype(np.float32)
            else:
                crop      = frame[y1:y2, x1:x2].copy()
                pill_mask = None   # no mask available → use whole crop

            detections.append({
                "bbox":      (x1, y1, x2, y2),
                "conf":      float(conf),
                "center":    (cx, cy),
                "crop":      crop,
                "pill_mask": pill_mask,
            })

        return detections


# =============================================================================
#  FCDD Anomaly Scorer
# =============================================================================
class FCDDScorer:
    """Score pill crops using trained FCDD model."""

    def __init__(
        self,
        model_path: str,
        backbone_path: str = None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        img_size = checkpoint.get("img_size", cfg.IMG_SIZE)
        hook_indices = checkpoint.get("hook_indices", cfg.HOOK_INDICES)
        self.threshold = checkpoint.get("anomaly_threshold", cfg.ANOMALY_THRESHOLD)
        self.img_size = img_size

        print(f"[FCDD] Loading model from {model_path}")
        print(f"[FCDD] Threshold: {self.threshold:.4f}")

        # Build model
        self.model = FCDDNet(
            backbone_path=backbone_path,
            img_size=img_size,
            hook_indices=hook_indices,
            freeze_backbone=True,
        ).to(self.device)

        # Load trained head weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "head_state_dict" in checkpoint:
            self.model.head.load_state_dict(checkpoint["head_state_dict"])

        self.model.eval()

        # Transform
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _make_square(
        self, crop: np.ndarray, pill_mask: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Pad crop (and optional mask) to square with WHITE background,
        then resize to img_size.
        Returns (square_crop, square_mask_or_None)
        """
        h, w = crop.shape[:2]
        max_side = max(h, w)
        y_off = (max_side - h) // 2
        x_off = (max_side - w) // 2

        # White background (matches white_back_cropped training data)
        square = np.full((max_side, max_side, 3), 255, dtype=np.uint8)
        square[y_off : y_off + h, x_off : x_off + w] = crop
        square = cv2.resize(square, (self.img_size, self.img_size))

        if pill_mask is not None:
            sq_mask = np.zeros((max_side, max_side), dtype=np.float32)
            sq_mask[y_off : y_off + h, x_off : x_off + w] = pill_mask
            sq_mask = cv2.resize(
                sq_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
            )
            sq_mask = (sq_mask > 0.5).astype(np.float32)
        else:
            sq_mask = None

        return square, sq_mask

    @torch.no_grad()
    def score(
        self, crop: np.ndarray, pill_mask: np.ndarray = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute anomaly score and heatmap for a pill crop.

        Args:
            crop:      BGR image (H, W, 3)
            pill_mask: (H, W) float32 binary mask (1 = pill pixel) หรือ None
        Returns:
            score:   float, anomaly score คำนวณเฉพาะ pill region (0-1)
            heatmap: (H, W) float32 anomaly map (0-1)
        """
        if crop is None or crop.size == 0:
            return 0.0, np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Pad to square (white bg) + resize mask
        square, sq_mask = self._make_square(crop, pill_mask)

        rgb    = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        # Forward
        anomaly_map = self.model(tensor)  # (1, 1, H, W)

        # Apply sigmoid → (H, W) in [0, 1]
        heatmap = torch.sigmoid(anomaly_map).squeeze().cpu().numpy()

        # ---- Score เฉพาะ pill region (ถ้ามี mask) ----
        if sq_mask is not None and sq_mask.sum() > 0:
            # Weighted mean – ignore background pixels
            score = float((heatmap * sq_mask).sum() / sq_mask.sum())
            # Zero-out background ใน heatmap เพื่อ connected-component
            heatmap = heatmap * sq_mask
        else:
            score = float(heatmap.mean())

        return score, heatmap

    def classify(
        self,
        crop: np.ndarray,
        threshold: float = None,
        min_component_area: int = None,
        **kwargs,
    ) -> Dict:
        """
        Classify a pill crop as normal or anomaly.

        Pipeline:
          FCDD → heatmap → global mean score → threshold → classify
          + connected components for anomaly region counting

        Returns dict with:
          - status: "NORMAL" or "ANOMALY"
          - score: float (global mean anomaly score, 0-1)
          - heatmap: (H, W) array
          - n_components: int (anomalous regions found)
        """
        thr = threshold if threshold is not None else self.threshold
        min_area = min_component_area if min_component_area is not None else cfg.MIN_COMPONENT_AREA

        pill_mask = kwargs.get("pill_mask", None)
        score, heatmap = self.score(crop, pill_mask=pill_mask)

        # Primary classification: global mean score vs calibrated threshold
        status = "ANOMALY" if score >= thr else "NORMAL"

        # Secondary: connected components on heatmap for region counting
        # Use a higher pixel threshold (0.5) for heatmap binarization
        pixel_thr = max(thr, 0.5)
        binary = (heatmap > pixel_thr).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        n_anomaly_regions = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                n_anomaly_regions += 1

        return {
            "status": status,
            "score": score,
            "heatmap": heatmap,
            "binary_mask": binary,
            "n_components": n_anomaly_regions,
        }


# =============================================================================
#  Visualization
# =============================================================================
def draw_results(
    image: np.ndarray,
    detections: List[Dict],
    pill_count: int,
    anomaly_count: int,
) -> np.ndarray:
    """
    Draw results on image:
      - Top-left: pill count & anomaly count
      - Center dots: green (normal) / red (anomaly)
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Colors
    GREEN = (0, 200, 0)
    RED = (0, 0, 220)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # ---- Draw center dots ----
    for det in detections:
        cx, cy = int(det["center"][0]), int(det["center"][1])
        status = det.get("status", "NORMAL")
        color = RED if status == "ANOMALY" else GREEN

        # Filled circle
        cv2.circle(vis, (cx, cy), cfg.CENTER_DOT_RADIUS, color, -1)
        # Border
        cv2.circle(vis, (cx, cy), cfg.CENTER_DOT_RADIUS, BLACK, 2)

    # ---- Draw text overlay (top-left) ----
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = cfg.FONT_SCALE
    ft = cfg.FONT_THICKNESS

    text1 = f"pills = {pill_count}"
    text2 = f"anomaly = {anomaly_count}"

    # Text sizes
    (tw1, th1), _ = cv2.getTextSize(text1, font, fs, ft)
    (tw2, th2), _ = cv2.getTextSize(text2, font, fs, ft)

    max_tw = max(tw1, tw2)
    box_h = th1 + th2 + 30
    box_w = max_tw + 20

    # Semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # Text
    cv2.putText(vis, text1, (15, 5 + th1 + 5), font, fs, WHITE, ft)
    cv2.putText(vis, text2, (15, 5 + th1 + th2 + 20), font, fs, RED, ft)

    return vis


# =============================================================================
#  Main Inference Pipeline
# =============================================================================
def run_inference():
    """
    Full inference pipeline:
      data_yolo/test/* → YOLO detect → FCDD classify → result_fcdd/*
    """
    print("=" * 60)
    print("  FCDD Inference - Pill Anomaly Detection")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Check paths ----
    if not cfg.FCDD_MODEL_PATH.exists():
        print(f"ERROR: Trained model not found at {cfg.FCDD_MODEL_PATH}")
        print("Please run training first: python run_train_fcdd.py")
        return

    test_dir = Path(cfg.TEST_DATA_DIR)
    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        print("Please create the directory and add test images.")
        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created empty directory: {test_dir}")
        return

    # Collect test images
    test_images = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in cfg.IMAGE_EXTS]
    )
    if not test_images:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(test_images)} test images in {test_dir}")

    # ---- Create output directory ----
    result_dir = Path(cfg.RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {result_dir}")

    # ---- Load models ----
    print("\nLoading models...")

    # YOLO detector
    yolo = YOLOPillDetector(
        model_path=str(cfg.SEGMENTATION_MODEL_PATH),
        conf=cfg.YOLO_CONF,
        img_size=cfg.YOLO_IMG_SIZE,
    )

    # FCDD scorer
    scorer = FCDDScorer(
        model_path=str(cfg.FCDD_MODEL_PATH),
        backbone_path=str(cfg.BACKBONE_PATH),
        device=device,
    )

    # ---- Process each image ----
    print(f"\nProcessing {len(test_images)} images...")
    print("-" * 60)

    total_pills = 0
    total_anomalies = 0

    for img_idx, img_path in enumerate(test_images):
        print(f"\n[{img_idx+1}/{len(test_images)}] {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: Could not load {img_path}")
            continue

        # Detect pills with YOLO
        detections = yolo.detect(image)
        n_pills = len(detections)

        if n_pills == 0:
            # No YOLO detections → treat entire image as single pill
            print("  No pills detected by YOLO, treating whole image as one pill")
            h, w = image.shape[:2]
            detections = [{
                "bbox": (0, 0, w, h),
                "conf": 1.0,
                "center": (w / 2.0, h / 2.0),
                "crop": image.copy(),
            }]
            n_pills = 1

        # Classify each pill
        n_anomalies = 0
        for det in detections:
            result = scorer.classify(
                det["crop"],
                pill_mask=det.get("pill_mask"),
            )
            det["status"] = result["status"]
            det["score"] = result["score"]
            det["heatmap"] = result["heatmap"]
            det["n_components"] = result["n_components"]

            if result["status"] == "ANOMALY":
                n_anomalies += 1

            status_str = (
                f"ANOMALY (score={result['score']:.3f}, regions={result['n_components']})"
                if result["status"] == "ANOMALY"
                else f"NORMAL  (score={result['score']:.3f})"
            )
            print(f"  Pill: {status_str}")

        total_pills += n_pills
        total_anomalies += n_anomalies

        # Draw results
        vis = draw_results(image, detections, n_pills, n_anomalies)

        # Save result
        out_path = result_dir / img_path.name
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved: {out_path}")

        # Save heatmaps (optional debug)
        heatmap_dir = result_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        for i, det in enumerate(detections):
            hm = det["heatmap"]
            hm_vis = (hm * 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
            hm_path = heatmap_dir / f"{img_path.stem}_pill{i}_{det['status']}.jpg"
            cv2.imwrite(str(hm_path), hm_color)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"  Results Summary")
    print(f"  Images processed:  {len(test_images)}")
    print(f"  Total pills:       {total_pills}")
    print(f"  Total anomalies:   {total_anomalies}")
    print(f"  Total normal:      {total_pills - total_anomalies}")
    print(f"  Output directory:  {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()
