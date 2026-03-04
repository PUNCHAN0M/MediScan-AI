#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pill Detection & Cropping System (Production Ready)
====================================================
Uses ``YOLOTracking`` — unified YOLO-seg + preprocess + tracking.
Supports: .pt / .onnx, Real-time / Batch mode, Grid Display
"""

import cv2
import sys
import numpy as np
import torch
import logging
from pathlib import Path
from datetime import datetime

from Core_ResnetPatchCore.segmentation.yolo_tracking import YOLOTracking

# =============================================================================
# ⚙️ CONFIGURATION
# =============================================================================
CONFIG = {
    # Model Path (รองรับ .pt หรือ .onnx)
    "MODEL_PATH": "model/SEGMENTATION/pill-detection-best-2.onnx",

    # Input: None → Real-time (Webcam), Path → Batch Mode
    "INPUT_DIR": None,

    # Output Directory
    "OUTPUT_DIR": "data/Androxsil/Androxsil",

    # Confidence Threshold (0.0 - 1.0)
    "CONFIDENCE": 0.25,

    # ขนาดภาพสุดท้าย (256×256)
    "FINAL_SIZE": 256,

    # Real-time Settings
    "WEBCAM_ID": 0,
    "GRID_MAX_COLS": 5,
    "DISPLAY_SCALE": 1.0,

    # Inference Settings
    "IMG_SIZE": 1280,
    "RETINA_MASKS": True,
    "USE_CUDA": True,
}

# =============================================================================
# 🛠️ SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class PillCollector:
    """
    Pill collection using ``YOLOTracking``.

    All detection, mask preprocessing, cropping, and grid building are
    delegated to the shared ``YOLOTracking`` class.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.output_dir = Path(config["OUTPUT_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if (config["USE_CUDA"] and torch.cuda.is_available()) else "cpu"
        logger.info(f"🔧 Using device: {device.upper()}")

        self.yolo = YOLOTracking(
            model_path=config["MODEL_PATH"],
            img_size=config["IMG_SIZE"],
            conf=config["CONFIDENCE"],
            iou=0.6,
            device=device,
            target_size=config["FINAL_SIZE"],
            pad=5,
            bg_value=0,
            retina_masks=config["RETINA_MASKS"],
            enable_tracking=False,          # pill_collect ไม่ต้อง track
        )

        self.save_counter = 0

    def process_frame(self, frame: np.ndarray, *, grid: bool = False):
        """
        Detect + crop pills from one frame.

        Returns
        -------
        crops    : list of (256,256,3) uint8
        infos    : list of detection info dicts
        preview  : annotated frame (with optional grid)
        """
        crops, infos, preview = self.yolo.process_frame(
            frame,
            draw=True,
            grid=grid,
            grid_max_cols=self.cfg["GRID_MAX_COLS"],
        )
        return crops, infos, preview

    def save_pill(self, pill_img: np.ndarray, source_name: str = "realtime") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_name}_pill_{self.save_counter:04d}_{timestamp}.png"
        save_path = self.output_dir / filename
        cv2.imwrite(str(save_path), pill_img)
        self.save_counter += 1
        logger.info(f"💾 Saved: {filename}")
        return str(save_path)


# =============================================================================
# 🎥 REALTIME
# =============================================================================
def digital_zoom(frame, zoom=1.5):
    if zoom <= 1:
        return frame
    h, w = frame.shape[:2]
    nw, nh = int(w / zoom), int(h / zoom)
    x1, y1 = (w - nw) // 2, (h - nh) // 2
    return cv2.resize(frame[y1:y1 + nh, x1:x1 + nw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


def run_realtime(collector: PillCollector, config: dict):
    cap = cv2.VideoCapture(config["WEBCAM_ID"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open webcam ID {config['WEBCAM_ID']}")
        return

    logger.info("🎬 Real-time mode | 's'=save  'g'=toggle grid  'q'=quit")
    show_grid = True

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("⚠️ Failed to grab frame")
            break
        frame = digital_zoom(frame, zoom=1.5)

        crops, infos, preview = collector.process_frame(frame, grid=show_grid)
        display = preview if preview is not None else frame

        cv2.putText(
            display,
            f"Pills: {len(crops)} | 'S'=save  'G'=grid  'Q'=quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

        scale = config["DISPLAY_SCALE"]
        if scale != 1.0:
            display = cv2.resize(display, (0, 0), fx=scale, fy=scale)

        cv2.imshow("Pill Detection System", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and crops:
            logger.info(f"💾 Saving {len(crops)} pill(s)...")
            for c in crops:
                collector.save_pill(c, source_name="realtime")
        elif key == ord("g"):
            show_grid = not show_grid
            logger.info(f"Grid: {'ON' if show_grid else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("🔚 Real-time session ended")


# =============================================================================
# 📁 BATCH
# =============================================================================
def run_batch(collector: PillCollector, config: dict):
    input_dir = Path(config["INPUT_DIR"])
    if not input_dir.exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        return

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    for e in exts:
        files.extend(input_dir.glob(e))

    if not files:
        logger.warning(f"⚠️ No images in {input_dir}")
        return

    logger.info(f"🚀 Batch mode: {len(files)} images...")

    for idx, img_path in enumerate(files, 1):
        logger.info(f"[{idx}/{len(files)}] {img_path.name}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"⚠️ Could not read: {img_path}")
            continue

        crops, _, _ = collector.process_frame(frame)
        if crops:
            logger.info(f"✅ Found {len(crops)} pill(s)")
            for c in crops:
                collector.save_pill(c, source_name=img_path.stem)
        else:
            logger.info("ℹ️  No pills detected")

    logger.info(f"✅ Batch complete! Output: {config['OUTPUT_DIR']}")


# =============================================================================
# 🚀 MAIN
# =============================================================================
def main():
    logger.info("🔷 Pill Detection System Starting...")
    try:
        collector = PillCollector(CONFIG)

        if not CONFIG["INPUT_DIR"]:
            logger.info("📹 Mode: REAL-TIME (Webcam)")
            run_realtime(collector, CONFIG)
        else:
            logger.info(f"📁 Mode: BATCH (Folder: {CONFIG['INPUT_DIR']})")
            run_batch(collector, CONFIG)

    except KeyboardInterrupt:
        logger.info("⚡ Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        logger.info("🔚 System shutdown complete")


if __name__ == "__main__":
    main()