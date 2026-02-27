#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pill Detection & Cropping System (Production Ready)
Support: YOLOv11 Segmentation (.pt / .onnx), Real-time / Batch mode, Grid Display
"""

import cv2
import os
import sys
import numpy as np
import torch
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# =============================================================================
# ⚙️ CONFIGURATION (แก้ไขค่าต่างๆ ที่นี่)
# =============================================================================
CONFIG = {
    # Model Path (รองรับ .pt หรือ .onnx)
    "MODEL_PATH": "model/SEGMENTATION/pill-detection-best-2.onnx",
    
    # Input: ถ้าเป็น None หรือ "" จะใช้ Real-time (Webcam), ถ้าใส่ Path จะใช้ Batch Mode
    "INPUT_DIR": None,  # Example: "data_yolo/dataset/" หรือ None
    
    # Output Directory สำหรับบันทึกภาพที่ Crop แล้ว
    "OUTPUT_DIR": "data/Androxsil/Androxsil",
    
    # Confidence Threshold (0.0 - 1.0)
    "CONFIDENCE": 0.25,
    
    # ขนาดภาพสุดท้ายที่ต้องการ (256x256)
    "FINAL_SIZE": 256,
    
    # Real-time Settings
    "WEBCAM_ID": 0,           # ID ของกล้อง (0 = กล้องหลัก)
    "GRID_MAX_COLS": 5,       # จำนวนคอลัมน์สูงสุดก่อนขึ้นแถวใหม่
    "DISPLAY_SCALE": 1.0,     # Scale ของหน้าต่างแสดงผล (ปรับถ้าจอเล็ก/ใหญ่เกิน)
    
    # Inference Settings
    "IMG_SIZE": 1280,         # ขนาดภาพที่ส่งเข้าโมเดล (ยิ่งมากยิ่งช้าแต่แม่น)
    "RETINA_MASKS": True,     # ใช้ High-res masks
    "USE_CUDA": True,         # ใช้ GPU ถ้ามี
}

# =============================================================================
# 🛠️ SETUP & UTILITIES
# =============================================================================

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class PillProcessor:
    """Class สำหรับจัดการ Detection, Cropping และ Post-processing"""
    
    def __init__(self, config: dict):
        self.cfg = config
        self.output_dir = Path(config["OUTPUT_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Device
        self.device = 'cuda' if (config["USE_CUDA"] and torch.cuda.is_available()) else 'cpu'
        self.is_onnx = False  # Flag เพื่อตรวจสอบรูปแบบโมเดล
        logger.info(f"🔧 Using device: {self.device.upper()}")
        
        # Load Model
        self._load_model()
        
        # Counters for naming saved files
        self.save_counter = 0
        
    def _load_model(self):
        """โหลดโมเดลพร้อมตรวจสอบรูปแบบไฟล์ (.pt vs .onnx)"""
        model_path = Path(self.cfg["MODEL_PATH"])
        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        
        logger.info(f"📦 Loading model: {model_path.name}")
        
        try:
            # ✅ ตรวจสอบนามสกุลไฟล์เพื่อเลือกวิธีโหลด
            if model_path.suffix.lower() == '.onnx':
                self.is_onnx = True
                # ONNX: ต้องระบุ task='segment' และห้ามใช้ .to()
                self.model = YOLO(str(model_path), task='segment')
                logger.info(f"✅ ONNX Model loaded (Task: Segmentation)")
            else:
                self.is_onnx = False
                # PyTorch (.pt): ใช้ .to() เพื่อย้ายไป GPU ได้ตามปกติ
                self.model = YOLO(str(model_path)).to(self.device)
                logger.info(f"✅ PyTorch Model loaded | Format: {model_path.suffix}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
            
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """ทำ Post-processing ให้ Mask นุ่มนวลและสะอาด"""
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        return mask
    
    def _crop_and_square(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Crop ภาพตาม Mask และทำ Square Padding พื้นหลังดำ (ไม่โปร่งใส)"""
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop ภาพและ mask
        crop_img = image[y:y+h, x:x+w].copy()
        crop_mask = mask[y:y+h, x:x+w].copy()
        
        # ✅ สร้างพื้นหลังสีดำ (BGR 3-channel) แทนการใช้ BGRA โปร่งใส
        max_dim = max(crop_img.shape[0], crop_img.shape[1])
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)  # ✅ พื้นหลังดำ [0,0,0]
        
        # คำนวณตำแหน่งวางภาพ crop ลงกลางพื้นหลังดำ
        pad_top = (max_dim - crop_img.shape[0]) // 2
        pad_left = (max_dim - crop_img.shape[1]) // 2
        
        # ✅ วางภาพ crop ลงบนพื้นหลังดำ โดยใช้ mask เป็น alpha blending
        # แปลง mask เป็น float [0,1] สำหรับ blending
        alpha = crop_mask.astype(np.float32) / 255.0
        
        # วางทีละ channel (BGR)
        for c in range(3):
            # สูตร: result = pill * alpha + black * (1-alpha) → black=0 จึงเหลือแค่ pill * alpha
            square_img[
                pad_top:pad_top+crop_img.shape[0], 
                pad_left:pad_left+crop_img.shape[1], 
                c
            ] = (crop_img[:, :, c] * alpha).astype(np.uint8)
        
        # ✅ Resize เป็นขนาดสุดท้าย (256x256)
        final_img = cv2.resize(
            square_img, 
            (self.cfg["FINAL_SIZE"], self.cfg["FINAL_SIZE"]), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        return final_img  # ✅ คืนค่า BGR 3-channel (พื้นหลังดำ)
    
    def process_frame(self, frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """ประมวลผลหนึ่งเฟรม"""
        cropped_pills = []
        display_frame = frame.copy()
        
        # ✅ แก้ไขสำคัญ: ส่ง device เข้าไปใน predict() โดยตรง (ใช้ได้ทั้ง .pt และ .onnx)
        # ถ้าเป็น ONNX Ultralytics จะจัดการ Device ให้เองเมื่อพารามิเตอร์นี้ถูกส่งมา
        infer_device = None if self.is_onnx else self.device
            
        results = self.model.predict(
            source=frame,
            conf=self.cfg["CONFIDENCE"],
            save=False,
            retina_masks=self.cfg["RETINA_MASKS"],
            imgsz=self.cfg["IMG_SIZE"],
            device=infer_device,  # ✅ จุดที่แก้ไข: ONNX ห้ามส่ง device ผ่าน .to() แต่ส่งที่นี่ได้
            verbose=False,
            task='segment' if self.is_onnx else None # บังคับ task สำหรับ onnx
        )
        
        if results[0].masks is None:
            return [], display_frame
            
        h_orig, w_orig = frame.shape[:2]
        masks_tensor = results[0].masks.data  # (N, H, W)
        
        for j in range(masks_tensor.shape[0]):
            # Resize mask ให้เท่าภาพต้นฉบับ
            m = masks_tensor[j].unsqueeze(0).unsqueeze(0)
            # ย้ายไป CPU เพื่อประมวลผลต่อด้วย OpenCV (จำเป็นสำหรับทั้ง 2 รูปแบบ)
            m = torch.nn.functional.interpolate(
                m, size=(h_orig, w_orig), mode='bilinear', align_corners=False
            )
            mask = (m.squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            # Preprocess mask
            mask = self._preprocess_mask(mask)
            
            # Crop & Square
            cropped = self._crop_and_square(frame, mask)
            if cropped is not None:
                cropped_pills.append(cropped)
                
            # Draw bounding box
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
        return cropped_pills, display_frame
    
    def save_pill(self, pill_img: np.ndarray, source_name: str = "realtime") -> str:
        """บันทึกภาพ Pill ที่ Crop แล้ว"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_name}_pill_{self.save_counter:04d}_{timestamp}.png"
        save_path = self.output_dir / filename
        cv2.imwrite(str(save_path), pill_img)
        self.save_counter += 1
        logger.info(f"💾 Saved: {filename}")
        return str(save_path)


# ในฟังก์ชัน create_grid_display
def create_grid_display(pills: list[np.ndarray], max_cols: int) -> np.ndarray | None:
    if not pills:
        return None
        
    size = pills[0].shape[0]
    n = len(pills)
    cols = min(n, max_cols)
    rows = (n + max_cols - 1) // max_cols
    
    # ✅ สร้างภาพดำสำหรับพื้นหลัง Grid (0 = ดำ)
    # ใช้ 3 channels (BGR) พอเพราะพื้นหลังดำไม่ต้องการ Alpha
    grid = np.zeros((rows * size, cols * size, 3), dtype=np.uint8) 
    
    for idx, pill in enumerate(pills):
        row = idx // max_cols
        col = idx % max_cols
        y = row * size
        x = col * size
        
        # ✅ วางภาพลง Grid โดยจัดการ Alpha Blending กับพื้นหลังดำ
        if pill.shape[2] == 4:  # มี Alpha channel
            alpha = pill[:, :, 3:] / 255.0
            # สูตร: Final = Pill * Alpha + Black * (1-Alpha) -> Black คือ 0 จึงตัดทิ้งได้
            for c in range(3):  # BGR channels
                grid[y:y+size, x:x+size, c] = (pill[:, :, c] * alpha.squeeze()).astype(np.uint8)
        else:
            grid[y:y+size, x:x+size, :3] = pill
            
    return grid

def digital_zoom(frame, zoom=1.5):
    if zoom <= 1:
        return frame

    h, w = frame.shape[:2]

    new_w = int(w / zoom)
    new_h = int(h / zoom)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2

    cropped = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def run_realtime(processor: PillProcessor, config: dict):
    """โหมด Real-time: เปิดกล้องและแสดงผล"""
    cap = cv2.VideoCapture(config["WEBCAM_ID"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open webcam ID {config['WEBCAM_ID']}")
        return
        
    logger.info("🎬 Real-time mode started | Press 's' to save | 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        frame = digital_zoom(frame, zoom=1.5)
        if not ret:
            logger.warning("⚠️ Failed to grab frame")
            break
            
        # Process
        pills, display_frame = processor.process_frame(frame)
        
        # Create Grid Display
        if pills:
            grid = create_grid_display(pills, config["GRID_MAX_COLS"])
            if grid is not None:
                # Convert BGRA -> BGR สำหรับแสดงด้วย OpenCV
                grid_bgr = cv2.cvtColor(grid, cv2.COLOR_BGRA2BGR)
                
                # แสดง Grid ด้านข้างหรือด้านล่างของเฟรมหลัก
                # Resize grid ให้สูงเท่ากับเฟรมหลักเพื่อจัดวางง่าย
                h_frame = display_frame.shape[0]
                grid_resized = cv2.resize(grid_bgr, (h_frame, h_frame))
                
                # วาง Grid ทางขวาของเฟรมหลัก
                combined = np.hstack([display_frame, grid_resized])
            else:
                combined = display_frame
        else:
            combined = display_frame
            
        # แสดงข้อความสถานะ
        cv2.putText(
            combined, 
            f"Pills: {len(pills)} | Press 'S' to save | 'Q' to quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        # แสดงผล
        scale = config["DISPLAY_SCALE"]
        if scale != 1.0:
            combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale)
            
        cv2.imshow("Pill Detection System", combined)
        
        # Handle Key Press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("👋 Quitting...")
            break
        elif key == ord('s') and pills:
            logger.info(f"💾 Saving {len(pills)} pill(s)...")
            for pill in pills:
                processor.save_pill(pill, source_name="realtime")
                
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("🔚 Real-time session ended")


def run_batch(processor: PillProcessor, config: dict):
    """โหมด Batch: ประมวลผลภาพจากโฟลเดอร์"""
    input_dir = Path(config["INPUT_DIR"])
    if not input_dir.exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        return
        
    # หาไฟล์ภาพทั้งหมด
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
        
    if not image_files:
        logger.warning(f"⚠️ No images found in {input_dir}")
        return
        
    logger.info(f"🚀 Batch mode: Processing {len(image_files)} images...")
    
    for idx, img_path in enumerate(image_files, 1):
        logger.info(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"⚠️ Could not read: {img_path}")
            continue
            
        pills, _ = processor.process_frame(frame)
        
        if pills:
            logger.info(f"✅ Found {len(pills)} pill(s)")
            for pill in pills:
                processor.save_pill(pill, source_name=img_path.stem)
        else:
            logger.info("ℹ️  No pills detected")
            
    logger.info(f"✅ Batch complete! Output: {config['OUTPUT_DIR']}")


def main():
    """Entry Point"""
    logger.info("🔷 Pill Detection System Starting...")
    
    try:
        # Initialize Processor
        processor = PillProcessor(CONFIG)
        
        # เลือกโหมดการทำงาน
        if CONFIG["INPUT_DIR"] is None or CONFIG["INPUT_DIR"] == "":
            logger.info("📹 Mode: REAL-TIME (Webcam)")
            run_realtime(processor, CONFIG)
        else:
            logger.info(f"📁 Mode: BATCH (Folder: {CONFIG['INPUT_DIR']})")
            run_batch(processor, CONFIG)
            
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