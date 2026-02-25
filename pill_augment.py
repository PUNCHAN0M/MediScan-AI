#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pill Dataset Augmentation System (Minimal Version)
สร้างเฉพาะภาพ Combined Augmented และ Original เท่านั้น
"""

import cv2
import os
import numpy as np
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

# =============================================================================
# ⚙️ CONFIGURATION
# =============================================================================
CONFIG = {
    # 📁 Input/Output Paths
    "INPUT_DIR": "data_yolo/test_cropped_pill/",
    "OUTPUT_DIR": "data_yolo/augmented_dataset/",
    
    # 🎯 Augmentation Settings
    "AUGMENT_COUNT": 5,
    
    # ☀️ Brightness Adjustment
    "USE_BRIGHTNESS": True,
    "BRIGHTNESS_MIN": 0.5,
    "BRIGHTNESS_MAX": 1.5,
    
    # 🔄 Rotation
    "USE_ROTATION": True,
    "ROTATION_MIN": 0,
    "ROTATION_MAX": 360,
    
    # ↔️ Flip
    "USE_FLIP": True,
    "FLIP_OPTIONS": ["horizontal", "vertical", "none"],
    
    # 🖼️ Image Settings
    "FINAL_SIZE": 256,
    "SAVE_FORMAT": "png",
    "JPEG_QUALITY": 95,
    
    # 🎨 Padding Settings
    "PADDING_COLOR": [255, 255, 255],  # [B, G, R] - ขาว
    
    # 🗑️ Output Mode: True = เก็บแค่ combined + original, False = เก็บทุกแบบแยกโฟลเดอร์
    "COMBINED_ONLY": True,
}

# =============================================================================
# 🛠️ SETUP & LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class PillAugmenter:
    """Class สำหรับทำ Data Augmentation บนภาพยา"""
    
    def __init__(self, config: dict):
        self.cfg = config
        self.input_dir = Path(config["INPUT_DIR"])
        self.output_dir = Path(config["OUTPUT_DIR"])
        
        # ✅ สร้างแค่โฟลเดอร์หลักโฟลเดอร์เดียว (Minimal)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Input: {self.input_dir}")
        logger.info(f"📁 Output: {self.output_dir} (Minimal Mode: {config.get('COMBINED_ONLY', True)})")
        logger.info(f"🔢 Augment Count per Image: {config['AUGMENT_COUNT']}")
        
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        if not self.cfg["USE_BRIGHTNESS"]:
            return image.copy()
        factor = random.uniform(self.cfg["BRIGHTNESS_MIN"], self.cfg["BRIGHTNESS_MAX"])
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
        
    def _resize_with_padding(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = image.shape[:2]
        is_grayscale = (len(image.shape) == 2)
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        if is_grayscale:
            padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            padding_color = self.cfg.get("PADDING_COLOR", [255, 255, 255])
            padded = np.ones((target_h, target_w, 3), dtype=np.uint8)
            padded[:] = padding_color
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def _rotate_image(self, image: np.ndarray) -> np.ndarray:
        if not self.cfg["USE_ROTATION"]:
            return image.copy()
        angle = random.uniform(self.cfg["ROTATION_MIN"], self.cfg["ROTATION_MAX"])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return rotated
    
    def _flip_image(self, image: np.ndarray) -> np.ndarray:
        if not self.cfg["USE_FLIP"]:
            return image.copy()
        flip_type = random.choice(self.cfg["FLIP_OPTIONS"])
        if flip_type == "horizontal":
            return cv2.flip(image.copy(), 1)
        elif flip_type == "vertical":
            return cv2.flip(image.copy(), 0)
        else:
            return image.copy()
    
    def augment_image(self, image: np.ndarray, img_name: str, aug_idx: int) -> dict:
        """ทำ Augmentation และคืนค่าเฉพาะ combined"""
        combined_img = image.copy()
        
        if self.cfg["USE_BRIGHTNESS"]:
            combined_img = self._adjust_brightness(combined_img)
        if self.cfg["USE_ROTATION"]:
            combined_img = self._rotate_image(combined_img)
        if self.cfg["USE_FLIP"]:
            combined_img = self._flip_image(combined_img)
        
        # ✅ รับประกันขนาดสุดท้าย
        if combined_img.shape[0] != self.cfg["FINAL_SIZE"] or combined_img.shape[1] != self.cfg["FINAL_SIZE"]:
            combined_img = self._resize_with_padding(combined_img, self.cfg["FINAL_SIZE"], self.cfg["FINAL_SIZE"])
        
        return {"combined": combined_img}
    
    def save_image(self, image: np.ndarray, filename: str) -> str:
        """บันทึกภาพลงโฟลเดอร์ output โดยตรง"""
        save_path = self.output_dir / filename
        fmt = self.cfg["SAVE_FORMAT"].lower()
        if fmt in ["jpg", "jpeg"]:
            cv2.imwrite(str(save_path), image, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["JPEG_QUALITY"]])
        else:
            cv2.imwrite(str(save_path), image)
        return str(save_path)
    
    def process_dataset(self):
        """ประมวลผลทั้งหมดในโฟลเดอร์ Input"""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(self.input_dir.glob(ext))
            
        if not image_files:
            logger.error(f"❌ No images found in {self.input_dir}")
            return
            
        logger.info(f"🚀 Found {len(image_files)} images")
        total_saved = 0
        
        for idx, img_path in enumerate(image_files, 1):
            logger.info(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
            
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"⚠️ Could not read: {img_path}")
                continue
            
            # ✅ Resize ต้นฉบับด้วย padding
            if img.shape[0] != self.cfg["FINAL_SIZE"] or img.shape[1] != self.cfg["FINAL_SIZE"]:
                img = self._resize_with_padding(img, self.cfg["FINAL_SIZE"], self.cfg["FINAL_SIZE"])
            
            img_name = img_path.stem
            
            # ✅ บันทึกเฉพาะ Combined Augmented Images
            for aug_idx in range(self.cfg["AUGMENT_COUNT"]):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                aug_results = self.augment_image(img, img_name, aug_idx)
                
                # บันทึกเฉพาะ combined
                filename = f"{img_name}_aug{aug_idx:02d}_combined_{timestamp}.{self.cfg['SAVE_FORMAT']}"
                self.save_image(aug_results["combined"], filename)
                total_saved += 1
            
            # ✅ บันทึกภาพต้นฉบับ (เพื่อใช้อ้างอิงหรือทำ backbone)
            orig_filename = f"{img_name}_original.{self.cfg['SAVE_FORMAT']}"
            self.save_image(img, orig_filename)
            total_saved += 1
            
        logger.info(f"✅ Augmentation Complete!")
        logger.info(f"💾 Total images saved: {total_saved}")
        self._print_summary(len(image_files), total_saved)
        
    def _print_summary(self, original_count: int, augmented_count: int):
        print("\n" + "="*60)
        print("📊 AUGMENTATION SUMMARY (Minimal)")
        print("="*60)
        print(f"Original Images:     {original_count}")
        print(f"Augmented Images:    {augmented_count - original_count}")
        print(f"Total Output:        {augmented_count}")
        print("-"*60)
        print(f"📁 Output Directory: {self.output_dir}")
        print("🗂️  Files saved directly (no subfolders)")
        print("="*60 + "\n")


# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

def main():
    logger.info("🔷 Pill Augmentation System Starting...")
    try:
        augmenter = PillAugmenter(CONFIG)
        augmenter.process_dataset()
    except KeyboardInterrupt:
        logger.info("⚡ Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        import sys
        sys.exit(1)
    logger.info("🔚 System shutdown complete")

if __name__ == "__main__":
    main()