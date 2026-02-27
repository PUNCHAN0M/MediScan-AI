#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pill Dataset Augmentation System (Minimal Version)
สร้างเฉพาะภาพ Combined Augmented และ Original เท่านั้น

Input: data_scrap_resize/{main_class}*/
Output: data_backbone_augment/{main_class}*/
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
    "INPUT_DIR": "data_scrap_resize/",  # เปลี่ยนเป็น data_scrap_resize
    "OUTPUT_DIR": "data_backbone_augment/",  # เปลี่ยนเป็น data_backbone_augment
    
    # 🎯 Augmentation Settings
    "AUGMENT_COUNT": 500,
    
    # ☀️ Brightness Adjustment
    "USE_BRIGHTNESS": True,
    "BRIGHTNESS_MIN": 0.8,
    "BRIGHTNESS_MAX": 1.2,
    
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
    "PADDING_COLOR": [0, 0, 0],  # [B, G, R] - BLACK padding
    
    # 🗑️ Output Mode: True = เก็บแค่ combined + original, False = เก็บทุกแบบแยกโฟลเดอร์
    "COMBINED_ONLY": True,
    
    # 📁 Pattern: รูปแบบของ main_class (ใช้ * แทน wildcard)
    "MAIN_CLASS_PATTERN": "*",  # จะ match ทุกโฟลเดอร์ที่อยู่ใน INPUT_DIR
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
        
        # ✅ สร้างโฟลเดอร์ output หลัก
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Input: {self.input_dir} (พร้อม main_class ย่อย)")
        logger.info(f"📁 Output: {self.output_dir} (จะสร้างโฟลเดอร์ตาม main_class)")
        logger.info(f"🔢 Augment Count per Image: {config['AUGMENT_COUNT']}")
        
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        if not self.cfg["USE_BRIGHTNESS"]:
            return image.copy()
        factor = random.uniform(self.cfg["BRIGHTNESS_MIN"], self.cfg["BRIGHTNESS_MAX"])
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    def _add_black_padding_to_square(self, image: np.ndarray) -> np.ndarray:
        """เพิ่ม padding สีดำให้ภาพเป็นสี่เหลี่ยมจัตุรัส โดยไม่ยืดภาพ"""
        h, w = image.shape[:2]
        
        # ถ้าเป็นสี่เหลี่ยมจัตุรัสอยู่แล้ว ไม่ต้องทำ padding
        if h == w:
            return image.copy()
        
        # หาขนาดด้านที่ยาวที่สุด
        max_side = max(h, w)
        
        # สร้างภาพพื้นหลังสีดำ
        if len(image.shape) == 2:  # Grayscale
            squared = np.zeros((max_side, max_side), dtype=np.uint8)
        else:  # Color
            squared = np.zeros((max_side, max_side, 3), dtype=np.uint8)
            squared[:] = self.cfg.get("PADDING_COLOR", [0, 0, 0])
        
        # วางภาพเดิมตรงกลาง
        y_offset = (max_side - h) // 2
        x_offset = (max_side - w) // 2
        squared[y_offset:y_offset+h, x_offset:x_offset+w] = image
        
        return squared
    
    def _rotate_image_any_angle(self, image: np.ndarray) -> np.ndarray:
        """หมุนภาพด้วยมุม 0-360 องศา"""
        if not self.cfg["USE_ROTATION"]:
            return image.copy()
        
        # สุ่มมุม 0-360
        angle = random.uniform(self.cfg["ROTATION_MIN"], self.cfg["ROTATION_MAX"])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # คำนวณ rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # คำนวณขนาดใหม่หลังจากหมุน
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # ปรับ rotation matrix สำหรับ center crop
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # ทำการหมุนด้วย padding สีดำ
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=self.cfg.get("PADDING_COLOR", [0, 0, 0]))
        
        return rotated
    
    def _crop_center(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """crop ตรงกลางภาพให้ได้ขนาด target_size x target_size"""
        h, w = image.shape[:2]
        
        # คำนวณตำแหน่ง crop
        start_y = (h - target_size) // 2
        start_x = (w - target_size) // 2
        
        # crop ตรงกลาง
        if len(image.shape) == 2:  # Grayscale
            cropped = image[start_y:start_y+target_size, start_x:start_x+target_size]
        else:  # Color
            cropped = image[start_y:start_y+target_size, start_x:start_x+target_size, :]
        
        return cropped
    
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
        """ทำ Augmentation ตามที่ต้องการ:
        1. ทำ padding สีดำให้เป็นสี่เหลี่ยมจัตุรัส
        2. หมุน 0-360 องศา
        3. crop ตรงกลาง 256x256
        """
        # ขั้นตอนที่ 1: ทำ padding สีดำให้เป็นสี่เหลี่ยมจัตุรัส
        squared_img = self._add_black_padding_to_square(image)
        
        # ขั้นตอนที่ 2: หมุนภาพด้วยมุม 0-360
        rotated_img = self._rotate_image_any_angle(squared_img)
        
        # ขั้นตอนที่ 3: crop ตรงกลางให้ได้ 256x256
        cropped_img = self._crop_center(rotated_img, self.cfg["FINAL_SIZE"])
        
        combined_img = cropped_img.copy()
        
        # ทำ brightness และ flip เพิ่มเติม
        if self.cfg["USE_BRIGHTNESS"]:
            combined_img = self._adjust_brightness(combined_img)
        if self.cfg["USE_FLIP"]:
            combined_img = self._flip_image(combined_img)
        
        return {"combined": combined_img}
    
    def save_image(self, image: np.ndarray, main_class: str, filename: str) -> str:
        """บันทึกภาพลงโฟลเดอร์ output ตาม main_class"""
        # สร้างโฟลเดอร์ตาม main_class
        class_output_dir = self.output_dir / main_class
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = class_output_dir / filename
        fmt = self.cfg["SAVE_FORMAT"].lower()
        if fmt in ["jpg", "jpeg"]:
            cv2.imwrite(str(save_path), image, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["JPEG_QUALITY"]])
        else:
            cv2.imwrite(str(save_path), image)
        return str(save_path)
    
    def process_main_class(self, main_class_dir: Path):
        """ประมวลผลภาพทั้งหมดใน main_class ที่กำหนด"""
        main_class = main_class_dir.name
        logger.info(f"📂 Processing main class: {main_class}")
        
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(main_class_dir.glob(ext))
        
        if not image_files:
            logger.warning(f"⚠️ No images found in {main_class_dir}")
            return 0, 0
        
        logger.info(f"   Found {len(image_files)} images in {main_class}")
        class_saved = 0
        
        for idx, img_path in enumerate(image_files, 1):
            logger.info(f"   [{idx}/{len(image_files)}] Processing: {img_path.name}")
            
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"   ⚠️ Could not read: {img_path}")
                continue
            
            img_name = img_path.stem
            
            # ✅ บันทึกภาพต้นฉบับ (หลังจากทำ padding สีดำให้เป็นสี่เหลี่ยม)
            original_squared = self._add_black_padding_to_square(img)
            orig_filename = f"{img_name}_original.{self.cfg['SAVE_FORMAT']}"
            self.save_image(original_squared, main_class, orig_filename)
            class_saved += 1
            
            # ✅ สร้างและบันทึก Augmented Images
            for aug_idx in range(self.cfg["AUGMENT_COUNT"]):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                aug_results = self.augment_image(img, img_name, aug_idx)
                
                # บันทึก augmented image
                filename = f"{img_name}_aug{aug_idx:02d}_combined_{timestamp}.{self.cfg['SAVE_FORMAT']}"
                self.save_image(aug_results["combined"], main_class, filename)
                class_saved += 1
        
        return len(image_files), class_saved
    
    def process_dataset(self):
        """ประมวลผลทั้งหมดในโฟลเดอร์ Input โดยวนทุก main_class"""
        # หาโฟลเดอร์ main_class ทั้งหมดใน input_dir
        pattern = self.cfg.get("MAIN_CLASS_PATTERN", "*")
        main_class_dirs = [d for d in self.input_dir.glob(f"{pattern}") if d.is_dir()]
        
        if not main_class_dirs:
            logger.error(f"❌ No main_class directories found in {self.input_dir}")
            logger.info(f"   Looking for pattern: {pattern}")
            return
        
        logger.info(f"🚀 Found {len(main_class_dirs)} main class directories")
        
        total_original = 0
        total_saved = 0
        
        for main_class_dir in sorted(main_class_dirs):
            orig_count, saved_count = self.process_main_class(main_class_dir)
            total_original += orig_count
            total_saved += saved_count
        
        logger.info(f"✅ Augmentation Complete!")
        logger.info(f"💾 Total images saved: {total_saved}")
        self._print_summary(total_original, total_saved)
        
    def _print_summary(self, original_count: int, augmented_count: int):
        print("\n" + "="*60)
        print("📊 AUGMENTATION SUMMARY")
        print("="*60)
        print(f"Original Images:     {original_count}")
        print(f"Augmented Images:    {augmented_count - original_count}")
        print(f"Total Output:        {augmented_count}")
        print("-"*60)
        print(f"📁 Output Directory: {self.output_dir}")
        print("🗂️  Files saved with main_class subfolders")
        print("="*60 + "\n")


# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

def main():
    logger.info("🔷 Pill Augmentation System Starting...")
    logger.info(f"📂 Input pattern: {CONFIG['INPUT_DIR']}{{main_class}}*/")
    logger.info(f"📂 Output: {CONFIG['OUTPUT_DIR']}{{main_class}}*/")
    
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