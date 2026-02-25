#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Image Resizer
Resize ภาพจากโฟลเดอร์ย่อยทั้งหมดให้เป็นขนาดที่กำหนด
"""

import cv2
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List

# =============================================================================
# ⚙️ CONFIGURATION (แก้ไขค่าต่างๆ ที่นี่)
# =============================================================================
CONFIG = {
    # 📁 Input/Output Paths
    "INPUT_DIR": "result/",              # โฟลเดอร์ต้นทาง (มีโฟลเดอร์ย่อยข้างใน)
    "OUTPUT_DIR": "resize_result/",      # โฟลเดอร์ปลายทาง
    
    # 🖼️ Image Settings
    "FINAL_SIZE": 256,                   # ขนาดภาพที่ต้องการ (256x256)
    "SAVE_FORMAT": "jpg",                # jpg หรือ png
    "JPEG_QUALITY": 95,                  # คุณภาพ JPEG (0-100)
    
    # 🔄 Behavior
    "OVERWRITE": False,                  # ถ้า True จะทับไฟล์เดิม, False จะข้าม
    "PRESERVE_STRUCTURE": True,          # ถ้า True จะรักษาโครงสร้างโฟลเดอร์ย่อย
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


class ImageResizer:
    """Class สำหรับ Resize ภาพแบบ Batch"""
    
    def __init__(self, config: dict):
        self.cfg = config
        self.input_dir = Path(config["INPUT_DIR"])
        self.output_dir = Path(config["OUTPUT_DIR"])
        
        # สร้างโฟลเดอร์ Output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Input: {self.input_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
        logger.info(f"📐 Target Size: {config['FINAL_SIZE']}x{config['FINAL_SIZE']}")
        
    def find_all_images(self) -> List[Path]:
        """ค้นหาไฟล์ภาพทั้งหมดในโฟลเดอร์ย่อย"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_files = []
        
        for ext in extensions:
            # ** = ค้นหาในโฟลเดอร์ย่อยทั้งหมด (recursive)
            image_files.extend(self.input_dir.rglob(ext))
            
        return sorted(image_files)
    
    def resize_image(self, img_path: Path) -> bool:
        """Resize ภาพเดียว"""
        try:
            # อ่านภาพ
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"⚠️ Could not read: {img_path}")
                return False
            
            # Resize
            resized = cv2.resize(
                img, 
                (self.cfg["FINAL_SIZE"], self.cfg["FINAL_SIZE"]), 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # กำหนด path ปลายทาง
            if self.cfg["PRESERVE_STRUCTURE"]:
                # รักษาโครงสร้างโฟลเดอร์ย่อย
                relative_path = img_path.relative_to(self.input_dir)
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # รวมทั้งหมดในโฟลเดอร์เดียว
                output_path = self.output_dir / img_path.name
            
            # เปลี่ยนนามสกุลถ้าต้องการ
            if self.cfg["SAVE_FORMAT"].lower() in ['jpg', 'jpeg']:
                output_path = output_path.with_suffix('.jpg')
            elif self.cfg["SAVE_FORMAT"].lower() == 'png':
                output_path = output_path.with_suffix('.png')
            
            # ตรวจสอบว่ามีไฟล์อยู่แล้วหรือไม่
            if output_path.exists() and not self.cfg["OVERWRITE"]:
                logger.debug(f"⏭️  Skipped (exists): {output_path.name}")
                return True
            
            # บันทึกภาพ
            fmt = self.cfg["SAVE_FORMAT"].lower()
            if fmt in ['jpg', 'jpeg']:
                cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["JPEG_QUALITY"]])
            else:
                cv2.imwrite(str(output_path), resized)
            
            logger.debug(f"✅ Resized: {img_path.name} -> {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {img_path}: {e}")
            return False
    
    def process_all(self):
        """ประมวลผลภาพทั้งหมด"""
        
        # หาไฟล์ภาพทั้งหมด
        image_files = self.find_all_images()
        
        if not image_files:
            logger.error(f"❌ No images found in {self.input_dir}")
            return
            
        logger.info(f"🚀 Found {len(image_files)} images")
        
        success_count = 0
        fail_count = 0
        
        for idx, img_path in enumerate(image_files, 1):
            logger.info(f"[{idx}/{len(image_files)}] Processing: {img_path.relative_to(self.input_dir)}")
            
            if self.resize_image(img_path):
                success_count += 1
            else:
                fail_count += 1
                
        # สรุปผล
        logger.info(f"✅ Resize Complete!")
        logger.info(f"📊 Success: {success_count} | Failed: {fail_count}")
        logger.info(f"📂 Output: {self.output_dir}")
        
        # พิมพ์สรุป
        self._print_summary(len(image_files), success_count, fail_count)
        
    def _print_summary(self, total: int, success: int, fail: int):
        """พิมพ์สรุปสถิติ"""
        print("\n" + "="*60)
        print("📊 RESIZE SUMMARY")
        print("="*60)
        print(f"Total Images:        {total}")
        print(f"Success:             {success}")
        print(f"Failed:              {fail}")
        print(f"Success Rate:        {(success/total*100):.1f}%")
        print("-"*60)
        print("🔧 Settings:")
        print(f"  • Target Size:     {self.cfg['FINAL_SIZE']}x{self.cfg['FINAL_SIZE']}")
        print(f"  • Output Format:   {self.cfg['SAVE_FORMAT']}")
        print(f"  • Preserve Structure: {self.cfg['PRESERVE_STRUCTURE']}")
        print("="*60 + "\n")


# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

def main():
    logger.info("🔷 Image Resizer Starting...")
    
    try:
        # ตรวจสอบ Input Directory
        if not Path(CONFIG["INPUT_DIR"]).exists():
            logger.error(f"❌ Input directory not found: {CONFIG['INPUT_DIR']}")
            return
            
        resizer = ImageResizer(CONFIG)
        resizer.process_all()
        
    except KeyboardInterrupt:
        logger.info("⚡ Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        import sys
        sys.exit(1)
    
    logger.info("🔚 System shutdown complete")


if __name__ == "__main__":
    main()