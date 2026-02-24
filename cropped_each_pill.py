import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from pillow_heif import register_heif_opener
from PIL import Image

# เปิดใช้งานการอ่านไฟล์ HEIC สำหรับ Pillow
register_heif_opener()

# โหลดโมเดล YOLO Segmentation ของคุณ
model = YOLO('model\\pill-detection-best-2.pt') 

# ตั้งค่า Path
input_dir = Path("data_yolo/panadal")
output_dir = Path("data_yolo/panadal_cropped")

# สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
output_dir.mkdir(parents=True, exist_ok=True)

# ค้นหาไฟล์ .heic (และ .heci)
image_paths = list(input_dir.glob("*.heic")) + list(input_dir.glob("*.heci"))
imgz_pill = 256

for img_path in image_paths:
    # 1. อ่านไฟล์ HEIC ด้วย Pillow และแปลงเป็น Numpy Array สำหรับ OpenCV
    try:
        img_pil = Image.open(img_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"อ่านไฟล์ {img_path.name} ไม่สำเร็จ: {e}")
        continue

    # 2. ให้ YOLO ทำนาย (Segmentation)
    results = model(img_cv)

    for result in results:
        if result.masks is not None:
            # วนลูปตามจำนวนเม็ดยา (Masks) ที่ตรวจพบในรูปนี้
            for i, mask_tensor in enumerate(result.masks.data):
                
                # ดึง Mask แต่ละอันออกมาและปรับขนาดให้ตรงกับรูปต้นฉบับ
                mask = mask_tensor.cpu().numpy()
                mask = cv2.resize(mask, (img_cv.shape[1], img_cv.shape[0]))
                
                # 3. สร้างพื้นหลังสีดำและดึงเฉพาะส่วนที่เป็นเม็ดยามา
                black_bg = np.zeros_like(img_cv)
                pill_isolated = np.where(mask[..., None] > 0.25, img_cv, black_bg)

                # 4. หา Bounding Box ของเม็ดยาเพื่อนำมา Crop
                coords = np.column_stack(np.where(mask > 0.25))
                if coords.size == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # ตัดภาพ (Crop) ให้เหลือแค่กรอบของเม็ดยา
                cropped_pill = pill_isolated[y_min:y_max, x_min:x_max]
                
                # 5. ทำ Center Crop แบบเติมพื้นหลังให้เป็นจัตุรัส
                h, w = cropped_pill.shape[:2]
                max_side = max(h, w)
                square_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)
                
                y_offset = (max_side - h) // 2
                x_offset = (max_side - w) // 2
                square_img[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_pill
                
                # 6. ย่อขนาดเป็น imgz_pill x imgz_pill
                final_img = cv2.resize(square_img, (imgz_pill, imgz_pill))
                
                # 7. บันทึกเป็นไฟล์ .jpg โดยตั้งชื่อแยกตามลำดับเม็ดยา (เช่น image_1.jpg, image_2.jpg)
                save_path = output_dir / f"{img_path.stem}_{i+1}.jpg"
                cv2.imwrite(str(save_path), final_img)
                print(f"บันทึกสำเร็จ: {save_path}")