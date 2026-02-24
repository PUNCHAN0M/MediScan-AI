import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime

# ------------------- Configuration -------------------
MODEL_PATH = "D:\\project\\Medicine-AI\\MediScan-AI\\model\\pill-detection-best-2.pt"
IMG_SIZE = 256
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
PAD = 0          # padding รอบ segmentation mask (pixel)

# โฟลเดอร์สำหรับบันทึกภาพ (จะสร้างอัตโนมัติถ้ายังไม่มี)
SAVE_BASE_DIR = "data"
PILL_NAME = "yellow_cap/yellow_cap"          # ← เปลี่ยนตรงนี้เป็นชื่อยา/ประเภทที่ต้องการ เช่น "paracetamol", "amoxicillin" เป็นต้น
SAVE_DIR = os.path.join(SAVE_BASE_DIR, PILL_NAME, "train", "good")
# SAVE_DIR = os.path.join(SAVE_BASE_DIR, PILL_NAME, "test", "good")
# SAVE_DIR = os.path.join(SAVE_BASE_DIR, PILL_NAME, "test", "cracked")  # เปลี่ยนโฟลเดอร์ย่อยตามต้องการ เช่น "good", "cracked", "dirty"

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"ภาพที่บันทึกจะไปที่: {SAVE_DIR}")

# ตัวนับเพื่อให้ชื่อไฟล์ไม่ซ้ำ (ถ้าไม่ต้องการใช้ timestamp)
global_counter = 0


def load_model():
    print("กำลังโหลดโมเดล YOLO...")
    model = YOLO(MODEL_PATH)
    print("โหลดโมเดลเสร็จสิ้น")
    return model


def detect_pills(model, frame):
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        retina_masks=True,
        verbose=False
    )
    return results
def make_square_with_black_padding(image, target_size=None):
    """
    ทำให้ภาพเป็นสี่เหลี่ยมจัตุรัส โดยเพิ่ม padding สีดำรอบด้าน
    ถ้าไม่ระบุ target_size จะใช้ขนาดด้านยาวที่สุดของภาพเดิม
    
    Parameters:
        image: numpy array (ภาพที่ต้องการทำให้เป็น square)
        target_size: int (ขนาดด้านของ square ที่ต้องการ ถ้า None จะใช้ max(h,w))
    
    Returns:
        square_image: numpy array ขนาด target_size x target_size
    """
    if image is None or image.size == 0:
        return None
        
    h, w = image.shape[:2]
    
    # ถ้าไม่ได้ระบุ target_size ให้ใช้ขนาดด้านยาวที่สุด
    if target_size is None:
        target_size = max(h, w)
    
    # คำนวณ padding แต่ละด้าน
    pad_top = (target_size - h) // 2
    pad_bottom = target_size - h - pad_top
    pad_left = (target_size - w) // 2
    pad_right = target_size - w - pad_left
    
    # สร้างภาพพื้นหลังสีดำขนาด target
    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # วางภาพเดิมลงตรงกลาง
    square_img[pad_top:pad_top + h, pad_left:pad_left + w] = image
    
    return square_img

def crop_segmented_pills(frame, results, pad=PAD):
    if not results or results[0].masks is None:
        return [], []

    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    orig_h, orig_w = frame.shape[:2]
    cropped_list = []
    info_list = []

    for i, (mask, box, conf, cls) in enumerate(zip(masks, boxes, confs, classes)):
        mask = (mask > 0.5).astype(np.uint8) * 255

        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            continue

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x1 = max(0, x_min - pad)
        y1 = max(0, y_min - pad)
        x2 = min(orig_w, x_max + pad)
        y2 = min(orig_h, y_max + pad)

        crop_img = frame[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]

        # สร้างภาพเม็ดยาบนพื้นหลังดำ
        pill_on_black = np.zeros_like(crop_img)
        pill_on_black[crop_mask == 255] = crop_img[crop_mask == 255]

        # ─── ใช้ฟังก์ชันใหม่ตรงนี้ ───
        square_img = make_square_with_black_padding(pill_on_black)
        # หรือถ้าต้องการกำหนดขนาดตายตัว เช่น 512x512
        # square_img = make_square_with_black_padding(pill_on_black, target_size=512)

        cropped_list.append(square_img)

        info_list.append({
            'original_bbox': box.astype(int).tolist(),
            'padded_crop_region': (x1, y1, x2, y2),
            'confidence': float(conf),
            'class_id': int(cls),
            'crop_shape': square_img.shape[:2],
            'square_target_size': square_img.shape[0]
        })

    return cropped_list, info_list

def draw_detections(frame, pill_info):
    vis = frame.copy()

    for info in pill_info:
        x1, y1, x2, y2 = info['padded_crop_region']
        conf = info['confidence']
        color = (100, 255, 100)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(vis, f"Pills: {len(pill_info)}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 255), 3)

    return vis


def preprocess_realtime(img):
    out = img.astype(np.float32)

    # ===== อ่านค่าจาก slider =====
    brightness = cv2.getTrackbarPos("brightness", "Controls") - 100
    contrast   = cv2.getTrackbarPos("contrast", "Controls") / 100
    r_gain     = cv2.getTrackbarPos("red", "Controls") / 100
    g_gain     = cv2.getTrackbarPos("green", "Controls") / 100
    b_gain     = cv2.getTrackbarPos("blue", "Controls") / 100
    clahe_val  = cv2.getTrackbarPos("clahe", "Controls")
    sharp_val  = cv2.getTrackbarPos("sharp", "Controls")
    gamma_val  = cv2.getTrackbarPos("gamma", "Controls") / 100
    denoise    = cv2.getTrackbarPos("denoise", "Controls")
    lap_edge   = cv2.getTrackbarPos("lap_edge", "Controls")

    # ===== brightness + contrast =====
    out = out * contrast + brightness

    # ===== RGB gain =====
    out[:,:,2] *= r_gain
    out[:,:,1] *= g_gain
    out[:,:,0] *= b_gain

    out = np.clip(out, 0, 255).astype(np.uint8)

    # ===== CLAHE =====
    if clahe_val > 0:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_val/10+1, tileGridSize=(8,8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

    # ===== Denoise =====
    if denoise > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None, denoise, denoise, 7, 21)

    # ===== Sharpen =====
    if sharp_val > 0:
        blur = cv2.GaussianBlur(out,(0,0),3)
        out = cv2.addWeighted(out, 1+sharp_val/30, blur, -sharp_val/30, 0)

    # ===== Laplacian Edge Boost (คล้าย SIFE) =====
    if lap_edge > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        lap = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 1.0, lap, lap_edge/100, 0)

    # ===== Gamma =====
    if gamma_val != 1:
        table = np.array([(i/255.0)**(1/gamma_val)*255 for i in range(256)]).astype("uint8")
        out = cv2.LUT(out, table)

    return out



def save_cropped_pills(cropped_pills, pill_info):
    """
    บันทึกภาพ cropped ทั้งหมดที่ตรวจพบในเฟรมล่าสุด
    ใช้ timestamp + confidence เพื่อให้ชื่อไฟล์ไม่ซ้ำและมีข้อมูลคร่าวๆ
    """
    global global_counter

    if not cropped_pills:
        print("ไม่มีภาพเม็ดยาที่จะบันทึก")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_count = 0
    
    for i, (crop_img, info) in enumerate(zip(cropped_pills, pill_info)):
        conf_str = f"{info['confidence']:.3f}".replace(".", "")
        # ตัวเลือก 1: ใช้ timestamp + ลำดับ + conf (แนะนำ)
        filename = f"good_{timestamp}_c{i+1}_conf{conf_str}.jpg"
        
        # ตัวเลือก 2: ใช้ counter แบบเพิ่มเรื่อยๆ (ถ้าต้องการลำดับตลอดการรัน)
        # filename = f"good_{global_counter:05d}.jpg"
        # global_counter += 1

        save_path = os.path.join(SAVE_DIR, filename)
        
        success = cv2.imwrite(save_path, crop_img)
        if success:
            print(f"บันทึกสำเร็จ: {save_path}")
            saved_count += 1
        else:
            print(f"บันทึกล้มเหลว: {save_path}")

    print(f"→ บันทึกทั้งหมด {saved_count} ภาพในเฟรมนี้")

# ------------------- Realtime Controller -------------------
def nothing(x):
    pass

def create_control_panel():
    """
    มีแสงอยู่ข้าง
    BRIGHT_DEFAULT = 75
    CONTRAST_DEFAULT = 66  # 1.0
    CLAHE_DEFAULT = 50
    SHARP_DEFAULT = 27
    GAMMA_DEFAULT = 136     # 1.0
    ไม่มีแสงอยู่ข้าง
    BRIGHT_DEFAULT = 200
    CONTRAST_DEFAULT = 66  # 1.0
    CLAHE_DEFAULT = 18
    SHARP_DEFAULT = 0
    GAMMA_DEFAULT = 43     # 1.0

    """
    BRIGHT_DEFAULT = 75
    CONTRAST_DEFAULT = 66  # 1.0
    CLAHE_DEFAULT = 50
    SHARP_DEFAULT = 27
    GAMMA_DEFAULT = 136     # 1.0

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 420, 500)

    cv2.createTrackbar("red", "Controls", 100, 200, nothing)
    cv2.createTrackbar("green", "Controls", 100, 200, nothing)
    cv2.createTrackbar("blue", "Controls", 100, 200, nothing)

    cv2.createTrackbar("brightness", "Controls", BRIGHT_DEFAULT, 200, nothing)
    cv2.createTrackbar("contrast", "Controls", CONTRAST_DEFAULT, 300, nothing)

    cv2.createTrackbar("clahe", "Controls", CLAHE_DEFAULT, 50, nothing)
    cv2.createTrackbar("sharp", "Controls", SHARP_DEFAULT, 100, nothing)
    cv2.createTrackbar("gamma", "Controls", GAMMA_DEFAULT, 300, nothing)

    cv2.createTrackbar("denoise", "Controls", 0, 20, nothing)
    cv2.createTrackbar("lap_edge", "Controls", 0, 100, nothing)

# ------------------- Main Realtime Loop -------------------
def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 780)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้!")
        return

    print("กด ESC เพื่อออก")
    print("กด p เพื่อดู crop เซกเมนต์ทั้งหมด (สูงสุด 6)")
    print("กด s เพื่อบันทึกภาพ cropped ทุกเม็ดในเฟรมปัจจุบัน ลง", SAVE_DIR)
    create_control_panel()

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("อ่านเฟรมล้มเหลว")
            break

        frame_pre = preprocess_realtime(frame)

        results = detect_pills(model, frame_pre)

        cropped_pills, pill_info = crop_segmented_pills(frame_pre, results, pad=PAD)

        vis_frame = draw_detections(frame_pre, pill_info)

        cv2.imshow("Pill Detection - Realtime", vis_frame)

        if cropped_pills:
            cv2.imshow("Segmented Pill (white bg) - First", 
                       cv2.resize(cropped_pills[0], None, fx=1.8, fy=1.8))

        fps = 1 / (time.time() - start_time + 1e-6)
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break

        elif key == ord('p') and cropped_pills:
            for i, crop in enumerate(cropped_pills[:6]):
                title = f"Seg Pill {i+1} conf:{pill_info[i]['confidence']:.2f}"
                cv2.imshow(title, cv2.resize(crop, None, fx=2.0, fy=2.0))

        elif key == ord('s'):
            save_cropped_pills(cropped_pills, pill_info)

    cap.release()
    cv2.destroyAllWindows()
    print("โปรแกรมจบการทำงาน")


if __name__ == "__main__":
    main()