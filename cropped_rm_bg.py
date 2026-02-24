import cv2
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime

# ─── Import จาก project ของคุณ ───
from config.base import SEGMENTATION_MODEL_PATH
from mobile_sife_cuda.core_predict.yolo_detector import OnnxYOLOModel, _load_yolo_model

# ─── Configuration ────────────────────────────────────────────────────────
MODEL_PATH = SEGMENTATION_MODEL_PATH

IMG_SIZE       = 1280
CONF_THRESHOLD = 0.4
IOU_THRESHOLD  = 0.45

SAVE_BASE_DIR = "data"
PILL_CATEGORY = "brown_cap_test\\brown_cap_test\\train"               # ← เปลี่ยนตามต้องการ เช่น "paracetamol", "amoxicillin"
SAVE_SUBDIR   = "good"                   # หรือ "cracked", "dirty", etc.
SAVE_DIR      = os.path.join(SAVE_BASE_DIR, PILL_CATEGORY, SAVE_SUBDIR)

CROP_MODE = "segmentation"                  # "segmentation" หรือ "detection"
PAD_SEG   = 0                            # ใช้เฉพาะ segmentation (ป้องกันขอบเม็ดยาติด)
PAD       = PAD_SEG if CROP_MODE == "segmentation" else 0

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Save to       : {SAVE_DIR}")
print(f"Mode          : {CROP_MODE.upper()}")
print(f"Padding       : {PAD}px (applied only in segmentation)")
print("Controls      : s = save crops | p = preview up to 6 | ESC = exit\n")

# ─── Helpers ──────────────────────────────────────────────────────────────

def make_square_black_pad(img: np.ndarray) -> np.ndarray:
    """ทำให้ภาพเป็น square ด้วย padding สีดำรอบนอก (ใช้ max side เป็นขนาด)"""
    if img is None or img.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    size = max(h, w)

    pad_top    = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left   = (size - w) // 2
    pad_right  = size - w - pad_left

    square = np.zeros((size, size, 3), dtype=np.uint8)
    square[pad_top:pad_top + h, pad_left:pad_left + w] = img
    return cv2.resize(square, (256, 256))


def preprocess_light(frame: np.ndarray) -> np.ndarray:
    """CLAHE ปรับความสว่างใน HSV"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def load_model():
    """โหลด YOLO (.pt หรือ .onnx)"""
    ext = Path(MODEL_PATH).suffix.lower()
    print(f"Loading model ({ext}): {MODEL_PATH}")

    model = _load_yolo_model(MODEL_PATH)

    if isinstance(model, OnnxYOLOModel):
        print("  → ONNX | " +
              ("segment" if model.is_segmentation else "detect") + " | " +
              ("dynamic" if model.is_dynamic else "fixed") + " | " +
              ("FP16" if model.is_fp16 else "FP32") + " | " +
              ("CUDA" if model.using_cuda else "CPU"))
    else:
        print("  → Ultralytics .pt")

    print("Model loaded\n")
    return model


def validate_imgsz(model, req_size: int) -> int:
    """ปรับ imgsz ให้เหมาะกับ ONNX ถ้าเป็น fixed/dynamic shape"""
    if not isinstance(model, OnnxYOLOModel):
        return req_size

    if model.is_dynamic:
        aligned = max(32, (req_size + 15) // 32 * 32)
        if aligned != req_size:
            print(f"imgsz adjusted → {aligned} (multiple of 32)")
        return aligned

    model_h = model._model_h
    if req_size != model_h:
        print(f"Using fixed ONNX shape: {model_h}")
    return model_h


def crop_pills(frame: np.ndarray, results, mode: str, pad: int) -> tuple[list, list]:
    """
    Crop ตามโหมด:
      segmentation → mask + ลบ bg + pad เล็กน้อย
      detection    → bbox เดิมเป๊ะ ๆ (pad=0 ไม่ขยาย)
    """
    if not results or not results[0].boxes:
        return [], []

    res = results[0] if isinstance(results, list) else results
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss  = res.boxes.cls.cpu().numpy().astype(int)

    use_mask = (mode == "segmentation") and (res.masks is not None)
    masks = res.masks.data.cpu().numpy() if use_mask else [None] * len(boxes)

    h, w = frame.shape[:2]
    cropped = []
    infos = []

    for box, conf, cls_id, mask in zip(boxes, confs, clss, masks):
        if use_mask and mask is not None:
            # Segmentation ── ลบ background
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), cv2.INTER_LINEAR)

            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            ys, xs = np.where(mask_bin)
            if len(xs) == 0:
                continue

            x1 = max(0, xs.min() - pad)
            y1 = max(0, ys.min() - pad)
            x2 = min(w, xs.max() + pad)
            y2 = min(h, ys.max() + pad)

            crop = frame[y1:y2, x1:x2].copy()
            m = mask_bin[y1:y2, x1:x2]

            pill = np.zeros_like(crop)
            pill[m == 255] = crop[m == 255]
            mode_str = "segmentation"

        else:
            # Detection ── tight bbox (ไม่ pad)
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            pill = frame[y1:y2, x1:x2].copy()
            mode_str = "detection (tight)"

        square = make_square_black_pad(pill)

        cropped.append(square)
        infos.append({
            "bbox": box.astype(int).tolist(),
            "crop_rect": (x1, y1, x2, y2),
            "conf": float(conf),
            "class": int(cls_id),
            "mode": mode_str,
        })

    return cropped, infos


def draw_results(frame: np.ndarray, infos: list) -> np.ndarray:
    vis = frame.copy()
    for info in infos:
        x1, y1, x2, y2 = info["crop_rect"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 255, 100), 2)
        cv2.putText(vis, f"{info['conf']:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    cv2.putText(vis, f"Pills: {len(infos)}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 255), 3)
    return vis


def save_crops(crops: list, infos: list):
    if not crops:
        print("No pills detected to save")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = 0

    for i, (img, info) in enumerate(zip(crops, infos)):
        conf_str = f"{info['conf']:.3f}".replace(".", "")
        name = f"{ts}_c{i+1}_conf{conf_str}.jpg"
        path = os.path.join(SAVE_DIR, name)

        if cv2.imwrite(path, img):
            print(f"Saved → {path}")
            saved += 1

    print(f"→ Saved {saved} images\n")


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    model = load_model()
    actual_size = validate_imgsz(model, IMG_SIZE)
    print(f"Input size: {actual_size}\n")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # Reset เป็น Auto Focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) 

    # cap.release()
    if not cap.isOpened():
        print("Camera open failed")
        return

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_proc = preprocess_light(frame)
        results = model(frame_proc, imgsz=actual_size, conf=CONF_THRESHOLD,
                        iou=IOU_THRESHOLD, retina_masks=True, verbose=False)

        crops, infos = crop_pills(frame_proc, results, CROP_MODE, PAD)

        vis = draw_results(frame_proc, infos)
        fps = 1 / (time.time() - t0 + 1e-6)
        cv2.putText(vis, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

        cv2.imshow("Pill Detection", vis)

        if crops:
            cv2.imshow("First Crop", cv2.resize(crops[0], None, fx=1.8, fy=1.8))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('p') and crops:
            for i, crop in enumerate(crops[:6]):
                title = f"Pill {i+1}  {infos[i]['conf']:.2f}"
                cv2.imshow(title, cv2.resize(crop, None, fx=2.0, fy=2.0))
        elif key == ord('s'):
            save_crops(crops, infos)

    cap.release()
    cv2.destroyAllWindows()
    print("Exit")


if __name__ == "__main__":
    main()