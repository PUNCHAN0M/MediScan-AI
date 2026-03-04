import cv2
import time
import threading
import numpy as np
from yolo_engine import YOLOEngine
from collections import OrderedDict


# ─────────────────────────────────────────────
# GRID (เร็วขึ้น: ใช้ INTER_LINEAR)
# ─────────────────────────────────────────────
def build_crop_grid(crops: dict, cell_size: int = 96, max_cols: int = 5):

    if not crops:
        return None

    items = list(crops.items())
    n = len(items)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    label_h = 18

    grid = np.zeros(
        (rows * (cell_size + label_h), cols * cell_size, 3),
        dtype=np.uint8
    )

    for idx, (tid, img) in enumerate(items):
        r = idx // cols
        c = idx % cols
        y = r * (cell_size + label_h)
        x = c * cell_size

        cell = cv2.resize(img, (cell_size, cell_size),
                          interpolation=cv2.INTER_LINEAR)

        grid[y + label_h:y + label_h + cell_size, x:x + cell_size] = cell

        cv2.putText(grid, f"ID:{tid}", (x + 4, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1)

    return grid


# ─────────────────────────────────────────────
# REALTIME (Threaded Inference)
# ─────────────────────────────────────────────
def main():

    engine = YOLOEngine(
        model_path="model\\SEGMENTATION\\pill-detection-best-2.onnx",   # .pt / .onnx / .engine
        mode="segment",
        device="cuda",
        retina_masks=False,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลด latency

    if not cap.isOpened():
        print("Camera error")
        return

    latest_frame = None
    latest_dets = []
    lock = threading.Lock()
    running = True

    # ─────────────────────────────────────────
    # INFERENCE THREAD
    # ─────────────────────────────────────────
    def inference_loop():
        nonlocal latest_frame, latest_dets, running

        while running:
            if latest_frame is None:
                continue

            with lock:
                frame_copy = latest_frame.copy()

            detections = engine.inference(frame_copy)

            with lock:
                latest_dets = detections

    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()

    # ─────────────────────────────────────────
    # DISPLAY LOOP
    # ─────────────────────────────────────────
    prev_time = time.time()

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            latest_frame = frame

        display = frame.copy()
        crops_dict = OrderedDict()

        with lock:
            detections = list(latest_dets)

        for idx, det in enumerate(detections):  # จำกัด max 10 objects

            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]

            cv2.rectangle(display, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(display,
                        f"{conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1)

            crop = engine.crop_object(frame, det)
            crops_dict[idx] = crop

        # Build grid (เร็วขึ้นเพราะ cell_size 96)
        grid = build_crop_grid(crops_dict)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.imshow("YOLO Live", display)

        if grid is not None:
            cv2.imshow("Patch Grid", grid)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()