import cv2
import os
from datetime import datetime

# =========================
# Config
# =========================
SAVE_DIR = "data_yolo"
CAM_INDEX = 1

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

counter = 0
print("Press 's' to save | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # save full image
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{counter}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(save_path, frame)
        print("Saved:", save_path)

        counter += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
