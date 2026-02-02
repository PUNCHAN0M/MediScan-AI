# 💊 MediScan AI - PatchCore Pill Anomaly Detection

> 🎯 ระบบตรวจจับความผิดปกติของยาเม็ดแบบ Realtime โดยใช้ PatchCore Algorithm + YOLO

## 📁 Project Structure

```
MediScan-AI/
├── 🧠 core_shared/              # อัลกอริทึม PatchCore
│   └── patchcore.py            # Memory bank & Feature extraction
│
├── 🎓 core_train/               # Training utilities
│   └── trainer.py              # PatchCore trainer
│
├── 🔍 core_predict/             # Inference utilities
│   ├── inspector.py            # PillInspector class (voting system)
│   ├── visualizer.py           # Drawing & visualization
│   └── yolo_detector.py        # YOLO segmentation wrapper
│
├── 🤖 model/                    # โมเดลที่ฝึกเสร็จ
│   ├── yolo12-seg.pt           # YOLO segmentation
│   └── patchcore/              # PatchCore models
│       └── {pill_name}/        # 📦 Parent class folder
│           ├── {pill_name}_front.pth
│           ├── {pill_name}_back.pth
│           └── {pill_name}_side.pth
│
├── 📊 data/                     # Dataset สำหรับ training
│   └── {pill_name}/            # 📦 Parent class folder (เช่น vitaminc)
│       ├── {pill_name}_front/  # 🔄 Subclass folder (ด้านหน้า)
│       │   ├── train/good/     # ✅ ภาพปกติสำหรับ train
│       │   └── test/good/      # 🎯 ภาพปกติสำหรับ calibrate threshold
│       ├── {pill_name}_back/   # 🔄 Subclass folder (ด้านหลัง)
│       └── {pill_name}_side/   # 🔄 Subclass folder (ด้านข้าง)
│
├── 🎓 train_patchcore_multi.py  # Script สำหรับ train model
└── 📹 predict_camera.py         # Script สำหรับตรวจสอบผ่านกล้อง
```

## 🚀 Quick Start

### 1️⃣ ติดตั้ง Environment

```bash
# สร้าง conda environment
conda create --name MediScan python=3.10
conda activate MediScan

# ติดตั้ง dependencies
pip install ultralytics opencv-python numpy torch faiss-cpu
```

## 🎓 Training Models

### ⚙️ การตั้งค่า (train_patchcore_multi.py)

```python
# 📂 ที่อยู่ไฟล์
DATA_ROOT = Path("./data")                      # โฟลเดอร์ข้อมูล
MODEL_OUTPUT_DIR = Path("./model/patchcore")    # โฟลเดอร์บันทึกโมเดล

# 🎛️ พารามิเตอร์โมเดล
IMG_SIZE = 512          # ขนาดภาพที่ใช้
GRID_SIZE = 20          # แบ่งภาพเป็น 20×20 = 400 patches
CORESSET_RATIO = 0.12   # เก็บ 12% ของ features (ลดขนาดโมเดล)
K_NEAREST = 4           # เปรียบเทียบกับ 4 ตัวอย่างที่ใกล้ที่สุด

# 🎯 เลือกยาที่ต้องการ train (parent class)
SELECTED_CLASSES = ["vitaminc", "white", "paracap"]
```

### 📝 ขั้นตอนการ Train

| Step | คำอธิบาย |
|------|----------|
| 1️⃣ **Scan** | หา subclass folders ภายใต้ parent class |
| 2️⃣ **Extract** | ดึง features จากภาพใน `train/good/` |
| 3️⃣ **Subsample** | ลดจำนวน patches ตาม `CORESSET_RATIO` |
| 4️⃣ **Calibrate** | คำนวณ threshold จากภาพใน `test/good/` |
| 5️⃣ **Save** | บันทึกเป็นไฟล์ `.pth` แยกตาม subclass |

### 🏃 วิธีใช้งาน

```bash
python train_patchcore_multi.py
```

### 📦 โครงสร้างไฟล์ที่ได้

```
model/patchcore/
├── 💊 vitaminc/
│   ├── vitaminc_front.pth    # โมเดลด้านหน้า
│   ├── vitaminc_back.pth     # โมเดลด้านหลัง
│   └── vitaminc_side.pth     # โมเดลด้านข้าง
├── 💊 white/
│   ├── white_front.pth
│   └── white_back.pth
└── 💊 paracap/
    └── paracap.pth
```

### 📋 รูปแบบไฟล์ .pth

```python
{
    "memory_bank": torch.Tensor,    # Feature vectors (n_patches × feature_dim)
    "threshold": float,              # เกณฑ์การตัดสิน (ปกติ/ผิดปกติ)
    "meta": {
        "n_patches": int,            # จำนวน patches ที่เก็บไว้
        "created_at": str,           # วันที่สร้างโมเดล
        "parent_class": str,         # ชื่อยา (parent class)
        # ... metadata อื่นๆ
    }
}
```

## 📹 Realtime Inspection

### ⚙️ การตั้งค่า (predict_camera.py)

```python
# 🎯 เลือกยาที่ต้องการตรวจ (โหลดทุก subclass จาก model/patchcore/{class_name}/)
COMPARE_CLASSES = ["vitaminc", "paracap"]

# 📂 ตำแหน่งโมเดล
MODEL_DIR = Path("./model/patchcore")
SEGMENTATION_MODEL_PATH = "model/yolo12-seg.pt"
```

### 🏃 วิธีใช้งาน

```bash
python predict_camera.py
```

### ⌨️ Hotkeys

| Key | หน้าที่ |
|-----|---------|
| `s` | 💾 บันทึกภาพยาที่ตรวจพบ |
| `r` | 🔄 รีเซ็ตการตรวจสอบ |
| `Enter` | 📊 แสดงสรุปผลทันที |
| `ESC` / `q` | 🚪 ออกจากโปรแกรม |

### 🎯 วิธีการทำงาน

1. 📷 เปิดกล้อง → ตรวจจับยาด้วย YOLO
2. 🔍 Extract features จากแต่ละยาที่พบ
3. 🧮 คำนวณ anomaly score เปรียบเทียบกับ memory bank
4. 🗳️ Voting system → สะสมผลหลายเฟรม
5. ✅ / ❌ แสดงผลว่ายาปกติหรือผิดปกติ

---

## 🧩 Core Modules

| Module | คำอธิบาย |
|--------|----------|
| 🧠 `patchcore.py` | **PatchCore Algorithm**<br>• Memory bank management<br>• Feature extraction<br>• Anomaly scoring |
| 🔍 `inspector.py` | **PillInspector Class**<br>• Inference pipeline<br>• Voting system<br>• Multi-class comparison |
| 🎨 `visualizer.py` | **Visualization**<br>• Drawing functions<br>• Status overlay<br>• Summary display |
| 🤖 `yolo_detector.py` | **YOLO Wrapper**<br>• Pill detection<br>• Segmentation<br>• Bounding box extraction |

---

## 📚 คำอธิบายเทคนิค

### 🎯 PatchCore Algorithm

- แบ่งภาพเป็น **patches** ขนาดเล็ก (เช่น 20×20 = 400 patches)
- Extract **features** จากแต่ละ patch ด้วย pre-trained model
- เก็บ features ของภาพปกติไว้ใน **memory bank**
- เปรียบเทียบภาพใหม่กับ memory bank → คำนวณ **anomaly score**
- ถ้า score > threshold → **ผิดปกติ** ❌
- ถ้า score ≤ threshold → **ปกติ** ✅

### 🗳️ Voting System

- สะสมผลหลายเฟรม (เช่น 5 เฟรม) ก่อนตัดสิน
- แต่ละเฟรมที่ผิดปกติจะ +1 คะแนน
- ถ้าผิดปกติ ≥ 1 เฟรม → ยาเม็ดนั้น**ผิดปกติ** ❌
- ช่วยลด false positive และเพิ่มความแม่นยำ

---

## 💡 Tips & Best Practices

### 📸 การเตรียมข้อมูล
- ใช้ภาพคุณภาพสูง ความละเอียดชัด
- ถ่ายยาในมุมที่ต่างกัน (หน้า/หลัง/ข้าง)
- แสงสม่ำเสมอ ไม่มีเงารบกวน
- อย่างน้อย **20-30 ภาพ** ต่อ class สำหรับ training

### ⚙️ การปรับ Parameters
- **IMG_SIZE** ↑ → ละเอียดมากขึ้น แต่ช้าลง
- **GRID_SIZE** ↑ → patches เยอะขึ้น แม่นยำขึ้น
- **CORESSET_RATIO** ↓ → โมเดลเล็กลง เร็วขึ้น
- **K_NEAREST** ↑ → พิจารณาตัวอย่างเยอะขึ้น

### 🎯 การปรับ Threshold
- Test ด้วยภาพปกติ → ดู anomaly score
- ตั้ง threshold ให้สูงกว่า max score ของภาพปกติเล็กน้อย
- ถ้า false positive มาก → เพิ่ม threshold
- ถ้า false negative มาก → ลด threshold

---

## 📞 Support

มีปัญหาหรือต้องการความช่วยเหลือ? 
- 📖 อ่าน documentation ใน code comments
- 🐛 รายงาน bug ผ่าน Issues
- 💬 ถาม questions ในทีม

---

**Made with ❤️ for quality assurance in pharmaceutical industry**
