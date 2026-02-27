# pill_process.py
import os
import shutil
import random
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Tuple
from config.base import SELECTED_CLASSES, SEED

# =============================================================================
# ⚙️ CONFIGURATION
# =============================================================================

ROOT_INPUT_DIR = "data"
INPUT_MAINCLASS_DIR: Optional[str] = None  #['black_sphere']
INPUT_SUBCLASS_DIR: Optional[str] = None  #['black_sphere','brown_cap']

ROOT_BACKBONE_OUTPUT_DIR = "data_train_backbone"
ROOT_TRAIN_OUTPUT_DIR = "data_train_defection"

AUGMENT_COUNT_PER_IMAGE = 5
DEFECTION_MIN_IMAGES = 800
DEFECTION_MAX_IMAGES = 800
BACKBONE_MIN_IMAGES = 500
BACKBONE_MAX_IMAGES = 500
TRAIN_TEST_RATIO = 0.8  # 80% Train, 20% Test

# =============================================================================
# 🛠️ HELPER FUNCTIONS
# =============================================================================

def get_image_files(directory: Path) -> list[Path]:
    """ดึงรายชื่อไฟล์ภาพทั้งหมดใน directory"""
    if not directory.exists():
        return []
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
    files = []
    for ext in extensions:
        files.extend(directory.glob(ext))
    return files

def get_class_structure(
    root_dir: Path,
    main_filter: Optional[List[str] | str] = None,
    sub_filter: Optional[List[str] | str] = None
) -> List[Tuple[str, str]]:

    classes = []

    if not root_dir.exists():
        print(f"❌ Input directory not found: {root_dir}")
        return classes

    # 👉 ทำให้เป็น list เสมอ
    if isinstance(main_filter, str):
        main_filter = [main_filter]
    if isinstance(sub_filter, str):
        sub_filter = [sub_filter]

    main_classes = [d.name for d in root_dir.iterdir() if d.is_dir()]

    # ================= MAIN CLASS FILTER =================
    if main_filter:
        main_classes = [m for m in main_classes if m in main_filter]

        if not main_classes:
            print(f"⚠️ Main class not found. Available: {[d.name for d in root_dir.iterdir() if d.is_dir()]}")
            return classes

    # ================= LOOP =================
    for main_cls in sorted(main_classes):
        main_path = root_dir / main_cls
        sub_classes = [d.name for d in main_path.iterdir() if d.is_dir()]

        if sub_filter:
            filtered_sub = [s for s in sub_classes if s in sub_filter]

            if not filtered_sub:
                print(f"⚠️ Sub class not found in '{main_cls}', available: {sub_classes}")
                continue

            sub_classes = filtered_sub

        for sub_cls in sorted(sub_classes):
            classes.append((main_cls, sub_cls))
            print(f"📁 Found class: {main_cls}/{sub_cls}")

    return classes

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def limit_images(images: list[Path], min_count: int, max_count: int, label: str) -> list[Path]:
    """จำกัดจำนวนภาพตาม min/max config"""
    original_count = len(images)
    
    if original_count < min_count and images:
        print(f"⚠️ [{label}] Images ({original_count}) < min ({min_count}), duplicating...")
        needed = min_count - original_count
        extra = random.choices(images, k=needed)
        images = images + extra
        print(f"   → After duplication: {len(images)} images")
    elif original_count < min_count:
        print(f"⚠️ [{label}] No images to duplicate, skipping min requirement")
    
    if len(images) > max_count:
        print(f"⚠️ [{label}] Images ({len(images)}) > max ({max_count}), sampling down...")
        images = random.sample(images, max_count)
        print(f"   → After sampling: {len(images)} images")
    
    return images

def split_and_organize(images: list[Path], output_base: Path, label: str):
    """แบ่ง Train/Test จาก list ภาพที่ส่งเข้ามา (ไม่ทำ limit ที่นี่ เพราะทำตอนเรียกแล้ว)"""
    ensure_dir(output_base / "train" / "good")
    ensure_dir(output_base / "test" / "good")
    
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_TEST_RATIO)
    
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    
    for img in train_images:
        shutil.copy2(img, output_base / "train" / "good" / img.name)
    for img in test_images:
        shutil.copy2(img, output_base / "test" / "good" / img.name)
    
    print(f"✅ [{label}] Train: {len(train_images)}, Test: {len(test_images)} | Total: {len(images)}")

# =============================================================================
# 🔄 AUGMENTATION (Updated for Minimal Output)
# =============================================================================

def run_augmentation(input_dir: Path, output_dir: Path, augment_count: int) -> list[Path]:
    """
    เรียกใช้ pill_augment.py และคืนค่า list ของภาพทั้งหมดจาก output_dir
    (รองรับโหมด Minimal ที่ไม่มี subfolder)
    """
    from pill_augment import PillAugmenter, CONFIG as AUG_CONFIG
    
    aug_config = AUG_CONFIG.copy()
    aug_config["INPUT_DIR"] = str(input_dir)
    aug_config["OUTPUT_DIR"] = str(output_dir)
    aug_config["AUGMENT_COUNT"] = augment_count
    aug_config["COMBINED_ONLY"] = True  # ✅ บังคับใช้โหมด Minimal
    
    augmenter = PillAugmenter(aug_config)
    augmenter.process_dataset()
    
    # ✅ อ่านไฟล์จาก output_dir โดยตรง (ไม่มี subfolder 'combined')
    if output_dir.exists():
        augmented_images = get_image_files(output_dir)
        # กรองเอาเฉพาะไฟล์ augmented (ตัด original ออกถ้าต้องการแยกจัดการ)
        # แต่ในที่นี้เราส่งกลับทั้งหมดไปให้ limit_images จัดการต่อ
    else:
        augmented_images = []
        logging.warning(f"⚠️ Output folder not found: {output_dir}")
    
    return augmented_images

def copy_to_backbone(images: list[Path], target_dir: Path, min_count: int, max_count: int):
    """คัดลอกภาพไปยัง backbone directory ตาม min/max constraints"""
    ensure_dir(target_dir)
    
    if not images:
        print(f"⚠️ [backbone] No images to copy")
        return
    
    selected = limit_images(images, min_count, max_count, "backbone")
    
    for idx, img in enumerate(selected):
        new_name = f"aug_{idx:04d}_{img.name}"
        shutil.copy2(img, target_dir / new_name)
    
    print(f"🔧 Backbone: Copied {len(selected)} images to {target_dir}")

# =============================================================================
# 🚀 PROCESS SINGLE CLASS (Fixed Split Logic)
# =============================================================================

def process_class(main_cls: str, sub_cls: str):
    print(f"\n{'='*60}")
    print(f"🔷 Processing: {main_cls}/{sub_cls}")
    print(f"{'='*60}")

    input_dir = Path(ROOT_INPUT_DIR) / main_cls / sub_cls
    output_train_dir = Path(ROOT_TRAIN_OUTPUT_DIR) / main_cls / sub_cls
    output_backbone_dir = Path(ROOT_BACKBONE_OUTPUT_DIR) / f"{main_cls}_{sub_cls}"

    ensure_dir(output_train_dir / "train" / "good")
    ensure_dir(output_train_dir / "test" / "good")
    ensure_dir(output_backbone_dir)

    original_images = get_image_files(input_dir)
    print(f"📸 Found {len(original_images)} original images")

    if not original_images:
        print("⚠️ No images found, skipping...")
        return

    # ==========================================================
    # 🔥 STEP 1: AUGMENT → ได้ภาพทั้งหมดกองเดียว
    # ==========================================================
    aug_output_dir = output_train_dir / "aug_all"
    ensure_dir(aug_output_dir)

    print("🎨 Running augmentation...")
    all_augmented = run_augmentation(
        input_dir=input_dir,
        output_dir=aug_output_dir,
        augment_count=AUGMENT_COUNT_PER_IMAGE
    )

    print(f"📦 Total augmented images: {len(all_augmented)}")

    # limit จำนวนสำหรับ defection
    all_augmented = limit_images(
        all_augmented,
        DEFECTION_MIN_IMAGES,
        DEFECTION_MAX_IMAGES,
        "Defection All"
    )

    random.shuffle(all_augmented)

    # ==========================================================
    # 🔥 STEP 2: SPLIT → 800 / 200
    # ==========================================================
    split_idx = int(len(all_augmented) * TRAIN_TEST_RATIO)

    train_imgs = all_augmented[:split_idx]
    test_imgs = all_augmented[split_idx:]

    for img in train_imgs:
        shutil.copy2(img, output_train_dir / "train" / "good" / img.name)

    for img in test_imgs:
        shutil.copy2(img, output_train_dir / "test" / "good" / img.name)

    print(f"✅ Train: {len(train_imgs)} | Test: {len(test_imgs)}")

    # ==========================================================
    # 🔥 STEP 3: BACKBONE → สุ่มใหม่จาก ALL (ซ้ำได้)
    # ==========================================================
    copy_to_backbone(
        all_augmented,
        output_backbone_dir,
        BACKBONE_MIN_IMAGES,
        BACKBONE_MAX_IMAGES
    )

    # ==========================================================
    # 🧹 CLEANUP
    # ==========================================================
    shutil.rmtree(aug_output_dir)

    print(f"✅ Completed: {main_cls}/{sub_cls}")

# =============================================================================
# 🚀 MAIN ENTRY POINT
# =============================================================================

def main():
    print(f"🔷 Pill Processing Pipeline Starting...")
    print(f"📂 Root Input: {ROOT_INPUT_DIR}")
    print(f"🎯 Main Class Filter: {INPUT_MAINCLASS_DIR or 'ALL'}")
    print(f"🎯 Sub Class Filter: {INPUT_SUBCLASS_DIR or 'ALL'}")
    
    class_list = get_class_structure(
        Path(ROOT_INPUT_DIR), 
        main_filter=INPUT_MAINCLASS_DIR, 
        sub_filter=INPUT_SUBCLASS_DIR
    )
    
    if not class_list:
        print("❌ No classes found to process. Please check your data structure.")
        return
    
    print(f"\n📋 Total classes to process: {len(class_list)}")
    
    for main_cls, sub_cls in class_list:
        try:
            process_class(main_cls, sub_cls)
        except Exception as e:
            print(f"💥 Error processing {main_cls}/{sub_cls}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"🎉 Pipeline Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    random.seed(SEED)
    main()