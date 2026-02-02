import os
import cv2
import numpy as np
from pathlib import Path
import random

def augment_images(input_dir, augment_count=5, rotation_step=30):
    """
    Create augmented images with brightness adjustment, rotation, and flips.
    No contour detection - just augmentation.
    
    Args:
        input_dir: Directory containing images to process
        augment_count: Number of augmented images to create per original
        rotation_step: Rotation angle step (degrees) - will create rotations at multiples of this
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    # Filter out previously augmented images
    image_files = [f for f in image_files if '_aug' not in f.stem]
    
    print(f"Found {len(image_files)} original images to process")
    
    processed_count = 0
    failed_count = 0
    augmented_created = 0
    
    for img_path in image_files:
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read: {img_path}")
                failed_count += 1
                continue
            
            processed_count += 1
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            
            # Create augmentations
            stem = img_path.stem
            suffix = img_path.suffix
            
            for aug_idx in range(1, augment_count + 1):
                aug_img = img.copy()
                
                # 1. Rotation with configurable step
                # Random multiple of rotation_step (e.g., 0, 30, 60, 90, etc. if step=30)
                rotation_options = list(range(0, 360, rotation_step))
                angle = random.choice(rotation_options)
                if angle != 0:
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aug_img = cv2.warpAffine(
                        aug_img, M, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0)
                    )
                
                # 2. Random flip vertical
                if random.random() > 0.5:
                    aug_img = cv2.flip(aug_img, 0)
                
                # 3. Random flip horizontal
                if random.random() > 0.5:
                    aug_img = cv2.flip(aug_img, 1)
                
                # 4. Brightness adjustment
                brightness_delta = random.uniform(-50, 50)
                aug_img = cv2.convertScaleAbs(aug_img, alpha=1.0, beta=brightness_delta)
                
                # Save augmented image
                aug_path = img_path.parent / f"{stem}_aug{aug_idx}{suffix}"
                cv2.imwrite(str(aug_path), aug_img)
                augmented_created += 1
            
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} images, created {augmented_created} augmentations...")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            failed_count += 1
    
    print(f"\nCompleted!")
    print(f"Successfully processed: {processed_count}")
    print(f"Augmentations created: {augmented_created}")
    print(f"Failed: {failed_count}")


def process_directory_recursive(base_dir, augment_count=5, rotation_step=5):
    base_path = Path(base_dir)
    image_dirs = set()
    for img_file in base_path.rglob('*.jpg'):
        image_dirs.add(img_file.parent)
    for img_file in base_path.rglob('*.png'):
        image_dirs.add(img_file.parent)
    
    print(f"Found {len(image_dirs)} directories with images")
    
    for img_dir in sorted(image_dirs):
        print(f"\n{'='*60}")
        print(f"Processing: {img_dir}")
        print(f"{'='*60}")
        augment_images(str(img_dir), augment_count=augment_count, rotation_step=rotation_step)


if __name__ == "__main__":
    # Configuration
    BASE_DIR = "data/vitaminc/vitaminc_front/train/good"
    AUGMENT_COUNT = 10
    ROTATION_STEP = 10  # Rotation angle step (degrees) - will rotate at 0, 5, 10, 15, ..., 355
    
    process_directory_recursive(BASE_DIR, augment_count=AUGMENT_COUNT, rotation_step=ROTATION_STEP)