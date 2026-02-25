"""
FCDD Dataset - Good-only training with pseudo-anomaly generation
================================================================
โหลด pill images (good only) + สร้าง pseudo-anomaly สำหรับ contrastive training.

Pseudo-anomaly types:
  1. CutPaste  - ตัด patch แปะที่อื่นในภาพ
  2. Gaussian Noise - เพิ่ม noise ในบริเวณสุ่ม
  3. Random Erasing - ลบบริเวณสุ่ม
"""
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional


class PseudoAnomalyGenerator:
    """Generate synthetic anomalies for one-class contrastive training."""

    def __init__(self, img_size: int = 256):
        self.img_size = img_size

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random pseudo-anomaly to image.

        Args:
            img: (H, W, 3) uint8 image
        Returns:
            corrupted: (H, W, 3) uint8 image with anomaly
            mask: (H, W) float32 binary mask (1 = anomaly region)
        """
        method = random.choice(["cutpaste", "noise", "erase", "smooth"])
        if method == "cutpaste":
            return self._cutpaste(img)
        elif method == "noise":
            return self._noise(img)
        elif method == "erase":
            return self._erase(img)
        else:
            return self._smooth(img)

    def _random_rect(
        self, h: int, w: int, min_ratio: float = 0.05, max_ratio: float = 0.25
    ) -> Tuple[int, int, int, int]:
        """Generate random rectangle coordinates."""
        rh = random.randint(int(h * min_ratio), int(h * max_ratio))
        rw = random.randint(int(w * min_ratio), int(w * max_ratio))
        y = random.randint(0, h - rh)
        x = random.randint(0, w - rw)
        return y, x, rh, rw

    def _cutpaste(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        out = img.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        # Source patch
        sy, sx, sh, sw = self._random_rect(h, w, 0.08, 0.20)
        patch = img[sy : sy + sh, sx : sx + sw].copy()

        # Random transform on patch
        if random.random() > 0.5:
            patch = cv2.flip(patch, 1)
        if random.random() > 0.5:
            patch = cv2.flip(patch, 0)
        angle = random.uniform(-45, 45)
        M = cv2.getRotationMatrix2D((sw // 2, sh // 2), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (sw, sh), borderMode=cv2.BORDER_REFLECT)

        # Adjust brightness
        beta = random.randint(-30, 30)
        patch = np.clip(patch.astype(np.int16) + beta, 0, 255).astype(np.uint8)

        # Destination
        dy = random.randint(0, max(0, h - sh))
        dx = random.randint(0, max(0, w - sw))
        out[dy : dy + sh, dx : dx + sw] = patch
        mask[dy : dy + sh, dx : dx + sw] = 1.0

        return out, mask

    def _noise(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        out = img.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        y, x, rh, rw = self._random_rect(h, w, 0.10, 0.30)
        noise = np.random.normal(0, random.randint(30, 80), (rh, rw, 3))
        region = out[y : y + rh, x : x + rw].astype(np.float32) + noise
        out[y : y + rh, x : x + rw] = np.clip(region, 0, 255).astype(np.uint8)
        mask[y : y + rh, x : x + rw] = 1.0

        return out, mask

    def _erase(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        out = img.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        y, x, rh, rw = self._random_rect(h, w, 0.08, 0.25)
        fill = random.choice(["mean", "random", "zero"])
        if fill == "mean":
            mean_val = img.mean(axis=(0, 1)).astype(np.uint8)
            out[y : y + rh, x : x + rw] = mean_val
        elif fill == "random":
            out[y : y + rh, x : x + rw] = np.random.randint(0, 256, (rh, rw, 3), dtype=np.uint8)
        else:
            out[y : y + rh, x : x + rw] = 0
        mask[y : y + rh, x : x + rw] = 1.0

        return out, mask

    def _smooth(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        out = img.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        y, x, rh, rw = self._random_rect(h, w, 0.10, 0.30)
        ksize = random.choice([11, 15, 21, 27])
        region = out[y : y + rh, x : x + rw]
        out[y : y + rh, x : x + rw] = cv2.GaussianBlur(region, (ksize, ksize), 0)
        mask[y : y + rh, x : x + rw] = 1.0

        return out, mask


class FCDDDataset(Dataset):
    """
    Dataset for FCDD training.

    - Loads good pill images only
    - Generates pseudo-anomalies on-the-fly
    - Returns (image_tensor, label, anomaly_mask)
      label = 0 → normal, 1 → pseudo-anomaly
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        good_dir: str,
        img_size: int = 256,
        pseudo_anomaly_ratio: float = 0.5,
        augment: bool = True,
    ):
        self.img_size = img_size
        self.pseudo_anomaly_ratio = pseudo_anomaly_ratio
        self.augment = augment
        self.anomaly_gen = PseudoAnomalyGenerator(img_size)

        # Collect image paths
        good_path = Path(good_dir)
        self.image_paths = sorted(
            [p for p in good_path.iterdir() if p.suffix.lower() in self.IMAGE_EXTS]
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {good_dir}")
        print(f"[FCDDDataset] Loaded {len(self.image_paths)} good images from {good_dir}")

        # Transforms
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])

    def __len__(self) -> int:
        # x3 for more training samples per epoch
        return len(self.image_paths) * 3

    def __getitem__(self, idx: int):
        real_idx = idx % len(self.image_paths)
        img_path = self.image_paths[real_idx]

        # Load and resize
        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback to random valid image
            img = cv2.imread(str(self.image_paths[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Decide: normal or pseudo-anomaly
        is_anomaly = random.random() < self.pseudo_anomaly_ratio

        if is_anomaly:
            img, mask = self.anomaly_gen(img)
            label = 1
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            label = 0

        # To PIL for augmentation
        pil_img = Image.fromarray(img)
        if self.augment:
            pil_img = self.augment_transform(pil_img)

        # To tensor
        img_tensor = T.ToTensor()(pil_img)  # (3, H, W) [0, 1]
        img_tensor = self.normalize(img_tensor)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return img_tensor, label, mask_tensor


class FCDDTestDataset(Dataset):
    """Simple dataset for loading test images (individual pill crops)."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, image_dir: str, img_size: int = 256):
        self.img_size = img_size
        img_path = Path(image_dir)
        self.image_paths = sorted(
            [p for p in img_path.iterdir() if p.suffix.lower() in self.IMAGE_EXTS]
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        img_tensor = T.ToTensor()(Image.fromarray(img))
        img_tensor = self.normalize(img_tensor)

        return img_tensor, str(img_path)
