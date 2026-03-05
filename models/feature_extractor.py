# models/feature_extractor.py
"""
ResNet50 Feature Extractor for PatchCore
=========================================
Pure compute — receives image, returns patches.
No I/O. No dataset loops. No recursive calls.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from typing import List, Optional


class ResNet50FeatureExtractor:
    """
    Extract multi-layer CNN + optional color features from images.

    Input:  PIL.Image  or  list[np.ndarray (BGR)]
    Output: np.ndarray (N_patches, feature_dim)
    """

    LAYERS = ("layer1", "layer2", "layer3")
    LAYER_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024}

    def __init__(
        self,
        img_size: int = 256,
        grid_size: int = 16,
        device: Optional[str] = None,
        use_half: Optional[bool] = None,
        use_color_features: bool = False,
        use_hsv: bool = False,
        color_weight: float = 1.0,
        backbone_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        if use_half is None:
            self.use_half = self.device == "cuda"
        else:
            self.use_half = use_half and self.device == "cuda"

        self.img_size = img_size
        self.grid_size = grid_size
        self.use_color_features = use_color_features
        self.use_hsv = use_hsv
        self.color_weight = color_weight

        # ── backbone ──
        backbone = models.resnet50(
            weights="IMAGENET1K_V1" if not backbone_path else None
        )
        if backbone_path:
            state = torch.load(backbone_path, map_location="cpu")
            if isinstance(state, dict):
                for key in ("state_dict", "model", "full_state_dict", "features_state_dict"):
                    if key in state:
                        state = state[key]
                        break
            state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            backbone.load_state_dict(state, strict=False)

        return_nodes = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3"}
        self.backbone = create_feature_extractor(backbone, return_nodes)
        self.backbone = self.backbone.to(self.device).eval()

        if self.use_half:
            self.backbone = self.backbone.half()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ── dimensions ──
        self.cnn_dim = sum(self.LAYER_CHANNELS[l] for l in self.LAYERS)
        self.color_dim = 0
        if use_color_features:
            self.color_dim += 6
        if use_hsv:
            self.color_dim += 6
        self.feature_dim = self.cnn_dim + self.color_dim

        # ── transforms ──
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.raw_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

        print(
            f"[FeatureExtractor] grid={grid_size} "
            f"cnn={self.cnn_dim} color={self.color_dim} "
            f"total={self.feature_dim} FP16={self.use_half}"
        )

    # ─────────────── Single PIL Image ───────────────
    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract patches from a single PIL Image (RGB)."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        if self.use_half:
            x = x.half()

        with torch.amp.autocast("cuda", enabled=self.use_half):
            feats = self.backbone(x)

        patches = self._cnn_from_feats(feats)

        if self.use_color_features or self.use_hsv:
            raw = self.raw_transform(image).unsqueeze(0).to(self.device)
            color = self._color_patches(raw)
            patches = torch.cat([patches, color], dim=1)

        return patches.cpu().numpy()

    # ─────────────── Single BGR numpy ───────────────
    def extract_from_numpy(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract patches from a single BGR numpy array."""
        if bgr is None or bgr.size == 0:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.extract(Image.fromarray(rgb))

    # ─────────────── Batch BGR numpy ───────────────
    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract patches from a batch of BGR numpy arrays."""
        if not images:
            return []

        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    for img in images]

        batch = torch.stack([self.transform(p) for p in pil_imgs]).to(self.device)
        if self.use_half:
            batch = batch.half()

        with torch.amp.autocast("cuda", enabled=self.use_half):
            feats = self.backbone(batch)

        cnn = self._cnn_from_feats(feats, batch_mode=True)

        if not (self.use_color_features or self.use_hsv):
            return [c.cpu().numpy() for c in cnn]

        raw_batch = torch.stack([self.raw_transform(p) for p in pil_imgs]).to(self.device)
        results = []
        for i in range(len(images)):
            col = self._color_patches(raw_batch[i:i + 1])
            out = torch.cat([cnn[i], col], dim=1)
            results.append(out.cpu().numpy())
        return results

    # ─────────────── CNN Patch Builder ───────────────
    def _cnn_from_feats(self, feats: dict, batch_mode: bool = False):
        pooled = []
        for ln in self.LAYERS:
            f = feats[ln]
            f = F.adaptive_avg_pool2d(f, (self.grid_size, self.grid_size))
            pooled.append(f)

        concat = torch.cat(pooled, dim=1)
        if batch_mode:
            B = concat.shape[0]
            return concat.permute(0, 2, 3, 1).reshape(B, -1, self.cnn_dim)
        return concat[0].permute(1, 2, 0).reshape(-1, self.cnn_dim)

    # ─────────────── Color Patches ───────────────
    def _color_patches(self, raw: torch.Tensor) -> torch.Tensor:
        _, c, h, w = raw.shape
        ph = h // self.grid_size
        pw = w // self.grid_size
        x = raw[:, :, :ph * self.grid_size, :pw * self.grid_size]

        patches = (
            x.unfold(2, ph, ph)
             .unfold(3, pw, pw)
             .contiguous()
             .view(1, c, self.grid_size * self.grid_size, ph, pw)
             .permute(0, 2, 1, 3, 4)
             .reshape(-1, c, ph, pw)
        )

        feats = []
        if self.use_color_features:
            feats.append(patches.mean(dim=(2, 3)))
            feats.append(patches.std(dim=(2, 3)))
        if self.use_hsv:
            hsv = self._rgb_to_hsv(patches)
            feats.append(hsv.mean(dim=(2, 3)))
            feats.append(hsv.std(dim=(2, 3)))

        out = torch.cat(feats, dim=1)
        if self.color_weight != 1.0:
            out = out * self.color_weight
        return out

    @staticmethod
    def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        mx, argmx = rgb.max(dim=1)
        mn = rgb.min(dim=1)[0]
        diff = mx - mn

        v = mx
        s = torch.zeros_like(mx)
        pos = mx > 0
        s[pos] = diff[pos] / mx[pos]

        h = torch.zeros_like(mx)
        d_pos = diff > 0

        m_r = (argmx == 0) & d_pos
        h[m_r] = ((g[m_r] - b[m_r]) / diff[m_r] / 6.0) % 1.0
        m_g = (argmx == 1) & d_pos
        h[m_g] = (b[m_g] - r[m_g]) / diff[m_g] / 6.0 + 1 / 3
        m_b = (argmx == 2) & d_pos
        h[m_b] = (r[m_b] - g[m_b]) / diff[m_b] / 6.0 + 2 / 3

        return torch.stack([h % 1.0, s, v], dim=1)
