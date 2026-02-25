"""
ResNet50 Feature Extractor
==========================

Multi-layer patch feature extraction from pretrained ResNet50.

Architecture
------------
::

    input 256×256
        ↓
    ResNet50 (frozen)
        ├── layer2  →  [B, 512,  32, 32]
        └── layer3  →  [B, 1024, 16, 16]
               ↓ upsample to 32×32
        concat →  [B, 1536, 32, 32]
        adaptive_avg_pool2d  →  [B, 1536, grid, grid]
        flatten  →  [B × grid², 1536]

Optional color features (appended per patch):
    +6  RGB mean/std
    +6  HSV mean/std

Single image  → ``extract(pil_image) → (grid², D)``
Batch images  → ``extract_batch(list_of_bgr) → list[(grid², D)]``
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from typing import List, Optional


class ResNet50FeatureExtractor:
    """
    Frozen ResNet50 backbone → multi-layer patch features.

    Parameters
    ----------
    img_size : int
        Input image resolution (default 256).
    grid_size : int
        Number of patches per side.  Total patches = grid × grid (default 16 → 256).
    device : str or None
        ``"cuda"`` / ``"cpu"`` (auto-detect if ``None``).
    use_color_features : bool
        Append per-patch RGB mean/std (+6 dims).
    use_hsv : bool
        Append per-patch HSV mean/std (+6 dims).
    color_weight : float
        Scale factor for color dims (1.0 = same scale as CNN features).
    backbone_path : str or None
        Path to a custom ``.pth`` backbone weights file.  When provided the
        ImageNet pretrained weights are NOT loaded; instead the state-dict
        (or a checkpoint containing ``"state_dict"`` / ``"model"`` key) is
        loaded from this file.  Useful for domain-finetuned backbones such as
        ``resnet_backbone.pth``.
    """

    # backbone layers to hook
    LAYERS = ("layer2", "layer3")
    LAYER_CHANNELS = {"layer2": 512, "layer3": 1024}

    def __init__(
        self,
        img_size: int = 256,
        grid_size: int = 16,
        device: Optional[str] = None,
        use_color_features: bool = False,
        use_hsv: bool = False,
        color_weight: float = 1.0,
        backbone_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.grid_size = grid_size
        self.use_color_features = use_color_features
        self.use_hsv = use_hsv
        self.color_weight = color_weight

        # ── backbone ──
        if backbone_path:
            print(f"[FeatureExtractor] Loading custom backbone from: {backbone_path}")
            self.backbone = models.resnet50(weights=None)
            state = torch.load(backbone_path, map_location="cpu")
            # support raw state_dict or checkpoint dicts
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing:
                print(f"[FeatureExtractor] Missing keys ({len(missing)}): {missing[:5]} ...")
            if unexpected:
                print(f"[FeatureExtractor] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        else:
            self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.backbone = self.backbone.to(self.device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._act: dict = {}
        self._register_hooks()

        # ── dimensions ──
        self.cnn_dim = sum(self.LAYER_CHANNELS[l] for l in self.LAYERS)  # 1536
        self.color_dim = 0
        if use_color_features:
            self.color_dim += 6
        if use_hsv:
            self.color_dim += 6
        self.feature_dim = self.cnn_dim + self.color_dim

        # ── transforms ──
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.raw_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

        backbone_label = backbone_path if backbone_path else "ImageNet pretrained"
        print(
            f"[FeatureExtractor] ResNet50 | backbone={backbone_label} | grid={grid_size} | "
            f"cnn={self.cnn_dim} color={self.color_dim} total={self.feature_dim}"
        )

    # ─────────────────── hooks ───────────────────
    def _register_hooks(self) -> None:
        def make_hook(name: str):
            def hook(_, __, out):
                self._act[name] = out.detach()
            return hook

        for ln in self.LAYERS:
            layer = dict(self.backbone.named_modules())[ln]
            layer.register_forward_hook(make_hook(ln))

    # ─────────────────── single image ───────────────────
    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extract patch features from one PIL image.

        Returns
        -------
        np.ndarray  (grid², feature_dim)
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        self.backbone(x)

        patches = self._cnn_patches()  # (grid², cnn_dim)

        if self.use_color_features or self.use_hsv:
            x_raw = self.raw_transform(image).unsqueeze(0).to(self.device)
            color = self._color_patches(x_raw)  # (grid², color_dim)
            patches = torch.cat([patches, color], dim=1)

        return patches.cpu().numpy()

    def extract_from_numpy(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract from OpenCV BGR image."""
        if bgr is None or bgr.size == 0:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.extract(Image.fromarray(rgb))

    # ─────────────────── batch ───────────────────
    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Batch feature extraction — stack pills → one ResNet50 forward.

        Parameters
        ----------
        images : list of BGR ``np.ndarray``

        Returns
        -------
        list of ``np.ndarray``  each ``(grid², feature_dim)``
        """
        if not images:
            return []

        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                     for img in images]
        batch = torch.stack([self.transform(p) for p in pil_imgs]).to(self.device)

        self.backbone(batch)

        # ── gather CNN features for whole batch ──
        target_h, target_w = self._act["layer2"].shape[2:]
        feats = []
        for ln in self.LAYERS:
            f = self._act[ln]
            if f.shape[2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w),
                                  mode="bilinear", align_corners=False)
            feats.append(f)

        concat = torch.cat(feats, dim=1)                          # (B, 1536, H', W')
        pooled = F.adaptive_avg_pool2d(concat,
                                       (self.grid_size, self.grid_size))
        # pooled: (B, 1536, grid, grid)

        results: List[np.ndarray] = []
        need_color = self.use_color_features or self.use_hsv

        if need_color:
            raw_batch = torch.stack(
                [self.raw_transform(p) for p in pil_imgs]
            ).to(self.device)

        for i in range(len(images)):
            cnn = (pooled[i]
                   .permute(1, 2, 0)
                   .reshape(-1, self.cnn_dim))   # (grid², 1536)

            if need_color:
                col = self._color_patches(raw_batch[i:i+1])  # (grid², color_dim)
                out = torch.cat([cnn, col], dim=1)
            else:
                out = cnn

            results.append(out.cpu().numpy())

        return results

    # ─────────────────── CNN patch helper ───────────────────
    def _cnn_patches(self) -> torch.Tensor:
        """
        Combine layer2 + layer3 → pooled → flat patches for the *last*
        forward pass (single-image or use index 0).

        Returns (grid², cnn_dim) tensor on device.
        """
        target_size = self._act["layer2"].shape[2:]  # (H/8, W/8)

        parts = []
        for ln in self.LAYERS:
            f = self._act[ln]
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size,
                                  mode="bilinear", align_corners=False)
            parts.append(f)

        concat = torch.cat(parts, dim=1)              # (1, 1536, H', W')
        pooled = F.adaptive_avg_pool2d(
            concat, (self.grid_size, self.grid_size))  # (1, 1536, g, g)

        return (pooled[0]
                .permute(1, 2, 0)
                .reshape(-1, self.cnn_dim))             # (g², 1536)

    # ─────────────────── color patch helper ───────────────────
    def _color_patches(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Per-patch color statistics from un-normalised image tensor.

        Parameters
        ----------
        raw : (1, 3, H, W) tensor [0, 1]

        Returns
        -------
        (grid², color_dim) tensor on device
        """
        _, c, h, w = raw.shape
        ph = h // self.grid_size
        pw = w // self.grid_size

        x = raw[:, :, :ph * self.grid_size, :pw * self.grid_size]
        # unfold into patches
        patches = (x.unfold(2, ph, ph)
                    .unfold(3, pw, pw)
                    .contiguous()
                    .view(1, c, self.grid_size * self.grid_size, ph, pw)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(-1, c, ph, pw))  # (n_patches, 3, ph, pw)

        feats: list = []

        # RGB mean / std
        if self.use_color_features:
            feats.append(patches.mean(dim=(2, 3)))   # (n, 3)
            feats.append(patches.std(dim=(2, 3)))    # (n, 3)

        # HSV mean / std
        if self.use_hsv:
            hsv = self._rgb_to_hsv(patches)
            feats.append(hsv.mean(dim=(2, 3)))
            feats.append(hsv.std(dim=(2, 3)))

        out = torch.cat(feats, dim=1)
        if self.color_weight != 1.0:
            out = out * self.color_weight

        return out

    # ─────────────────── RGB → HSV ───────────────────
    @staticmethod
    def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
        """Batch RGB [0,1] → HSV [0,1]."""
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
        h[m_g] = (b[m_g] - r[m_g]) / diff[m_g] / 6.0 + 1.0 / 3.0

        m_b = (argmx == 2) & d_pos
        h[m_b] = (r[m_b] - g[m_b]) / diff[m_b] / 6.0 + 2.0 / 3.0

        h = h % 1.0
        return torch.stack([h, s, v], dim=1)
