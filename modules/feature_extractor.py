# models/feature_extractor.py
"""
ResNet50 Feature Extractor for PatchCore
=========================================
Pure compute — receives image, returns patches.
No I/O. No dataset loops. No recursive calls.

Performance notes:
    - extract_batch() bypasses PIL entirely (direct NumPy→Tensor)
    - Color features computed in full-batch mode (no per-image loop)
    - _rgb_to_hsv() uses branchless torch.where ops
    - Normalization constants pre-allocated on device
    - CUDA streams used when available for async D2H transfers
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
            state = torch.load(backbone_path, map_location="cpu", weights_only=False)
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

        # ── Pre-allocated normalisation tensors on device (avoid per-call alloc) ──
        self._norm_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(1, 3, 1, 1)
        self._norm_std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(1, 3, 1, 1)
        if self.use_half:
            self._norm_mean = self._norm_mean.half()
            self._norm_std = self._norm_std.half()

        # ── CUDA stream for async D2H ──
        self._d2h_stream: Optional[torch.cuda.Stream] = None
        if self.device == "cuda":
            self._d2h_stream = torch.cuda.Stream()

        # ── PIL transforms (kept for single-image extract()) ──
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

    # ─────────── Tensor-based batch preprocessing (no PIL) ───────────
    def _preprocess_batch_tensor(
        self, images: List[np.ndarray], normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convert list of BGR uint8 ndarrays → (B, 3, H, W) float tensor.

        Bypasses PIL entirely:
          1. BGR → RGB via numpy slice (no copy if contiguous)
          2. Stack → single numpy array
          3. One H2D transfer
          4. Resize + normalise on GPU
        """
        sz = self.img_size
        resized: List[np.ndarray] = []
        for img in images:
            # BGR → RGB (fast channel flip, no copy)
            rgb = img[:, :, ::-1]
            # Resize directly via OpenCV (faster than PIL)
            if rgb.shape[0] != sz or rgb.shape[1] != sz:
                rgb = cv2.resize(rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
            resized.append(rgb)

        # Stack to (B, H, W, 3) → contiguous
        arr = np.stack(resized, axis=0)  # (B, H, W, 3), uint8
        # Single H2D transfer — much faster than per-image
        t = torch.from_numpy(arr).to(self.device, non_blocking=True)
        # (B, H, W, 3) → (B, 3, H, W), float [0, 1]
        t = t.permute(0, 3, 1, 2).float().div_(255.0)

        if self.use_half:
            t = t.half()

        if normalize:
            t = (t - self._norm_mean) / self._norm_std

        return t

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

    # ─────────────── Batch BGR numpy (OPTIMISED) ───────────────
    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract patches from a batch of BGR numpy arrays.

        Optimisations vs. old path:
          - No PIL conversion — direct NumPy→Tensor
          - Single H2D transfer for the whole batch
          - Resize via cv2 (INTER_LINEAR, ~2-3× faster than PIL bilinear)
          - Color features computed in batch (no per-image loop)
          - Async D2H via CUDA stream
        """
        if not images:
            return []

        # ── Normalised batch for CNN ──
        batch = self._preprocess_batch_tensor(images, normalize=True)

        with torch.amp.autocast("cuda", enabled=self.use_half):
            feats = self.backbone(batch)

        cnn = self._cnn_from_feats(feats, batch_mode=True)  # (B, P, cnn_dim)

        need_color = self.use_color_features or self.use_hsv

        if not need_color:
            # Fast path: move all at once, split on CPU
            stacked = cnn.cpu().numpy()                      # (B, P, D)
            return [stacked[i] for i in range(stacked.shape[0])]

        # ── Raw (un-normalised) batch for color ──
        raw_batch = self._preprocess_batch_tensor(images, normalize=False)
        color = self._color_patches_batch(raw_batch)         # (B, P, color_dim)

        combined = torch.cat([cnn, color], dim=2)            # (B, P, total_dim)

        # Async D2H if available
        if self._d2h_stream is not None:
            with torch.cuda.stream(self._d2h_stream):
                out = combined.cpu().numpy()
            self._d2h_stream.synchronize()
        else:
            out = combined.cpu().numpy()

        return [out[i] for i in range(out.shape[0])]

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

    # ─────────────── Color Patches (single — kept for extract()) ───────────────
    def _color_patches(self, raw: torch.Tensor) -> torch.Tensor:
        """Color features for a single image (B=1)."""
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

    # ─────────────── Color Patches BATCH (no per-image loop) ───────────────
    def _color_patches_batch(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Compute color patch features for the entire batch at once.

        Input:  (B, 3, H, W) — un-normalised [0, 1]
        Output: (B, grid², color_dim)
        """
        B, c, h, w = raw.shape
        gs = self.grid_size
        ph = h // gs
        pw = w // gs

        # Crop to exact grid multiple
        x = raw[:, :, :ph * gs, :pw * gs]   # (B, 3, ph*gs, pw*gs)

        # Unfold into grid patches:  (B, 3, gs, gs, ph, pw)
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        # → (B, gs*gs, 3, ph, pw)
        x = x.contiguous().view(B, c, gs * gs, ph, pw).permute(0, 2, 1, 3, 4)

        feats = []
        if self.use_color_features:
            feats.append(x.mean(dim=(3, 4)))   # (B, gs², 3)
            feats.append(x.std(dim=(3, 4)))    # (B, gs², 3)
        if self.use_hsv:
            # Reshape for HSV: (B*gs², 3, ph, pw)
            flat = x.reshape(B * gs * gs, c, ph, pw)
            hsv = self._rgb_to_hsv(flat)       # (B*gs², 3, ph, pw)
            hsv = hsv.view(B, gs * gs, c, ph, pw)
            feats.append(hsv.mean(dim=(3, 4)))
            feats.append(hsv.std(dim=(3, 4)))

        out = torch.cat(feats, dim=2)          # (B, gs², color_dim)
        if self.color_weight != 1.0:
            out = out * self.color_weight
        return out

    # ─────────────── Branchless HSV (torch.where) ───────────────
    @staticmethod
    def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
        """
        Vectorised RGB→HSV using torch.where (no boolean index writes).

        Faster than the masked-assignment approach, especially on GPU,
        because torch.where compiles to a single fused kernel.
        """
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        mx, argmx = rgb.max(dim=1)
        mn = rgb.min(dim=1)[0]
        diff = mx - mn

        # Value
        v = mx

        # Saturation — branchless
        s = torch.where(mx > 0, diff / mx, torch.zeros_like(mx))

        # Hue — compute all three cases, select via argmax
        safe_diff = torch.where(diff > 0, diff, torch.ones_like(diff))
        h_r = ((g - b) / safe_diff / 6.0) % 1.0
        h_g = (b - r) / safe_diff / 6.0 + 1.0 / 3.0
        h_b = (r - g) / safe_diff / 6.0 + 2.0 / 3.0

        h = torch.where(argmx == 0, h_r, torch.where(argmx == 1, h_g, h_b))
        h = torch.where(diff > 0, h % 1.0, torch.zeros_like(h))

        return torch.stack([h, s, v], dim=1)
