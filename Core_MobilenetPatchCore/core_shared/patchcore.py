# core_shared/patchcore.py
"""
PatchCore Feature Extractor - MobileNetV3 backbone with multi-layer features + Color Features.

🔥 Updated: Added optional color features (RGB/HSV mean/std) to improve color anomaly detection.

Single Responsibility:
- Extract patch features from images (CNN + optional Color)
- Compute anomaly scores using FAISS index
"""
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import faiss
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


class PatchCore:
    """Feature extractor using MobileNetV3-Large backbone + optional Color Features."""
    
    def __init__(
        self,
        model_size: int = 256,
        grid_size: int = 20,
        k_nearest: int = 19,
        device: str = None,
        use_color_features: bool = False,  # 🔥 NEW: Add RGB mean/std
        use_hsv: bool = False,              # 🔥 NEW: Add HSV mean/std
        color_weight: float = 1.0,          # 🔥 NEW: Weight for color features
    ):
        """
        Args:
            model_size: Input image size (512, 640, etc.)
            grid_size: Grid size for patch features (14, 20, etc.)
            k_nearest: Number of k-nearest neighbors for anomaly score
            device: "cuda" or "cpu" (auto-detect if None)
            use_color_features: Whether to add RGB mean/std per patch
            use_hsv: Whether to also add HSV mean/std per patch
            color_weight: Multiplier for color features (increase if color matters more)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.grid_size = grid_size
        self.k_nearest = k_nearest
        self.use_color_features = use_color_features
        self.use_hsv = use_hsv
        self.color_weight = color_weight

        # Feature extractor
        self.backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.backbone = self.backbone.to(self.device).eval()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.selected_layers = [
            "features.3",   # edge, texture, noise
            "features.6",   # surface pattern
            "features.9",   # shape, geometry
            "features.12",  # structure, spatial relation
            "features.15",  # semantic, global context
        ]
        
        self.activation = {}
        self._register_hooks()
        
        # Calculate feature dimensions
        self.color_feature_dim = 0
        if use_color_features:
            self.color_feature_dim += 6  # RGB mean (3) + RGB std (3)
        if use_hsv:
            self.color_feature_dim += 6  # HSV mean (3) + HSV std (3)

        # Transform for backbone
        self.transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(model_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Raw transform (for color extraction, no normalization)
        self.raw_transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(model_size),
            T.ToTensor(),
        ])
        
        if use_color_features or use_hsv:
            print(f"[PatchCore] Color features enabled: RGB={use_color_features}, HSV={use_hsv}")

    def _register_hooks(self):
        def get_hook(name):
            def hook(_, __, output):
                self.activation[name] = output.detach()
            return hook

        for layer_name in self.selected_layers:
            layer = dict([*self.backbone.named_modules()])[layer_name]
            layer.register_forward_hook(get_hook(layer_name))

    def _extract_color_features(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract color statistics per patch from raw (un-normalized) image tensor.
        
        Args:
            raw_tensor: (1, 3, H, W) tensor with values in [0, 1]
            
        Returns:
            (grid_size*grid_size, color_dim) color features
        """
        b, c, h, w = raw_tensor.shape
        
        # Unfold into patches
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size
        
        # Reshape to patches
        x = raw_tensor[:, :, :patch_h * self.grid_size, :patch_w * self.grid_size]
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(b, c, self.grid_size * self.grid_size, patch_h, patch_w)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.reshape(-1, c, patch_h, patch_w)
        
        color_feats = []
        
        # RGB features
        if self.use_color_features:
            rgb_mean = patches.mean(dim=(2, 3))
            rgb_std = patches.std(dim=(2, 3))
            color_feats.extend([rgb_mean, rgb_std])
        
        if self.use_hsv:
            hsv_patches = self._rgb_to_hsv_batch(patches)
            hsv_mean = hsv_patches.mean(dim=(2, 3))
            hsv_std = hsv_patches.std(dim=(2, 3))
            color_feats.extend([hsv_mean, hsv_std])
        
        if not color_feats:
            return None
        
        color_features = torch.cat(color_feats, dim=1)
        
        if self.color_weight != 1.0:
            color_features = color_features * self.color_weight
        
        return color_features

    @staticmethod
    def _rgb_to_hsv_batch(rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB tensor to HSV."""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        max_rgb, argmax_rgb = rgb.max(dim=1)
        min_rgb = rgb.min(dim=1)[0]
        diff = max_rgb - min_rgb
        
        v = max_rgb
        s = torch.zeros_like(max_rgb)
        mask = max_rgb > 0
        s[mask] = diff[mask] / max_rgb[mask]
        
        h = torch.zeros_like(max_rgb)
        mask_r = (argmax_rgb == 0) & (diff > 0)
        h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r] / 6.0
        mask_g = (argmax_rgb == 1) & (diff > 0)
        h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] / 6.0 + 1/3
        mask_b = (argmax_rgb == 2) & (diff > 0)
        h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] / 6.0 + 2/3
        h = h % 1.0
        
        return torch.stack([h, s, v], dim=1)

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract patch features (CNN + optional Color) from PIL Image."""
        x = self.transform(image).unsqueeze(0).to(self.device)

        _ = self.backbone(x)
        features = [
            F.adaptive_avg_pool2d(self.activation[ln], (self.grid_size, self.grid_size))
            for ln in self.selected_layers
        ]
        concat = torch.cat(features, dim=1)
        patches = concat.permute(0, 2, 3, 1).reshape(-1, concat.shape[1])
        
        # Add color features if enabled
        if self.use_color_features or self.use_hsv:
            x_raw = self.raw_transform(image).unsqueeze(0).to(self.device)
            color_features = self._extract_color_features(x_raw)
            if color_features is not None:
                patches = torch.cat([patches, color_features], dim=1)
        
        # L2 normalize
        patches = patches / torch.norm(patches, p=2, dim=-1, keepdim=True)

        return patches.contiguous().cpu().numpy()

    def extract_from_numpy(self, np_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from OpenCV BGR image."""
        if np_img is None or np_img.size == 0:
            return None
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.extract_features(pil)

    # =========================================================
    # FAISS Index & Scoring
    # =========================================================
    @staticmethod
    def build_faiss_index(memory_bank: np.ndarray) -> faiss.Index:
        """Create IndexFlatIP for normalized memory bank."""
        memory_bank = np.ascontiguousarray(memory_bank.astype(np.float32))
        faiss.normalize_L2(memory_bank)
        d = memory_bank.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(memory_bank)
        return index

    def get_max_anomaly_score(self, patch_features: np.ndarray, index: faiss.Index) -> float:
        """Compute max anomaly score (1 - similarity)."""
        if patch_features is None or patch_features.shape[0] == 0:
            return 0.0

        patch_features = np.ascontiguousarray(patch_features)
        faiss.normalize_L2(patch_features)

        sim, _ = index.search(patch_features.astype(np.float32), self.k_nearest)
        scores = 1.0 - np.mean(sim, axis=1)
        return float(scores.max())
    
    def get_anomaly_heatmap(
        self,
        patch_features: np.ndarray,
        index: faiss.Index,
        image_size: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """Generate anomaly heatmap showing defect locations."""
        if patch_features is None or patch_features.shape[0] == 0:
            return np.zeros(image_size, dtype=np.float32)
        
        patch_features = np.ascontiguousarray(patch_features)
        faiss.normalize_L2(patch_features)
        
        sim, _ = index.search(patch_features.astype(np.float32), self.k_nearest)
        scores = 1.0 - np.mean(sim, axis=1)
        
        heatmap = scores.reshape(self.grid_size, self.grid_size)
        heatmap = cv2.resize(
            heatmap.astype(np.float32),
            (image_size[1], image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        
        return heatmap
