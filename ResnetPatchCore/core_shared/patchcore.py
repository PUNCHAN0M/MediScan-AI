# ResnetPatchCore/core_shared/patchcore.py
"""
ResNet18 PatchCore Feature Extractor with Color Features.

🔥 Key Innovation: Combines CNN features with explicit color statistics
This solves the classic problem where deep networks ignore color.

Features extracted per patch:
1. ResNet18 conv features (from shallow layers to preserve color)
2. RGB mean per patch
3. RGB std per patch  
4. HSV mean per patch (better color separation)

Single Responsibility:
- Extract patch features from images (CNN + Color)
- Compute anomaly scores using FAISS index
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import faiss
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


class ResNetPatchCore:
    """
    Feature extractor using ResNet18 backbone + Color Features.
    
    Why ResNet18 + Color Features?
    1. ResNet shallower layers preserve color better than MobileNet/DINOv2
    2. Explicit color statistics (RGB/HSV mean/std) ensure color differences are captured
    3. Fast inference - ResNet18 is lightweight
    4. Perfect for pill inspection where color = critical indicator
    """
    
    def __init__(
        self,
        model_size: int = 256,
        grid_size: int = 28,  # Smaller patches = better small defect detection
        k_nearest: int = 19,
        device: str = None,
        use_color_features: bool = True,  # Add color statistics to features
        use_hsv: bool = True,  # Add HSV color space
        color_weight: float = 1.0,  # Weight for color features (increase if color matters more)
    ):
        """
        Args:
            model_size: Input image size
            grid_size: Grid size for patch features (28-32 recommended for small defects)
            k_nearest: Number of k-nearest neighbors for anomaly score
            device: "cuda" or "cpu" (auto-detect if None)
            use_color_features: Whether to concatenate RGB mean/std
            use_hsv: Whether to also add HSV mean/std
            color_weight: Multiplier for color features (1.0 = equal weight)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.grid_size = grid_size
        self.k_nearest = k_nearest
        self.use_color_features = use_color_features
        self.use_hsv = use_hsv
        self.color_weight = color_weight

        # Load ResNet18 backbone
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = self.backbone.to(self.device).eval()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Use shallow + mid layers (preserve color info)
        # layer1: 64 channels - edge, color, texture
        # layer2: 128 channels - texture, pattern
        # layer3: 256 channels - shape, structure
        self.selected_layers = ["layer1", "layer2", "layer3"]
        
        self.activation = {}
        self._register_hooks()
        
        # Calculate feature dimension
        self.cnn_feature_dim = 64 + 128 + 256  # From ResNet layers
        self.color_feature_dim = 0
        if use_color_features:
            self.color_feature_dim += 6  # RGB mean (3) + RGB std (3)
        if use_hsv:
            self.color_feature_dim += 6  # HSV mean (3) + HSV std (3)
        
        self.feature_dim = self.cnn_feature_dim + self.color_feature_dim

        # Transform for backbone
        self.transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Raw transform (for color extraction, no normalization)
        self.raw_transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

        print(f"[ResNetPatchCore] Loaded ResNet18")
        print(f"[ResNetPatchCore] Grid size: {grid_size}x{grid_size}")
        print(f"[ResNetPatchCore] CNN features: {self.cnn_feature_dim}")
        print(f"[ResNetPatchCore] Color features: {self.color_feature_dim}")
        print(f"[ResNetPatchCore] Total features: {self.feature_dim}")
        print(f"[ResNetPatchCore] Device: {self.device}")

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
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
        
        # Unfold into patches: (1, 3, grid_size, patch_h, grid_size, patch_w)
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size
        
        # Reshape to patches
        x = raw_tensor[:, :, :patch_h * self.grid_size, :patch_w * self.grid_size]
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        # Shape: (1, 3, grid_size, grid_size, patch_h, patch_w)
        
        patches = patches.contiguous().view(b, c, self.grid_size * self.grid_size, patch_h, patch_w)
        # Shape: (1, 3, n_patches, patch_h, patch_w)
        
        patches = patches.permute(0, 2, 1, 3, 4)  # (1, n_patches, 3, patch_h, patch_w)
        patches = patches.reshape(-1, c, patch_h, patch_w)  # (n_patches, 3, patch_h, patch_w)
        
        color_feats = []
        
        # RGB features
        rgb_mean = patches.mean(dim=(2, 3))  # (n_patches, 3)
        rgb_std = patches.std(dim=(2, 3))    # (n_patches, 3)
        color_feats.extend([rgb_mean, rgb_std])
        
        if self.use_hsv:
            # Convert to HSV
            hsv_patches = self._rgb_to_hsv_batch(patches)  # (n_patches, 3, H, W)
            hsv_mean = hsv_patches.mean(dim=(2, 3))  # (n_patches, 3)
            hsv_std = hsv_patches.std(dim=(2, 3))    # (n_patches, 3)
            color_feats.extend([hsv_mean, hsv_std])
        
        # Concatenate all color features
        color_features = torch.cat(color_feats, dim=1)  # (n_patches, color_dim)
        
        # Apply weight
        if self.color_weight != 1.0:
            color_features = color_features * self.color_weight
        
        return color_features

    @staticmethod
    def _rgb_to_hsv_batch(rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB tensor to HSV.
        
        Args:
            rgb: (N, 3, H, W) tensor with values in [0, 1]
            
        Returns:
            hsv: (N, 3, H, W) tensor with H in [0, 1], S in [0, 1], V in [0, 1]
        """
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        max_rgb, argmax_rgb = rgb.max(dim=1)
        min_rgb = rgb.min(dim=1)[0]
        diff = max_rgb - min_rgb
        
        # Value
        v = max_rgb
        
        # Saturation
        s = torch.zeros_like(max_rgb)
        mask = max_rgb > 0
        s[mask] = diff[mask] / max_rgb[mask]
        
        # Hue
        h = torch.zeros_like(max_rgb)
        
        # When max is R
        mask_r = (argmax_rgb == 0) & (diff > 0)
        h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r] / 6.0
        
        # When max is G
        mask_g = (argmax_rgb == 1) & (diff > 0)
        h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] / 6.0 + 1/3
        
        # When max is B
        mask_b = (argmax_rgb == 2) & (diff > 0)
        h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] / 6.0 + 2/3
        
        # Handle negative hue
        h = h % 1.0
        
        return torch.stack([h, s, v], dim=1)

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract patch features (CNN + Color) from PIL Image.
        
        Returns:
            numpy array of shape (grid_size * grid_size, feature_dim)
        """
        # Get normalized tensor for CNN
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get raw tensor for color extraction
        x_raw = self.raw_transform(image).unsqueeze(0).to(self.device)
        
        # Forward through backbone
        _ = self.backbone(x)
        
        # Collect CNN features from selected layers
        cnn_features_list = []
        for layer_name in self.selected_layers:
            feat = self.activation[layer_name]
            feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
            cnn_features_list.append(feat)
        
        # Concatenate CNN features
        cnn_concat = torch.cat(cnn_features_list, dim=1)  # (1, 448, grid, grid)
        
        # Reshape to patches: (n_patches, cnn_dim)
        cnn_patches = cnn_concat.permute(0, 2, 3, 1).reshape(-1, self.cnn_feature_dim)
        
        # Add color features if enabled
        if self.use_color_features or self.use_hsv:
            color_features = self._extract_color_features(x_raw)  # (n_patches, color_dim)
            
            # Concatenate CNN + Color
            all_features = torch.cat([cnn_patches, color_features], dim=1)
        else:
            all_features = cnn_patches
        
        # L2 normalize
        all_features = all_features / torch.norm(all_features, p=2, dim=-1, keepdim=True)
        
        return all_features.contiguous().cpu().numpy()

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
        # Score per patch = 1 - mean(similarity), then take max
        scores = 1.0 - np.mean(sim, axis=1)
        return float(scores.max())

    def get_anomaly_heatmap(
        self,
        patch_features: np.ndarray,
        index: faiss.Index,
        image_size: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """
        Generate anomaly heatmap showing defect locations.
        
        Returns:
            Heatmap numpy array of shape (H, W)
        """
        if patch_features is None or patch_features.shape[0] == 0:
            return np.zeros(image_size, dtype=np.float32)
        
        patch_features = np.ascontiguousarray(patch_features)
        faiss.normalize_L2(patch_features)
        
        sim, _ = index.search(patch_features.astype(np.float32), self.k_nearest)
        scores = 1.0 - np.mean(sim, axis=1)
        
        # Reshape to grid
        heatmap = scores.reshape(self.grid_size, self.grid_size)
        
        # Upscale to image size
        heatmap = cv2.resize(
            heatmap.astype(np.float32),
            (image_size[1], image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        
        return heatmap
