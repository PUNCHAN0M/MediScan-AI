# DINOv2PatchCore/core_dinov2.py
"""
DINOv2 PatchCore Feature Extractor + Color Features.

Uses DINOv2 (ViT) backbone for superior feature extraction:
- Better color/texture discrimination than MobileNet
- Multi-scale patch tokens with rich semantic information
- Self-supervised features work great for anomaly detection

🔥 Updated: Added optional color features (RGB/HSV mean/std) to improve color anomaly detection.

Single Responsibility:
- Extract patch features from images using DINOv2 (+ optional Color)
- Compute anomaly scores using FAISS index
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import faiss
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple


class DINOv2PatchCore:
    """
    Feature extractor using DINOv2 ViT backbone + optional Color Features.
    
    DINOv2 advantages over MobileNet:
    1. Better texture discrimination - critical for defect detection
    2. Better color awareness - can differentiate similar shapes with different colors
    3. Richer patch-level features from Vision Transformer architecture
    4. Self-supervised pretraining captures fine-grained details
    
    🔥 NEW: Optional color features for cases where DINOv2 still ignores color
    """
    
    # DINOv2 model sizes
    MODELS = {
        "small": "dinov2_vits14",   # 21M params, 384-dim features
        "base": "dinov2_vitb14",    # 86M params, 768-dim features  
        "large": "dinov2_vitl14",   # 300M params, 1024-dim features
        "giant": "dinov2_vitg14",   # 1.1B params, 1536-dim features (need lots of VRAM)
    }
    
    def __init__(
        self,
        model_size: int = 256,
        grid_size: int = 16,  # DINOv2 outputs 14x14 or 16x16 patches
        k_nearest: int = 19,
        device: str = None,
        backbone_size: str = "small",  # "small", "base", "large", "giant"
        use_registers: bool = False,   # Use DINOv2 with registers (v2)
        multi_scale: bool = True,      # Use multi-scale features
        use_color_features: bool = False,  # 🔥 NEW: Add RGB mean/std
        use_hsv: bool = False,              # 🔥 NEW: Add HSV mean/std
        color_weight: float = 1.0,          # 🔥 NEW: Weight for color features
    ):
        """
        Args:
            model_size: Input image size (should be divisible by 14 for DINOv2)
            grid_size: Target grid size for output patches
            k_nearest: Number of k-nearest neighbors for anomaly score
            device: "cuda" or "cpu" (auto-detect if None)
            backbone_size: DINOv2 model size ("small", "base", "large", "giant")
            use_registers: Use DINOv2 with registers for better features
            multi_scale: Extract features at multiple scales
            use_color_features: Whether to add RGB mean/std per patch
            use_hsv: Whether to also add HSV mean/std per patch
            color_weight: Multiplier for color features
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.grid_size = grid_size
        self.k_nearest = k_nearest
        self.backbone_size = backbone_size
        self.multi_scale = multi_scale
        self.use_color_features = use_color_features
        self.use_hsv = use_hsv
        self.color_weight = color_weight
        
        # Load DINOv2 backbone
        self._load_backbone(backbone_size, use_registers)
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Calculate color feature dimension
        self.color_feature_dim = 0
        if use_color_features:
            self.color_feature_dim += 6  # RGB mean (3) + RGB std (3)
        if use_hsv:
            self.color_feature_dim += 6  # HSV mean (3) + HSV std (3)
        
        # Transform - DINOv2 expects 224x224 or multiples of 14
        # For best results, use 224, 336, 448, 518
        target_size = self._round_to_patch_size(model_size)
        self.target_size = target_size
        
        self.transform = T.Compose([
            T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Raw transform for color extraction
        self.raw_transform = T.Compose([
            T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        
        print(f"[DINOv2PatchCore] Loaded {backbone_size} model")
        print(f"[DINOv2PatchCore] Input size: {target_size}x{target_size}")
        print(f"[DINOv2PatchCore] Feature dim: {self.feature_dim}")
        if use_color_features or use_hsv:
            print(f"[DINOv2PatchCore] Color features: RGB={use_color_features}, HSV={use_hsv}")
        print(f"[DINOv2PatchCore] Device: {self.device}")
    
    def _round_to_patch_size(self, size: int, patch_size: int = 14) -> int:
        """Round size to nearest multiple of patch_size."""
        return max(patch_size, (size // patch_size) * patch_size)
    
    def _load_backbone(self, size: str, use_registers: bool) -> None:
        """Load DINOv2 backbone from torch hub."""
        if size not in self.MODELS:
            raise ValueError(f"Unknown backbone size: {size}. Choose from {list(self.MODELS.keys())}")
        
        model_name = self.MODELS[size]
        
        # Add _reg suffix for register version
        if use_registers:
            model_name = model_name.replace("dinov2_", "dinov2_") + "_reg"
        
        try:
            # Load from Facebook's DINOv2 repo
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=True,
            )
        except Exception as e:
            print(f"[Warning] Failed to load {model_name} from hub: {e}")
            print("[Warning] Falling back to dinov2_vits14")
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vits14',
                pretrained=True,
            )
        
        self.backbone = self.backbone.to(self.device).eval()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _get_feature_dim(self) -> int:
        """Get output feature dimension."""
        dims = {"small": 384, "base": 768, "large": 1024, "giant": 1536}
        return dims.get(self.backbone_size, 384)

    def _extract_color_features(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """Extract color statistics per patch from raw image tensor."""
        b, c, h, w = raw_tensor.shape
        
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size
        
        x = raw_tensor[:, :, :patch_h * self.grid_size, :patch_w * self.grid_size]
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(b, c, self.grid_size * self.grid_size, patch_h, patch_w)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.reshape(-1, c, patch_h, patch_w)
        
        color_feats = []
        
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
        """
        Extract patch features (DINOv2 + optional Color) from PIL Image.
        
        Returns:
            numpy array of shape (grid_size * grid_size, feature_dim + color_dim)
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        if self.multi_scale:
            features = self._extract_multiscale(x)
        else:
            features = self._extract_single_scale(x)
        
        # Interpolate to target grid size
        b, n_patches, dim = features.shape
        h = w = int(np.sqrt(n_patches))
        
        # Reshape to spatial format: (B, H, W, C)
        features = features.view(b, h, w, dim).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Interpolate to target grid size
        features = F.interpolate(
            features,
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False,
        )
        
        # Reshape back to patches: (N, C)
        patches = features.permute(0, 2, 3, 1).reshape(-1, dim)
        
        # Add color features if enabled
        if self.use_color_features or self.use_hsv:
            x_raw = self.raw_transform(image).unsqueeze(0).to(self.device)
            color_features = self._extract_color_features(x_raw)
            if color_features is not None:
                patches = torch.cat([patches, color_features], dim=1)
        
        # L2 normalize
        patches = patches / torch.norm(patches, p=2, dim=-1, keepdim=True)
        
        return patches.contiguous().cpu().numpy()
    
    def _extract_single_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features at single scale."""
        # DINOv2 returns dict with patch tokens and class token
        output = self.backbone.forward_features(x)
        
        if isinstance(output, dict):
            # Get patch tokens (exclude CLS token)
            patch_tokens = output.get('x_norm_patchtokens', output.get('x_prenorm', None))
            if patch_tokens is None:
                # Fallback: get from full sequence
                full = output.get('x_norm', output.get('x_prenorm', None))
                if full is not None:
                    patch_tokens = full[:, 1:]  # Remove CLS token
                else:
                    raise ValueError("Cannot extract patch tokens from DINOv2 output")
        else:
            # Older format: just tensor
            patch_tokens = output[:, 1:]  # Remove CLS token
        
        return patch_tokens
    
    def _extract_multiscale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features at multiple scales for richer representation.
        
        This helps capture both fine-grained texture and global structure.
        """
        features_list: List[torch.Tensor] = []
        
        # Scale 1: Original
        features_list.append(self._extract_single_scale(x))
        
        # Scale 2: Smaller (captures more context per patch)
        h, w = x.shape[2:]
        if h >= 140 and w >= 140:  # Only if image is large enough
            x_small = F.interpolate(x, size=(h // 2 + h // 4, w // 2 + w // 4), mode='bilinear', align_corners=False)
            x_small = self._round_tensor_size(x_small)
            if x_small.shape[2] >= 70:
                feat_small = self._extract_single_scale(x_small)
                features_list.append(feat_small)
        
        # Concatenate all scales
        # Each scale has different number of patches, so we need to interpolate
        target_patches = features_list[0].shape[1]
        
        aligned_features: List[torch.Tensor] = []
        for feat in features_list:
            if feat.shape[1] != target_patches:
                # Reshape and interpolate
                h_feat = w_feat = int(np.sqrt(feat.shape[1]))
                feat = feat.view(1, h_feat, w_feat, -1).permute(0, 3, 1, 2)
                h_target = w_target = int(np.sqrt(target_patches))
                feat = F.interpolate(feat, size=(h_target, w_target), mode='bilinear', align_corners=False)
                feat = feat.permute(0, 2, 3, 1).view(1, target_patches, -1)
            aligned_features.append(feat)
        
        # Average features from all scales
        stacked = torch.stack(aligned_features, dim=0)
        combined = stacked.mean(dim=0)
        
        return combined
    
    def _round_tensor_size(self, x: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
        """Round tensor spatial size to multiple of patch_size."""
        h, w = x.shape[2:]
        new_h = (h // patch_size) * patch_size
        new_w = (w // patch_size) * patch_size
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return x
    
    def extract_from_numpy(self, np_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from OpenCV BGR image."""
        if np_img is None or np_img.size == 0:
            return None
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.extract_features(pil)
    
    # =========================================================
    # FAISS Index & Scoring (same as MobileNet version)
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
        image_size: tuple = (256, 256),
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
