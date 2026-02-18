# CNNMultiScale/core_shared/patchcore_multiscale.py
"""
CNN Multi-Scale PatchCore Feature Extractor.

🔥 Architecture optimized for 2-5px pill defect detection:

1. Modified ResNet34 backbone
   - conv1 stride=2 → stride=1 (preserve resolution)
   - Remove initial maxpool (prevent information loss)
   - Optional dilated layer3 (expand receptive field without downsampling)

2. Multi-Scale Feature Extraction
   - layer1: 64ch, high resolution → detects 2-5px cracks
   - layer2: 128ch, medium resolution → texture patterns
   - layer3: 256ch, semantic → shape context

3. Per-Scale Processing
   - L2 normalize per scale (prevent magnitude imbalance)
   - Separate FAISS index per scale
   - Independent anomaly scoring

4. Score Fusion
   - max(score_s1, score_s2, score_s3) — defect in ANY scale triggers alarm

5. Preprocessing
   - CLAHE contrast boost for micro-crack visibility
   - Optional Laplacian edge enhancement

Single Responsibility:
- Extract multi-scale features from images
- Compute per-scale anomaly scores using FAISS
- Fuse scores across scales
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
from typing import Optional, Tuple, List, Dict


# =============================================================================
#                        SE ATTENTION BLOCK
# =============================================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Helps the network focus on channels that highlight defects.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# =============================================================================
#                   MODIFIED RESNET BACKBONE
# =============================================================================
class ModifiedResNet(nn.Module):
    """
    Modified ResNet34/50 that preserves spatial resolution.
    
    Changes from standard ResNet:
    1. conv1: stride=2 → stride=1 (doubles output resolution)
    2. maxpool: removed (doubles output resolution again)
    3. Optional: layer3 uses dilation=2 instead of stride=2
    4. Optional: SE attention blocks after each layer
    
    Result: 512→512→512→256→128 instead of 512→256→128→64→32
    4x more resolution = can detect 4x smaller defects
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        remove_maxpool: bool = True,
        stride1_conv1: bool = True,
        use_dilated_layer3: bool = True,
        use_se_attention: bool = True,
        se_reduction: int = 16,
        selected_layers: List[str] = None,
    ):
        super().__init__()
        self.selected_layers = selected_layers or ["layer1", "layer2", "layer3"]

        # Load pretrained backbone
        if backbone == "resnet34":
            base = models.resnet34(weights="IMAGENET1K_V1")
        elif backbone == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet34' or 'resnet50'.")

        # ── Modify conv1: stride=2 → stride=1 ──
        if stride1_conv1:
            old_conv1 = base.conv1
            base.conv1 = nn.Conv2d(
                3, old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=1,  # Changed from 2 to 1
                padding=old_conv1.padding,
                bias=False,
            )
            # Copy weights (still valid, just different stride)
            with torch.no_grad():
                base.conv1.weight.copy_(old_conv1.weight)

        # ── Build sequential stem ──
        stem_layers = [base.conv1, base.bn1, base.relu]
        if not remove_maxpool:
            stem_layers.append(base.maxpool)
        self.stem = nn.Sequential(*stem_layers)

        # ── Feature layers ──
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # Keep but don't use for features

        # ── Dilated layer3 (preserve resolution) ──
        if use_dilated_layer3:
            self._make_dilated(self.layer3, dilation=2)

        # ── SE Attention ──
        self.se_blocks = nn.ModuleDict()
        if use_se_attention:
            layer_channels = {
                "layer1": 64 if backbone == "resnet34" else 256,
                "layer2": 128 if backbone == "resnet34" else 512,
                "layer3": 256 if backbone == "resnet34" else 1024,
            }
            for name in self.selected_layers:
                if name in layer_channels:
                    self.se_blocks[name] = SEBlock(layer_channels[name], se_reduction)

        # Freeze all backbone weights (we're using as feature extractor)
        for param in self.parameters():
            param.requires_grad = False
        # But SE blocks should be trainable if you want to fine-tune
        # For PatchCore, we freeze everything (no training needed)

    @staticmethod
    def _make_dilated(layer: nn.Sequential, dilation: int = 2):
        """Replace stride=2 convolutions with dilation=2 in a ResNet layer."""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                # Only apply dilation to 3x3+ convs, NOT 1x1 (downsample)
                if module.stride == (2, 2) and module.kernel_size[0] > 1:
                    module.stride = (1, 1)
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)
        # Also fix the downsample (shortcut) path: just change stride, no dilation
        if hasattr(layer[0], 'downsample') and layer[0].downsample is not None:
            for module in layer[0].downsample.modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning feature maps from selected layers.
        
        Returns:
            Dict mapping layer name → feature tensor
        """
        features = {}
        x = self.stem(x)

        x = self.layer1(x)
        if "layer1" in self.selected_layers:
            feat = x
            if "layer1" in self.se_blocks:
                feat = self.se_blocks["layer1"](feat)
            features["layer1"] = feat

        x = self.layer2(x)
        if "layer2" in self.selected_layers:
            feat = x
            if "layer2" in self.se_blocks:
                feat = self.se_blocks["layer2"](feat)
            features["layer2"] = feat

        x = self.layer3(x)
        if "layer3" in self.selected_layers:
            feat = x
            if "layer3" in self.se_blocks:
                feat = self.se_blocks["layer3"](feat)
            features["layer3"] = feat

        return features


# =============================================================================
#                    CLAHE PREPROCESSOR
# =============================================================================
class CLAHEPreprocessor:
    """
    CLAHE preprocessing to boost contrast of micro-cracks.
    
    Particularly effective for:
    - Smooth pill surfaces where cracks have very low contrast
    - White/light colored pills where defects are subtle
    """

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to an image (BGR or RGB).
        
        Args:
            image: np.ndarray (H, W, 3) uint8
        
        Returns:
            Enhanced image (same format)
        """
        if image is None or image.size == 0:
            return image

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size),
        )
        l_enhanced = clahe.apply(l_channel)

        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return result


# =============================================================================
#                CNN MULTI-SCALE PATCHCORE
# =============================================================================
class CNNMultiScalePatchCore:
    """
    Multi-Scale PatchCore with Modified ResNet backbone.
    
    🔥 Designed specifically for detecting tiny defects (2-5px cracks).
    
    Architecture:
        Input (512×512) → CLAHE → Modified ResNet34
        → layer1 features (L2 norm) → PatchCore memory 1 → score_s1
        → layer2 features (L2 norm) → PatchCore memory 2 → score_s2
        → layer3 features (L2 norm) → PatchCore memory 3 → score_s3
        → final_score = max(score_s1, score_s2, score_s3)
    
    Multi-Resolution (optional):
        Also process at 768×768 → extract features → fuse scores
        2px defect becomes 3-4px at larger scale → easier to detect
    """

    def __init__(
        self,
        model_size: int = 512,
        model_size_secondary: int = 768,
        enable_multi_resolution: bool = True,
        grid_size: int = 32,
        k_nearest: int = 9,
        device: str = None,
        # Backbone
        backbone: str = "resnet34",
        remove_maxpool: bool = True,
        stride1_conv1: bool = True,
        use_dilated_layer3: bool = True,
        selected_layers: List[str] = None,
        # Fusion
        score_fusion: str = "max",
        scale_weights: List[float] = None,
        separate_memory_per_scale: bool = True,
        # Preprocessing
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        use_laplacian_boost: bool = False,
        laplacian_weight: float = 0.3,
        # Attention
        use_se_attention: bool = True,
        se_reduction: int = 16,
        # Color
        use_color_features: bool = True,
        use_hsv: bool = True,
        color_weight: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model_size_secondary = model_size_secondary
        self.enable_multi_resolution = enable_multi_resolution
        self.grid_size = grid_size
        self.k_nearest = k_nearest
        self.score_fusion = score_fusion
        self.scale_weights = scale_weights or [0.5, 0.3, 0.2]
        self.separate_memory_per_scale = separate_memory_per_scale
        self.selected_layers = selected_layers or ["layer1", "layer2", "layer3"]
        self.use_color_features = use_color_features
        self.use_hsv = use_hsv
        self.color_weight = color_weight
        self.use_laplacian_boost = use_laplacian_boost
        self.laplacian_weight = laplacian_weight

        # ── Build Modified Backbone ──
        self.backbone = ModifiedResNet(
            backbone=backbone,
            remove_maxpool=remove_maxpool,
            stride1_conv1=stride1_conv1,
            use_dilated_layer3=use_dilated_layer3,
            use_se_attention=use_se_attention,
            se_reduction=se_reduction,
            selected_layers=self.selected_layers,
        ).to(self.device).eval()

        # ── CLAHE ──
        self.clahe = CLAHEPreprocessor(clahe_clip_limit, clahe_tile_size) if use_clahe else None

        # ── Feature dimensions per layer ──
        is_resnet34 = backbone == "resnet34"
        self.layer_dims = {
            "layer1": 64 if is_resnet34 else 256,
            "layer2": 128 if is_resnet34 else 512,
            "layer3": 256 if is_resnet34 else 1024,
        }

        # Color feature dim
        self.color_feature_dim = 0
        if use_color_features:
            self.color_feature_dim += 6  # RGB mean + std
        if use_hsv:
            self.color_feature_dim += 6  # HSV mean + std

        # Per-scale feature dimensions
        self.scale_feature_dims = {}
        for layer_name in self.selected_layers:
            dim = self.layer_dims.get(layer_name, 0)
            if self.separate_memory_per_scale:
                # Each scale has its own dim (+ color if enabled)
                self.scale_feature_dims[layer_name] = dim + self.color_feature_dim
            else:
                self.scale_feature_dims[layer_name] = dim

        # Concatenated feature dim (for non-separate mode)
        self.concat_feature_dim = sum(
            self.layer_dims[l] for l in self.selected_layers
        ) + self.color_feature_dim

        # ── Transforms ──
        self.transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_secondary = T.Compose([
            T.Resize((model_size_secondary, model_size_secondary), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.raw_transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

        # ── Print summary ──
        print(f"[CNNMultiScale] Backbone: Modified {backbone}")
        print(f"[CNNMultiScale]   stride1_conv1={stride1_conv1}, remove_maxpool={remove_maxpool}")
        print(f"[CNNMultiScale]   dilated_layer3={use_dilated_layer3}, se_attention={use_se_attention}")
        print(f"[CNNMultiScale] Input: {model_size}×{model_size}" +
              (f" + {model_size_secondary}×{model_size_secondary}" if enable_multi_resolution else ""))
        print(f"[CNNMultiScale] Grid: {grid_size}×{grid_size} = {grid_size**2} patches")
        print(f"[CNNMultiScale] Layers: {self.selected_layers}")
        print(f"[CNNMultiScale] Per-scale dims: {self.scale_feature_dims}")
        print(f"[CNNMultiScale] Score fusion: {score_fusion}")
        print(f"[CNNMultiScale] CLAHE: {'ON' if use_clahe else 'OFF'}")
        print(f"[CNNMultiScale] Multi-resolution: {'ON' if enable_multi_resolution else 'OFF'}")
        print(f"[CNNMultiScale] Separate memory/scale: {separate_memory_per_scale}")
        print(f"[CNNMultiScale] Device: {self.device}")

    # =========================================================
    #                    PREPROCESSING
    # =========================================================
    def _preprocess_numpy(self, np_img: np.ndarray) -> np.ndarray:
        """Apply CLAHE and optional Laplacian boost to BGR image."""
        if np_img is None or np_img.size == 0:
            return np_img

        result = np_img.copy()

        # CLAHE
        if self.clahe is not None:
            result = self.clahe.apply(result)

        # Laplacian edge boost
        if self.use_laplacian_boost:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.abs(laplacian)
            laplacian = (laplacian / (laplacian.max() + 1e-8) * 255).astype(np.uint8)
            # Blend edge map into each channel
            for c in range(3):
                result[:, :, c] = cv2.addWeighted(
                    result[:, :, c], 1.0,
                    laplacian, self.laplacian_weight,
                    0,
                )

        return result

    def _preprocess_pil(self, pil_img: Image.Image) -> Image.Image:
        """Apply preprocessing to PIL Image (converts to numpy and back)."""
        np_img = np.array(pil_img)
        # PIL is RGB, CLAHE expects BGR
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        processed = self._preprocess_numpy(bgr)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # =========================================================
    #                COLOR FEATURE EXTRACTION
    # =========================================================
    def _extract_color_features(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """Extract per-patch color statistics from raw image tensor."""
        b, c, h, w = raw_tensor.shape
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size

        x = raw_tensor[:, :, :patch_h * self.grid_size, :patch_w * self.grid_size]
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(b, c, self.grid_size * self.grid_size, patch_h, patch_w)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, c, patch_h, patch_w)

        color_feats = []

        # RGB stats
        if self.use_color_features:
            rgb_mean = patches.mean(dim=(2, 3))
            rgb_std = patches.std(dim=(2, 3))
            color_feats.extend([rgb_mean, rgb_std])

        # HSV stats
        if self.use_hsv:
            hsv = self._rgb_to_hsv_batch(patches)
            hsv_mean = hsv.mean(dim=(2, 3))
            hsv_std = hsv.std(dim=(2, 3))
            color_feats.extend([hsv_mean, hsv_std])

        if not color_feats:
            return torch.zeros(self.grid_size * self.grid_size, 0, device=raw_tensor.device)

        color_features = torch.cat(color_feats, dim=1)
        if self.color_weight != 1.0:
            color_features = color_features * self.color_weight
        return color_features

    @staticmethod
    def _rgb_to_hsv_batch(rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB tensor to HSV (batch)."""
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
        h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] / 6.0 + 1 / 3
        mask_b = (argmax_rgb == 2) & (diff > 0)
        h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] / 6.0 + 2 / 3
        h = h % 1.0

        return torch.stack([h, s, v], dim=1)

    # =========================================================
    #          MULTI-SCALE FEATURE EXTRACTION
    # =========================================================
    @torch.no_grad()
    def extract_multiscale_features(
        self, image: Image.Image
    ) -> Dict[str, np.ndarray]:
        """
        Extract per-scale patch features from PIL Image.
        
        Returns:
            Dict[layer_name → numpy array of shape (n_patches, feature_dim_per_scale)]
            
        Each scale's features are L2-normalized independently.
        """
        # Preprocess
        image_processed = self._preprocess_pil(image)

        # Forward through backbone
        x = self.transform(image_processed).unsqueeze(0).to(self.device)
        layer_features = self.backbone(x)

        # Optional: raw tensor for color features
        x_raw = self.raw_transform(image).unsqueeze(0).to(self.device)
        color_feats = None
        if self.use_color_features or self.use_hsv:
            color_feats = self._extract_color_features(x_raw)

        # Process each scale
        scale_features = {}
        for layer_name in self.selected_layers:
            feat = layer_features[layer_name]
            # Adaptive pool to grid_size
            feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
            # Reshape to (n_patches, dim)
            feat = feat.permute(0, 2, 3, 1).reshape(-1, self.layer_dims[layer_name])

            # ★ L2 normalize per scale (critical for balanced multi-scale)
            feat = F.normalize(feat, p=2, dim=-1)

            # Append color features if separate memory per scale
            if self.separate_memory_per_scale and color_feats is not None:
                color_normed = F.normalize(color_feats, p=2, dim=-1)
                feat = torch.cat([feat, color_normed], dim=1)

            scale_features[layer_name] = feat.contiguous().cpu().numpy()

        return scale_features

    @torch.no_grad()
    def extract_multiscale_features_multiresolution(
        self, image: Image.Image
    ) -> Dict[str, np.ndarray]:
        """
        Extract features at multiple resolutions and combine.
        
        Process image at both primary (512) and secondary (768) size.
        Average the features for more robust representation.
        """
        # Primary resolution
        features_primary = self.extract_multiscale_features(image)

        if not self.enable_multi_resolution:
            return features_primary

        # Secondary resolution — use larger transform
        image_processed = self._preprocess_pil(image)
        x2 = self.transform_secondary(image_processed).unsqueeze(0).to(self.device)
        layer_features_2 = self.backbone(x2)

        x_raw_2 = T.Compose([
            T.Resize(
                (self.model_size_secondary, self.model_size_secondary),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
        ])(image).unsqueeze(0).to(self.device)

        color_feats_2 = None
        if self.use_color_features or self.use_hsv:
            # Temporarily adjust grid_size for larger image
            color_feats_2 = self._extract_color_features_at_size(
                x_raw_2, self.grid_size
            )

        # Process secondary features
        features_secondary = {}
        for layer_name in self.selected_layers:
            feat = layer_features_2[layer_name]
            feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
            feat = feat.permute(0, 2, 3, 1).reshape(-1, self.layer_dims[layer_name])
            feat = F.normalize(feat, p=2, dim=-1)

            if self.separate_memory_per_scale and color_feats_2 is not None:
                color_normed = F.normalize(color_feats_2, p=2, dim=-1)
                feat = torch.cat([feat, color_normed], dim=1)

            features_secondary[layer_name] = feat.contiguous().cpu().numpy()

        # Average primary + secondary features
        combined = {}
        for layer_name in self.selected_layers:
            f1 = features_primary[layer_name]
            f2 = features_secondary[layer_name]
            avg = (f1 + f2) / 2.0
            # Re-normalize after averaging
            norms = np.linalg.norm(avg, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            combined[layer_name] = (avg / norms).astype(np.float32)

        return combined

    def _extract_color_features_at_size(
        self, raw_tensor: torch.Tensor, grid_size: int
    ) -> torch.Tensor:
        """Extract color features with a specific grid size."""
        b, c, h, w = raw_tensor.shape
        patch_h = h // grid_size
        patch_w = w // grid_size

        x = raw_tensor[:, :, :patch_h * grid_size, :patch_w * grid_size]
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(b, c, grid_size * grid_size, patch_h, patch_w)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, c, patch_h, patch_w)

        color_feats = []
        if self.use_color_features:
            color_feats.extend([patches.mean(dim=(2, 3)), patches.std(dim=(2, 3))])
        if self.use_hsv:
            hsv = self._rgb_to_hsv_batch(patches)
            color_feats.extend([hsv.mean(dim=(2, 3)), hsv.std(dim=(2, 3))])

        if not color_feats:
            return torch.zeros(grid_size * grid_size, 0, device=raw_tensor.device)

        result = torch.cat(color_feats, dim=1)
        if self.color_weight != 1.0:
            result = result * self.color_weight
        return result

    @torch.no_grad()
    def extract_concat_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract concatenated multi-scale features (non-separate mode).
        
        Returns:
            numpy array of shape (n_patches, concat_feature_dim)
        """
        image_processed = self._preprocess_pil(image)
        x = self.transform(image_processed).unsqueeze(0).to(self.device)
        layer_features = self.backbone(x)

        all_feats = []
        for layer_name in self.selected_layers:
            feat = layer_features[layer_name]
            feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
            feat = feat.permute(0, 2, 3, 1).reshape(-1, self.layer_dims[layer_name])
            # L2 normalize per scale before concat
            feat = F.normalize(feat, p=2, dim=-1)
            all_feats.append(feat)

        concat = torch.cat(all_feats, dim=1)

        # Color features
        if self.use_color_features or self.use_hsv:
            x_raw = self.raw_transform(image).unsqueeze(0).to(self.device)
            color_feats = self._extract_color_features(x_raw)
            color_normed = F.normalize(color_feats, p=2, dim=-1)
            concat = torch.cat([concat, color_normed], dim=1)

        # Final L2 normalize
        concat = F.normalize(concat, p=2, dim=-1)
        return concat.contiguous().cpu().numpy()

    def extract_from_numpy(
        self, np_img: np.ndarray, multi_resolution: bool = False
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract multi-scale features from OpenCV BGR image."""
        if np_img is None or np_img.size == 0:
            return None
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if multi_resolution:
            return self.extract_multiscale_features_multiresolution(pil)
        return self.extract_multiscale_features(pil)

    def extract_concat_from_numpy(self, np_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract concatenated features from OpenCV BGR image."""
        if np_img is None or np_img.size == 0:
            return None
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.extract_concat_features(pil)

    # =========================================================
    #           FAISS INDEX & SCORING (PER-SCALE)
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

    def get_per_scale_anomaly_scores(
        self,
        scale_features: Dict[str, np.ndarray],
        scale_indices: Dict[str, faiss.Index],
    ) -> Dict[str, float]:
        """
        Compute anomaly score per scale.
        
        Returns:
            Dict[layer_name → max_anomaly_score]
        """
        scores = {}
        for layer_name in self.selected_layers:
            if layer_name not in scale_features or layer_name not in scale_indices:
                continue

            feats = np.ascontiguousarray(scale_features[layer_name])
            faiss.normalize_L2(feats)

            sim, _ = scale_indices[layer_name].search(
                feats.astype(np.float32), self.k_nearest
            )
            patch_scores = 1.0 - np.mean(sim, axis=1)
            scores[layer_name] = float(patch_scores.max())

        return scores

    def fuse_scores(self, per_scale_scores: Dict[str, float]) -> float:
        """
        Fuse per-scale anomaly scores into a single score.
        
        Strategies:
        - "max": max of all scales → most sensitive
        - "mean": average → more stable
        - "weighted": weighted sum using scale_weights
        """
        if not per_scale_scores:
            return 0.0

        scores_list = [per_scale_scores.get(l, 0.0) for l in self.selected_layers]

        if self.score_fusion == "max":
            return max(scores_list)
        elif self.score_fusion == "mean":
            return sum(scores_list) / len(scores_list)
        elif self.score_fusion == "weighted":
            weighted = sum(
                w * s for w, s in zip(self.scale_weights, scores_list)
            )
            return weighted
        else:
            return max(scores_list)

    def get_max_anomaly_score(
        self,
        scale_features: Dict[str, np.ndarray],
        scale_indices: Dict[str, faiss.Index],
    ) -> float:
        """Convenience: compute fused anomaly score."""
        per_scale = self.get_per_scale_anomaly_scores(scale_features, scale_indices)
        return self.fuse_scores(per_scale)

    # =========================================================
    #                 ANOMALY HEATMAP
    # =========================================================
    def get_anomaly_heatmap(
        self,
        scale_features: Dict[str, np.ndarray],
        scale_indices: Dict[str, faiss.Index],
        image_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Generate fused anomaly heatmap from all scales.
        
        Takes max heatmap across scales (per-pixel max).
        """
        heatmaps = []

        for layer_name in self.selected_layers:
            if layer_name not in scale_features or layer_name not in scale_indices:
                continue

            feats = np.ascontiguousarray(scale_features[layer_name])
            faiss.normalize_L2(feats)

            sim, _ = scale_indices[layer_name].search(
                feats.astype(np.float32), self.k_nearest
            )
            patch_scores = 1.0 - np.mean(sim, axis=1)

            heatmap = patch_scores.reshape(self.grid_size, self.grid_size)
            heatmap = cv2.resize(
                heatmap.astype(np.float32),
                (image_size[1], image_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            heatmaps.append(heatmap)

        if not heatmaps:
            return np.zeros(image_size, dtype=np.float32)

        # Max fusion across scales
        return np.maximum.reduce(heatmaps)
