# core_shared/patchcore_sife.py
"""
PatchCore Feature Extractor with SIFE (Spatial Information Feature Enhancement).

🔥 Key Enhancement:
- MobileNetV3 backbone for CNN features
- SIFE adds spatial/positional encoding to patch features
- Better localization and position-aware anomaly detection

SIFE Components:
1. Sinusoidal/Coordinate positional encoding
2. Distance from center encoding
3. Local gradient information
4. Optional color features (RGB/HSV)
"""
import math
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import faiss
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


class PatchCoreSIFE:
    """
    Feature extractor using MobileNetV3-Large backbone + SIFE.
    
    SIFE = Spatial Information Feature Enhancement
    - Adds positional encoding to help model understand WHERE defects occur
    - Improves small defect detection at edges/corners
    """
    
    def __init__(
        self,
        model_size: int = 256,
        grid_size: int = 20,
        k_nearest: int = 11,
        device: str = None,
        # SIFE settings
        use_sife: bool = True,
        sife_dim: int = 32,
        sife_encoding_type: str = "sinusoidal",  # "sinusoidal", "coordinate", "learned"
        sife_weight: float = 1.0,
        use_center_distance: bool = True,
        use_local_gradient: bool = True,
        # 🔥 NEW: CNN vs SIFE balancing
        cnn_weight: float = 1.0,
        # 🔥 NEW: Laplacian variance for crack detection
        use_laplacian_variance: bool = False,
        laplacian_weight: float = 1.0,
        # Color features (optional)
        use_color_features: bool = False,
        use_hsv: bool = False,
        color_weight: float = 1.0,
        # 🔥 NEW: Multi-scale & Edge Enhancement
        use_multi_scale: bool = False,
        multi_scale_grids: list = None,  # e.g. [16, 32, 48]
        use_edge_enhancement: bool = False,
        edge_weight: float = 1.5,
        # 🔥 NEW: Scoring weights
        score_weight_max: float = 0.3,
        score_weight_top_k: float = 0.5,
        score_weight_percentile: float = 0.2,
        top_k_percent: float = 0.05,
    ):
        """
        Args:
            model_size: Input image size
            grid_size: Grid size for patch features
            k_nearest: Number of k-nearest neighbors for anomaly score
            device: "cuda" or "cpu"
            use_sife: Enable SIFE spatial encoding
            sife_dim: Dimension of SIFE features
            sife_encoding_type: Type of positional encoding
            sife_weight: Weight multiplier for SIFE features
            use_center_distance: Add distance from center feature
            use_local_gradient: Add local gradient magnitude
            cnn_weight: Weight for CNN features (default 1.0, reduce to ~0.7 for better SIFE balance)
            use_laplacian_variance: Add Laplacian variance for crack detection
            laplacian_weight: Weight for Laplacian features
            use_color_features: Add RGB mean/std per patch
            use_hsv: Add HSV mean/std per patch
            color_weight: Weight for color features
            use_multi_scale: Enable multi-scale feature extraction
            multi_scale_grids: List of grid sizes for multi-scale
            use_edge_enhancement: Enable edge-aware feature boost
            edge_weight: Weight multiplier for edge regions
            score_weight_max: Weight for max score in scoring
            score_weight_top_k: Weight for top-k mean in scoring
            score_weight_percentile: Weight for percentile score
            top_k_percent: Percentage of top patches for top-k mean
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.grid_size = grid_size
        self.k_nearest = k_nearest
        
        # SIFE settings
        self.use_sife = use_sife
        self.sife_dim = sife_dim
        self.sife_encoding_type = sife_encoding_type
        self.sife_weight = sife_weight
        self.use_center_distance = use_center_distance
        self.use_local_gradient = use_local_gradient
        
        # 🔥 NEW: CNN vs SIFE balancing
        self.cnn_weight = cnn_weight
        
        # 🔥 NEW: Laplacian variance for crack detection
        self.use_laplacian_variance = use_laplacian_variance
        self.laplacian_weight = laplacian_weight
        
        # Color settings
        self.use_color_features = use_color_features
        self.use_hsv = use_hsv
        self.color_weight = color_weight
        
        # 🔥 NEW: Multi-scale & Edge settings
        self.use_multi_scale = use_multi_scale
        self.multi_scale_grids = multi_scale_grids or [16, 32, 48]
        self.use_edge_enhancement = use_edge_enhancement
        self.edge_weight = edge_weight
        
        # 🔥 NEW: Scoring weights
        self.score_weight_max = score_weight_max
        self.score_weight_top_k = score_weight_top_k
        self.score_weight_percentile = score_weight_percentile
        self.top_k_percent = top_k_percent

        # Feature extractor (MobileNetV3)
        self.backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.backbone = self.backbone.to(self.device).eval()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.selected_layers = [
            "features.3",   # edge, texture
            "features.6",   # surface pattern
            "features.9",   # shape
            "features.12",  # structure
            "features.15",  # semantic
        ]
        
        self.activation = {}
        self._register_hooks()
        
        # Calculate total feature dimensions
        self._calc_feature_dims()
        
        # Create positional encoding
        if self.use_sife:
            self._init_positional_encoding()

        # Transforms
        self.transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(model_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.raw_transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(model_size),
            T.ToTensor(),
        ])
        
        self._print_config()

    def _print_config(self):
        """Print configuration info."""
        print(f"[PatchCoreSIFE] Initialized")
        print(f"  SIFE: {self.use_sife} (dim={self.sife_dim}, type={self.sife_encoding_type})")
        print(f"  CNN/SIFE weights: CNN={self.cnn_weight}, SIFE={self.sife_weight}")
        print(f"  Center distance: {self.use_center_distance}")
        print(f"  Local gradient: {self.use_local_gradient}")
        print(f"  Laplacian variance: {self.use_laplacian_variance} (weight={self.laplacian_weight})")
        print(f"  Color features: RGB={self.use_color_features}, HSV={self.use_hsv}")
        print(f"  Multi-scale: {self.use_multi_scale} ({self.multi_scale_grids if self.use_multi_scale else 'disabled'})")
        print(f"  Edge enhancement: {self.use_edge_enhancement} (weight={self.edge_weight})")
        print(f"  Scoring weights: max={self.score_weight_max}, top_k={self.score_weight_top_k}, percentile={self.score_weight_percentile}")
        print(f"  Total SIFE features: {self.sife_feature_dim}")

    def _calc_feature_dims(self):
        """Calculate additional feature dimensions."""
        self.sife_feature_dim = 0
        
        if self.use_sife:
            self.sife_feature_dim += self.sife_dim
        
        if self.use_center_distance:
            self.sife_feature_dim += 1  # distance from center
        
        if self.use_local_gradient:
            self.sife_feature_dim += 2  # gradient magnitude + direction
        
        # 🔥 NEW: Laplacian variance
        if self.use_laplacian_variance:
            self.sife_feature_dim += 1  # Laplacian variance per patch
        
        if self.use_color_features:
            self.sife_feature_dim += 6  # RGB mean + std
        
        if self.use_hsv:
            self.sife_feature_dim += 6  # HSV mean + std

    def _init_positional_encoding(self):
        """Initialize positional encoding for SIFE."""
        n_patches = self.grid_size * self.grid_size
        
        if self.sife_encoding_type == "sinusoidal":
            # Sinusoidal positional encoding (like Transformer)
            pe = self._create_sinusoidal_pe(self.grid_size, self.sife_dim)
        elif self.sife_encoding_type == "coordinate":
            # Direct coordinate encoding
            pe = self._create_coordinate_pe(self.grid_size, self.sife_dim)
        else:
            # Learnable (still use sinusoidal as base, can be fine-tuned)
            pe = self._create_sinusoidal_pe(self.grid_size, self.sife_dim)
        
        self.positional_encoding = pe.to(self.device)

    def _create_sinusoidal_pe(self, grid_size: int, dim: int) -> torch.Tensor:
        """
        Create 2D sinusoidal positional encoding.
        
        Returns: (grid_size*grid_size, dim) tensor
        """
        # Create position indices
        y_pos = torch.arange(grid_size).unsqueeze(1).expand(grid_size, grid_size)
        x_pos = torch.arange(grid_size).unsqueeze(0).expand(grid_size, grid_size)
        
        # Normalize to [-1, 1]
        y_pos = 2 * y_pos.float() / (grid_size - 1) - 1
        x_pos = 2 * x_pos.float() / (grid_size - 1) - 1
        
        # Create frequency bands
        half_dim = dim // 4
        freqs = torch.exp(
            torch.arange(half_dim) * -(math.log(10000.0) / half_dim)
        )
        
        # Apply sinusoidal functions
        y_pe = y_pos.flatten().unsqueeze(1) * freqs.unsqueeze(0)
        x_pe = x_pos.flatten().unsqueeze(1) * freqs.unsqueeze(0)
        
        # Concatenate sin and cos for both x and y
        pe = torch.cat([
            torch.sin(y_pe),
            torch.cos(y_pe),
            torch.sin(x_pe),
            torch.cos(x_pe),
        ], dim=1)
        
        # Trim or pad to exact dimension
        if pe.shape[1] > dim:
            pe = pe[:, :dim]
        elif pe.shape[1] < dim:
            pe = F.pad(pe, (0, dim - pe.shape[1]))
        
        return pe

    def _create_coordinate_pe(self, grid_size: int, dim: int) -> torch.Tensor:
        """
        Create coordinate-based positional encoding.
        
        Returns: (grid_size*grid_size, dim) tensor
        """
        # Create normalized coordinates
        y_pos = torch.linspace(-1, 1, grid_size).unsqueeze(1).expand(grid_size, grid_size)
        x_pos = torch.linspace(-1, 1, grid_size).unsqueeze(0).expand(grid_size, grid_size)
        
        # Flatten
        y_flat = y_pos.flatten().unsqueeze(1)
        x_flat = x_pos.flatten().unsqueeze(1)
        
        # Create multiple frequency components
        n_freqs = dim // 4
        features = [y_flat, x_flat]
        
        for i in range(1, n_freqs):
            freq = 2 ** i
            features.extend([
                torch.sin(y_flat * freq * math.pi),
                torch.cos(x_flat * freq * math.pi),
            ])
        
        pe = torch.cat(features, dim=1)
        
        # Trim or pad
        if pe.shape[1] > dim:
            pe = pe[:, :dim]
        elif pe.shape[1] < dim:
            pe = F.pad(pe, (0, dim - pe.shape[1]))
        
        return pe

    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        def get_hook(name):
            def hook(_, __, output):
                self.activation[name] = output.detach()
            return hook

        for layer_name in self.selected_layers:
            layer = dict([*self.backbone.named_modules()])[layer_name]
            layer.register_forward_hook(get_hook(layer_name))

    def _extract_center_distance(self) -> torch.Tensor:
        """
        Compute distance from center for each patch.
        
        Returns: (grid_size*grid_size, 1) tensor
        """
        y = torch.linspace(-1, 1, self.grid_size)
        x = torch.linspace(-1, 1, self.grid_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Euclidean distance from center
        dist = torch.sqrt(xx**2 + yy**2).flatten().unsqueeze(1)
        
        # Normalize to [0, 1]
        dist = dist / dist.max()
        
        return dist.to(self.device)

    def _extract_local_gradient(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract local gradient information per patch.
        
        Args:
            raw_tensor: (1, 3, H, W) tensor
            
        Returns: (grid_size*grid_size, 2) tensor [magnitude, direction]
        """
        # Convert to grayscale
        gray = 0.299 * raw_tensor[:, 0] + 0.587 * raw_tensor[:, 1] + 0.114 * raw_tensor[:, 2]
        gray = gray.unsqueeze(1)  # (1, 1, H, W)
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Apply Sobel
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        
        # Magnitude and direction
        magnitude = torch.sqrt(gx**2 + gy**2)
        direction = torch.atan2(gy, gx) / math.pi  # Normalize to [-1, 1]
        
        # Pool to grid size
        magnitude = F.adaptive_avg_pool2d(magnitude, (self.grid_size, self.grid_size))
        direction = F.adaptive_avg_pool2d(direction, (self.grid_size, self.grid_size))
        
        # Flatten and concatenate
        mag_flat = magnitude.flatten().unsqueeze(1)
        dir_flat = direction.flatten().unsqueeze(1)
        
        # Normalize magnitude
        mag_flat = mag_flat / (mag_flat.max() + 1e-8)
        
        return torch.cat([mag_flat, dir_flat], dim=1)

    def _extract_laplacian_variance(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        🔥 NEW: Extract Laplacian variance per patch for crack/scratch detection.
        
        Laplacian variance highlights micro-texture variations that indicate
        surface defects like hairline cracks, scratches, and texture anomalies.
        
        Args:
            raw_tensor: (1, 3, H, W) tensor
            
        Returns: (grid_size*grid_size, 1) tensor [laplacian_variance]
        """
        # Convert to grayscale
        gray = 0.299 * raw_tensor[:, 0] + 0.587 * raw_tensor[:, 1] + 0.114 * raw_tensor[:, 2]
        
        # Convert to numpy for OpenCV Laplacian
        gray_np = (gray[0] * 255).cpu().numpy().astype(np.uint8)
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray_np, cv2.CV_32F)
        
        # Convert back to tensor
        laplacian_tensor = torch.from_numpy(laplacian).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Compute variance per patch using unfold
        h, w = laplacian_tensor.shape[2], laplacian_tensor.shape[3]
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size
        
        # Crop to exact size
        laplacian_cropped = laplacian_tensor[:, :, :patch_h * self.grid_size, :patch_w * self.grid_size]
        
        # Reshape to patches and compute variance
        patches = laplacian_cropped.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(1, 1, self.grid_size * self.grid_size, patch_h, patch_w)
        patches = patches.squeeze(0).squeeze(0)  # (grid_size*grid_size, patch_h, patch_w)
        
        # Compute variance per patch
        variance = patches.var(dim=(1, 2), keepdim=False).unsqueeze(1)  # (grid_size*grid_size, 1)
        
        # Normalize
        variance = variance / (variance.max() + 1e-8)
        
        # Apply weight
        variance = variance * self.laplacian_weight
        
        return variance

    def _extract_color_features(self, raw_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract color statistics per patch."""
        if not (self.use_color_features or self.use_hsv):
            return None
        
        b, c, h, w = raw_tensor.shape
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size
        
        # Reshape to patches
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
        
        features = torch.cat(color_feats, dim=1)
        
        if self.color_weight != 1.0:
            features = features * self.color_weight
        
        return features

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

    def _extract_edge_weights(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        🔥 Extract edge intensity map to weight features.
        
        High values at edges help detect cracks, scratches, defects.
        Returns: (grid_size*grid_size,) tensor with edge weights
        """
        # Convert to grayscale
        gray = 0.299 * raw_tensor[:, 0] + 0.587 * raw_tensor[:, 1] + 0.114 * raw_tensor[:, 2]
        gray = gray.unsqueeze(1)
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(gx**2 + gy**2)
        
        # Pool to grid and normalize
        edge_pooled = F.adaptive_avg_pool2d(edge_magnitude, (self.grid_size, self.grid_size))
        edge_flat = edge_pooled.flatten()
        
        # Normalize to [1, edge_weight] - so base weight is 1, high edges get more
        edge_min = edge_flat.min()
        edge_max = edge_flat.max()
        if edge_max > edge_min:
            normalized = (edge_flat - edge_min) / (edge_max - edge_min)
            weights = 1.0 + normalized * (self.edge_weight - 1.0)
        else:
            weights = torch.ones_like(edge_flat)
        
        return weights

    @torch.no_grad()
    def _extract_multi_scale_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        🔥 Extract features at multiple grid scales and combine.
        
        Helps detect defects of varying sizes:
        - Small grid (e.g. 16): better for large defects
        - Large grid (e.g. 48): better for small/fine defects
        """
        all_features = []
        
        for scale_grid in self.multi_scale_grids:
            features = [
                F.adaptive_avg_pool2d(self.activation[ln], (scale_grid, scale_grid))
                for ln in self.selected_layers
            ]
            concat = torch.cat(features, dim=1)
            # Resize to main grid size
            concat = F.adaptive_avg_pool2d(concat, (self.grid_size, self.grid_size))
            all_features.append(concat)
        
        # Average across scales
        combined = torch.stack(all_features, dim=0).mean(dim=0)
        patches = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])
        
        return patches

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract patch features with SIFE from PIL Image.
        
        🔥 Key fix: Apply CNN_WEIGHT before concat to balance CNN vs SIFE
        
        Returns: (grid_size*grid_size, feature_dim) numpy array
        """
        # CNN features
        x = self.transform(image).unsqueeze(0).to(self.device)
        _ = self.backbone(x)
        
        # 🔥 Multi-scale or single-scale feature extraction
        if self.use_multi_scale:
            patches = self._extract_multi_scale_features(x)
        else:
            features = [
                F.adaptive_avg_pool2d(self.activation[ln], (self.grid_size, self.grid_size))
                for ln in self.selected_layers
            ]
            concat = torch.cat(features, dim=1)
            patches = concat.permute(0, 2, 3, 1).reshape(-1, concat.shape[1])
        
        # 🔥 KEY FIX: Apply CNN weight BEFORE concat with SIFE
        # This prevents L2 normalize from washing out spatial signal
        patches = patches * self.cnn_weight
        
        # Add SIFE features
        sife_parts = []
        
        if self.use_sife:
            pe = self.positional_encoding * self.sife_weight
            sife_parts.append(pe)
        
        if self.use_center_distance:
            center_dist = self._extract_center_distance()
            sife_parts.append(center_dist)
        
        # Get raw tensor for gradient and color
        x_raw = self.raw_transform(image).unsqueeze(0).to(self.device)
        
        if self.use_local_gradient:
            gradient = self._extract_local_gradient(x_raw)
            sife_parts.append(gradient)
        
        # 🔥 NEW: Laplacian variance for crack/scratch detection
        if self.use_laplacian_variance:
            lap_var = self._extract_laplacian_variance(x_raw)
            sife_parts.append(lap_var)
        
        # Color features
        color_feats = self._extract_color_features(x_raw)
        if color_feats is not None:
            sife_parts.append(color_feats)
        
        # Concatenate all features
        if sife_parts:
            sife_features = torch.cat(sife_parts, dim=1)
            patches = torch.cat([patches, sife_features], dim=1)
        
        # 🔥 Edge Enhancement: weight features by edge intensity
        if self.use_edge_enhancement:
            edge_weights = self._extract_edge_weights(x_raw)
            patches = patches * edge_weights.unsqueeze(1)
        
        # L2 normalize
        patches = patches / (torch.norm(patches, p=2, dim=-1, keepdim=True) + 1e-8)
        
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
    
    def get_anomaly_score_detailed(
        self, 
        patch_features: np.ndarray, 
        index: faiss.Index,
        use_percentile: bool = True,
        percentile: float = 95,
    ) -> dict:
        """
        🔥 Enhanced scoring with multiple metrics for better defect detection.
        
        Uses configurable weights (self.score_weight_*) for fine-tuning sensitivity.
        Default: top_k weighted highest for tiny defect detection.
        
        Returns:
            dict with keys: max, mean, percentile, top_k_mean, score (recommended)
        """
        if patch_features is None or patch_features.shape[0] == 0:
            return {"max": 0.0, "mean": 0.0, "percentile": 0.0, "top_k_mean": 0.0, "score": 0.0}

        patch_features = np.ascontiguousarray(patch_features)
        faiss.normalize_L2(patch_features)

        sim, _ = index.search(patch_features.astype(np.float32), self.k_nearest)
        scores = 1.0 - np.mean(sim, axis=1)
        
        # Multiple scoring methods
        max_score = float(scores.max())
        mean_score = float(scores.mean())
        percentile_score = float(np.percentile(scores, percentile))
        
        # 🔥 Top-K mean (configurable percentage)
        top_k = max(1, int(len(scores) * self.top_k_percent))
        top_k_scores = np.sort(scores)[-top_k:]
        top_k_mean = float(top_k_scores.mean())
        
        # 🔥 Recommended score: weighted combination (configurable weights)
        # เน้น top_k_mean เพื่อจับ defect เล็กๆ ได้ดี
        recommended_score = (
            self.score_weight_max * max_score + 
            self.score_weight_top_k * top_k_mean + 
            self.score_weight_percentile * percentile_score
        )
        
        return {
            "max": max_score,
            "mean": mean_score,
            "percentile": percentile_score,
            "top_k_mean": top_k_mean,
            "score": recommended_score,  # Use this for thresholding
        }
    
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
