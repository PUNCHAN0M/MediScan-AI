# core_shared/patchcore_sife.py
"""
PatchCore Feature Extractor with SIFE (Spatial Information Feature Enhancement).

CUDA-Optimized Version:
- Pre-allocated GPU tensors (sobel kernels, positional encoding, center distance)
- Automatic Mixed Precision (AMP) for ~1.5-2x backbone speedup
- Batch processing for multiple images in one forward pass
- Pure PyTorch ops (no CPU roundtrips for laplacian/gradient)
- Optional FAISS GPU index for faster nearest-neighbor search

Responsibilities:
- MobileNetV3 backbone feature extraction
- SIFE spatial/positional encoding
- Multi-scale feature aggregation
- Edge enhancement, gradient, laplacian, color features
- FAISS index construction and anomaly scoring
"""
import math
import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from typing import Dict, List, Optional, Tuple

from .cuda_utils import (
    get_optimal_device,
    warmup_cuda,
    amp_autocast,
    supports_amp,
    check_faiss_gpu,
    faiss_index_to_gpu,
)


class PatchCoreSIFE:
    """
    Feature extractor: MobileNetV3-Large backbone + SIFE.
    
    SIFE = Spatial Information Feature Enhancement
    - Positional encoding for WHERE defects occur
    - Gradient/Laplacian for micro-texture variations
    - Multi-scale for defects of varying sizes
    - Edge enhancement for border anomalies
    
    All tensors pre-allocated on GPU. Supports single + batch extraction.
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
        sife_encoding_type: str = "sinusoidal",
        sife_weight: float = 1.0,
        use_center_distance: bool = True,
        use_local_gradient: bool = True,
        # CNN vs SIFE balancing
        cnn_weight: float = 1.0,
        # Laplacian variance
        use_laplacian_variance: bool = False,
        laplacian_weight: float = 1.0,
        # Color features
        use_color_features: bool = False,
        use_hsv: bool = False,
        color_weight: float = 1.0,
        # Multi-scale & Edge
        use_multi_scale: bool = False,
        multi_scale_grids: list = None,
        use_edge_enhancement: bool = False,
        edge_weight: float = 1.5,
        # Scoring weights
        score_weight_max: float = 0.3,
        score_weight_top_k: float = 0.5,
        score_weight_percentile: float = 0.2,
        top_k_percent: float = 0.05,
        # FAISS GPU
        use_faiss_gpu: bool = True,
    ):
        # --- Device ---
        self.device = get_optimal_device() if device is None else torch.device(device)
        self._use_amp = supports_amp(self.device)
        warmup_cuda(self.device)

        # --- Config ---
        self.model_size = model_size
        self.grid_size = grid_size
        self.k_nearest = k_nearest
        self.cnn_weight = cnn_weight

        # SIFE
        self.use_sife = use_sife
        self.sife_dim = sife_dim
        self.sife_encoding_type = sife_encoding_type
        self.sife_weight = sife_weight
        self.use_center_distance = use_center_distance
        self.use_local_gradient = use_local_gradient

        # Laplacian
        self.use_laplacian_variance = use_laplacian_variance
        self.laplacian_weight = laplacian_weight

        # Color
        self.use_color_features = use_color_features
        self.use_hsv = use_hsv
        self.color_weight = color_weight

        # Multi-scale & Edge
        self.use_multi_scale = use_multi_scale
        self.multi_scale_grids = multi_scale_grids or [16, 32, 48]
        self.use_edge_enhancement = use_edge_enhancement
        self.edge_weight = edge_weight

        # Scoring
        self.score_weight_max = score_weight_max
        self.score_weight_top_k = score_weight_top_k
        self.score_weight_percentile = score_weight_percentile
        self.top_k_percent = top_k_percent

        # FAISS
        self.use_faiss_gpu = use_faiss_gpu and check_faiss_gpu()

        # --- Backbone ---
        self._init_backbone()

        # --- Pre-allocated GPU Tensors ---
        self._init_static_tensors()

        # --- Transforms ---
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

        # --- Feature dims ---
        self._calc_sife_dim()
        self._print_config()

    # =============================================================
    # INITIALIZATION
    # =============================================================

    def _init_backbone(self) -> None:
        """Load MobileNetV3-Large and register hooks for multi-layer features."""
        backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.backbone = backbone.to(self.device).eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.selected_layers = [
            "features.3",   # edge / texture      (24 ch)
            "features.6",   # surface pattern     (40 ch)
            "features.9",   # shape               (80 ch)
            "features.12",  # structure            (112 ch)
            "features.15",  # semantic             (160 ch)
        ]

        self.activation: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks for intermediate feature extraction."""
        def _make_hook(name: str):
            def hook(_, __, output):
                self.activation[name] = output.detach()
            return hook

        modules = dict(self.backbone.named_modules())
        for layer_name in self.selected_layers:
            modules[layer_name].register_forward_hook(_make_hook(layer_name))

    def _init_static_tensors(self) -> None:
        """
        Pre-allocate and cache all static tensors on GPU.
        
        Avoids re-creation every frame:
        - Sobel kernels for gradient/edge detection
        - Laplacian kernel for variance
        - Positional encoding matrix
        - Center distance matrix
        """
        G = self.grid_size
        dev = self.device

        # --- Sobel kernels (for gradient & edge) ---
        self._sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=dev,
        ).view(1, 1, 3, 3)
        self._sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=dev,
        ).view(1, 1, 3, 3)

        # --- Laplacian kernel ---
        self._laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32, device=dev,
        ).view(1, 1, 3, 3)

        # --- Positional encoding (G*G, sife_dim) ---
        self._pos_encoding: Optional[torch.Tensor] = None
        if self.use_sife:
            if self.sife_encoding_type == "coordinate":
                pe = self._create_coordinate_pe(G, self.sife_dim)
            else:
                pe = self._create_sinusoidal_pe(G, self.sife_dim)
            self._pos_encoding = pe.to(dev)

        # --- Center distance (G*G, 1) ---
        self._center_dist: Optional[torch.Tensor] = None
        if self.use_center_distance:
            y = torch.linspace(-1, 1, G, device=dev)
            x = torch.linspace(-1, 1, G, device=dev)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            dist = torch.sqrt(xx**2 + yy**2).flatten().unsqueeze(1)
            self._center_dist = dist / dist.max()

    def _calc_sife_dim(self) -> None:
        """Calculate total SIFE feature dimension."""
        dim = 0
        if self.use_sife:
            dim += self.sife_dim
        if self.use_center_distance:
            dim += 1
        if self.use_local_gradient:
            dim += 2
        if self.use_laplacian_variance:
            dim += 1
        if self.use_color_features:
            dim += 6
        if self.use_hsv:
            dim += 6
        self.sife_feature_dim = dim

    def _print_config(self) -> None:
        """Print initialization summary."""
        print(f"[PatchCoreSIFE] CUDA={self.device.type == 'cuda'} | AMP={self._use_amp} | FAISS-GPU={self.use_faiss_gpu}")
        print(f"  Grid={self.grid_size}x{self.grid_size} | CNN_w={self.cnn_weight} | SIFE_w={self.sife_weight}")
        print(f"  SIFE={self.use_sife}(dim={self.sife_dim}) | Grad={self.use_local_gradient} | Lap={self.use_laplacian_variance}")
        print(f"  MultiScale={self.use_multi_scale}({self.multi_scale_grids if self.use_multi_scale else '-'})")
        print(f"  Edge={self.use_edge_enhancement}(w={self.edge_weight}) | Color={self.use_color_features} | HSV={self.use_hsv}")
        print(f"  Scoring: max={self.score_weight_max} top_k={self.score_weight_top_k} pct={self.score_weight_percentile}")
        print(f"  Total SIFE dim: {self.sife_feature_dim}")

    # =============================================================
    # POSITIONAL ENCODING
    # =============================================================

    @staticmethod
    def _create_sinusoidal_pe(grid_size: int, dim: int) -> torch.Tensor:
        """2D sinusoidal positional encoding (Transformer-style). → (G*G, dim)"""
        y_pos = torch.arange(grid_size).unsqueeze(1).expand(grid_size, grid_size)
        x_pos = torch.arange(grid_size).unsqueeze(0).expand(grid_size, grid_size)

        y_pos = 2.0 * y_pos.float() / (grid_size - 1) - 1.0
        x_pos = 2.0 * x_pos.float() / (grid_size - 1) - 1.0

        half_dim = dim // 4
        freqs = torch.exp(torch.arange(half_dim) * -(math.log(10000.0) / half_dim))

        y_pe = y_pos.flatten().unsqueeze(1) * freqs.unsqueeze(0)
        x_pe = x_pos.flatten().unsqueeze(1) * freqs.unsqueeze(0)

        pe = torch.cat([torch.sin(y_pe), torch.cos(y_pe),
                        torch.sin(x_pe), torch.cos(x_pe)], dim=1)

        if pe.shape[1] > dim:
            pe = pe[:, :dim]
        elif pe.shape[1] < dim:
            pe = F.pad(pe, (0, dim - pe.shape[1]))
        return pe

    @staticmethod
    def _create_coordinate_pe(grid_size: int, dim: int) -> torch.Tensor:
        """Coordinate-based positional encoding. → (G*G, dim)"""
        y = torch.linspace(-1, 1, grid_size).unsqueeze(1).expand(grid_size, grid_size)
        x = torch.linspace(-1, 1, grid_size).unsqueeze(0).expand(grid_size, grid_size)

        y_flat = y.flatten().unsqueeze(1)
        x_flat = x.flatten().unsqueeze(1)

        n_freqs = dim // 4
        features = [y_flat, x_flat]
        for i in range(1, n_freqs):
            freq = 2 ** i
            features.extend([
                torch.sin(y_flat * freq * math.pi),
                torch.cos(x_flat * freq * math.pi),
            ])

        pe = torch.cat(features, dim=1)
        if pe.shape[1] > dim:
            pe = pe[:, :dim]
        elif pe.shape[1] < dim:
            pe = F.pad(pe, (0, dim - pe.shape[1]))
        return pe

    # =============================================================
    # FEATURE EXTRACTION — SINGLE IMAGE
    # =============================================================

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract patch features from a single PIL Image.
        
        Delegates to batch extraction for unified code path.
        
        Returns:
            (grid_size*grid_size, feature_dim) float32 numpy array
        """
        return self.extract_features_batch([image])[0]

    def extract_from_numpy(self, np_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from a single OpenCV BGR image."""
        if np_img is None or np_img.size == 0:
            return None
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.extract_features(pil)

    def extract_from_numpy_batch(self, np_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract features from multiple OpenCV BGR images.
        
        Handles None/empty images gracefully by returning None at those indices.
        """
        pil_images: List[Image.Image] = []
        valid_indices: List[int] = []

        for i, img in enumerate(np_images):
            if img is not None and img.size > 0:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))
                valid_indices.append(i)

        if not pil_images:
            return [None] * len(np_images)

        valid_results = self.extract_features_batch(pil_images)

        results: List[Optional[np.ndarray]] = [None] * len(np_images)
        for idx, val_idx in enumerate(valid_indices):
            results[val_idx] = valid_results[idx]
        return results

    # =============================================================
    # FEATURE EXTRACTION — BATCH (CUDA OPTIMIZED)
    # =============================================================

    @torch.no_grad()
    def extract_features_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Batch feature extraction with full CUDA + AMP optimization.
        
        Pipeline:
            1. Stack images → single batch tensor
            2. Backbone forward with AMP (FP16 on CUDA)
            3. Multi-layer adaptive pooling → CNN features
            4. SIFE features: PE, center dist, gradient, laplacian, color
            5. Edge enhancement weighting
            6. L2 normalize
        
        Args:
            images: List of PIL Images
        Returns:
            List of (grid_size*grid_size, feature_dim) numpy arrays
        """
        if not images:
            return []

        B = len(images)
        G = self.grid_size
        dev = self.device

        # --- 1. Prepare batch tensors (pinned memory → GPU) ---
        batch = torch.stack([self.transform(img) for img in images])
        batch = batch.to(dev, non_blocking=True)

        need_raw = (
            self.use_local_gradient
            or self.use_laplacian_variance
            or self.use_color_features
            or self.use_hsv
            or self.use_edge_enhancement
        )
        raw_batch: Optional[torch.Tensor] = None
        if need_raw:
            raw_batch = torch.stack([self.raw_transform(img) for img in images])
            raw_batch = raw_batch.to(dev, non_blocking=True)

        # --- 2. Backbone forward with AMP ---
        with amp_autocast(dev):
            self.backbone(batch)

        # --- 3. CNN features ---
        if self.use_multi_scale:
            patches = self._extract_multi_scale_batch(B, G)
        else:
            features = [
                F.adaptive_avg_pool2d(self.activation[ln], (G, G))
                for ln in self.selected_layers
            ]
            concat = torch.cat(features, dim=1).float()  # (B, C, G, G) → float32
            patches = concat.permute(0, 2, 3, 1).reshape(B, G * G, -1)

        patches = patches * self.cnn_weight

        # --- 4. SIFE features ---
        sife_parts: List[torch.Tensor] = []

        if self.use_sife and self._pos_encoding is not None:
            pe = self._pos_encoding.unsqueeze(0).expand(B, -1, -1) * self.sife_weight
            sife_parts.append(pe)

        if self.use_center_distance and self._center_dist is not None:
            cd = self._center_dist.unsqueeze(0).expand(B, -1, -1)
            sife_parts.append(cd)

        if self.use_local_gradient and raw_batch is not None:
            grad = self._compute_gradient_batch(raw_batch)  # (B, G*G, 2)
            sife_parts.append(grad)

        if self.use_laplacian_variance and raw_batch is not None:
            lap = self._compute_laplacian_batch(raw_batch)  # (B, G*G, 1)
            sife_parts.append(lap)

        if (self.use_color_features or self.use_hsv) and raw_batch is not None:
            color = self._compute_color_batch(raw_batch)
            if color is not None:
                sife_parts.append(color)

        if sife_parts:
            sife = torch.cat(sife_parts, dim=2)
            patches = torch.cat([patches, sife], dim=2)

        # --- 5. Edge enhancement ---
        if self.use_edge_enhancement and raw_batch is not None:
            edge_w = self._compute_edge_weights_batch(raw_batch)  # (B, G*G)
            patches = patches * edge_w.unsqueeze(2)

        # --- 6. L2 normalize ---
        patches = F.normalize(patches, p=2, dim=2)

        # --- 7. Output as numpy ---
        patches_np = patches.cpu().numpy()
        return [np.ascontiguousarray(patches_np[i]) for i in range(B)]

    def _extract_multi_scale_batch(self, B: int, G: int) -> torch.Tensor:
        """
        Multi-scale feature extraction: pool at multiple grids, average.
        
        Returns: (B, G*G, C) float32 tensor
        """
        all_features = []
        for scale_grid in self.multi_scale_grids:
            layer_feats = [
                F.adaptive_avg_pool2d(self.activation[ln], (scale_grid, scale_grid))
                for ln in self.selected_layers
            ]
            concat = torch.cat(layer_feats, dim=1)
            resized = F.adaptive_avg_pool2d(concat, (G, G))
            all_features.append(resized)

        combined = torch.stack(all_features, dim=0).mean(dim=0).float()
        return combined.permute(0, 2, 3, 1).reshape(B, G * G, -1)

    # =============================================================
    # SIFE COMPONENT EXTRACTORS (ALL BATCHED, ALL GPU)
    # =============================================================

    def _compute_gradient_batch(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Batch local gradient extraction. Pure PyTorch, stays on GPU.
        
        Args:
            raw: (B, 3, H, W) raw image tensor
        Returns:
            (B, G*G, 2) tensor [magnitude, direction]
        """
        G = self.grid_size
        gray = (0.299 * raw[:, 0] + 0.587 * raw[:, 1] + 0.114 * raw[:, 2]).unsqueeze(1)

        gx = F.conv2d(gray, self._sobel_x, padding=1)
        gy = F.conv2d(gray, self._sobel_y, padding=1)

        magnitude = torch.sqrt(gx ** 2 + gy ** 2)
        direction = torch.atan2(gy, gx) / math.pi

        mag = F.adaptive_avg_pool2d(magnitude, (G, G)).view(-1, G * G)
        dir_ = F.adaptive_avg_pool2d(direction, (G, G)).view(-1, G * G)

        # Normalize magnitude per image
        mag_max = mag.max(dim=1, keepdim=True)[0] + 1e-8
        mag = mag / mag_max

        return torch.stack([mag, dir_], dim=2)  # (B, G*G, 2)

    def _compute_laplacian_batch(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Batch Laplacian variance per patch. Pure PyTorch (no OpenCV).
        
        Highlights micro-texture variations: cracks, scratches, surface defects.
        
        Args:
            raw: (B, 3, H, W) tensor
        Returns:
            (B, G*G, 1) tensor
        """
        G = self.grid_size
        gray = (0.299 * raw[:, 0] + 0.587 * raw[:, 1] + 0.114 * raw[:, 2]).unsqueeze(1)

        laplacian = F.conv2d(gray, self._laplacian_kernel, padding=1)

        B, _, H, W = laplacian.shape
        pH = H // G
        pW = W // G

        # Crop to exact grid-aligned size
        cropped = laplacian[:, :, :pH * G, :pW * G]

        # Unfold into patches → (B, 1*pH*pW, G*G)
        patches = F.unfold(cropped, kernel_size=(pH, pW), stride=(pH, pW))

        # Variance per patch → (B, G*G)
        variance = patches.var(dim=1)
        var_max = variance.max(dim=1, keepdim=True)[0] + 1e-8
        variance = (variance / var_max) * self.laplacian_weight

        return variance.unsqueeze(2)  # (B, G*G, 1)

    def _compute_color_batch(self, raw: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Batch color statistics (mean + std) per patch.
        
        Args:
            raw: (B, 3, H, W) tensor (RGB normalized 0-1)
        Returns:
            (B, G*G, color_dim) tensor or None
        """
        if not (self.use_color_features or self.use_hsv):
            return None

        B, C, H, W = raw.shape
        G = self.grid_size
        pH = H // G
        pW = W // G

        cropped = raw[:, :, :pH * G, :pW * G]

        # Unfold → (B, C*pH*pW, G*G)
        patches = F.unfold(cropped, kernel_size=(pH, pW), stride=(pH, pW))
        # → (B, C, pH*pW, G*G) → (B, G*G, C, pH*pW)
        patches = patches.view(B, C, pH * pW, G * G).permute(0, 3, 1, 2)

        color_parts: List[torch.Tensor] = []

        if self.use_color_features:
            rgb_mean = patches.mean(dim=3)  # (B, G*G, 3)
            rgb_std = patches.std(dim=3)
            color_parts.extend([rgb_mean, rgb_std])

        if self.use_hsv:
            # Convert full image to HSV, then extract patches
            hsv = self._rgb_to_hsv_batch(raw)
            hsv_cropped = hsv[:, :, :pH * G, :pW * G]
            hsv_patches = F.unfold(hsv_cropped, kernel_size=(pH, pW), stride=(pH, pW))
            hsv_patches = hsv_patches.view(B, C, pH * pW, G * G).permute(0, 3, 1, 2)
            hsv_mean = hsv_patches.mean(dim=3)
            hsv_std = hsv_patches.std(dim=3)
            color_parts.extend([hsv_mean, hsv_std])

        result = torch.cat(color_parts, dim=2)
        return result * self.color_weight

    @staticmethod
    def _rgb_to_hsv_batch(rgb: torch.Tensor) -> torch.Tensor:
        """Convert (B, 3, H, W) RGB tensor to HSV. All on GPU."""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        max_rgb, argmax_rgb = rgb.max(dim=1)
        min_rgb = rgb.min(dim=1)[0]
        diff = max_rgb - min_rgb

        v = max_rgb
        s = torch.zeros_like(v)
        mask = v > 0
        s[mask] = diff[mask] / v[mask]

        h = torch.zeros_like(v)
        nonzero = diff > 0
        mask_r = (argmax_rgb == 0) & nonzero
        h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r] / 6.0
        mask_g = (argmax_rgb == 1) & nonzero
        h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] / 6.0 + 1 / 3
        mask_b = (argmax_rgb == 2) & nonzero
        h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] / 6.0 + 2 / 3
        h = h % 1.0

        return torch.stack([h, s, v], dim=1)

    def _compute_edge_weights_batch(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Batch edge-intensity weighting. High edges get more feature emphasis.
        
        Args:
            raw: (B, 3, H, W)
        Returns:
            (B, G*G) weights in range [1.0, edge_weight]
        """
        G = self.grid_size
        gray = (0.299 * raw[:, 0] + 0.587 * raw[:, 1] + 0.114 * raw[:, 2]).unsqueeze(1)

        gx = F.conv2d(gray, self._sobel_x, padding=1)
        gy = F.conv2d(gray, self._sobel_y, padding=1)
        edge_mag = torch.sqrt(gx ** 2 + gy ** 2)

        pooled = F.adaptive_avg_pool2d(edge_mag, (G, G)).view(-1, G * G)

        # Normalize per image → [0, 1]
        e_min = pooled.min(dim=1, keepdim=True)[0]
        e_max = pooled.max(dim=1, keepdim=True)[0]
        normalized = (pooled - e_min) / (e_max - e_min + 1e-8)

        # Map to [1.0, edge_weight]
        return 1.0 + normalized * (self.edge_weight - 1.0)

    # =============================================================
    # FAISS INDEX & ANOMALY SCORING
    # =============================================================

    def build_faiss_index(self, memory_bank: np.ndarray) -> faiss.Index:
        """
        Build FAISS inner-product index from normalized memory bank.
        
        Uses GPU index when FAISS-GPU is available for faster search.
        """
        bank = np.ascontiguousarray(memory_bank.astype(np.float32))
        faiss.normalize_L2(bank)

        d = bank.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(bank)

        if self.use_faiss_gpu:
            index = faiss_index_to_gpu(index)

        return index

    def get_max_anomaly_score(self, patch_features: np.ndarray, index: faiss.Index) -> float:
        """
        Max anomaly score: 1 - similarity of worst patch.
        
        Backward-compatible single-score method.
        """
        if patch_features is None or patch_features.shape[0] == 0:
            return 0.0

        feats = np.ascontiguousarray(patch_features.astype(np.float32))
        faiss.normalize_L2(feats)

        sim, _ = index.search(feats, self.k_nearest)
        scores = 1.0 - np.mean(sim, axis=1)
        return float(scores.max())

    def get_anomaly_score_detailed(
        self,
        patch_features: np.ndarray,
        index: faiss.Index,
        percentile: float = 95,
    ) -> Dict[str, float]:
        """
        Detailed anomaly scoring with configurable weight combination.
        
        Returns:
            Dict with max, mean, percentile, top_k_mean, score (weighted combo)
        """
        empty = {"max": 0.0, "mean": 0.0, "percentile": 0.0, "top_k_mean": 0.0, "score": 0.0}
        if patch_features is None or patch_features.shape[0] == 0:
            return empty

        feats = np.ascontiguousarray(patch_features.astype(np.float32))
        faiss.normalize_L2(feats)

        sim, _ = index.search(feats, self.k_nearest)
        scores = 1.0 - np.mean(sim, axis=1)

        max_score = float(scores.max())
        mean_score = float(scores.mean())
        pct_score = float(np.percentile(scores, percentile))

        top_k = max(1, int(len(scores) * self.top_k_percent))
        top_k_mean = float(np.sort(scores)[-top_k:].mean())

        recommended = (
            self.score_weight_max * max_score
            + self.score_weight_top_k * top_k_mean
            + self.score_weight_percentile * pct_score
        )

        return {
            "max": max_score,
            "mean": mean_score,
            "percentile": pct_score,
            "top_k_mean": top_k_mean,
            "score": recommended,
        }

    def get_anomaly_heatmap(
        self,
        patch_features: np.ndarray,
        index: faiss.Index,
        image_size: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """Generate spatial anomaly heatmap for visualization."""
        if patch_features is None or patch_features.shape[0] == 0:
            return np.zeros(image_size, dtype=np.float32)

        feats = np.ascontiguousarray(patch_features.astype(np.float32))
        faiss.normalize_L2(feats)

        sim, _ = index.search(feats, self.k_nearest)
        scores = 1.0 - np.mean(sim, axis=1)

        heatmap = scores.reshape(self.grid_size, self.grid_size).astype(np.float32)
        return cv2.resize(heatmap, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
