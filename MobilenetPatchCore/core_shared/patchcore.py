# core_shared/patchcore.py
"""
PatchCore Feature Extractor - MobileNetV3 backbone with multi-layer features.

Single Responsibility:
- Extract patch features from images
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
from typing import Optional


class PatchCore:
    """Feature extractor using MobileNetV3-Large backbone."""
    
    def __init__(
        self,
        model_size: int = 256,
        grid_size: int = 20,
        k_nearest: int = 19,
        device: str = None,
    ):
        """
        Args:
            model_size: Input image size (512, 640, etc.)
            grid_size: Grid size for patch features (14, 20, etc.)
            k_nearest: Number of k-nearest neighbors for anomaly score
            device: "cuda" or "cpu" (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.grid_size = grid_size
        self.k_nearest = k_nearest

        # Feature extractor
        self.backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.backbone = self.backbone.to(self.device).eval()

        self.selected_layers = [
            "features.3",   # edge, texture, noise
            "features.6",   # surface pattern
            "features.9",   # shape, geometry
            "features.12",  # structure, spatial relation
            "features.15",  # semantic, global context
        ]
        
        self.activation = {}
        self._register_hooks()

        # Transform
        self.transform = T.Compose([
            T.Resize((model_size, model_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(model_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _register_hooks(self):
        def get_hook(name):
            def hook(_, __, output):
                self.activation[name] = output.detach()
            return hook

        for layer_name in self.selected_layers:
            layer = dict([*self.backbone.named_modules()])[layer_name]
            layer.register_forward_hook(get_hook(layer_name))

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract patch features from PIL Image."""
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.backbone(x)
            features = [
                F.adaptive_avg_pool2d(self.activation[ln], (self.grid_size, self.grid_size))
                for ln in self.selected_layers
            ]
            concat = torch.cat(features, dim=1)
            patches = concat.permute(0, 2, 3, 1).reshape(-1, concat.shape[1])
            patches = patches / torch.norm(patches, p=2, dim=-1, keepdim=True)

        return patches.contiguous().cpu().numpy()

    def extract_from_numpy(self, np_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from OpenCV BGR image."""
        if np_img.size == 0:
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
