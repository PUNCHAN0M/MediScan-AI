"""
FCDD Model - Fully Convolutional Data Description
==================================================
MobileNetV3-Large backbone + lightweight decoder head
สร้าง anomaly heatmap แบบ pixel-level

Architecture:
  MobileNetV3 features (frozen) → multi-scale hooks
      → 1x1 conv projectors per scale
      → bilinear upsample → average
      → anomaly heatmap (B, 1, H, W)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FCDDHead(nn.Module):
    """Lightweight decoder head: project each feature map to 1 channel and fuse."""

    def __init__(self, layer_channels: list, img_size: int = 256):
        super().__init__()
        self.img_size = img_size

        self.projectors = nn.ModuleList()
        for ch in layer_channels:
            mid = max(ch // 4, 16)
            self.projectors.append(
                nn.Sequential(
                    nn.Conv2d(ch, mid, 1, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid, 1, 1, bias=False),
                )
            )

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: list of feature tensors from backbone hooks
        Returns:
            anomaly_map: (B, 1, H, W)
        """
        maps = []
        for feat, proj in zip(features, self.projectors):
            m = proj(feat)
            m = F.interpolate(
                m,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            maps.append(m)
        # Average across scales
        anomaly_map = torch.stack(maps, dim=0).mean(dim=0)
        return anomaly_map


class FCDDNet(nn.Module):
    """
    FCDD network with MobileNetV3-Large backbone.

    - Backbone is frozen (pretrained)
    - Only decoder head is trained
    - Produces pixel-level anomaly heatmap
    """

    def __init__(
        self,
        backbone_path: str = None,
        img_size: int = 256,
        hook_indices: list = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.hook_indices = hook_indices or [3, 6, 9, 12, 15]

        # ---- Load MobileNetV3-Large backbone ----
        backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.features = backbone.features  # nn.Sequential of InvertedResidual blocks

        # Try loading custom pill-pretrained weights
        if backbone_path and os.path.exists(backbone_path):
            self._load_backbone_weights(backbone_path)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
            self.features.eval()

        self.freeze_backbone = freeze_backbone

        # ---- Register hooks for multi-scale features ----
        self._activations = {}
        self._hooks = []
        for idx in self.hook_indices:
            handle = self.features[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

        # ---- Determine channel dimensions via dummy forward ----
        layer_channels = self._get_layer_channels()

        # ---- Build FCDD decoder head ----
        self.head = FCDDHead(layer_channels, img_size)

    def _load_backbone_weights(self, path: str):
        """Load backbone weights with multiple format support."""
        state = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(state, dict):
            # Try common keys
            for key in ["backbone", "features", "state_dict", "model_state_dict", "model"]:
                if key in state:
                    state = state[key]
                    break

            # Filter for 'features.' prefix if present
            if any(k.startswith("features.") for k in state.keys()):
                state = {
                    k.replace("features.", "", 1): v
                    for k, v in state.items()
                    if k.startswith("features.")
                }

            try:
                self.features.load_state_dict(state, strict=False)
                print(f"[FCDD] Loaded backbone weights from {path}")
            except Exception as e:
                print(f"[FCDD] Warning: Could not load backbone weights: {e}")
                print("[FCDD] Using ImageNet pretrained weights instead")
        else:
            print(f"[FCDD] Warning: Unexpected format in {path}, using ImageNet weights")

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._activations[idx] = output
        return hook

    @torch.no_grad()
    def _get_layer_channels(self) -> list:
        """Run dummy forward to get channel dims for each hooked layer."""
        dummy = torch.zeros(1, 3, self.img_size, self.img_size)
        device = next(self.features.parameters()).device
        dummy = dummy.to(device)
        self.features(dummy)
        channels = [self._activations[idx].shape[1] for idx in self.hook_indices]
        self._activations.clear()
        return channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            anomaly_map: (B, 1, H, W) anomaly heatmap
        """
        self.features(x)
        feats = [self._activations[idx] for idx in self.hook_indices]
        anomaly_map = self.head(feats)
        return anomaly_map

    def train(self, mode: bool = True):
        """Override to keep backbone frozen during training."""
        super().train(mode)
        if self.freeze_backbone:
            self.features.eval()
        return self
