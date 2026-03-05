"""
Settings Manager — JSON-based persistent settings for MediScan-AI
=================================================================
Reads/writes ``app_settings.json`` in the project root.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_FILE = _ROOT / "app_settings.json"

# ── defaults (match app_settings.json shipped with the repo) ──
_DEFAULTS: Dict[str, Any] = {
    "backbone_model": "weights\\backbone\\resnet_last.pth",
    "segmentation_model": "weights\\detection\\pill-detection-best-2.onnx",
    "detection_model": "weights\\detection\\pill-detection-best-2.onnx",
    "seg_conf": 0.5,
    "seg_iou": 0.6,
    "seg_pad": 5,
    "seg_img_size": 512,
    "det_conf": 0.5,
    "det_iou": 0.6,
    "det_img_size": 512,
    "track_max_dist": 80.0,
    "track_iou_thr": 0.8,
    "track_max_age": 10,
    "coreset_ratio": 0.25,
    "k_nearest": 3,
    "threshold_mult": 1.0,
    "fallback_thr": 0.5,
    "coreset_min_keep": 1000,
    "coreset_max_keep": 20000,
    "grid_size": 16,
    "img_size": 256,
    "use_color": True,
    "use_hsv": True,
    "color_weight": 1.5,
    "enable_layer1": True,
    "enable_layer2": True,
    "enable_layer3": True,
    "n_steps": 351,
    "seed": 42,
    "font_size": 14,
    "camera_index": 0,
    "background_color": "#ffffff",
    "text_color": "#000000",
    "primary_color": "#33FF00",
    "secondary_color": "#FF6A00",
    "font_color": "#000000",
}


class SettingsManager:
    """Singleton-ish JSON settings store."""

    _instance: "SettingsManager | None" = None

    def __new__(cls) -> "SettingsManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = dict(_DEFAULTS)
            cls._instance._load()
        return cls._instance

    # ── persistence ──────────────────────────────────────
    def _load(self) -> None:
        if _SETTINGS_FILE.exists():
            try:
                with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    self._data.update(json.load(f))
            except Exception:
                pass

    def save(self) -> None:
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # ── dict-like access ─────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def all(self) -> Dict[str, Any]:
        return dict(self._data)
