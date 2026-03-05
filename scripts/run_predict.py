# scripts/run_predict.py
"""
Entry point — Realtime camera prediction.
===========================================
Usage::

    python -m scripts.run_predict
    python -m scripts.run_predict --classes Androxsil ANTA1mg

Reads all model/detection/scoring parameters from ``app_settings.json``
so results are identical to the desktop app classifier.
"""
from __future__ import annotations

import argparse

from app.settings_manager import SettingsManager
from core.device import get_device, setup_cuda, setup_seed
from pipeline.infer_pipeline import PillInspector, InspectorConfig
from pipeline.predict_pipeline import run_camera


def main():
    parser = argparse.ArgumentParser(description="Realtime Pill Inspection")
    parser.add_argument("--classes", "-c", nargs="*", default=None)
    args = parser.parse_args()

    settings = SettingsManager()
    device = get_device()
    setup_cuda()
    setup_seed(int(settings.get("seed", 42)))

    compare_classes = args.classes or []

    inspector_cfg = InspectorConfig.from_settings(settings.all(), compare_classes)
    inspector_cfg.device = device

    inspector = PillInspector(inspector_cfg)

    run_camera(
        inspector=inspector,
        compare_classes=compare_classes,
        camera_index=int(settings.get("camera_index", 0)),
        save_dir=None,
        window_name="Pill Inspector",
    )


if __name__ == "__main__":
    main()
