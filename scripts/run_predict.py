# scripts/run_predict.py
"""
Entry point — Realtime camera prediction.
===========================================
Usage::

    python -m scripts.run_predict
    python -m scripts.run_predict --classes Androxsil ANTA1mg
"""
from __future__ import annotations

import argparse

from core.config import Config
from core.device import get_device, setup_cuda, setup_seed
from pipeline.infer_pipeline import PillInspector, InspectorConfig
from pipeline.predict_pipeline import run_camera


def main():
    parser = argparse.ArgumentParser(description="Realtime Pill Inspection")
    parser.add_argument("--classes", "-c", nargs="*", default=None)
    args = parser.parse_args()

    cfg = Config()
    device = get_device()
    setup_cuda()
    setup_seed(cfg.train.seed)

    compare_classes = args.classes or list(cfg.train.selected_classes)

    inspector_cfg = InspectorConfig.from_config(cfg, compare_classes)
    inspector_cfg.device = device

    inspector = PillInspector(inspector_cfg)

    run_camera(
        inspector=inspector,
        compare_classes=compare_classes,
        camera_index=cfg.camera.camera_index,
        frames_before_summary=cfg.camera.frames_before_summary,
        save_dir=cfg.paths.save_dir,
        window_name=cfg.camera.window_name,
    )


if __name__ == "__main__":
    main()
