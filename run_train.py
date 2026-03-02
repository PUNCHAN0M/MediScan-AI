#!/usr/bin/env python3
"""
ResNet50 PatchCore — Training
==============================
Train ResNet50 PatchCore model for pill anomaly detection.

Usage:
    python run_train.py
    python run_train.py --backbone=model/backbone/resnet_last.pth
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPT = ROOT / "Core_ResnetPatchCore" / "train_patchcore.py"


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 PatchCore — MediScan-AI",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a custom backbone .pth file",
    )
    args = parser.parse_args()

    if not SCRIPT.exists():
        print(f"Error: Script not found: {SCRIPT}")
        sys.exit(1)

    print("Training: ResNet50 PatchCore (Color-Aware)")
    print(f"Script  : {SCRIPT}")
    if args.backbone:
        print(f"Backbone: {args.backbone}")
    print("=" * 50)

    cmd = [sys.executable, str(SCRIPT)]
    if args.backbone:
        cmd += ["--backbone", args.backbone]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
