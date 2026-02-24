#!/usr/bin/env python3
"""
Unified Training Runner
=======================
Train any model with a single command.

Usage:
    python run_train.py --model=mobile_sife_cuda
    python run_train.py --model=fcdd
    python run_train.py --list

Available models:
    mobile_sife_cuda  - MobileNet + SIFE PatchCore (CUDA Optimized)
    mobilenet_sife    - MobileNet + SIFE PatchCore
    mobilenet         - MobileNet PatchCore
    resnet            - ResNet18 PatchCore (Color-Aware)
    dinov2            - DINOv2 PatchCore
    cnn_multiscale    - CNN Multi-Scale PatchCore (Tiny Defect)
    wideresnet        - WideResNet50 PatchCore
    fcdd              - FCDD Anomaly Detection
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

MODEL_REGISTRY = {
    "mobile_sife_cuda": {
        "script": "mobile_sife_cuda/train_patchcore.py",
        "desc": "MobileNet + SIFE PatchCore (CUDA Optimized)",
    },
    "mobilenet_sife": {
        "script": "MobilenetSIFE/train_patchcore.py",
        "desc": "MobileNet + SIFE PatchCore",
    },
    "mobilenet": {
        "script": "MobilenetPatchCore/train_patchcore.py",
        "desc": "MobileNet PatchCore",
    },
    "resnet": {
        "script": "ResnetPatchCore/train_patchcore.py",
        "desc": "ResNet50 PatchCore (Color-Aware)",
    },
    "dinov2": {
        "script": "DINOv2PatchCore/train_patchcore_dinov2.py",
        "desc": "DINOv2 PatchCore",
    },
    "cnn_multiscale": {
        "script": "CNNMultiScale/train_patchcore.py",
        "desc": "CNN Multi-Scale PatchCore (Tiny Defect)",
    },
    "wideresnet": {
        "script": "WideResnetAnomalyCore/train_patchcore.py",
        "desc": "WideResNet50 PatchCore",
    },
    "fcdd": {
        "script": "FCDD/fcdd_train.py",
        "desc": "FCDD Anomaly Detection",
    },
}


def list_models():
    print("Available models:")
    print("-" * 50)
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name:<20s}  {info['desc']}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Training Runner for MediScan-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list or args.model is None:
        list_models()
        if args.model is None:
            print("\nUsage: python run_train.py --model=<model_name>")
        return

    info = MODEL_REGISTRY[args.model]
    script = ROOT / info["script"]

    if not script.exists():
        print(f"Error: Script not found: {script}")
        sys.exit(1)

    print(f"Training: {info['desc']}")
    print(f"Script  : {script}")
    print("=" * 50)
    subprocess.run([sys.executable, str(script)])


if __name__ == "__main__":
    main()
