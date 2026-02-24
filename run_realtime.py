#!/usr/bin/env python3
"""
Unified Realtime Camera Prediction
===================================
Run camera-based pill inspection with any PatchCore model.

Usage:
    python run_realtime.py --model=mobile_sife_cuda
    python run_realtime.py --model=resnet
    python run_realtime.py --list

Note: FCDD ไม่รองรับ realtime camera — ใช้ run_predict.py แทน

Available models:
    mobile_sife_cuda  - MobileNet + SIFE PatchCore (CUDA Optimized)
    mobilenet_sife    - MobileNet + SIFE PatchCorea
    mobilenet         - MobileNet PatchCore
    resnet            - ResNet18 PatchCore (Color-Aware)
    dinov2            - DINOv2 PatchCore
    cnn_multiscale    - CNN Multi-Scale PatchCore (Tiny Defect)
    wideresnet        - WideResNet50 PatchCore
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

MODEL_REGISTRY = {
    "mobile_sife_cuda": {
        "script": "mobile_sife_cuda/predict_camera.py",
        "desc": "MobileNet + SIFE PatchCore (CUDA Optimized)",
    },
    "mobilenet_sife": {
        "script": "MobilenetSIFE/predict_camera.py",
        "desc": "MobileNet + SIFE PatchCore",
    },
    "mobilenet": {
        "script": "MobilenetPatchCore/predict_camera.py",
        "desc": "MobileNet PatchCore",
    },
    "resnet": {
        "script": "ResnetPatchCore/predict_camera.py",
        "desc": "ResNet50 PatchCore (Color-Aware)",
    },
    "dinov2": {
        "script": "DINOv2PatchCore/predict_camera_dinov2.py",
        "desc": "DINOv2 PatchCore",
    },
    "cnn_multiscale": {
        "script": "CNNMultiScale/predict_camera.py",
        "desc": "CNN Multi-Scale PatchCore (Tiny Defect)",
    },
    "wideresnet": {
        "script": "WideResnetAnomalyCore/predict_camera.py",
        "desc": "WideResNet50 PatchCore",
    },
}


def list_models():
    print("Available models (realtime camera):")
    print("-" * 50)
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name:<20s}  {info['desc']}")
    print("-" * 50)
    print("\nNote: FCDD ใช้ run_predict.py (folder-based)")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Realtime Camera Prediction for MediScan-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use for prediction",
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
            print("\nUsage: python run_realtime.py --model=<model_name>")
        return

    info = MODEL_REGISTRY[args.model]
    script = ROOT / info["script"]

    if not script.exists():
        print(f"Error: Script not found: {script}")
        sys.exit(1)

    print(f"Realtime: {info['desc']}")
    print(f"Script  : {script}")
    print("=" * 50)
    subprocess.run([sys.executable, str(script)])


if __name__ == "__main__":
    main()
