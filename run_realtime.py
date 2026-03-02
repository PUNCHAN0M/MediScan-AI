#!/usr/bin/env python3
"""
ResNet50 PatchCore — Realtime Camera Prediction
================================================
Run camera-based pill inspection with ResNet50 PatchCore.

Usage:
    python run_realtime.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPT = ROOT / "Core_ResnetPatchCore" / "predict_camera.py"


def main():
    if not SCRIPT.exists():
        print(f"Error: Script not found: {SCRIPT}")
        sys.exit(1)

    print("Realtime: ResNet50 PatchCore (Color-Aware)")
    print(f"Script  : {SCRIPT}")
    print("=" * 50)
    subprocess.run([sys.executable, str(SCRIPT)])


if __name__ == "__main__":
    main()
