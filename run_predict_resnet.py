#!/usr/bin/env python3
"""
Run ResNet18 PatchCore Camera Prediction (Color-Aware)
Usage: python run_predict_resnet.py

🔥 Best for: Color anomaly detection (white vs black pills, colored defects)
"""
import subprocess
import sys
from pathlib import Path

# Change to ResnetPatchCore directory and run
script_path = Path(__file__).parent / "ResnetPatchCore" / "predict_camera.py"
subprocess.run([sys.executable, str(script_path)])
