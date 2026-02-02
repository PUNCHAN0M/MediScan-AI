#!/usr/bin/env python3
"""
Run DINOv2 PatchCore Camera Prediction
Usage: python run_predict_dinov2.py
"""
import subprocess
import sys
from pathlib import Path

# Change to DINOv2PatchCore directory and run
script_path = Path(__file__).parent / "DINOv2PatchCore" / "predict_camera_dinov2.py"
subprocess.run([sys.executable, str(script_path)])
