#!/usr/bin/env python3
"""
Run MobileNet PatchCore Camera Prediction
Usage: python run_predict_mobilenet.py
"""
import subprocess
import sys
from pathlib import Path

# Change to MobilenetPatchCore directory and run
script_path = Path(__file__).parent / "MobilenetPatchCore" / "predict_camera.py"
subprocess.run([sys.executable, str(script_path)])
