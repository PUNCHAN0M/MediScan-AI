#!/usr/bin/env python3
"""
Run MobileNet + SIFE PatchCore Camera Prediction (CUDA Optimized)
Usage: python run_predict_sife_cuda.py
"""
import subprocess
import sys
from pathlib import Path

script_path = Path(__file__).parent / "mobile_sife_cuda" / "predict_camera.py"
subprocess.run([sys.executable, str(script_path)])
