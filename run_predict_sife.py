#!/usr/bin/env python3
"""
Run MobileNet + SIFE PatchCore Camera Prediction
Usage: python run_predict_sife.py

🎯 Best for:
- Enhanced spatial awareness for defect localization
- Better small defect detection with position encoding
"""
import subprocess
import sys
from pathlib import Path

# Change to MobilenetSIFE directory and run
script_path = Path(__file__).parent / "MobilenetSIFE" / "predict_camera.py"
subprocess.run([sys.executable, str(script_path)])
