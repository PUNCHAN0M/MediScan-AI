#!/usr/bin/env python3
"""
Run ResNet18 PatchCore Training (Color-Aware)
Usage: python run_train_resnet.py

🔥 Best for: Color anomaly detection (white vs black pills, colored defects)
"""
import subprocess
import sys
from pathlib import Path

# Change to ResnetPatchCore directory and run
script_path = Path(__file__).parent / "ResnetPatchCore" / "train_patchcore.py"
subprocess.run([sys.executable, str(script_path)])
