#!/usr/bin/env python3
"""
Run DINOv2 PatchCore Training
Usage: python run_train_dinov2.py
"""
import subprocess
import sys
from pathlib import Path

# Change to DINOv2PatchCore directory and run
script_path = Path(__file__).parent / "DINOv2PatchCore" / "train_patchcore_dinov2.py"
subprocess.run([sys.executable, str(script_path)])
