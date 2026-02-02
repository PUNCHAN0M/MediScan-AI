#!/usr/bin/env python3
"""
Run MobileNet PatchCore Training
Usage: python run_train_mobilenet.py
"""
import subprocess
import sys
from pathlib import Path

# Change to MobilenetPatchCore directory and run
script_path = Path(__file__).parent / "MobilenetPatchCore" / "train_patchcore.py"
subprocess.run([sys.executable, str(script_path)])
