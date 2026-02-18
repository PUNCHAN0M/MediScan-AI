#!/usr/bin/env python3
"""
Run CNN Multi-Scale PatchCore Training (Tiny Defect Optimized)
Usage: python run_train_cnnmultiscale.py

🔥 Best for: Tiny crack detection (2-5px), micro defects, surface scratches
"""
import subprocess
import sys
from pathlib import Path

# Change to CNNMultiScale directory and run
script_path = Path(__file__).parent / "CNNMultiScale" / "train_patchcore.py"
subprocess.run([sys.executable, str(script_path)])
