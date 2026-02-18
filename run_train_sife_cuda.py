#!/usr/bin/env python3
"""
Run MobileNet + SIFE PatchCore Training (CUDA Optimized)
Usage: python run_train_sife_cuda.py
"""
import subprocess
import sys
from pathlib import Path

script_path = Path(__file__).parent / "mobile_sife_cuda" / "train_patchcore.py"
subprocess.run([sys.executable, str(script_path)])
