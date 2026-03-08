# main.py
"""
MediScan-AI — Main entry point.
=================================
Dispatch to training, prediction, or the desktop app.

Usage::

    python main.py train
    python main.py predict --classes=Androxsil --show-morph-stage / --no-show-morph-stage
    python main.py infer --image test.jpg --classes Androxsil
    python main.py app
    
"""
from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py {train|predict|infer|app}")
        print("  train   — Train PatchCore models")
        print("  predict — Realtime camera prediction")
        print("  infer   — Single-image / folder inference")
        print("  app     — Launch PyQt6 desktop app")
        sys.exit(1)

    command = sys.argv[1].lower()
    # remove subcommand so argparse in sub-scripts works correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        from scripts.train_patchcore import main as train_main
        train_main()

    elif command == "predict":
        from scripts.run_predict import main as predict_main
        predict_main()

    elif command == "infer":
        from scripts.run_infer import main as infer_main
        infer_main()

    elif command == "app":
        from run_app import main as app_main
        app_main()

    else:
        print(f"Unknown command: '{command}'")
        print("Use one of: train, predict, infer, app")
        sys.exit(1)


if __name__ == "__main__":
    main()
