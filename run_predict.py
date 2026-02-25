#!/usr/bin/env python3
"""
Unified Folder-based Prediction
================================
Predict anomalies from a folder of images (not realtime camera).

Usage:
    python run_predict.py --model=fcdd
    python run_predict.py --model=mobile_sife_cuda --input=data_yolo/test --output=result_sife
    python run_predict.py --model=resnet --input=data_yolo/test --output=result_resnet
    python run_predict.py --list

Available models:
    mobile_sife_cuda  - MobileNet + SIFE PatchCore (CUDA Optimized)
    mobilenet_sife    - MobileNet + SIFE PatchCore
    mobilenet         - MobileNet PatchCore
    resnet            - ResNet18 PatchCore (Color-Aware)
    dinov2            - DINOv2 PatchCore
    cnn_multiscale    - CNN Multi-Scale PatchCore (Tiny Defect)
    wideresnet        - WideResNet50 PatchCore
    fcdd              - FCDD Anomaly Detection
"""
import argparse
import subprocess
import sys
import os
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# =============================================================================
#  Model registry
# =============================================================================

# FCDD uses its own inference script
FCDD_SCRIPT = "FCDD/fcdd_inference.py"

# PatchCore-based models share the same inspector pattern
PATCHCORE_MODELS = {
    "mobile_sife_cuda": {
        "module":     "mobile_sife_cuda",
        "config_mod": "config.sife",
        "desc":       "MobileNet + SIFE PatchCore (CUDA Optimized)",
    },
    "mobilenet_sife": {
        "module":     "MobilenetSIFE",
        "config_mod": "config.mobilenet",
        "desc":       "MobileNet + SIFE PatchCore",
    },
    "mobilenet": {
        "module":     "MobilenetPatchCore",
        "config_mod": "config.mobilenet",
        "desc":       "MobileNet PatchCore",
    },
    "resnet": {
        "module":     "ResnetPatchCore",
        "config_mod": "config.resnet",
        "desc":       "ResNet50 PatchCore (Color-Aware)",
    },
    "dinov2": {
        "module":     "DINOv2PatchCore",
        "config_mod": "config.dinov2",
        "desc":       "DINOv2 PatchCore",
    },
    "cnn_multiscale": {
        "module":     "CNNMultiScale",
        "config_mod": "config.cnnmultiscale",
        "desc":       "CNN Multi-Scale PatchCore (Tiny Defect)",
    },
    "wideresnet": {
        "module":     "WideResnetAnomalyCore",
        "config_mod": "config.wideresnet",
        "desc":       "WideResNet50 PatchCore",
    },
}

ALL_MODELS = {
    **{k: v["desc"] for k, v in PATCHCORE_MODELS.items()},
    "fcdd": "FCDD Anomaly Detection",
}


# =============================================================================
#  FCDD prediction (delegates to existing script)
# =============================================================================

def run_fcdd(input_dir: str | None, output_dir: str | None):
    """Run FCDD inference. Optionally override input/output dirs."""
    env = os.environ.copy()
    if input_dir:
        env["FCDD_INPUT_DIR"] = str(input_dir)
    if output_dir:
        env["FCDD_OUTPUT_DIR"] = str(output_dir)

    script = ROOT / FCDD_SCRIPT
    if not script.exists():
        print(f"Error: FCDD script not found: {script}")
        sys.exit(1)

    subprocess.run([sys.executable, str(script)], env=env)


# =============================================================================
#  PatchCore folder prediction (generic for all PatchCore models)
# =============================================================================

def run_patchcore_predict(model_name: str, input_dir: Path, output_dir: Path):
    """
    Generic folder-based prediction for any PatchCore model.

    Pipeline:
      1. Load inspector (same as realtime camera uses)
      2. Read each image from input folder
      3. Run YOLO detect + PatchCore classify
      4. Save annotated result image + print scores
    """
    info = PATCHCORE_MODELS[model_name]
    module_name = info["module"]

    # --- Dynamic import of inspector and config ---
    try:
        inspector_mod = __import__(
            f"{module_name}.core_predict.inspector",
            fromlist=["PillInspectorSIFE", "InspectorConfig"],
        )
        InspectorConfig = inspector_mod.InspectorConfig
        PillInspectorSIFE = inspector_mod.PillInspectorSIFE
    except ImportError:
        # DINOv2 uses different class names
        try:
            inspector_mod = __import__(
                f"{module_name}.inspector_dinov2",
                fromlist=["PillInspectorDINOv2", "InspectorConfig"],
            )
            InspectorConfig = inspector_mod.InspectorConfig
            PillInspectorSIFE = inspector_mod.PillInspectorDINOv2
        except ImportError as e:
            print(f"Error: Cannot import inspector for {model_name}: {e}")
            sys.exit(1)

    # --- Build config from the model's config module ---
    config = _build_config(model_name, InspectorConfig)

    print("=" * 60)
    print(f"  Folder Prediction — {info['desc']}")
    print("=" * 60)
    print(f"Model   : {model_name}")
    print(f"Input   : {input_dir}")
    print(f"Output  : {output_dir}")
    print("-" * 60)

    # Collect images
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Init inspector ---
    print("Loading model...")
    inspector = PillInspectorSIFE(config)

    # --- Process each image ---
    total_pills = 0
    total_anomalies = 0
    total_normal = 0

    for idx, img_path in enumerate(images):
        print(f"\n[{idx+1}/{len(images)}] {img_path.name}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Warning: Could not load {img_path}")
            continue

        # Reset state for each image (no cross-image voting)
        inspector.reset()

        # Run classification (same pipeline as camera)
        preview = inspector.classify_anomaly(frame)
        results = inspector._last_results

        n_pills = len(results)
        n_anomalies = 0

        for r in results:
            tid = r.get("id", -1)
            status = r.get("status", "UNKNOWN")
            scores = r.get("class_scores", {})
            normal_from = r.get("normal_from", [])

            if status == "ANOMALY":
                n_anomalies += 1

            # Print per-pill result
            score_strs = []
            for cls, score in scores.items():
                thr = inspector._thresholds.get(cls, 0)
                tag = "NORMAL" if score <= thr else "ANOMALY"
                score_strs.append(f"    {cls}: {score:.4f} {'<=' if score <= thr else '>'} {thr:.4f} -> {tag}")

            mark = "O" if status == "NORMAL" else "X"
            print(f"  [ID:{tid}] {mark} {status}")
            for s in score_strs:
                print(s)
            if normal_from:
                print(f"    Normal from: {', '.join(normal_from)}")

        total_pills += n_pills
        total_anomalies += n_anomalies
        total_normal += (n_pills - n_anomalies)

        # Save annotated image
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), preview)
        print(f"  Saved: {out_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  Results Summary")
    print(f"  Model:             {info['desc']}")
    print(f"  Images processed:  {len(images)}")
    print(f"  Total pills:       {total_pills}")
    print(f"  Total anomalies:   {total_anomalies}")
    print(f"  Total normal:      {total_normal}")
    print(f"  Output directory:  {output_dir}")
    print("=" * 60)


def _build_config(model_name: str, InspectorConfig):
    """
    Build InspectorConfig from the model's config module.

    Auto-detect feature settings from saved model metadata so extraction
    at inference matches exactly what was used during training.
    """
    import inspect
    import torch as _torch

    from config.base import (
        SEGMENTATION_MODEL_PATH,
        DETECTION_MODEL_PATH,
        COMPARE_CLASSES,
    )

    # Try to import model-specific config
    info = PATCHCORE_MODELS[model_name]
    config_mod_name = info["config_mod"]
    try:
        cfg = __import__(config_mod_name, fromlist=["*"])
    except ImportError:
        cfg = __import__("config.sife", fromlist=["*"])

    def g(attr, default=None):
        return getattr(cfg, attr, default)

    model_dir = Path(g("MODEL_OUTPUT_DIR", "./model/patchcore_sife"))

    # --- Auto-detect feature config from saved model metadata ---
    # Find ANY .pth in model_dir and read its meta to ensure feature
    # extraction matches what was used during training.
    saved_meta = {}
    pth_files = sorted(model_dir.rglob("*.pth"))  # sorted → deterministic across runs
    if pth_files:
        try:
            data = _torch.load(str(pth_files[0]), map_location="cpu", weights_only=False)
            saved_meta = data.get("meta", {})
            if saved_meta:
                print(f"[Config] Auto-detected feature config from {pth_files[0].name}")
        except Exception:
            pass

    def m(key, config_attr, default):
        """Prefer saved model meta > config file > default."""
        if key in saved_meta:
            return saved_meta[key]
        return getattr(cfg, config_attr, default)

    kwargs = dict(
        compare_classes=list(COMPARE_CLASSES),
        model_dir=model_dir,
        yolo_model_path=str(SEGMENTATION_MODEL_PATH),
        yolo_det_model_path=str(DETECTION_MODEL_PATH) if DETECTION_MODEL_PATH != SEGMENTATION_MODEL_PATH else None,
        model_size=m("model_size", "IMG_SIZE", 256),
        grid_size=m("grid_size", "GRID_SIZE", 16),
        k_nearest=m("k_nearest", "K_NEAREST", 3),
        use_sife=m("use_sife", "USE_SIFE", False),
        sife_dim=m("sife_dim", "SIFE_DIM", 32),
        sife_encoding_type=m("sife_encoding_type", "SIFE_ENCODING_TYPE", "sinusoidal"),
        sife_weight=m("sife_weight", "SIFE_WEIGHT", 1.0),
        use_center_distance=m("use_center_distance", "USE_CENTER_DISTANCE", False),
        use_local_gradient=m("use_local_gradient", "USE_LOCAL_GRADIENT", False),
        cnn_weight=g("CNN_WEIGHT", 1.0),
        use_color_features=m("use_color_features", "USE_COLOR_FEATURES", False),
        use_hsv=m("use_hsv", "USE_HSV", False),
        color_weight=m("color_weight", "COLOR_WEIGHT", 0.3),
        use_multi_scale=g("USE_MULTI_SCALE", False),
        multi_scale_grids=g("MULTI_SCALE_GRIDS", [16, 32, 48]),
        use_edge_enhancement=g("USE_EDGE_ENHANCEMENT", False),
        edge_weight=g("EDGE_WEIGHT", 1.5),
        finetuned_backbone_path=g("FINETUNED_BACKBONE_PATH", None),
        # Prefer score_method saved in model meta (matches calibration exactly)
        # fall back to config file, then to the universal default
        score_method=saved_meta.get("score_method") or g("SCORE_METHOD", "top10_mean"),
        threshold_multiplier=g("THRESHOLD_MULTIPLIER", 1.5),
    )

    # Features that may NOT exist in model meta → use False if not in meta
    # (prevents dim mismatch when config has new features the model wasn't trained with)
    use_lap = saved_meta.get("use_laplacian_variance", False) if saved_meta else g("USE_LAPLACIAN_VARIANCE", False)
    lap_w   = saved_meta.get("laplacian_weight", 1.0)        if saved_meta else g("LAPLACIAN_WEIGHT", 1.0)

    optional = {
        "use_laplacian_variance": use_lap,
        "laplacian_weight": lap_w,
        "score_weight_max": g("SCORE_WEIGHT_MAX", 0.3),
        "score_weight_top_k": g("SCORE_WEIGHT_TOP_K", 0.5),
        "score_weight_percentile": g("SCORE_WEIGHT_PERCENTILE", 0.2),
        "top_k_percent": g("TOP_K_PERCENT", 0.05),
    }

    # Only add optional fields if InspectorConfig accepts them
    sig = inspect.signature(InspectorConfig)
    for key, val in optional.items():
        if key in sig.parameters:
            kwargs[key] = val

    # Filter out any kwargs that this InspectorConfig doesn't accept
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return InspectorConfig(**kwargs)


# =============================================================================
#  CLI
# =============================================================================

def list_models():
    print("Available models (folder prediction):")
    print("-" * 55)
    for name, desc in ALL_MODELS.items():
        print(f"  {name:<20s}  {desc}")
    print("-" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Folder-based Prediction for MediScan-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(ALL_MODELS.keys()),
        help="Model to use for prediction",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input folder of images (default: data_yolo/test)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output folder for results (default: result_<model>)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list or args.model is None:
        list_models()
        if args.model is None:
            print("\nUsage: python run_predict.py --model=<model_name> [--input=<folder>] [--output=<folder>]")
        return

    input_dir = Path(args.input) if args.input else ROOT / "data_yolo" / "test"
    output_dir = Path(args.output) if args.output else ROOT / f"result_{args.model}"

    if args.model == "fcdd":
        run_fcdd(
            input_dir=str(input_dir) if args.input else None,
            output_dir=str(output_dir) if args.output else None,
        )
    else:
        run_patchcore_predict(args.model, input_dir, output_dir)


if __name__ == "__main__":
    main()
    #python run_predict.py --model=mobile_sife_cuda --input=data_yolo/test --output=result_sife
