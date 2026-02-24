#!/usr/bin/env python3
"""
Diagnostic: Show anomaly score distribution for good vs bad images
==================================================================

ใช้ตรวจสอบว่า model สกัด feature ได้ดีพอที่จะแยกยาดี/เสียหรือไม่
และ threshold ที่บันทึกไว้เหมาะสมหรือเปล่า

Usage:
    python run_diagnose_scores.py ^
        --model_dir model/patchcore_sife/yellow_cap ^
        --good_dir  data/yellow_cap/yellow_cap/train/good ^
        --bad_dir   data/yellow_cap/yellow_cap/test/bad
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config.sife import (
    IMG_SIZE, GRID_SIZE, K_NEAREST, FINETUNED_BACKBONE_PATH,
    USE_SIFE, SIFE_DIM, SIFE_ENCODING_TYPE, SIFE_WEIGHT,
    USE_CENTER_DISTANCE, USE_LOCAL_GRADIENT,
    USE_COLOR_FEATURES, USE_HSV, COLOR_WEIGHT,
    USE_MULTI_SCALE, MULTI_SCALE_GRIDS, USE_EDGE_ENHANCEMENT, EDGE_WEIGHT,
    SCORE_WEIGHT_MAX, SCORE_WEIGHT_TOP_K, SCORE_WEIGHT_PERCENTILE, TOP_K_PERCENT,
)
from mobile_sife_cuda.core_shared.patchcore_sife import PatchCoreSIFE
from mobile_sife_cuda.core_train.trainer import PatchCoreSIFETrainer


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_patchcore() -> PatchCoreSIFE:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PatchCoreSIFE(
        model_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        k_nearest=K_NEAREST,
        device=device,
        use_sife=USE_SIFE,
        sife_dim=SIFE_DIM,
        sife_encoding_type=SIFE_ENCODING_TYPE,
        sife_weight=SIFE_WEIGHT,
        use_center_distance=USE_CENTER_DISTANCE,
        use_local_gradient=USE_LOCAL_GRADIENT,
        use_color_features=USE_COLOR_FEATURES,
        use_hsv=USE_HSV,
        color_weight=COLOR_WEIGHT,
        use_multi_scale=USE_MULTI_SCALE,
        multi_scale_grids=MULTI_SCALE_GRIDS,
        use_edge_enhancement=USE_EDGE_ENHANCEMENT,
        edge_weight=EDGE_WEIGHT,
        score_weight_max=SCORE_WEIGHT_MAX,
        score_weight_top_k=SCORE_WEIGHT_TOP_K,
        score_weight_percentile=SCORE_WEIGHT_PERCENTILE,
        top_k_percent=TOP_K_PERCENT,
        finetuned_backbone_path=FINETUNED_BACKBONE_PATH,
    )


def score_dir(
    patchcore: PatchCoreSIFE,
    trainer: PatchCoreSIFETrainer,
    index,
    image_dir: Path,
) -> list[tuple[str, float, float]]:
    """Return list of (filename, max_score, detailed_score)."""
    paths = trainer.iter_images(image_dir, image_exts=IMAGE_EXTS)
    results = []
    for p in paths:
        from PIL import Image
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        feats = patchcore.extract_features(img)
        max_s = patchcore.get_max_anomaly_score(feats, index)
        det_s = patchcore.get_anomaly_score_detailed(feats, index)["score"]
        results.append((p.name, max_s, det_s))
    return results


def print_stats(label: str, scores: list[float]) -> None:
    if not scores:
        print(f"  {label}: (no images)")
        return
    a = np.array(scores)
    print(
        f"  {label:10s}  n={len(a):3d}  "
        f"min={a.min():.4f}  mean={a.mean():.4f}  "
        f"p90={np.percentile(a,90):.4f}  p95={np.percentile(a,95):.4f}  "
        f"max={a.max():.4f}"
    )


def suggest_threshold(good_scores: list[float], bad_scores: list[float]) -> None:
    if not good_scores:
        return
    g = np.array(good_scores)

    # Simple percentile + sigma threshold
    p95  = float(np.percentile(g, 95))
    sig  = float(g.mean() + 2.5 * g.std())
    auto = max(p95, sig)
    print(f"\n  === Suggested threshold (detailed_score) ===")
    print(f"  good p95 + 2.5σ  → {auto:.4f}")

    if bad_scores:
        b = np.array(bad_scores)
        overlap = (g > b.min()).mean()
        sep     = b.mean() - g.mean()
        print(f"  bad  mean        = {b.mean():.4f}")
        print(f"  separation       = {sep:+.4f}  (positive = bad scores higher)")
        print(f"  good above bad_min = {overlap*100:.1f}% (lower = better)")

        # F1-optimal search
        all_scores = np.concatenate([g, b])
        best_thr, best_f1 = 0.0, -1.0
        for thr in np.linspace(all_scores.min(), all_scores.max(), 500):
            tp = (b > thr).sum();  fp = (g > thr).sum();  fn = len(b) - tp
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        print(f"  F1-optimal thr   = {best_thr:.4f}  (F1={best_f1:.4f})")

        if sep < 0.05:
            print("\n  ⚠️  WARNING: Scores barely separated. Possible causes:")
            print("     - Too few training images (need 10+ good images per class)")
            print("     - Training images contain anomalies")
            print("     - Backbone features not discriminative for this domain")
            print("     → Try: re-train backbone, add more good images, or lower GRID_SIZE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True,
                        help="Folder containing .pth files, e.g. model/patchcore_sife/yellow_cap")
    parser.add_argument("--good_dir",  type=Path, default=None,
                        help="Directory with GOOD/normal images to score")
    parser.add_argument("--bad_dir",   type=Path, default=None,
                        help="Directory with BAD/anomalous images to score")
    parser.add_argument("--subclass",  type=str, default=None,
                        help="Specific .pth name to load (without extension). "
                             "Default: first .pth in model_dir")
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────
    pth_files = sorted(args.model_dir.glob("*.pth"))
    if not pth_files:
        print(f"[ERROR] No .pth files found in {args.model_dir}")
        sys.exit(1)

    if args.subclass:
        pth_path = args.model_dir / f"{args.subclass}.pth"
    else:
        pth_path = pth_files[0]
        if len(pth_files) > 1:
            print(f"[INFO] Multiple .pth found, using: {pth_path.name}")
            print(f"       Use --subclass NAME to pick a specific one")

    print(f"\nLoading model: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")
    bank     = ckpt["memory_bank"].numpy().astype("float32")
    saved_thr = float(ckpt.get("threshold", 0.0))
    meta     = ckpt.get("meta", {})

    print(f"  Memory bank  : {bank.shape[0]:,} patches, dim={bank.shape[1]}")
    print(f"  Saved thr    : {saved_thr:.4f}")
    print(f"  Meta         : {meta}")

    # ── Build PatchCore + index ──────────────────────────────────
    print("\nInitializing PatchCoreSIFE...")
    patchcore = load_patchcore()
    trainer   = PatchCoreSIFETrainer(patchcore)

    import faiss
    faiss.normalize_L2(bank)
    index = faiss.IndexFlatIP(bank.shape[1])
    index.add(bank)

    # ── Score images ─────────────────────────────────────────────
    good_det, bad_det = [], []
    good_max, bad_max = [], []

    if args.good_dir and args.good_dir.exists():
        print(f"\nScoring GOOD images from: {args.good_dir}")
        results = score_dir(patchcore, trainer, index, args.good_dir)
        for name, ms, ds in results:
            good_max.append(ms)
            good_det.append(ds)
            status = "✓ NORMAL" if ds <= saved_thr else "✗ ANOMALY (FP)"
            print(f"  {name:<45s} max={ms:.4f}  detailed={ds:.4f}  {status}")
    else:
        print("\n[INFO] No --good_dir provided, skipping good image scoring")

    if args.bad_dir and args.bad_dir.exists():
        print(f"\nScoring BAD images from: {args.bad_dir}")
        results = score_dir(patchcore, trainer, index, args.bad_dir)
        for name, ms, ds in results:
            bad_max.append(ms)
            bad_det.append(ds)
            status = "✗ ANOMALY" if ds > saved_thr else "✓ NORMAL (FN — missed!)"
            print(f"  {name:<45s} max={ms:.4f}  detailed={ds:.4f}  {status}")
    else:
        print("\n[INFO] No --bad_dir provided, skipping bad image scoring")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Score summary  (saved threshold = {saved_thr:.4f})")
    print(f"{'='*65}")
    print("  [max_score — calibration method BEFORE fix]")
    print_stats("GOOD", good_max)
    print_stats("BAD ", bad_max)
    print("  [detailed_score — method used by inspector]")
    print_stats("GOOD", good_det)
    print_stats("BAD ", bad_det)

    suggest_threshold(good_det, bad_det)
    print()


if __name__ == "__main__":
    main()
