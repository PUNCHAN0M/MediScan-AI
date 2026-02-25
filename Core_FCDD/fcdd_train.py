"""
FCDD Training Script
====================
Train FCDD model ด้วย good images only + pseudo-anomaly contrastive learning.

Loss function (HSC - Hypersphere Classification):
  - Normal pixels:  log(1 + exp(anomaly_map))   → push map to -∞
  - Anomaly pixels: log(1 + exp(-anomaly_map))  → push map to +∞

Usage:
  python -m FCDD.fcdd_train
  หรือ python run_train_fcdd.py
"""
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import fcdd as cfg
from FCDD.fcdd_model import FCDDNet
from FCDD.fcdd_dataset import FCDDDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hsc_loss(
    anomaly_map: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    Hypersphere Classification loss for FCDD.

    Args:
        anomaly_map: (B, 1, H, W) raw anomaly map output
        labels: (B,) 0=normal, 1=pseudo-anomaly
        masks: (B, 1, H, W) anomaly region mask (1=anomaly pixel)
    Returns:
        loss: scalar
    """
    B = anomaly_map.shape[0]
    loss = torch.tensor(0.0, device=anomaly_map.device)

    for i in range(B):
        a_map = anomaly_map[i]  # (1, H, W)

        if labels[i] == 0:
            # Normal: push all pixels to negative (low anomaly score)
            # L = log(1 + exp(a_map)).mean()
            loss_i = torch.log1p(torch.exp(a_map)).mean()
        else:
            # Pseudo-anomaly: use mask for spatial guidance
            mask = masks[i]  # (1, H, W)

            # Normal region (mask=0): push to negative
            normal_region = a_map * (1.0 - mask)
            normal_count = (1.0 - mask).sum().clamp(min=1)
            loss_normal = torch.log1p(torch.exp(normal_region)).sum() / normal_count

            # Anomaly region (mask=1): push to positive
            anomaly_region = a_map * mask
            anomaly_count = mask.sum().clamp(min=1)
            loss_anomaly = torch.log1p(torch.exp(-anomaly_region)).sum() / anomaly_count

            loss_i = loss_normal + loss_anomaly

        loss = loss + loss_i

    return loss / B


def train():
    """Main training function."""
    print("=" * 60)
    print("  FCDD Training - Pill Anomaly Detection")
    print("=" * 60)

    set_seed(cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Create output directory ----
    cfg.FCDD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model output: {cfg.FCDD_MODEL_DIR}")

    # ---- Dataset ----
    print(f"\nLoading training data from: {cfg.GOOD_DATA_DIR}")
    dataset = FCDDDataset(
        good_dir=str(cfg.GOOD_DATA_DIR),
        img_size=cfg.IMG_SIZE,
        pseudo_anomaly_ratio=cfg.PSEUDO_ANOMALY_RATIO,
        augment=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset size: {len(dataset)} (x3 augmented from {len(dataset)//3} images)")
    print(f"Batches per epoch: {len(dataloader)}")

    # ---- Model ----
    print(f"\nBuilding FCDD model...")
    print(f"Backbone: {cfg.BACKBONE_PATH}")
    model = FCDDNet(
        backbone_path=str(cfg.BACKBONE_PATH),
        img_size=cfg.IMG_SIZE,
        hook_indices=cfg.HOOK_INDICES,
        freeze_backbone=True,
    ).to(device)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ---- Optimizer ----
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )

    # ---- Training loop ----
    print(f"\nStarting training for {cfg.EPOCHS} epochs...")
    print("-" * 60)

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (images, labels, masks) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward
            anomaly_map = model(images)

            # Loss
            loss = hsc_loss(anomaly_map, labels, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        # Print progress
        print(
            f"Epoch [{epoch:3d}/{cfg.EPOCHS}]  "
            f"Loss: {avg_loss:.4f}  "
            f"LR: {lr:.2e}  "
            f"Time: {elapsed:.0f}s"
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "img_size": cfg.IMG_SIZE,
                "hook_indices": cfg.HOOK_INDICES,
                "anomaly_threshold": cfg.ANOMALY_THRESHOLD,
            }
            torch.save(save_dict, str(cfg.FCDD_MODEL_PATH))
            print(f"  -> Saved best model (loss={best_loss:.4f})")

    # ---- Calibrate threshold using bad data ----
    threshold = calibrate_threshold(model, device)

    # Update saved model with calibrated threshold
    save_dict = torch.load(str(cfg.FCDD_MODEL_PATH), map_location=device, weights_only=False)
    save_dict["anomaly_threshold"] = threshold
    torch.save(save_dict, str(cfg.FCDD_MODEL_PATH))

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Model saved to: {cfg.FCDD_MODEL_PATH}")
    print("=" * 60)


def calibrate_threshold(model: FCDDNet, device: str) -> float:
    """
    Calibrate anomaly threshold using good and bad images.
    Finds optimal threshold that maximizes F1 score.
    """
    import cv2
    from torchvision import transforms as T
    from PIL import Image

    print("\nCalibrating threshold...")

    transform = T.Compose([
        T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()
    scores_good = []
    scores_bad = []

    def compute_score(img_path: str) -> float:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            amap = model(tensor)
        # Global anomaly score = mean of positive part of heatmap
        score = torch.sigmoid(amap).mean().item()
        return score

    # Score good images
    good_dir = Path(cfg.GOOD_DATA_DIR)
    if good_dir.exists():
        for p in sorted(good_dir.iterdir()):
            if p.suffix.lower() in cfg.IMAGE_EXTS:
                s = compute_score(str(p))
                if s is not None:
                    scores_good.append(s)

    # Score bad images
    bad_dir = Path(cfg.BAD_DATA_DIR)
    if bad_dir.exists():
        for p in sorted(bad_dir.iterdir()):
            if p.suffix.lower() in cfg.IMAGE_EXTS:
                s = compute_score(str(p))
                if s is not None:
                    scores_bad.append(s)

    print(f"  Good scores: n={len(scores_good)}, mean={np.mean(scores_good):.4f}" if scores_good else "  No good images")
    print(f"  Bad scores:  n={len(scores_bad)}, mean={np.mean(scores_bad):.4f}" if scores_bad else "  No bad images")

    if not scores_good or not scores_bad:
        print(f"  Using default threshold: {cfg.ANOMALY_THRESHOLD}")
        return cfg.ANOMALY_THRESHOLD

    # Find optimal threshold via F1
    all_scores = scores_good + scores_bad
    all_labels = [0] * len(scores_good) + [1] * len(scores_bad)

    best_f1 = 0.0
    best_thr = cfg.ANOMALY_THRESHOLD

    for thr in np.linspace(min(all_scores), max(all_scores), 200):
        tp = sum(1 for s, l in zip(all_scores, all_labels) if s >= thr and l == 1)
        fp = sum(1 for s, l in zip(all_scores, all_labels) if s >= thr and l == 0)
        fn = sum(1 for s, l in zip(all_scores, all_labels) if s < thr and l == 1)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"  Optimal threshold: {best_thr:.4f} (F1={best_f1:.4f})")
    return float(best_thr)


if __name__ == "__main__":
    train()
