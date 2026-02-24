#!/usr/bin/env python3
"""
Fine-tune MobileNetV3-Large Backbone on Custom Dataset
=======================================================

ต่อยอด IMAGENET1K_V1 weights ด้วย dataset ของเราก่อนใช้กับ PatchCore SIFE

Dataset format expected:
    result/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── class2/
        ├── image1.jpg
        └── image2.jpg

Output:
    model/{model_name}_backbone.pth   ← โหลดไปใช้กับ PatchCoreSIFE ต่อ

Usage:
    python run_finetune_backbone.py --model=mobilenet
    python run_finetune_backbone.py --model=resnet
    python run_finetune_backbone.py --model=efficientnet
    python run_finetune_backbone.py --model=inception
    python run_finetune_backbone.py --model=dinov2

Output:
    backbone_path : model/{model_name}_backbone.pth

Flow:
    Stage 1 (Warmup)  : Freeze backbone → Train classifier only       (5 epochs)
    Stage 2 (Finetune): Unfreeze last 3 InvertedResidual blocks + train (remaining epochs)

ผลลัพธ์ที่ได้ backbone จะ "รู้จัก" domain ยาของเราดีขึ้น และ features
ที่สกัดได้จะ discriminative กว่าการใช้ raw IMAGENET1K_V1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from PIL import Image


# =============================================================================
#                              DEFAULTS
# =============================================================================

DEFAULT_DATA_DIR   = Path("augmentation_result_1")  # folder with subfolders for each class
DEFAULT_OUTPUT     = Path(f"model/backbone/{model}.pth")
DEFAULT_EPOCHS     = 1#50
DEFAULT_WARMUP     = 1#5          # epochs to train classifier-only first
DEFAULT_LR         = 1e-4       # learning rate for finetune stage
DEFAULT_LR_HEAD    = 1e-3       # learning rate for classifier head (warmup)
DEFAULT_BATCH      = 32
DEFAULT_VAL_SPLIT  = 0.25       # 15% of data used for validation
DEFAULT_IMG_SIZE   = 256
DEFAULT_WORKERS    = 4
DEFAULT_UNFREEZE   = 3          # number of feature blocks to unfreeze in stage 2


# =============================================================================
#                         MODULE-LEVEL DATASET HELPER
# =============================================================================
# Must be at module level (not inside a function) so Windows multiprocessing
# can pickle it for DataLoader workers.

class TransformSubset(Dataset):
    """
    Wraps a Subset and applies a different transform.
    Re-loads images from disk so val transforms don't see augmented tensors.
    """
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# =============================================================================
#                              TRANSFORMS
# =============================================================================

def get_transforms(img_size: int):
    """Return train / val transforms."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# =============================================================================
#                              DATASET HELPERS
# =============================================================================

def build_datasets(data_dir: Path, img_size: int, val_split: float, seed: int = 42):
    """
    Load ImageFolder dataset and split into train/val.

    Folders ending with ' - Copy' are merged into their original class
    by using the same label. ImageFolder assigns labels alphabetically,
    so we load with a custom target_transform to unify duplicates.
    """
    # Collect valid class dirs (skip empty folders)
    all_class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not all_class_dirs:
        raise ValueError(f"No subdirectories found in {data_dir}")

    # Each folder = its own class (including ' - Copy' folders)
    unique_classes = sorted([d.name for d in all_class_dirs])
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    print(f"\n  Dataset root : {data_dir}")
    print(f"  Total folders: {len(all_class_dirs)}")
    print(f"  Classes      : {num_classes}")

    # Load with full transform for display, but use custom loader
    train_tf, val_tf = get_transforms(img_size)

    # ImageFolder — already assigns one label per folder alphabetically
    raw_dataset = datasets.ImageFolder(str(data_dir), transform=train_tf)

    # Count per class
    from collections import Counter
    counts = Counter(raw_dataset.targets)
    for cls, idx in sorted(raw_dataset.class_to_idx.items(), key=lambda x: x[1]):
        print(f"    [{idx:3d}] {cls:<60s} {counts[idx]:4d} images")

    # Split train / val
    n_total = len(raw_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    torch.manual_seed(seed)
    train_ds, val_ds = random_split(raw_dataset, [n_train, n_val])

    # Val set needs different transform (module-level class so Windows can pickle it)
    val_ds_final = TransformSubset(val_ds, val_tf)

    print(f"\n  Train samples: {n_train}   Val samples: {n_val}")
    return train_ds, val_ds_final, num_classes


# =============================================================================
#                              MODEL SETUP
# =============================================================================

def build_model(num_classes: int, device: torch.device, model_name: str) -> nn.Module:
    """
    Build backbone model by name.
    Supported: mobilenet, resnet, dinov2, efficientnet, inception
    """
    if model_name == "mobilenet":
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "dinov2":
        try:
            from torchvision.models import dinov2_base
            model = dinov2_base(weights="IMAGENET1K_V1")
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        except Exception:
            raise ImportError("dinov2 not available in torchvision")
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "inception":
        model = models.inception_v3(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def _get_head_and_backbone_blocks(model: nn.Module):
    """
    Returns (head_module, list_of_backbone_block_modules) for any supported architecture.
    backbone_blocks are ordered first→last (deeper layers at the end).
    """
    if hasattr(model, "features"):
        # MobileNetV3, EfficientNet — backbone is model.features (Sequential)
        head = model.classifier if hasattr(model, "classifier") else model.head
        return head, list(model.features)
    elif hasattr(model, "layer4"):
        # ResNet — backbone blocks are layer1..layer4
        head   = model.fc
        blocks = [getattr(model, n) for n in ("layer1", "layer2", "layer3", "layer4")
                  if hasattr(model, n)]
        return head, blocks
    elif hasattr(model, "blocks"):
        # DINOv2 / ViT style
        head = model.head
        return head, list(model.blocks)
    else:
        raise AttributeError(
            f"Cannot determine backbone structure for {type(model).__name__}. "
            "Expected .features, .layer4, or .blocks attribute."
        )


def set_trainable(model: nn.Module, stage: str, n_unfreeze_blocks: int = 3) -> None:
    """
    Control which parts of the model are trainable.

    stage='warmup'  → only classifier head trainable
    stage='finetune'→ classifier + last n_unfreeze_blocks backbone blocks
    stage='full'    → everything trainable
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    head, backbone_blocks = _get_head_and_backbone_blocks(model)

    if stage == "warmup":
        for p in head.parameters():
            p.requires_grad = True
        print("  [Stage: WARMUP] Training classifier head only")

    elif stage == "finetune":
        for p in head.parameters():
            p.requires_grad = True
        to_unfreeze = backbone_blocks[-n_unfreeze_blocks:] if backbone_blocks else []
        for block in to_unfreeze:
            for p in block.parameters():
                p.requires_grad = True
        start_idx = len(backbone_blocks) - len(to_unfreeze)
        print(
            f"  [Stage: FINETUNE] Unfreezing {len(to_unfreeze)} backbone blocks "
            f"[{start_idx}–{len(backbone_blocks)-1}] + classifier"
        )

    elif stage == "full":
        for p in model.parameters():
            p.requires_grad = True
        print("  [Stage: FULL] All parameters trainable")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")


# =============================================================================
#                              TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler,
    epoch: int,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += bs

        if (batch_idx + 1) % 20 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}", end="\r")

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    n          = 0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += bs
    return total_loss / n, correct / n


# =============================================================================
#                              SAVE / LOAD
# =============================================================================

def save_backbone(model: nn.Module, output_path: Path, meta: dict) -> None:
    """
    Save fine-tuned backbone + metadata.
    Saves full state_dict always; also saves features_state_dict when available.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict = {
        "full_state_dict": model.state_dict(),
        "meta"           : meta,
    }
    # Save feature extractor portion separately when accessible
    if hasattr(model, "features"):
        checkpoint["features_state_dict"] = model.features.state_dict()
    elif hasattr(model, "layer4"):          # ResNet
        checkpoint["features_state_dict"] = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("fc.")
        }
    torch.save(checkpoint, output_path)
    print(f"\n  Backbone saved → {output_path}")
    print(f"  Metadata: {meta}")


# =============================================================================
#                              MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune backbone")
    parser.add_argument("--model", type=str, default="mobilenet",
                        choices=["mobilenet", "resnet", "dinov2", "efficientnet", "inception"],
                        help="Backbone model to finetune")
    parser.add_argument("--data_dir",  type=Path,  default=DEFAULT_DATA_DIR)
    parser.add_argument("--output",    type=Path,  default=None)
    parser.add_argument("--epochs",    type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup",    type=int,   default=DEFAULT_WARMUP,
                        help="Epochs for warmup stage (classifier only)")
    parser.add_argument("--lr",        type=float, default=DEFAULT_LR,
                        help="LR for fine-tune stage (backbone)")
    parser.add_argument("--lr_head",   type=float, default=DEFAULT_LR_HEAD,
                        help="LR for warmup stage (head only)")
    parser.add_argument("--batch",     type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--img_size",  type=int,   default=DEFAULT_IMG_SIZE)
    parser.add_argument("--workers",   type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--unfreeze",  type=int,   default=DEFAULT_UNFREEZE,
                        help="Number of feature blocks to unfreeze in stage 2")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--no_amp",    action="store_true", help="Disable AMP")
    args = parser.parse_args()

    # Set output path if not specified
    if args.output is None:
        args.output = Path(f"model/backbone/{args.model}_backbone.pth")

    # ── Device ──────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    # ── GPU optimisations ────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark   = True   # auto-tune conv kernels
        torch.backends.cudnn.deterministic = False  # allow non-deterministic (faster)
    print(f"\n{'='*60}")
    print(f"  Fine-tune {args.model} Backbone")
    print(f"{'='*60}")
    print(f"  Device   : {device}  (AMP={use_amp})")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Output   : {args.output}")
    print(f"  Epochs   : {args.epochs}  (warmup={args.warmup})")

    # ── Dataset ─────────────────────────────────────────────────────────────
    train_ds, val_ds, num_classes = build_datasets(
        args.data_dir, args.img_size, args.val_split, args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,   # 2× batch for eval
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model     = build_model(num_classes, device, args.model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_state   = None

    # ══════════════════════════════════════════════════════════════
    #  STAGE 1 — Warmup: train classifier head only
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  STAGE 1: WARMUP  ({args.warmup} epochs)")
    print(f"{'─'*60}")

    set_trainable(model, "warmup")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.warmup, eta_min=1e-6)

    for epoch in range(1, args.warmup + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp, scaler, epoch
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        print(
            f"  Warmup [{epoch:3d}/{args.warmup}]  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc*100:.1f}%  "
            f"({elapsed:.1f}s)"
        )
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # ══════════════════════════════════════════════════════════════
    #  STAGE 2 — Fine-tune: unfreeze last N blocks + classifier
    # ══════════════════════════════════════════════════════════════
    finetune_epochs = args.epochs - args.warmup
    if finetune_epochs > 0:
        print(f"\n{'─'*60}")
        print(f"  STAGE 2: FINETUNE  ({finetune_epochs} epochs, unfreeze={args.unfreeze} blocks)")
        print(f"{'─'*60}")

        set_trainable(model, "finetune", n_unfreeze_blocks=args.unfreeze)

        # Differential LR: backbone gets lower LR than head
        # Identify head param names dynamically
        head, _ = _get_head_and_backbone_blocks(model)
        head_param_ids = {id(p) for p in head.parameters()}
        backbone_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in head_param_ids
        ]
        head_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) in head_param_ids
        ]
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params,     "lr": args.lr * 5},
        ], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-7)

        for epoch in range(1, finetune_epochs + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, use_amp, scaler, epoch
            )
            va_loss, va_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.perf_counter() - t0
            print(
                f"  Fine-tune [{epoch:3d}/{finetune_epochs}]  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  "
                f"val_loss={va_loss:.4f}  val_acc={va_acc*100:.1f}%  "
                f"({elapsed:.1f}s)"
            )
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ✓ New best val_acc: {best_val_acc*100:.2f}%")

    # ── Restore best weights and save ────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best checkpoint  (val_acc={best_val_acc*100:.2f}%)")

    save_backbone(model, args.output, {
        "num_classes"    : num_classes,
        "img_size"       : args.img_size,
        "epochs"         : args.epochs,
        "best_val_acc"   : round(best_val_acc, 4),
        "data_dir"       : str(args.data_dir),
        "unfreeze_blocks": args.unfreeze,
    })

    print(f"\n  Done!  Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"  Load in PatchCoreSIFE by setting:\n")
    print(f"    FINETUNED_BACKBONE_PATH = Path(\"{args.output}\")")
    print(f"  in config/sife.py\n")


if __name__ == "__main__":
    main()
