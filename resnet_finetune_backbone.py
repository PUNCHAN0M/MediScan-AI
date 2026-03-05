#!/usr/bin/env python3
"""
Fine-tune ResNet50 Backbone on Custom Dataset
==============================================
ต่อยอด IMAGENET1K_V1 weights ด้วย dataset ของเราก่อนใช้กับ PatchCore

Dataset format expected:
    data_backbone_augment/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── class2/
        ├── image1.jpg
        └── image2.jpg

Output:
    model/backbone/resnet_backbone_{timestamp}.pth

Usage:
    python run_finetune_backbone.py
    python run_finetune_backbone.py --epochs=30
    python run_finetune_backbone.py --data_dir=my_data/

Flow:
    Stage 1 (Warmup)  : Freeze backbone → Train classifier only       (5 epochs)
    Stage 2 (Finetune): Unfreeze last 3 residual blocks + train       (remaining epochs)
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image

# =============================================================================
#                              DEFAULTS
# =============================================================================
DEFAULT_DATA_DIR   = Path("data_train_backbone/")
DEFAULT_OUTPUT_DIR = Path("model/backbone/")
DEFAULT_EPOCHS     = 20
DEFAULT_WARMUP     = 5
DEFAULT_LR         = 1e-4
DEFAULT_LR_HEAD    = 1e-3
DEFAULT_BATCH      = 32
DEFAULT_VAL_SPLIT  = 0.25
DEFAULT_IMG_SIZE   = 256
DEFAULT_WORKERS    = 4
DEFAULT_UNFREEZE   = 3
DEFAULT_PATIENCE   = 7
DEFAULT_MIN_DELTA  = 0.001


# =============================================================================
#                         CUSTOM DATASET (files without extensions)
# =============================================================================
class CustomImageFolder(Dataset):
    """Custom ImageFolder that accepts files without extensions"""
    
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if not self.classes:
            raise RuntimeError(f"No class directories found in {root}")
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        print(f"\n  Scanning for image files in {root}...")
        
        for class_name in self.classes:
            class_dir = self.root / class_name
            class_idx = self.class_to_idx[class_name]
            file_count = 0
            
            for file_path in class_dir.iterdir():
                if file_path.is_file():
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                        self.samples.append((str(file_path), class_idx))
                        self.targets.append(class_idx)
                        file_count += 1
                    except Exception as e:
                        print(f"  ⚠ Skipping non-image file: {file_path.name} - {e}")
            
            print(f"    Class {class_name}: {file_count} images")
        
        if not self.samples:
            raise RuntimeError(f"Found 0 valid image files in {root}")
        
        print(f"  Total images found: {len(self.samples)}")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise
        if self.transform:
            img = self.transform(img)
        return img, target


# =============================================================================
#                         TRANSFORMS
# =============================================================================
def get_transforms(img_size: int):
    """Return train / val transforms (Optimized for pre-augmented dataset)."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomRotation(10),
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
#                         DATASET HELPERS
# =============================================================================
def build_datasets(data_dir: Path, img_size: int, val_split: float, seed: int = 42):
    """Load CustomImageFolder dataset and split into train/val."""
    train_tf, val_tf = get_transforms(img_size)
    
    print(f"\n  Loading dataset from: {data_dir}")
    raw_dataset = CustomImageFolder(str(data_dir), transform=train_tf)
    
    from collections import Counter
    counts = Counter(raw_dataset.targets)
    idx_to_class = {v: k for k, v in raw_dataset.class_to_idx.items()}
    
    print(f"\n  Class distribution:")
    for idx, count in sorted(counts.items()):
        print(f"    [{idx:3d}] {idx_to_class[idx]:<60s} {count:4d} images")
    
    num_classes = len(raw_dataset.classes)
    n_total = len(raw_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    torch.manual_seed(seed)
    train_ds, val_ds = random_split(raw_dataset, [n_train, n_val])
    
    # Apply val transform to validation subset
    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset, self.transform = subset, transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
    
    val_ds_final = TransformSubset(val_ds, val_tf)
    print(f"\n  Train samples: {n_train}   Val samples: {n_val}")
    return train_ds, val_ds_final, num_classes


# =============================================================================
#                         RESNET50 MODEL SETUP (ONLY)
# =============================================================================
def build_resnet50(num_classes: int, device: torch.device) -> nn.Module:
    """Build ResNet50 model with custom classifier head."""
    model = models.resnet50(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def set_trainable(model: nn.Module, stage: str, n_unfreeze_blocks: int = 3) -> None:
    """Control which parts of ResNet50 are trainable."""
    for p in model.parameters():
        p.requires_grad = False
    
    if stage == "warmup":
        # Train classifier head only
        for p in model.fc.parameters():
            p.requires_grad = True
        print("  [Stage: WARMUP] Training classifier head only")
    
    elif stage == "finetune":
        # Unfreeze last N residual blocks + classifier
        for p in model.fc.parameters():
            p.requires_grad = True
        # layer4, layer3, layer2 (in that order)
        blocks = [model.layer4, model.layer3, model.layer2]
        for block in blocks[:n_unfreeze_blocks]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"  [Stage: FINETUNE] Unfreezing {n_unfreeze_blocks} backbone blocks + classifier")
    
    elif stage == "full":
        for p in model.parameters():
            p.requires_grad = True
        print("  [Stage: FULL] All parameters trainable")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")


# =============================================================================
#                         EARLY STOPPING
# =============================================================================
class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, verbose: bool = True):
        self.patience, self.min_delta, self.verbose = patience, min_delta, verbose
        self.counter, self.best_score, self.early_stop = 0, None, False
        self.val_acc_max = -float('inf')
    
    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None:
            self.best_score = self.val_acc_max = val_acc
            return False
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} - no improvement "
                      f"(best={self.best_score*100:.2f}%, current={val_acc*100:.2f}%)")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  ⚠ Early stopping triggered")
                return True
        else:
            self.best_score = self.val_acc_max = max(self.val_acc_max, val_acc)
            self.counter = 0
            if self.verbose:
                print(f"  ✓ EarlyStopping: improvement! best={self.best_score*100:.2f}%")
        return False
    
    def get_best_score(self) -> float:
        return self.val_acc_max


# =============================================================================
#                         TRAINING LOOP
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, use_amp, scaler, epoch):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct += (model(imgs).argmax(1) == labels).sum().item()
        n += bs
        
        if (batch_idx + 1) % 20 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}", end="\r")
    
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        n += bs
    
    return total_loss / n, correct / n


# =============================================================================
#                         SAVE BACKBONE
# =============================================================================
def save_backbone(model: nn.Module, output_dir: Path, meta: dict) -> Path:
    """Save fine-tuned ResNet50 backbone + metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"resnet_backbone_{timestamp}.pth"
    
    checkpoint = {
        "full_state_dict": model.state_dict(),
        # Save only backbone features (exclude fc layer for PatchCore usage)
        "features_state_dict": {k: v for k, v in model.state_dict().items() if not k.startswith('fc.')},
        "meta": meta,
        "model_name": "resnet50",
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n  {'='*60}")
    print(f"  💾 Backbone saved → {output_path}")
    print(f"  {'='*60}\n  Meta: {meta}\n  {'='*60}\n")
    return output_path


# =============================================================================
#                         UTILITIES
# =============================================================================
def add_image_extensions(data_dir: Path):
    """Add image extensions to files without extensions"""
    print(f"\n  Scanning for files without extensions in {data_dir}...")
    renamed_count = 0
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            for file_path in class_dir.iterdir():
                if file_path.is_file() and not file_path.suffix:
                    try:
                        with Image.open(file_path) as img:
                            ext = { 'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp' }.get(img.format, '.png')
                            new_path = file_path.with_suffix(ext)
                            file_path.rename(new_path)
                            print(f"    Renamed: {file_path.name} -> {new_path.name}")
                            renamed_count += 1
                    except Exception as e:
                        print(f"    ⚠ Could not process {file_path.name}: {e}")
    
    print(f"  ✅ Renamed {renamed_count} files")
    return renamed_count


# =============================================================================
#                         MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 backbone")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr_head", type=float, default=DEFAULT_LR_HEAD)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--unfreeze", type=int, default=DEFAULT_UNFREEZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--fix_extensions", action="store_true")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min_delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--no_early_stop", action="store_true")
    
    args = parser.parse_args()
    
    if args.fix_extensions:
        add_image_extensions(args.data_dir)
        print("\n  Continuing with training...\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print(f"\n{'='*60}\n  Fine-tune RESNET50 Backbone\n{'='*60}")
    print(f"  Device: {device} (AMP={use_amp})\n  Data: {args.data_dir}\n  Output: {args.output_dir}")
    print(f"  Epochs: {args.epochs} (warmup={args.warmup})")
    if not args.no_early_stop:
        print(f"  Early stopping: patience={args.patience}, min_delta={args.min_delta}")
    
    # Dataset
    train_ds, val_ds, num_classes = build_datasets(
        args.data_dir, args.img_size, args.val_split, args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type=="cuda"),
        persistent_workers=(args.workers>0), prefetch_factor=2 if args.workers>0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch*2, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type=="cuda"),
        persistent_workers=(args.workers>0), prefetch_factor=2 if args.workers>0 else None,
    )
    
    # Model
    model = build_resnet50(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc, best_state = 0.0, None
    early_stopping = EarlyStopping(args.patience, args.min_delta, verbose=True)
    stop_training = False
    
    # STAGE 1: Warmup
    print(f"\n{'─'*60}\n  STAGE 1: WARMUP ({args.warmup} epochs)\n{'─'*60}")
    set_trainable(model, "warmup")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.warmup, eta_min=1e-6)
    
    for epoch in range(1, args.warmup+1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler, epoch)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  Warmup [{epoch:3d}/{args.warmup}] train_loss={tr_loss:.4f} train_acc={tr_acc*100:.1f}% "
              f"val_loss={va_loss:.4f} val_acc={va_acc*100:.1f}% ({time.perf_counter()-t0:.1f}s)")
        if va_acc > best_val_acc:
            best_val_acc, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}
        if not args.no_early_stop and epoch > args.warmup//2 and early_stopping(va_acc):
            print(f"  ⚠ Early stopping during warmup at epoch {epoch}")
            stop_training = True
            break
    
    # STAGE 2: Finetune
    if not stop_training and (finetune_epochs := args.epochs - args.warmup) > 0:
        print(f"\n{'─'*60}\n  STAGE 2: FINETUNE ({finetune_epochs} epochs)\n{'─'*60}")
        set_trainable(model, "finetune", args.unfreeze)
        
        # Different LR for backbone vs classifier head
        head_params = list(model.fc.parameters())
        backbone_params = [p for p in model.parameters() if p.requires_grad and p not in head_params]
        
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 5},
        ], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-7)
        
        if not args.no_early_stop:
            early_stopping = EarlyStopping(args.patience, args.min_delta, verbose=True)
        
        for epoch in range(1, finetune_epochs+1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler, epoch)
            va_loss, va_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"  Fine-tune [{epoch:3d}/{finetune_epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc*100:.1f}% "
                  f"val_loss={va_loss:.4f} val_acc={va_acc*100:.1f}% ({time.perf_counter()-t0:.1f}s)")
            if va_acc > best_val_acc:
                best_val_acc, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ✓ New best val_acc: {best_val_acc*100:.2f}%")
            if not args.no_early_stop and early_stopping(va_acc):
                print(f"  ⚠ Early stopping at fine-tune epoch {epoch}")
                break
    
    # Save best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  Restored best checkpoint (val_acc={best_val_acc*100:.2f}%)")
        if not args.no_early_stop:
            best_val_acc = early_stopping.get_best_score()
    
    saved_path = save_backbone(model, args.output_dir, {
        "num_classes": num_classes,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "best_val_acc": round(best_val_acc, 4),
        "data_dir": str(args.data_dir),
        "unfreeze_blocks": args.unfreeze,
        "early_stopped": not args.no_early_stop,
        "patience": args.patience if not args.no_early_stop else None,
    })
    
    print(f"\n  ✅ Done! Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"    FINETUNED_BACKBONE_PATH = Path(\"{saved_path}\")")


if __name__ == "__main__":
    main()