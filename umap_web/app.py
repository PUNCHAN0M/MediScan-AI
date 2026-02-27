import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import seaborn as sns
from pathlib import Path
from PIL import Image
import torch
from sklearn.decomposition import PCA
import umap
import json
from datetime import datetime
from Core_ResnetPatchCore.patchcore.feature_extractor import ResNet50FeatureExtractor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =============================
    # 📂 PATH
    # =============================
    base_good_dir = Path("data_train_defection")
    bad_base_dir = Path("data_yolo/each_bad")

    # =============================
    # 🔍 SCAN GOOD
    # =============================
    good_classes = {}
    for main_class in base_good_dir.iterdir():
        if not main_class.is_dir():
            continue
        for sub_class in main_class.iterdir():
            good_folder = sub_class / "train" / "good"
            if good_folder.exists():
                display_name = f"{main_class.name}_{sub_class.name}"
                good_classes[display_name] = good_folder

    # =============================
    # 🔍 SCAN BAD
    # =============================
    bad_classes = {}
    for color_folder in bad_base_dir.iterdir():
        if not color_folder.is_dir():
            continue
        for sub_folder in color_folder.iterdir():
            display_name = f"{color_folder.name}_{sub_folder.name}"
            bad_classes[display_name] = sub_folder

    print(f"✅ Found {len(good_classes)} good classes, {len(bad_classes)} bad classes")

    # =============================
    # 🧠 BACKBONES
    # =============================
    BACKBONE_CONFIG = {
        "FineTuned_before": 'model/backbone/resnet_backbone_20260226_234833.pth',
        "FineTuned_after": "model/backbone/resnet_backbone_20260227_152310.pth",
    }

    extractors = {}
    for name, path in BACKBONE_CONFIG.items():
        extractor = ResNet50FeatureExtractor(
            img_size=256,
            grid_size=16,
            device=device,
        )

        if path:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            for key in ("state_dict", "model", "full_state_dict", "features_state_dict"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
            checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("fc.")}
            extractor.backbone.load_state_dict(checkpoint, strict=False)

        extractor.backbone.to(device).eval()
        extractors[name] = extractor

    # =============================
    # 🔥 EXTRACT FEATURE
    # =============================
    image_features = {name: [] for name in extractors}
    pill_classes = []
    pill_conditions = []

    print("🔥 Extracting features...")
    with torch.inference_mode():
        # GOOD
        for cls, folder in good_classes.items():
            for img_path in folder.glob("*.*"):
                img = Image.open(img_path).convert("RGB")
                for name, extractor in extractors.items():
                    patches = extractor.extract(img)
                    feat = patches.mean(axis=0)
                    image_features[name].append(feat)
                pill_classes.append(cls)
                pill_conditions.append("good")

        # BAD
        for cls, folder in bad_classes.items():
            for img_path in folder.glob("*.*"):
                img = Image.open(img_path).convert("RGB")
                for name, extractor in extractors.items():
                    patches = extractor.extract(img)
                    feat = patches.mean(axis=0)
                    image_features[name].append(feat)
                pill_classes.append(cls)
                pill_conditions.append("bad")

    X = {k: np.array(v) for k, v in image_features.items()}
    print(f"✅ Extracted {len(pill_classes)} samples")

    # =============================
    # 📉 PCA → UMAP
    # =============================
    X_2d = {}
    for name in X:
        print(f"  Processing {name}...")
        X_pca = PCA(n_components=50).fit_transform(X[name])
        X_2d[name] = umap.UMAP(metric="cosine", random_state=42).fit_transform(X_pca)

    # =============================
    # 🎨 COLOR PER CLASS
    # =============================
    unique_classes = sorted(set(pill_classes))
    palette = sns.color_palette("tab20", len(unique_classes))
    class_color_map = dict(zip(unique_classes, palette))

    classes_with_good = {c for c, cond in zip(pill_classes, pill_conditions) if cond == "good"}
    classes_with_bad = {c for c, cond in zip(pill_classes, pill_conditions) if cond == "bad"}

    # =============================
    # 💾 SAVE DATA
    # =============================
    save_dir = Path("umap_vectors")
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # แปลง colors เป็น list
    class_colors_list = {}
    for cls, color in class_color_map.items():
        class_colors_list[cls] = [float(c) for c in color]

    save_data = {
        'X_2d': {name: X_2d[name].tolist() for name in X_2d},
        'pill_classes': pill_classes,
        'pill_conditions': pill_conditions,
        'unique_classes': unique_classes,
        'class_colors': class_colors_list,
        'classes_with_good': list(classes_with_good),
        'classes_with_bad': list(classes_with_bad),
        'metadata': {
            'timestamp': timestamp,
            'n_samples': len(pill_classes),
            'n_classes': len(unique_classes),
            'n_good_classes': len(classes_with_good),
            'n_bad_classes': len(classes_with_bad),
            'backbones': list(extractors.keys())
        }
    }

    # Save as JSON
    json_path = save_dir / f"umap_data_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    # Save as latest
    latest_path = save_dir / "umap_data_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n💾 Saved to:")
    print(f"   - {json_path}")
    print(f"   - {latest_path}")
    print(f"\n📊 Stats:")
    print(f"   - Total samples: {len(pill_classes)}")
    print(f"   - Unique classes: {len(unique_classes)}")
    print(f"   - Good classes: {len(classes_with_good)}")
    print(f"   - Bad classes: {len(classes_with_bad)}")

if __name__ == "__main__":
    main()