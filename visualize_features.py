import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import torch
from sklearn.decomposition import PCA
import umap
from sklearn.metrics.pairwise import cosine_similarity
from modules.feature_extractor import ResNet50FeatureExtractor
from matplotlib.lines import Line2D
from core.utils import list_images_recursive

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # =============================
    # 📂 PATH
    # =============================
    base_good_dir = Path("data_train_defection")
    bad_base_dir = Path("data_bad")

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

    if bad_base_dir.exists():
        for class_dir in bad_base_dir.iterdir():
            if not class_dir.is_dir():
                continue
            bad_imgs = list_images_recursive(class_dir)
            if bad_imgs:
                bad_classes[class_dir.name] = bad_imgs

    # =============================
    # 🧠 BACKBONES
    # =============================
    BACKBONE_CONFIG = {
        "FineTuned_after": 'weights/backbone/resnet_backbone_20260307_172030.pth',
        "FineTuned_before": "weights/backbone/resnet_last.pth",
    }

    extractors = {}

    for name, path in BACKBONE_CONFIG.items():

        extractor = ResNet50FeatureExtractor(
            img_size=256,
            grid_size=16,
            device=device,
        )

        if path:
            checkpoint = torch.load(path, map_location="cpu")

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
        for cls, bad_paths in bad_classes.items():

            for img_path in bad_paths:

                img = Image.open(img_path).convert("RGB")

                for name, extractor in extractors.items():
                    patches = extractor.extract(img)
                    feat = patches.mean(axis=0)
                    image_features[name].append(feat)

                pill_classes.append(cls)
                pill_conditions.append("bad")

    X = {k: np.array(v) for k, v in image_features.items()}

    # =============================
    # 📉 PCA → UMAP
    # =============================
    X_2d = {}

    for name in X:
        X_pca = PCA(n_components=50).fit_transform(X[name])
        X_2d[name] = umap.UMAP(metric="cosine").fit_transform(X_pca)

    # =============================
    # 🎨 COLOR PER CLASS
    # =============================
    unique_classes = sorted(set(pill_classes))
    palette = sns.color_palette("tab20", len(unique_classes))
    class_color_map = dict(zip(unique_classes, palette))

    classes_with_good = {c for c, cond in zip(pill_classes, pill_conditions) if cond == "good"}
    classes_with_bad = {c for c, cond in zip(pill_classes, pill_conditions) if cond == "bad"}

    # =============================
    # 📊 PLOT (BAD ON TOP)
    # =============================
    fig, axes = plt.subplots(1, len(X_2d), figsize=(10 * len(X_2d), 8))

    if len(X_2d) == 1:
        axes = [axes]

    # เก็บ scatter collections สำหรับ toggle
    scatter_collections = {name: {'good': {}, 'bad': {}} for name in X_2d}

    for ax_idx, (name, data) in enumerate(X_2d.items()):
        ax = axes[ax_idx]
        
        # GOOD
        for cls in unique_classes:
            mask_good = [
                (c == cls) and (cond == "good")
                for c, cond in zip(pill_classes, pill_conditions)
            ]
            
            if any(mask_good):
                sc = ax.scatter(
                    data[mask_good, 0],
                    data[mask_good, 1],
                    marker='o',
                    s=80,
                    edgecolors='black',
                    facecolors=class_color_map[cls],
                    alpha=0.85,
                    zorder=1,
                    label=None,
                    picker=True  # เปิด picker
                )
                scatter_collections[name]['good'][cls] = sc
        
        # BAD
        for cls in unique_classes:
            mask_bad = [
                (c == cls) and (cond == "bad")
                for c, cond in zip(pill_classes, pill_conditions)
            ]
            
            if any(mask_bad):
                sc = ax.scatter(
                    data[mask_bad, 0],
                    data[mask_bad, 1],
                    marker='X',
                    s=120,
                    edgecolors='black',
                    linewidths=1.5,
                    facecolors=class_color_map[cls],
                    alpha=0.95,
                    zorder=2,
                    label=None,
                    picker=True  # เปิด picker
                )
                scatter_collections[name]['bad'][cls] = sc
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # =============================
    # 🏷️ INTERACTIVE LEGEND
    # =============================
    legend_elements = []
    legend_data = []  # เก็บข้อมูลสำหรับ toggle

    for cls in unique_classes:
        color = class_color_map[cls]

        if cls in classes_with_good:
            line_good = Line2D([0], [0],
                   marker='o',
                   color='w',
                   label=f"{cls} (good)",
                   markerfacecolor=color,
                   markeredgecolor='black',
                   markersize=9,
                   alpha=0.85)
            legend_elements.append(line_good)
            legend_data.append({'class': cls, 'condition': 'good', 'artist': line_good})

        if cls in classes_with_bad:
            line_bad = Line2D([0], [0],
                   marker='X',
                   color='w',
                   label=f"{cls} (bad)",
                   markerfacecolor=color,
                   markeredgecolor='black',
                   markersize=9,
                   alpha=0.95)
            legend_elements.append(line_bad)
            legend_data.append({'class': cls, 'condition': 'bad', 'artist': line_bad})

    plt.subplots_adjust(right=0.70) 
    
    if legend_elements:
        leg = fig.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(0.72, 0.5),
            fontsize=8,
            frameon=True,
            title="Click to toggle visibility",
            ncol=2,
            framealpha=1.0
        )
        
        # ทำให้ legend items pickable
        for line in leg.get_lines():
            line.set_picker(5)
        for path in leg.get_patches():
            path.set_picker(5)

    # =============================
    # 🖱️ TOGGLE FUNCTION
    # =============================
    def on_legend_pick(event):
        """Toggle visibility when clicking on legend"""
        artist = event.artist
        
        # หาข้อมูลที่ตรงกัน
        for item in legend_data:
            if item['artist'] == artist:
                cls = item['class']
                condition = item['condition']
                
                # Toggle visibility ในทุก axes
                for name in X_2d:
                    if condition in scatter_collections[name] and cls in scatter_collections[name][condition]:
                        sc = scatter_collections[name][condition][cls]
                        visible = sc.get_visible()
                        sc.set_visible(not visible)
                        
                        # ปรับ alpha ของ legend item
                        current_alpha = item['artist'].get_alpha() or 1.0
                        if visible:
                            item['artist'].set_alpha(0.3)
                        else:
                            item['artist'].set_alpha(1.0)
                
                fig.canvas.draw_idle()
                break

    fig.canvas.mpl_connect('pick_event', on_legend_pick)
    
    # เพิ่มคำอธิบาย
    print("\n💡 Interactive Mode:")
    print("   - Click on legend items to toggle class visibility")
    print("   - Close the window to exit")
    
    plt.show()


if __name__ == "__main__":
    main()