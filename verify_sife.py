#!/usr/bin/env python3
"""Verify model on its own training test set."""
import sys
sys.path.insert(0, '.')
import torch, numpy as np, cv2
from pathlib import Path
from config.sife import (IMG_SIZE, GRID_SIZE, K_NEAREST, USE_SIFE, SIFE_DIM,
    SIFE_ENCODING_TYPE, SIFE_WEIGHT, USE_CENTER_DISTANCE, USE_LOCAL_GRADIENT,
    USE_MULTI_SCALE, MULTI_SCALE_GRIDS, USE_EDGE_ENHANCEMENT, EDGE_WEIGHT,
    USE_COLOR_FEATURES, USE_HSV, COLOR_WEIGHT, FINETUNED_BACKBONE_PATH)
from mobile_sife_cuda.core_shared.patchcore_sife import PatchCoreSIFE

DEVICE = 'cuda'
data = torch.load('model/patchcore_sife/brown_cap/brown_cap.pth', map_location='cpu', weights_only=False)
bank_np = data['memory_bank'].numpy().astype(np.float32)
threshold = data['threshold']
print(f'Threshold: {threshold:.4f}')

pc = PatchCoreSIFE(
    model_size=IMG_SIZE, grid_size=GRID_SIZE, k_nearest=K_NEAREST,
    device=DEVICE, use_sife=USE_SIFE, sife_dim=SIFE_DIM,
    sife_encoding_type=SIFE_ENCODING_TYPE, sife_weight=SIFE_WEIGHT,
    use_center_distance=USE_CENTER_DISTANCE, use_local_gradient=USE_LOCAL_GRADIENT,
    use_color_features=USE_COLOR_FEATURES, use_hsv=USE_HSV, color_weight=COLOR_WEIGHT,
    use_multi_scale=USE_MULTI_SCALE, multi_scale_grids=MULTI_SCALE_GRIDS,
    use_edge_enhancement=USE_EDGE_ENHANCEMENT, edge_weight=EDGE_WEIGHT,
    finetuned_backbone_path=FINETUNED_BACKBONE_PATH)
index = pc.build_faiss_index(bank_np)

def score_dir(d, expect):
    files = sorted(Path(d).glob('*.jpg'))[:8]
    scores = []
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = pc.get_anomaly_score_detailed(pc.extract_from_numpy(img), index)['score']
        scores.append(s)
        tag = 'ANOMALY' if s > threshold else 'NORMAL'
        ok = 'OK' if (tag == expect) else 'WRONG'
        print(f'  {f.name:40s}  {s:.4f}  {tag}  [{ok}]')
    print(f'  Mean={np.mean(scores):.4f}  thr={threshold:.4f}')
    return scores

print('\n-- test/good (expect NORMAL) --')
g = score_dir('data/brown_cap/brown_cap/test/good', 'NORMAL')
print('\n-- test/bad  (expect ANOMALY) --')
b = score_dir('data/brown_cap/brown_cap/test/bad', 'ANOMALY')

print(f'\nGOOD correct: {sum(1 for s in g if s <= threshold)}/{len(g)}')
print(f'BAD  correct: {sum(1 for s in b if s > threshold)}/{len(b)}')
