import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.YOLO import detect_person
from util.getImage import getImage_YOLO




def show_crops(idx, orig_img, frame_, output_dir, max_cols=5):
    """將原圖和所有 crop 顯示在同一個 matplotlib 畫布，最多顯示 max_cols 個 crop。"""
    crops = frame_[:max_cols]
    n = len(crops)

    fig = plt.figure(figsize=(3 * (n + 1), 4))
    gs  = gridspec.GridSpec(1, n + 1, figure=fig)

    # 左側：原圖
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(np.array(orig_img))
    ax0.set_title(f"original\nid={idx}", fontsize=8)
    ax0.axis('off')

    # 右側：各 crop
    for k, crop in enumerate(crops):
        ax = fig.add_subplot(gs[0, k + 1])
        ax.imshow(np.array(crop))
        ax.set_title(f"crop_{k}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    canvas_path = output_dir / f"{idx}_canvas.jpg"
    plt.savefig(canvas_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return canvas_path


_ROOT = Path(__file__).resolve().parent.parent

with open(_ROOT / 'yolo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

image_dir    = Path(config['data']['image_dir'])
output_dir   = Path(config['data']['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

# model settings
model_cfg = config['model']
level     = model_cfg['level']
version   = model_cfg['version']

# detection settings
det_cfg      = config['detection']
conf_threshold = det_cfg['conf_threshold']
iou_threshold  = det_cfg['iou_threshold']
max_persons    = det_cfg['max_persons']
save_crops     = det_cfg['save_crops']
fallback       = det_cfg['fallback_full_image']

# debug settings
debug_cfg   = config['debug']
debug_mode  = debug_cfg['enabled']
sample_size = debug_cfg['sample_size']

# image_columns 取第一個（目前為 coverImage_medium）
col = config['data']['image_columns'][0]

rows  = getImage_YOLO(config, col)
if debug_mode:
    rows = rows.head(sample_size)

start= 5000
end=min(len(rows), 5050)

for i in range(start, end):
    row      = rows.iloc[i]
    idx      = row['idx']
    url      = row['url']

    img     = Image.open(requests.get(url, timeout=15, stream=True).raw).convert('RGB')

    w, h = img.size
    scale = max(640 / w, 640 / h)
    if scale > 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    results = detect_person(
        img,
        level=level,
        version=version,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    # 依信心分數排序，最多取 max_persons 個
    results = sorted(results, key=lambda x: x[2], reverse=True)[:max_persons]

    frame_ = []

    if results:
        for j, (bbox, label, conf) in enumerate(results):
            x0, y0, x1, y1 = bbox
            crop = img.crop((x0, y0, x1, y1))
            frame_.append(crop)
            if save_crops:
                save_path = output_dir / f"{idx}_{col}_crop_{j}.jpg"
                crop.save(save_path)
        print(f"[{idx}] detected {len(results)} person(s)")
    elif fallback:
        crop = img
        frame_.append(crop)
        if save_crops:
            save_path = output_dir / f"{idx}_{col}_crop_0.jpg"
            crop.save(save_path)
        print(f"[{idx}] no person detected, fallback to full image")

    if frame_:
        canvas_path = show_crops(idx, img, frame_, output_dir)
        print(f"[{idx}] canvas saved → {canvas_path}")


