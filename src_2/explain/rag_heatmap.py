"""
RAG Cross-Attention Heatmap

對指定動畫顯示 Cross Attention 在各 retrieved anime × modality 上的注意力權重。
KV layout：[meta(0-4), text(5-9), image(10-14)]

用法：
    python src_2/explain/rag_heatmap.py --target popularity --ids 12345 67890
    python src_2/explain/rag_heatmap.py --target popularity --n 5   # 從 val 隨機抽 5 筆
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fussion_training"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from dataset import AnimeDataset
from meta_encoder import MetaEncoder
from model import FusionModel


@torch.no_grad()
def extract_attn(model, batch, device):
    """回傳 [top_k, 3] attention weights（已對 batch 維度 squeeze）。"""
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    _, attn = model(batch, return_attn=True)   # attn: [1, top_k, 3]
    return attn[0].cpu().numpy()               # [top_k, 3]


def plot_heatmap(attn: np.ndarray, titles: list[str], anime_title: str, save_path: Path):
    """
    attn     : [top_k, 3]
    titles   : retrieved 動畫名稱列表
    """
    modalities = ["meta", "text", "image"]
    fig, ax = plt.subplots(figsize=(max(5, len(titles) * 1.5), 2.5))
    im = ax.imshow(attn.T, aspect="auto", cmap="Blues", vmin=0)   # [3, top_k]

    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels(titles, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(modalities, fontsize=9)

    for i in range(attn.shape[0]):
        for j in range(3):
            ax.text(i, j, f"{attn[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if attn[i, j] > 0.15 else "black")

    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(f"RAG Attention — {anime_title}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src_2/fussion_configs.yaml")
    parser.add_argument("--target", required=True, choices=["popularity", "meanScore"])
    parser.add_argument("--split",  default="val",
                        choices=["train", "val", "test", "holdout_unknown"])
    parser.add_argument("--ids",    nargs="+", type=int, default=None)
    parser.add_argument("--n",      type=int, default=5,
                        help="--ids 未指定時隨機抽幾筆")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg_out = config["output"]
    run_dir = Path(cfg_out["run_dir"]) / cfg_out["run_id"] / args.target
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "explain" / "rag"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_encoder = MetaEncoder.load(config["data"]["meta_encoder_path"])
    with open(run_dir / "target_scaler.json") as f:
        target_scaler = json.load(f)

    ds = AnimeDataset(args.split, config, meta_encoder,
                      target=args.target, target_scaler=target_scaler)

    model = FusionModel(config).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model.eval()

    # id → title lookup（從 train CSV）
    meta_suffix = config["data"].get("meta_suffix", "_v2")
    train_df    = pd.read_csv(
        Path(config["data"]["meta_dir"]) / f"fusion_meta_clean_train{meta_suffix}.csv"
    )
    id_to_title: dict = {}
    for col in ("title_romaji", "title_english", "id"):
        if col in train_df.columns:
            id_to_title = dict(zip(train_df["id"].astype(int),
                                   train_df[col].astype(str)))
            break

    # 選擇要解釋的 sample
    all_ids = ds.ids
    selected = args.ids if args.ids else list(
        np.random.choice(all_ids, min(args.n, len(all_ids)), replace=False)
    )

    for anime_id in selected:
        if anime_id not in all_ids:
            print(f"  [skip] {anime_id} not in {args.split} split")
            continue

        idx  = all_ids.index(anime_id)
        item = ds[idx]
        batch = {k: v.unsqueeze(0) for k, v in item.items()
                 if isinstance(v, torch.Tensor)}

        attn = extract_attn(model, batch, device)  # [top_k, 3]

        rids   = ds.retrieved_ids_map.get(anime_id, [])
        titles = [id_to_title.get(rid, str(rid))[:12] for rid in rids]
        titles += ["[pad]"] * (config.get("top_k_retrieved", 5) - len(titles))

        anime_title = id_to_title.get(anime_id, str(anime_id))[:20]
        save_path   = out_dir / f"{anime_id}_attn.png"
        plot_heatmap(attn, titles, anime_title, save_path)

        top_idx = int(np.argmax(attn.sum(axis=1)))
        print(f"  [{anime_id}] {anime_title} → top retrieved: {titles[top_idx]}  "
              f"(saved: {save_path})")


if __name__ == "__main__":
    main()
