"""
推論 Pipeline（order.md Step 14）

給定一部新動畫（封面圖 + metadata + 描述），即時走完完整推論流程：
  1. YOLO crop（封面 → 人物/臉部裁切）
  2. Swin-B embedding（cover + banner + yolo crops，各 1024-dim）
  3. e5-base-v2 text embedding（描述 → 768-dim）
  4. RAG query（Qdrant，top-k 相似舊動畫）
  5. FusionModel inference → popularity / meanScore 預測

最佳 checkpoint（test set）：
  popularity → runs/07/popularity/best_model.pt（log_MAE 0.8904）
  meanScore  → runs/02/meanScore/best_model.pt（MAE 7.2937）
兩者架構相同（full 四分支 / use_rag / TrendHead / cover_banner_yolo）。

用法：
    # metadata 用單列 CSV（欄位同訓練 schema，可無 popularity/meanScore）
    python src_2/inference.py \
        --cover  path/to/cover.jpg \
        --banner path/to/banner.jpg \
        --meta   path/to/new_anime.csv \
        --description "一段動畫劇情描述..."

    # 或用 test set 既有動畫驗證（--anime-id，從 test CSV 取 metadata + 描述 + 圖）
    python src_2/inference.py --anime-id 132806 --split test --verify
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_HERE = Path(__file__).resolve().parent           # src_2
_ROOT = _HERE.parent

# fussion_training 與 RAG 直接掛 sys.path（其 model.py/config 名稱在此唯一使用）
sys.path.insert(0, str(_HERE / "fussion_training"))
sys.path.insert(0, str(_HERE / "RAG"))

from meta_encoder import MetaEncoder
from model import FusionModel, make_model_config, apply_target_overrides
from dataset import (_build_rag_meta_lookup, _load_emb_parquet,
                     denormalize_target, _parse_list)
from sparse_encoder import (SparseEncoder, parse_genres, parse_studios,
                            parse_voice_actors, parse_source)
import rag_query as _ragq


# ── 隔離載入有命名衝突的元件模組（config.py / model.py 各 component 都有）────────

def _load_isolated(mod_name: str, file_path: Path, dep_dir: Path):
    """以唯一名稱載入模組，並把其所在目錄暫掛 sys.path 供其內部 import。"""
    sys.path.insert(0, str(dep_dir))
    # 清掉可能殘留的同名 config/model，避免抓到別的 component
    for clash in ("config", "model", "output", "text_preprocessor",
                  "embedding_generator", "image_process"):
        sys.modules.pop(clash, None)
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


# ── Inference Pipeline ────────────────────────────────────────────────────────

class InferencePipeline:
    def __init__(self, config: dict,
                 pop_run: str = "22", score_run: str = "22",
                 rag_use_image: bool = False):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k  = config.get("top_k_retrieved", 5)
        # image_rag 只有 train embedding，故 val/test 的 RAG 檢索為 sparse+text only；
        # 為與驗證指標一致，新動畫推論預設亦不使用 image modality 檢索。
        self.rag_use_image = rag_use_image

        cfg_data = config["data"]
        self.train_df = pd.read_csv(
            Path(cfg_data["meta_dir"]) / "fusion_meta_clean_train_v2.csv")

        # ── MetaEncoder ─────────────────────────────────────────────────────
        self.meta_encoder = MetaEncoder.load(cfg_data["meta_encoder_path"])

        # ── Text embedder（e5-base-v2）──────────────────────────────────────
        ct_dir = _HERE / "component_text"
        text_out = _load_isolated("ct_output", ct_dir / "output.py", ct_dir)
        self.text_embedder = text_out.TextEmbedder()

        # ── Swin image encoder ──────────────────────────────────────────────
        ci_dir = _HERE / "component_image"
        self._img_model = _load_isolated("ci_model", ci_dir / "model.py", ci_dir)
        self._img_proc  = _load_isolated("ci_imgproc", ci_dir / "image_process.py", ci_dir)
        self._img_cfg   = _load_isolated("ci_config", ci_dir / "config.py", ci_dir)
        self._yolo      = _load_isolated("ci_yolo", ci_dir / "YOLO.py", ci_dir)
        self._run_yolo  = _load_isolated("ci_runyolo", ci_dir / "run_yolo_crop.py", ci_dir)

        img_config = self._img_cfg.load_config(str(ci_dir / "image_encoder_config.yaml"))
        swin_path = Path(img_config["inference"]["model_path"])
        if not swin_path.is_absolute() and not swin_path.exists():
            swin_path = _ROOT / swin_path
        self.swin = self._img_model.load_model(
            {"model": {"name": str(swin_path)}}).to(self.device).eval()
        image_size = img_config["data"]["image_size"]
        self._resize    = self._img_proc.ResizeWithPad(image_size)
        self._transform = self._img_proc.get_transform_original(image_size)
        self._yolo_cfg  = self._img_cfg.load_yolo_config(str(ci_dir / "image_encoder_config.yaml"))

        # ── RAG（Qdrant + sparse encoder）───────────────────────────────────
        rag_cfg = _ragq._load_config()
        from qdrant_client import QdrantClient
        self._qdrant = QdrantClient(host=rag_cfg["qdrant"]["host"],
                                    port=rag_cfg["qdrant"]["port"])
        self._collection = rag_cfg["qdrant"]["collection_name"]
        self._sparse = SparseEncoder.load(rag_cfg["paths"]["encoder_path"])
        self._prefetch_k = rag_cfg["query"]["prefetch_k"]
        self._fetch_limit = max(rag_cfg["query"]["top_k"],
                                rag_cfg["query"].get("top_k_ids", 5))

        # RAG 知識庫：retrieved 動畫的 text / image embedding（來自 train）
        self.rag_text_map  = _load_emb_parquet(
            str(Path(cfg_data["text_emb_dir"]) / "text_embeddings_train.parquet"))
        self.rag_image_map = _load_emb_parquet(
            str(Path(cfg_data["image_rag_emb_dir"]) / "image_embeddings_train.parquet"))

        # ── 兩個最佳 FusionModel ─────────────────────────────────────────────
        self.models, self.scalers = {}, {}
        for target, run_id in (("popularity", pop_run), ("meanScore", score_run)):
            run_dir = Path(config["output"]["run_dir"]) / run_id / target
            tcfg = apply_target_overrides(config, target)   # 與訓練架構一致
            m = FusionModel(make_model_config(tcfg, target)).to(self.device)
            m.load_state_dict(torch.load(run_dir / "best_model.pt",
                                         map_location=self.device, weights_only=True))
            m.eval()
            self.models[target]  = m
            self.scalers[target] = json.load(open(run_dir / "target_scaler.json"))

    # ── 各階段 ───────────────────────────────────────────────────────────────

    def _swin_embed(self, imgs) -> np.ndarray:
        batch = torch.stack([self._transform(self._resize(i)) for i in imgs]).to(self.device)
        with torch.no_grad():
            return self._img_model.get_embedding(self.swin, batch).mean(dim=0).cpu().numpy()

    def _image_features(self, cover_path, banner_path):
        """回傳 image_emb [3,1024]（cover/banner/yolo）+ mask [3]（True=缺失）。"""
        load_image = self._img_proc.load_image
        zero = np.zeros(1024, dtype=np.float32)

        cover = load_image(str(cover_path)) if cover_path and Path(cover_path).exists() else None
        cover_emb = self._swin_embed([cover]).astype(np.float32) if cover else zero
        has_cover = cover is not None

        banner = load_image(str(banner_path)) if banner_path and Path(banner_path).exists() else None
        banner_emb = self._swin_embed([banner]).astype(np.float32) if banner else zero
        has_banner = banner is not None

        # YOLO crop（in-memory）→ Swin
        yolo_emb, has_yolo = zero, False
        if cover:
            crops, _ = self._run_yolo._get_crops(
                cover, self._yolo_cfg,
                self._yolo_cfg.get("detect_mode", "face"))
            crops = [c for c in crops if c is not None]
            if crops:
                yolo_emb = self._swin_embed(crops).astype(np.float32)
                has_yolo = True

        image_emb  = np.stack([cover_emb, banner_emb, yolo_emb], axis=0)  # [3,1024]
        image_mask = np.array([not has_cover, not has_banner, not has_yolo], dtype=bool)
        return image_emb, image_mask, cover_emb  # cover_emb 供 RAG query image_vec

    def _rag_retrieve(self, meta_row: pd.Series, text_vec, image_vec):
        """查 Qdrant，回傳 retrieved_ids（最多 top_k）。"""
        genres  = parse_genres(meta_row.get("genres"))
        studios = parse_studios(meta_row.get("studios"))
        vas     = parse_voice_actors(meta_row.get("voice_actor_names"))
        source  = parse_source(meta_row.get("source"))
        indices, values = self._sparse.encode(genres, studios, vas, source)
        if not indices:
            return []

        from qdrant_client import models
        anime_id = int(meta_row["id"])
        qfilter = _ragq._build_time_filter(
            int(meta_row["release_year"]), meta_row.get("release_quarter"), anime_id)

        prefetch = [models.Prefetch(query=models.SparseVector(indices=indices, values=values),
                                    using="genre_studio", limit=self._prefetch_k, filter=qfilter)]
        if text_vec is not None:
            prefetch.append(models.Prefetch(query=text_vec.tolist(), using="text",
                                            limit=self._prefetch_k, filter=qfilter))
        if self.rag_use_image and image_vec is not None:
            prefetch.append(models.Prefetch(query=image_vec.tolist(), using="image",
                                            limit=self._prefetch_k, filter=qfilter))

        if len(prefetch) > 1:
            results = self._qdrant.query_points(
                collection_name=self._collection, prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=qfilter, limit=self._fetch_limit).points
        else:
            results = self._qdrant.query_points(
                collection_name=self._collection,
                query=models.SparseVector(indices=indices, values=values),
                using="genre_studio", query_filter=qfilter,
                limit=self._fetch_limit).points
        return [int(r.id) for r in results[:self.top_k]]

    # ── 主入口 ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, meta_row: pd.Series, description: str,
                cover_path, banner_path=None) -> dict:
        # 1+2. image
        image_emb, image_mask, cover_emb = self._image_features(cover_path, banner_path)

        # 3. text
        text_emb = self.text_embedder.embed(description)
        if text_emb is None:
            text_emb = np.zeros(768, dtype=np.float32)
        text_emb = text_emb.astype(np.float32)

        # 4. RAG retrieve
        retrieved_ids = self._rag_retrieve(meta_row, text_emb, cover_emb)

        # rag_text / rag_image（從知識庫取回）
        rag_text  = np.zeros((self.top_k, 768),  dtype=np.float32)
        rag_image = np.zeros((self.top_k, 1024), dtype=np.float32)
        rag_mask  = np.ones(self.top_k, dtype=bool)
        for i, rid in enumerate(retrieved_ids):
            rag_text[i]  = self.rag_text_map.get(rid,  np.zeros(768))
            rag_image[i] = self.rag_image_map.get(rid, np.zeros(1024))
            rag_mask[i]  = False
        if rag_mask.all():
            rag_mask[0] = False

        # rag_meta [top_k,10]（複用 dataset 的 lookup）
        meta_df = meta_row.to_frame().T
        rag_df  = pd.DataFrame([{"id": int(meta_row["id"]),
                                 "retrieved_ids": json.dumps(retrieved_ids)}])
        rag_meta = _build_rag_meta_lookup(
            meta_df, rag_df, self.train_df, self.meta_encoder, self.top_k
        ).get(int(meta_row["id"]), np.zeros((self.top_k, 10), dtype=np.float32))

        # meta features [56]
        meta_feat = self.meta_encoder.transform(meta_df)[0].astype(np.float32)

        # batch（batch=1）
        def _t(arr):
            return torch.from_numpy(np.asarray(arr)).unsqueeze(0).to(self.device)
        batch = {
            "image_emb":  _t(image_emb), "image_mask": _t(image_mask),
            "text_emb":   _t(text_emb),  "meta_feat":  _t(meta_feat),
            "rag_meta":   _t(rag_meta),  "rag_text":   _t(rag_text),
            "rag_image":  _t(rag_image), "rag_mask":   _t(rag_mask),
        }

        # 5. FusionModel inference
        out = {"retrieved_ids": retrieved_ids}
        for target, model in self.models.items():
            pred_norm = model(batch).cpu().numpy()
            out[target] = float(denormalize_target(pred_norm, self.scalers[target])[0])
        return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src_2/fussion_configs.yaml")
    parser.add_argument("--cover",  default=None)
    parser.add_argument("--banner", default=None)
    parser.add_argument("--meta",   default=None, help="單列 CSV（訓練 schema）")
    parser.add_argument("--description", default=None)
    # 驗證模式：用既有 split 的某動畫
    parser.add_argument("--anime-id", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--verify", action="store_true",
                        help="與 pred_{split}.csv 對照預測值")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    pipe = InferencePipeline(config)

    # ── 取得 meta_row / description / 圖片路徑 ────────────────────────────────
    if args.anime_id is not None:
        meta_dir = Path(config["data"]["meta_dir"])
        df = pd.read_csv(meta_dir / f"fusion_meta_clean_{args.split}_v2.csv")
        meta_row = df[df["id"] == args.anime_id].iloc[0]
        description = meta_row.get("description", "")
        split_img = {"train": "train_image", "val": "validation_image",
                     "test": "test_image", "holdout_unknown": "holdout_unknow_image"}[args.split]
        img_dir = _ROOT / "src_2" / "data" / "image" / split_img
        cover  = img_dir / f"{args.anime_id}_coverImage_extraLarge.jpg"
        banner = img_dir / f"{args.anime_id}_bannerImage.jpg"
    else:
        if not (args.cover and args.meta):
            parser.error("需提供 --cover + --meta（或用 --anime-id 驗證模式）")
        meta_row = pd.read_csv(args.meta).iloc[0]
        if "id" not in meta_row:
            meta_row["id"] = -1
        description = args.description or meta_row.get("description", "")
        cover, banner = args.cover, args.banner

    result = pipe.predict(meta_row, description, cover, banner)

    print("\n" + "=" * 50)
    print(f"  popularity : {result['popularity']:,.0f}")
    print(f"  meanScore  : {result['meanScore']:.1f}")
    print(f"  retrieved  : {result['retrieved_ids']}")
    print("=" * 50)

    if args.verify and args.anime_id is not None:
        for target in ("popularity", "meanScore"):
            pred_csv = Path(config["output"]["run_dir"]) / "22" / target / f"pred_{args.split}.csv"
            if pred_csv.exists():
                pdf = pd.read_csv(pred_csv)
                ref = pdf[pdf["id"] == args.anime_id]
                if len(ref):
                    print(f"  [verify] {target}: pipeline={result[target]:.2f}  "
                          f"batch={ref.iloc[0]['pred']:.2f}  "
                          f"true={ref.iloc[0]['target']:.2f}")


if __name__ == "__main__":
    main()
