"""
AnimeDataset：載入並組合所有 embedding + metadata + RAG features

每筆 __getitem__ 回傳：
  text_emb   : [768]
  image_emb  : [N_modality, 1024]  N_modality ∈ {1,2,3} 依 image_mode
  image_mask : [N_modality] bool   True = 缺失模態（不參與 gate 計算）
  meta_feat  : [56]
  target     : scalar float
  # 若 use_rag=True：
  rag_meta   : [top_k, 10]
  rag_text   : [top_k, 768]
  rag_image  : [top_k, 1024]
  rag_mask   : [top_k] bool（True = padding 位置）
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# image_mode → 使用的欄位前綴（順序即 concat 順序）
_IMAGE_MODE_COLS = {
    "cover":              ["cover"],
    "banner":             ["banner"],            # 單模態：只有 banner
    "yolo":               ["yolo"],              # 單模態：只有 character（yolo crop）
    "cover_banner":       ["cover", "banner"],
    "cover_banner_yolo":  ["cover", "banner", "yolo"],
}


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _load_emb_parquet(path: str, id_col: str = "id") -> Dict[int, np.ndarray]:
    """parquet → {id: np.ndarray}，自動偵測 emb 欄位前綴。"""
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c != id_col and not c.startswith("has_")
                and not c.startswith("split") and not c.startswith("text_clean")]
    ids  = df[id_col].astype(int).tolist()
    embs = df[emb_cols].to_numpy(dtype=np.float32)
    return {aid: arr.copy() for aid, arr in zip(ids, embs)}


def _load_image_emb_stack(path: str, mode: str):
    """
    依 image_mode 載入各模態 embedding，輸出：
      emb_map  : {id: np.ndarray [N_modality, 1024]}
      mask_map : {id: np.ndarray [N_modality] bool}  True = 缺失（padding）
    """
    prefixes = _IMAGE_MODE_COLS[mode]
    df = pd.read_parquet(path)

    prefix_cols = {}
    for p in prefixes:
        cols = sorted([c for c in df.columns if c.startswith(f"{p}_") and c[len(p)+1:].isdigit()])
        prefix_cols[p] = cols

    ids = df["id"].astype(int).tolist()

    # 向量化：各模態分別抽出整塊 array，再 stack
    modality_embs  = [df[prefix_cols[p]].to_numpy(dtype=np.float32) for p in prefixes]  # [N, 1024] each
    stacked = np.stack(modality_embs, axis=1)   # [N, N_modality, 1024]

    # mask：has_{p} == 0 表示缺失
    mask_matrix = np.zeros((len(df), len(prefixes)), dtype=bool)
    for j, p in enumerate(prefixes):
        has_col = f"has_{p}"
        if has_col in df.columns:
            mask_matrix[:, j] = df[has_col].to_numpy() == 0

    emb_map  = {aid: arr.copy() for aid, arr in zip(ids, stacked)}
    mask_map = {aid: arr.copy() for aid, arr in zip(ids, mask_matrix)}
    return emb_map, mask_map


def _winsorize(arr: np.ndarray, pct: float) -> np.ndarray:
    cap = np.percentile(arr, pct)
    return np.clip(arr, None, cap)


def _build_target_scaler(values: np.ndarray, log_transform: bool, winsor_pct: float):
    if winsor_pct < 100:
        values = _winsorize(values, winsor_pct)
    if log_transform:
        values = np.log1p(values)
    return {
        "center": float(np.mean(values)),
        "scale":  max(float(np.std(values)), 1e-8),
        "log_transform": log_transform,
    }


def _normalize_target(y: np.ndarray, scaler: dict) -> np.ndarray:
    v = np.log1p(y) if scaler["log_transform"] else y.copy()
    return (v - scaler["center"]) / scaler["scale"]


def denormalize_target(y_norm: np.ndarray, scaler: dict) -> np.ndarray:
    y_norm = np.asarray(y_norm, dtype=np.float64)  # AMP 輸出 float16，expm1 會 overflow
    if scaler["log_transform"]:
        y_norm = np.clip(y_norm, -5.0, 5.0)
    y = y_norm * scaler["scale"] + scaler["center"]
    return np.expm1(y) if scaler["log_transform"] else y


# ── RAG Context ───────────────────────────────────────────────────────────────

def _parse_list(val) -> List:
    if isinstance(val, list):
        return val
    try:
        if pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass
    try:
        return json.loads(str(val))
    except Exception:
        return []


def _build_rag_meta_lookup(
    split_df: pd.DataFrame,
    rag_df: pd.DataFrame,
    train_df: pd.DataFrame,
    meta_encoder,
    top_k: int,
) -> Dict[int, np.ndarray]:
    """
    為每部動畫建立 rag_meta [top_k, 10]。
    10 dims = retrieved 動畫的 5 個數值特徵 + 5 個 pair-wise 特徵。
    """
    from meta_encoder import _parse_genres, _parse_studios, _te_lookup, _standardize_te
    from tqdm import tqdm

    fallback_pop   = meta_encoder.te_fallback["pop"]
    fallback_score = meta_encoder.te_fallback["score"]

    # 標準化參數（popularity → log1p + zscore，其餘 zscore）
    pop_vals = pd.to_numeric(train_df["popularity"],   errors="coerce").dropna()
    sc_vals  = pd.to_numeric(train_df["meanScore"],    errors="coerce").dropna()
    yr_vals  = pd.to_numeric(train_df["release_year"], errors="coerce").dropna()
    ep_vals  = pd.to_numeric(train_df["episodes"],     errors="coerce").dropna()

    def _zscore_params(arr):
        m, s = float(arr.median()), float(arr.std())
        return m, max(s, 1e-8)

    pop_m, pop_s = _zscore_params(pd.Series(np.log1p(pop_vals)))
    sc_m,  sc_s  = _zscore_params(sc_vals)
    yr_m,  yr_s  = _zscore_params(yr_vals)
    ep_m,  ep_s  = _zscore_params(ep_vals)

    def _norm_pop(v):
        return (math.log1p(max(v, 0)) - pop_m) / pop_s if not math.isnan(v) else 0.0

    def _norm(v, m, s):
        return (v - m) / s if not math.isnan(v) else 0.0

    # train lookup：預先 parse 所有欄位，避免重複解析字串
    train_lookup: Dict[int, dict] = {}
    for row in train_df.itertuples(index=False):
        aid = int(row.id)
        studio_list = _parse_studios(getattr(row, "studios", "") or "")
        raw_pop_te, raw_sc_te = _te_lookup(studio_list, meta_encoder.studio_te,
                                            fallback_pop, fallback_score)
        std_pop_te, std_sc_te = _standardize_te(raw_pop_te, raw_sc_te, meta_encoder.te_stats)
        pop = pd.to_numeric(getattr(row, "popularity",   None), errors="coerce")
        sc  = pd.to_numeric(getattr(row, "meanScore",    None), errors="coerce")
        yr  = pd.to_numeric(getattr(row, "release_year", None), errors="coerce")
        ep  = pd.to_numeric(getattr(row, "episodes",     None), errors="coerce")
        fmt = getattr(row, "format", None)
        train_lookup[aid] = {
            "pop": float(pop) if not pd.isna(pop) else float("nan"),
            "sc":  float(sc)  if not pd.isna(sc)  else float("nan"),
            "yr":  float(yr)  if not pd.isna(yr)  else float("nan"),
            "ep":  float(ep)  if not pd.isna(ep)  else float("nan"),
            "studios": set(studio_list),
            "genres":  set(_parse_genres(getattr(row, "genres", "") or "")),
            "format":  str(fmt) if fmt and pd.notna(fmt) else "",
            "ste_pop": std_pop_te,
            "ste_sc":  std_sc_te,
        }

    rag_meta_map = {}
    rag_indexed  = rag_df.set_index("id")

    for row in tqdm(split_df.itertuples(index=False),
                    total=len(split_df), desc="  RAG meta", leave=False, ncols=80):
        qid = int(row.id)
        rag_row = rag_indexed.loc[qid] if qid in rag_indexed.index else None
        retrieved_ids = _parse_list(rag_row["retrieved_ids"]) if rag_row is not None else []
        retrieved_ids = [int(x) for x in retrieved_ids[:top_k]]

        q_fmt     = getattr(row, "format", None)
        q_studios = set(_parse_studios(getattr(row, "studios", "") or ""))
        q_genres  = set(_parse_genres(getattr(row, "genres",  "") or ""))
        q_format  = str(q_fmt) if q_fmt and pd.notna(q_fmt) else ""

        mat = np.zeros((top_k, 10), dtype=np.float32)

        for i, rid in enumerate(retrieved_ids):
            t = train_lookup.get(rid)
            if t is None:
                continue
            mat[i, 0] = _norm_pop(t["pop"] if not math.isnan(t["pop"]) else fallback_pop)
            mat[i, 1] = _norm(t["sc"] if not math.isnan(t["sc"]) else fallback_score, sc_m, sc_s)
            mat[i, 2] = _norm(t["yr"] if not math.isnan(t["yr"]) else yr_m, yr_m, yr_s)
            mat[i, 3] = _norm(t["ep"] if not math.isnan(t["ep"]) else ep_m, ep_m, ep_s)
            mat[i, 4] = 1.0
            mat[i, 5] = 1.0 if q_studios & t["studios"] else 0.0
            union = q_genres | t["genres"]
            mat[i, 6] = len(q_genres & t["genres"]) / len(union) if union else 0.0
            mat[i, 7] = 1.0 if q_format and q_format == t["format"] else 0.0
            mat[i, 8] = t["ste_pop"]
            mat[i, 9] = t["ste_sc"]

        rag_meta_map[qid] = mat

    return rag_meta_map


# ── Dataset ───────────────────────────────────────────────────────────────────

class AnimeDataset(Dataset):
    def __init__(
        self,
        split: str,
        config: dict,
        meta_encoder,
        target: str = "popularity",
        target_scaler: Optional[dict] = None,
    ):
        """
        Args:
            split          : 'train' | 'val' | 'test' | 'holdout_unknown'
            config         : fussion_configs.yaml 完整 dict
            meta_encoder   : 已 fit 的 MetaEncoder 實例
            target         : 訓練目標欄位名稱
            target_scaler  : 正規化參數 dict（val/test 時傳入 train 的 scaler）
        """
        cfg_data  = config["data"]
        self.image_mode = config.get("image_mode", "cover")
        self.use_rag    = config.get("use_rag", True)
        self.top_k      = config.get("top_k_retrieved", 5)

        meta_suffix = cfg_data.get("meta_suffix", "_v2")
        meta_path   = Path(cfg_data["meta_dir"]) / f"fusion_meta_clean_{split}{meta_suffix}.csv"
        train_path  = Path(cfg_data["meta_dir"]) / f"fusion_meta_clean_train{meta_suffix}.csv"

        self.meta_df  = pd.read_csv(meta_path)
        self.train_df = pd.read_csv(train_path)
        self.ids      = self.meta_df["id"].astype(int).tolist()

        # ── metadata features ──────────────────────────────────────────────
        _all_meta = meta_encoder.transform(self.meta_df)   # [N, 56]，一次批次
        self.meta_feats = {
            int(self.meta_df.iloc[i]["id"]): _all_meta[i].copy()
            for i in range(len(self.meta_df))
        }

        # ── temporal sample weights（只有 train split 且有啟用時才計算）─────
        tw_cfg   = config.get("training", {}).get("temporal_weight", {})
        apply_to = tw_cfg.get("apply_to", None)   # None = 全部套用
        use_tw   = (tw_cfg.get("enabled", False)
                    and split == "train"
                    and (apply_to is None or target in apply_to))
        if use_tw:
            alpha   = float(tw_cfg.get("alpha", 0.2))
            max_yr  = pd.to_numeric(self.train_df["release_year"], errors="coerce").max()
            yr_vals = pd.to_numeric(self.meta_df["release_year"], errors="coerce").fillna(max_yr).values
            raw_w   = np.exp(-alpha * (max_yr - yr_vals)).astype(np.float32)
            self.sample_weights = (raw_w / raw_w.mean())   # normalize：mean=1
        else:
            self.sample_weights = np.ones(len(self.meta_df), dtype=np.float32)

        # ── text embeddings ────────────────────────────────────────────────
        text_path = Path(cfg_data["text_emb_dir"]) / f"text_embeddings_{split}.parquet"
        self.text_map = _load_emb_parquet(str(text_path))

        # ── image embeddings ───────────────────────────────────────────────
        img_path = Path(cfg_data["image_emb_dir"]) / f"image_embeddings_{split}.parquet"
        self.image_map, self.image_mask_map = _load_image_emb_stack(str(img_path), self.image_mode)
        _sample_img = next(iter(self.image_map.values()))
        self.n_image_modality = _sample_img.shape[0]  # 1, 2, or 3
        self.image_dim        = _sample_img.shape[1]  # 自動偵測：pooler=1024 / stage=1920

        # ── target ─────────────────────────────────────────────────────────
        self.target_col = target
        target_vals = pd.to_numeric(self.meta_df[target], errors="coerce").fillna(0).values

        if target_scaler is None:
            t_cfg = config.get("training", {}).get(target, {})
            target_scaler = _build_target_scaler(
                target_vals,
                log_transform = t_cfg.get("log_transform", False),
                winsor_pct    = t_cfg.get("winsor_pct", 100),
            )
        self.target_scaler = target_scaler
        self.target_vals   = _normalize_target(target_vals, target_scaler)
        self.target_ids    = self.meta_df["id"].astype(int).tolist()

        # ── RAG ────────────────────────────────────────────────────────────
        if self.use_rag:
            rag_path = Path(cfg_data["rag_return_dir"]) / f"rag_features_{split}.parquet"
            rag_df   = pd.read_parquet(rag_path)

            # retrieved text / image（來自 train set）
            text_train_path = Path(cfg_data["text_emb_dir"]) / "text_embeddings_train.parquet"
            img_rag_path    = Path(cfg_data["image_rag_emb_dir"]) / "image_embeddings_train.parquet"
            # Reuse text_map when split is train (same file, avoid double memory)
            self.rag_text_map  = (self.text_map if split == "train"
                                  else _load_emb_parquet(str(text_train_path)))
            self.rag_image_map = _load_emb_parquet(str(img_rag_path))
            # RAG image 維度自動偵測（可獨立於主 image：pooler=1024 / stage=1920）
            self.rag_image_dim = (len(next(iter(self.rag_image_map.values())))
                                  if self.rag_image_map else 1024)

            # retrieved ids
            rag_indexed  = rag_df.set_index("id")
            self.retrieved_ids_map: Dict[int, List[int]] = {}
            for anime_id in self.ids:
                if anime_id in rag_indexed.index:
                    ids_raw = _parse_list(rag_indexed.loc[anime_id, "retrieved_ids"])
                    self.retrieved_ids_map[anime_id] = [int(x) for x in ids_raw[:self.top_k]]
                else:
                    self.retrieved_ids_map[anime_id] = []

            # rag_meta [top_k, 10]
            self.rag_meta_map = _build_rag_meta_lookup(
                self.meta_df, rag_df, self.train_df, meta_encoder, self.top_k
            )

        # ── fallback dims ──────────────────────────────────────────────────
        sample_text = next(iter(self.text_map.values()))
        self.text_dim = len(sample_text)        # 768
        # self.image_dim 已於 image 載入時自動偵測（pooler=1024 / stage=1920）
        if not self.use_rag:
            self.rag_image_dim = self.image_dim

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        anime_id = self.ids[idx]

        text   = self.text_map.get(anime_id, np.zeros(self.text_dim, dtype=np.float32))
        # image_emb: [N_modality, image_dim]，image_mask: [N_modality] True=缺失
        image  = self.image_map.get(
            anime_id, np.zeros((self.n_image_modality, self.image_dim), dtype=np.float32))
        i_mask = self.image_mask_map.get(
            anime_id, np.ones(self.n_image_modality, dtype=bool))
        meta   = self.meta_feats[anime_id]
        target = float(self.target_vals[idx])

        item = {
            "text_emb":   torch.from_numpy(text),
            "image_emb":  torch.from_numpy(image),   # [N, 1024]
            "image_mask": torch.from_numpy(i_mask),  # [N] bool
            "meta_feat":  torch.from_numpy(meta),
            "target":     torch.tensor(target, dtype=torch.float32),
            "weight":     torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            "id":         anime_id,
        }

        if self.use_rag:
            rids = self.retrieved_ids_map.get(anime_id, [])

            rag_text  = np.zeros((self.top_k, self.text_dim),      dtype=np.float32)
            rag_image = np.zeros((self.top_k, self.rag_image_dim), dtype=np.float32)
            rag_mask  = np.ones(self.top_k, dtype=bool)   # True = padding

            for i, rid in enumerate(rids):
                rag_text[i]  = self.rag_text_map.get(rid,  np.zeros(self.text_dim))
                rag_image[i] = self.rag_image_map.get(rid, np.zeros(self.rag_image_dim))
                rag_mask[i]  = False  # valid

            # 全遮罩 → MultiheadAttention softmax 產生 NaN；強制第 0 格有效
            if rag_mask.all():
                rag_mask[0] = False

            rag_meta = self.rag_meta_map.get(anime_id, np.zeros((self.top_k, 10), dtype=np.float32))

            item["rag_meta"]  = torch.from_numpy(rag_meta)
            item["rag_text"]  = torch.from_numpy(rag_text)
            item["rag_image"] = torch.from_numpy(rag_image)
            item["rag_mask"]  = torch.from_numpy(rag_mask)

        return item

    @property
    def image_input_dim(self) -> int:
        return self.image_dim

    @property
    def text_input_dim(self) -> int:
        return self.text_dim
