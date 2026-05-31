# v2 建置順序

架構參考：`docs_new/` 下的各說明文件

---

## ✅ 已完成

### Step 0：Component 建置

| 元件 | 路徑 | 說明 | 狀態 |
|------|------|------|------|
| Text Encoder | `src_2/component_text/` | e5-base-v2（768-dim），TextEmbedder + CLI | ✅ |
| Image Encoder | `src_2/component_image/` | Swin-B fine-tuned（1024-dim），ImageEmbedder + YOLO（both: face+person） | ✅ |
| RAG 基礎建設 | `src_2/RAG/` | sparse_encoder, rag_builder, rag_query, run_build_embeddings, rag_config | ✅ |

### Step 1：RAG Embeddings 生成

```bash
python src_2/RAG/run_build_embeddings.py
```

| 輸出 | 說明 | 狀態 |
|------|------|------|
| `src_2/embedding/text/text_embeddings_{split}.parquet` | e5-base-v2 768-dim | ✅ train / val / test / holdout |
| `src_2/embedding/image_rag/image_embeddings_train.parquet` | Swin-B 1024-dim（no YOLO，RAG 用） | ✅ train only |

### Step 2：YOLO Crop 圖片

```bash
python src_2/component_image/run_yolo_crop.py --splits train val test holdout_unknown
```

| 輸出 | 狀態 |
|------|------|
| `src_2/data/image/crops/{split}/{id}_crop_{n}.jpg` | ✅ train / val / test / holdout |

### Step 3：Fusion Image Embeddings（yolo + cover + banner）

```bash
python src_2/component_image/run_swin_embedding.py --splits train val test holdout_unknown
```

| 輸出 | 說明 | 狀態 |
|------|------|------|
| `src_2/embedding/image/image_embeddings_{split}.parquet` | id, yolo_0…1023, cover_0…1023, banner_0…1023, has_* | ✅ train / val / test / holdout |

### Step 4：建立 Qdrant Collection

```bash
bash src_2/RAG/start_qdrant.sh
python src_2/RAG/rag_builder.py
```

| 輸出 | 狀態 |
|------|------|
| `src_2/RAG/qdrant_storage/`（collection: `anime_rag_v2`） | ✅ |
| `src_2/RAG/sparse_encoder.json` | ✅ |

### Step 5：查詢 RAG Features

```bash
python src_2/RAG/rag_query.py --splits train val test holdout_unknown
```

| 輸出 | 說明 | 狀態 |
|------|------|------|
| `src_2/RAG/return/rag_features_{split}.parquet` | id, rag_popularity, rag_score, rag_release_year, rag_episodes, rag_found, retrieved_ids … | ✅ train / val / test / holdout |

### Step 6：Fusion Model 元件

| 元件 | 路徑 | 說明 | 狀態 |
|------|------|------|------|
| MetaEncoder v2 | `src_2/fussion_training/meta_encoder.py` | 56-dim（移除舊版 RAG 10 dims），`fit(meta_df)` / `transform(meta_df)` | ✅ |
| Cross Attention | `src_2/fussion_training/cross_attention.py` | Q[batch,1,128] × KV[batch,15,128] → [batch,128]，4 heads | ✅ |

### Step 7：Fusion Model v2 訓練

架構：

```
Image  [batch, 3, 1024] → ImageProjection (Shared + Gate) → [batch, 128] ─┐
Text   [batch, 768]     → TextProjection                  → [batch, 128] ─┤
Meta   [batch, 56]      → MetaProjection                  → [batch, 128] ─┤─ concat → MLP → [1]
                          MetaProjection → Q [batch,1,128] ─┐              │
RAG retrieved items                                          │              │
  [batch,5,10]  → Rag_meta Proj  ─┐                        │              │
  [batch,5,768] → Rag_text Proj  ─┼─ KV [batch,15,128] → Cross Attention ─┘
  [batch,5,1024]→ Rag_image Proj ─┘
  concat_dim: use_rag=True → 512（128×4）; use_rag=False → 384（128×3）
```

設計決策：
- **Loss**：popularity → HuberLoss（delta=1.0）；meanScore → Log-Cosh Loss
- **Optimizer**：SAM (rho=0.05, pure wrapper) + AdamW；LR scheduler：ReduceLROnPlateau（factor=0.5, patience=3）
- **AMP**：autocast（float16 forward，float32 gradient）；不用 GradScaler（與 SAM 兩步更新不相容）
- **DataLoader**：`num_workers=min(4, cpu_count())`，`persistent_workers=True`
- **train_separately**：true → 同一 script 兩遍循環，各 target 獨立模型
- **Checkpoint**：每 epoch 若 val_loss 改善則儲存 `best_model.pt` + `target_scaler.json`

執行：

```bash
python src_2/train.py                    # 跑全部 targets
python src_2/train.py --target popularity
```

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/{target}/best_model.pt` | 最佳 val loss checkpoint |
| `src_2/runs/{run_id}/{target}/target_scaler.json` | 正規化參數 |
| `src_2/runs/{run_id}/{target}/history.json` | 訓練曲線（含 run_id / notes） |
| `src_2/runs/{run_id}/{target}/final_metrics.json` | train + val 全量 metrics（含 run_id / notes） |

建置清單：
- [x] `src_2/fussion_training/meta_encoder.py`（56-dim）
- [x] `src_2/fussion_training/cross_attention.py`（RAGCrossAttention）
- [x] `src_2/fussion_training/dataset.py`（AnimeDataset，輸出 image[3,1024] + mask）
- [x] `src_2/fussion_training/model.py`（FusionModel v2，image/text/meta dim 從 config 讀取）
- [x] `src_2/fussion_configs.yaml`（image_mode / use_rag / targets / mixed_precision / notes）
- [x] `src_2/train.py`（訓練主程式）
- [x] `src_2/evaluate.py`（評估）

Code Review 修正紀錄（已套用）：

| 檔案 | 問題 | 修正 |
|------|------|------|
| `train.py` | SAM `group["rho"]` KeyError（AdamW param_groups 無此 key） | 改存 `self.rho` |
| `train.py` | SAM 繼承 `torch.optim.Optimizer`，新版 PyTorch `state` property setter 衝突 | 改為 pure wrapper class，`e_w` 存在 `self._e_w` dict |
| `train.py` | SAM `second_step` 的 `@torch.no_grad()` 包住 `base_opt.step()`，新版 PyTorch 報錯 | `second_step` 移除 decorator，手動 `with torch.no_grad()` 包權重還原 |
| `train.py` | `train_separately: false` 未實作但 config 存在 | 加 `NotImplementedError` guard |
| `train.py` | GradScaler + SAM 雙 `unscale_()` → RuntimeError | 移除 GradScaler，只用 `autocast`（gradient 仍 float32） |
| `train.py` | YAML `lr: 1e-3` 被解析為字串 | 改為 `0.001` 明確小數 |
| `dataset.py` | RAG 無命中 `rag_mask` 全 True → MultiheadAttention NaN | 強制 `rag_mask[0] = False` |
| `dataset.py` | `_load_emb_parquet` / `_load_image_emb_stack` 用 `iterrows()` 載入 → 極慢 | 改為向量化 `to_numpy()` + `.copy()` |
| `dataset.py` | `meta_encoder.transform()` 逐 sample 呼叫（N 次）→ 載入卡頓 | 改為一次批次 `transform(meta_df)`，再切分 dict |
| `dataset.py` | `_build_rag_meta_lookup` 重複 parse studios/genres → 慢 | 預先計算 train_lookup 所有欄位，`itertuples()` 取代 `iterrows()`，加 tqdm |
| `dataset.py` | `denormalize_target` 無 clip → 極端預測值 expm1 overflow | 加 `np.clip(y_norm, -5, 5)` |
| `evaluate.py` | `spearmanr().statistic`（scipy < 1.7 不支援） | 改用 `.correlation` |
| `evaluate.py` | metrics：RMSE 不適合 skewed 分佈 | 改為 `log_R2`（popularity）/ `R2`（meanScore）+ `log_MAE`（popularity only） |
| `evaluate.py` | metrics 欄位順序不一致 | 統一順序：spearman_rho → R2/log_R2 → MAE → log_MAE |
| `evaluate.py` | test metrics 存為獨立 `eval_test.json` | 改為 merge 進 `final_metrics.json`，單一檔案含 train/val/test |
| `dataset.py` | AMP float16 output → `denormalize_target` 中 `expm1` overflow → Infinity | 加 `np.asarray(y_norm, dtype=np.float64)` 強制轉型 |

### Step 8：評估與推論

```bash
# test set 評估（兩個 target 一起）
python src_2/evaluate.py --split test

# holdout 推論（無標籤，只輸出預測）
python src_2/evaluate.py --split holdout_unknown
```

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/{target}/final_metrics.json` | train / val / test 完整 metrics 合併於此（spearman_rho / R2 or log_R2 / MAE / log_MAE） |
| `src_2/runs/{run_id}/{target}/pred_{split}.csv` | id, pred, target（原始 scale） |

- [x] `src_2/evaluate.py`（spearman_rho / R2 or log_R2 / MAE / log_MAE，results merge 進 final_metrics.json）
- [x] Test set 評估完成（final_metrics.json 含 train / val / test）
- [x] Holdout 推論（pred_holdout_unknown.csv）

### Step 9：可解釋性分析

```bash
# RAG attention heatmap
python src_2/explain/rag_heatmap.py --target popularity --n 5

# Captum（modality 貢獻）+ SHAP（meta feature 貢獻）
python src_2/explain/feature_attr.py --target popularity --n 20
```

| 輸出 | 說明 |
|------|------|
| `runs/{run_id}/{target}/explain/rag/{id}_attn.png` | Cross Attention heatmap：x = retrieved anime，y = meta/text/image modality |
| `runs/{run_id}/{target}/explain/feature/captum_modality.csv` | 各 sample 的 modality 歸一化重要性 |
| `runs/{run_id}/{target}/explain/feature/captum_modality.png` | 平均 modality 重要性長條圖 |
| `runs/{run_id}/{target}/explain/feature/shap_values.npy` | raw SHAP values [n, 56] |
| `runs/{run_id}/{target}/explain/feature/shap_summary.png` | top-k meta feature 重要性（release_year, genre_Action 等） |

建置清單：
- [x] `src_2/fussion_training/cross_attention.py`（`return_attn=True` 回傳 `[batch, top_k, 3]` weights）
- [x] `src_2/fussion_training/model.py`（`forward(batch, return_attn=True)`）
- [x] `src_2/explain/rag_heatmap.py`（RAG attention heatmap）
- [x] `src_2/explain/feature_attr.py`（Captum IG + SHAP DeepExplainer）
- [x] `src_2/requirements.txt`

前置安裝（Step 9）：
```bash
pip install captum shap
```

---

## ⏳ 待完成

### Step 10：推論 Pipeline

給定一部新動畫（封面圖 + metadata + 描述），即時走完完整推論流程：

- [ ] YOLO crop（人物／臉部）
- [ ] Swin-B embedding（cover + banner + yolo）
- [ ] e5-base-v2 text embedding
- [ ] RAG query（Qdrant）
- [ ] FusionModel inference → popularity / meanScore 預測

---

## 資料流總覽

```
src_2/data/dataset/fusion_meta_clean_{split}_v2.csv
    │
    ├─ description ──→ TextEmbedder（e5-base-v2）──→ embedding/text/
    │
    ├─ coverImage ──→ YOLO crop ──→ Swin-B ──→ embedding/image/（yolo/cover/banner）
    │
    ├─ coverImage ──→ Swin-B（no YOLO）──→ embedding/image_rag/（RAG knowledge base）
    │
    ├─ metadata（56 cols）──→ MetaEncoder v2 ──→ [batch, 56]
    │
    └─ RAG query ──→ Qdrant ──→ retrieved_ids ──→ RAG/return/rag_features_{split}.parquet
```
