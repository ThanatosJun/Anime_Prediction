# Fusion Branch

多模態融合預測模組。結合 text embedding、image embedding、metadata 與 RAG 檢索，預測動畫的 `popularity`（人氣）與 `meanScore`（評分）。

---

## 目錄結構

```
src/fussion_branch/
│
├── RAG/                          # RAG pipeline（Qdrant sparse + dense hybrid search）
│   ├── sparse_encoder.py         # genre + studio + voice_actor + source → sparse vector 詞表
│   ├── rag_builder.py            # 建立 Qdrant collection（訓練集 indexing）
│   ├── rag_query.py              # 查詢 Qdrant，批次產出 rag_features_{split}.parquet
│   ├── rag_query_single.py       # 單筆 inference 查詢（pre-release 預測用）
│   ├── sparse_encoder.json       # ← run_rag 產生
│   └── return/                   # ← run_rag 產生（已 gitignore）
│       └── rag_features_{split}.parquet
│
├── embedding/                    # embedding 產出（已 gitignore）
│   ├── text/                     # ← run_text_embedding 產生
│   │   └── text_embeddings_{split}.parquet
│   └── image/                    # ← run_image_embedding 產生
│       └── image_embeddings_{split}.parquet
│
├── fussion_training/             # MLP 訓練 pipeline
│   ├── dataset.py                # FusionDataset（組合 text + image + meta_rag → tensor）
│   ├── meta_encoder.py           # MetaEncoder：metadata + RAG → 232-dim float32
│   ├── model.py                  # FusionMLP（modality projection + backbone + head）
│   └── train.py                  # 訓練 loop、mixed precision、early stopping、評估
│
├── utilities/
│   ├── config.py                 # YAML config 載入 helper
│   └── evaluate.py               # MAE / RMSE / Spearman / MAPE / log_MAE / bucket_accuracy
│
├── text_components/              # text_branch 本地副本
│   ├── text_preprocessor.py
│   ├── embedding_generator.py
│   └── embedding_config.yaml
│
├── image_components/             # image_branch 本地副本
│   ├── image_process.py
│   └── image_process_config.yaml
│
├── text_embedding.py             # TextEmbedder（inference 用 wrapper）
├── image_embedding.py            # ImageEmbedder（Swin-base 1024-dim）
│
├── configs/
│   ├── fusion_config.yaml        # MLP 訓練設定
│   └── rag_config.yaml           # Qdrant / encoder / path 設定
│
├── run_meta_preprocess.py        # 從 data/processed/ 產生 fusion_meta_clean_{split}.csv
├── run_text_embedding.py         # 產生 embedding/text/text_embeddings_{split}.parquet
├── run_image_embedding.py        # 產生 embedding/image/image_embeddings_{split}.parquet
├── run_rag.py                    # 建 Qdrant collection + 查詢所有 split
└── run_train.py                  # 訓練 popularity / meanScore 模型
```

---

## 前置條件：Qdrant Docker

RAG pipeline 使用 Qdrant **server 模式**（payload index 需要 server 模式才能生效）。

### 首次啟動

```bash
docker run -d \
  -p 6333:6333 \
  -v $(pwd)/qdrant_db:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant
```

確認啟動成功：
```bash
curl http://localhost:6333/healthz
# 回傳 healthz check passed 代表正常
```

### 後續重啟 / 停止

```bash
docker start qdrant
docker stop qdrant
```

> `qdrant_db/` 已加入 `.gitignore`。每次執行 `run_rag.py` 自動重建 collection。

---

## 執行順序

```bash
conda activate animeprediction

# 前置：確認 Qdrant 已啟動
docker start qdrant

# Step 1：產生 text embedding（hybrid search 前置條件）
python -m src.fussion_branch.run_text_embedding
# → src/fussion_branch/embedding/text/text_embeddings_{train,val,test}.parquet

# Step 2（選用）：產生 image embedding
python -m src.fussion_branch.run_image_embedding
# → src/fussion_branch/embedding/image/image_embeddings_{train,val,test,holdout_unknown}.parquet

# Step 3：RAG query（只需重跑 query，不需重建 collection）
python -c "from src.fussion_branch.RAG.rag_query import query_all_splits; query_all_splits()"
# → src/fussion_branch/RAG/return/rag_features_{train,val,test}.parquet

# 或完整重建 collection + query：
python -m src.fussion_branch.run_rag

# Step 4：訓練 MLP
python -m src.fussion_branch.run_train                      # 讀 config 的 active_targets
python -m src.fussion_branch.run_train --target popularity  # CLI 覆蓋
python -m src.fussion_branch.run_train --target meanScore
# → .exp/fussion/results/{run_id}/{target}/
#     best_model.pt / model_config.json / target_scaler.json
#     training_log.jsonl / metrics_val.json / metrics_test.json
```

> Step 1 若跳過，Step 3 自動退回 **sparse-only** 模式。
> Step 2 若跳過，FusionDataset 自動補零向量（1024-dim）。

---

## 模型架構

### FusionMLP

```
text_emb  (384)  ──→ text_proj  (Linear→LN→GELU, 128-dim) ─┐
image_emb (1024) ──→ image_proj (Linear→LN→GELU, 256-dim) ──┤→ concat(512) → backbone → head
meta_rag  (232)  ──→ meta_proj  (Linear→LN→GELU, 128-dim) ─┘

backbone: Dropout → [Linear→LN→GELU→Dropout] × 3（512→256→128）
head:     Linear(128→1)
```

**設計重點：**
- 各模態先獨立 projection，解決 image（1024-dim）對梯度的主導問題
- LayerNorm 取代 BatchNorm（inference 時行為和 training 一致）
- GELU 取代 ReLU（平滑梯度）

### 訓練設定

| 項目 | 設定 |
|------|------|
| Loss | HuberLoss（delta=1.0，對離群值穩健） |
| Optimizer | AdamW（lr=1e-3, weight_decay=1e-4） |
| LR Schedule | warmup 5 epochs → ReduceLROnPlateau（patience=5, factor=0.5） |
| Early stopping | patience=20 |
| Mixed precision | FP16 autocast + GradScaler（RTX GPU） |

---

## 特徵維度

| 來源 | 維度 | 說明 |
|------|------|------|
| Text embedding | 384 | all-MiniLM-L6-v2，description |
| Image embedding | 1024 | Swin-base pooler_output（缺失補零）|
| MetaEncoder | 232 | 見下表 |
| **合計** | **1640** | |

### MetaEncoder 232-dim 明細

| 類型 | 欄位 | 維度 |
|------|------|------|
| 標準化 | release_year, episodes, duration, startDate_day, prequel_count, prequel_meanScore_mean | 6 |
| log1p + 標準化 | prequel_popularity_mean | 1 |
| Cyclical sin/cos | release_quarter（period=4）, startDate_month（period=12）| 4 |
| One-hot | format（7）, source（7）, countryOfOrigin（4）| 18 |
| Binary | isAdult, is_sequel, has_sequel | 3 |
| Multi-hot | genres（19 類）| 19 |
| Multi-hot | studios（top-50）| 50 |
| Multi-hot | voice_actor_names（top-50，缺值補零）| 50 |
| RAG 標準化 | rag_popularity, rag_score, rag_release_year, rag_episodes | 4 |
| RAG binary | rag_found | 1 |
| RAG multi-hot | rag_studios（同 studio vocab）| 50 |
| RAG multi-hot | rag_genres（同 genre vocab）| 19 |
| RAG one-hot | rag_format（同 format vocab）| 7 |

---

## 訓練目標轉換

| Target | 轉換 | 反轉 |
|--------|------|------|
| `popularity` | `log1p` → z-score | `expm1` ← 反標準化 |
| `meanScore` | z-score（直接）| 反標準化 |

mean/std 僅從訓練集計算，再套用到 val/test。

---

## 輸出檔案

```
.exp/fussion/results/{run_id}/{target}/
├── best_model.pt        ← 最佳 val checkpoint（state dict）
├── model_config.json    ← 架構參數（用於 inference 重建模型）
├── target_scaler.json   ← 標準化參數（mean, std, log_transform）
├── training_log.jsonl   ← 每 epoch 的 train_loss / val_MAE / lr
├── metrics_val.json     ← 最終 val 評估指標
└── metrics_test.json    ← 最終 test 評估指標
```

---

## RAG 查詢模式

| 條件 | 查詢方式 |
|------|---------|
| text_embeddings 存在 | **Hybrid**：sparse + dense → RRF fusion |
| text_embeddings 不存在 | **Sparse-only**：genre+studio+voice_actor+source |

Sparse 向量：genre、studio、voice_actor、source token
時間過濾：Qdrant server-side filter（`release_year/quarter < target` + self-exclusion）
結果：取 top-1 作為 RAG 輔助特徵

---

## 資料來源

| 資料 | 路徑 |
|------|------|
| Fusion metadata（25 欄）| `data/fussion/fusion_meta_clean_{split}.csv` |
| Text embedding | `src/fussion_branch/embedding/text/text_embeddings_{split}.parquet` |
| Image embedding | `src/fussion_branch/embedding/image/image_embeddings_{split}.parquet` |
| RAG features | `src/fussion_branch/RAG/return/rag_features_{split}.parquet` |
| MetaEncoder | `.exp/fussion/meta_encoder.json` |
| 訓練 checkpoint | `.exp/fussion/results/{run_id}/{target}/` |
