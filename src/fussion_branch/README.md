# Fusion Branch

多模態融合預測模組。結合 text embedding、image embedding、metadata 與 RAG 檢索，預測動畫的 `popularity`（人氣）與 `meanScore`（評分）。

---

## 目錄結構

```
src/fussion_branch/
│
├── RAG/                          # RAG pipeline（Qdrant sparse + dense hybrid search）
│   ├── sparse_encoder.py         # genre + studio → sparse vector 詞表
│   ├── rag_builder.py            # 建立 Qdrant collection（訓練集 indexing）
│   ├── rag_query.py              # 查詢 Qdrant，產出 rag_features_{split}.parquet
│   ├── meta_encoder.py           # fusion metadata → 155-dim float32 feature vector
│   ├── sparse_encoder.json       # ← run_rag 產生
│   └── rag_features_{split}.parquet  # ← run_rag 產生
│       text_embeddings_{split}.parquet  # ← run_text_embedding 產生
│
├── utilities/
│   ├── config.py                 # YAML config 載入 helper
│   └── evaluate.py               # MAE / RMSE / Spearman / MAPE / log_MAE / bucket_accuracy
│
├── fussion_training/             # MLP 訓練 pipeline
│   ├── dataset.py                # FusionDataset（組合 text + image + meta_rag → tensor）
│   ├── model.py                  # FusionMLP（Linear → BN → ReLU → Dropout × 3）
│   └── train.py                  # 訓練 loop、early stopping、最終評估
│
├── text_components/              # text_branch 本地副本（避免跨 branch import）
│   ├── text_preprocessor.py      # URL 清除、長度過濾等文字清洗
│   ├── embedding_generator.py    # SentenceTransformer 封裝
│   └── embedding_config.yaml     # 模型名稱、batch size、preprocessing 參數
│
├── image_components/             # image_branch 本地副本
│   ├── image_process.py          # ResizeWithPad、get_transform_original
│   └── image_process_config.yaml
│
├── text_embedding.py             # TextEmbedder（inference 用 wrapper）
├── image_embedding.py            # ImageEmbedder（inference 用 wrapper，Swin-base 1024-dim）
│
├── run_text_embedding.py         # 執行入口：產生 RAG/text_embeddings_{split}.parquet
├── run_rag.py                    # 執行入口：建 Qdrant + 查詢所有 split
└── run_train.py                  # 執行入口：訓練 popularity / meanScore 模型
```

---

## 執行順序

```bash
conda activate animeprediction

# Step 1：產生 text embedding（hybrid search 前置條件）
python -m src.fussion_branch.run_text_embedding
# → src/fussion_branch/RAG/text_embeddings_{train,val,test}.parquet

# Step 2：建 Qdrant collection + RAG 查詢（sparse-only 或 hybrid）
python -m src.fussion_branch.run_rag
# → src/fussion_branch/RAG/sparse_encoder.json
# → src/fussion_branch/RAG/rag_features_{train,val,test}.parquet

# Step 3：訓練 MLP（兩個 target 各自獨立）
python -m src.fussion_branch.run_train
python -m src.fussion_branch.run_train --target popularity
python -m src.fussion_branch.run_train --target meanScore
# → .exp/fussion/results/{run_id}/{target}/best_model.pt
```

> Step 1 若跳過，Step 2 自動退回 **sparse-only** 模式（genre + studio）。

---

## 特徵維度

| 來源 | 維度 | 說明 |
|------|------|------|
| Text embedding | 384 | all-MiniLM-L6-v2，description |
| Image embedding | 1024 | Swin-base pooler_output，coverImage（缺失補零）|
| MetaEncoder | 155 | 見下表 |
| **合計** | **1563** | |

### MetaEncoder 155-dim 明細

| 類型 | 欄位 | 維度 |
|------|------|------|
| 標準化 | release_year, episodes, duration, startDate_day, prequel_count, prequel_meanScore_mean | 6 |
| log1p + 標準化 | prequel_popularity_mean | 1 |
| Cyclical sin/cos | release_quarter（period=4）, startDate_month（period=12）| 4 |
| One-hot | format（7）, source（7）, countryOfOrigin（4）| 18 |
| Binary | isAdult, is_sequel, has_sequel | 3 |
| Multi-hot | genres（19 類）| 19 |
| Multi-hot | studios（top-50）| 50 |
| RAG 標準化 | rag_popularity, rag_score, rag_release_year | 3 |
| RAG binary | rag_found | 1 |
| RAG multi-hot | rag_studios（同 studio vocab）| 50 |

---

## RAG 查詢模式

| 條件 | 查詢方式 |
|------|---------|
| text_embeddings 存在 | **Hybrid**：sparse（genre+studio）+ dense（text 384-dim）→ RRF fusion |
| text_embeddings 不存在 | **Sparse-only**：genre+studio dot product |

查詢後 Python 端 post-filter：`release_year < target anime release_year`，取 top-1 作為 RAG 輔助特徵。

---

## 資料來源

| 資料 | 路徑 |
|------|------|
| Fusion metadata（23 欄）| `data/fussion/fusion_meta_clean_{split}.csv` |
| Text embedding | `src/fussion_branch/RAG/text_embeddings_{split}.parquet` |
| Image embedding | `.exp/image_embedding/image_embeddings.parquet` |
| RAG features | `src/fussion_branch/RAG/rag_features_{split}.parquet` |
| MetaEncoder | `.exp/fussion/meta_encoder.json` |
| 訓練 checkpoint | `.exp/fussion/results/{run_id}/{target}/` |
