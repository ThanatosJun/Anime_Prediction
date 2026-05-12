# Fusion Branch

多模態融合預測模組。結合 text embedding、image embedding、metadata 與 RAG 檢索，預測動畫的 `popularity`（人氣）與 `meanScore`（評分）。

---

## 目錄結構

```
src/fussion_branch/
│
├── model/                        # Swin-base 微調 checkpoint（已 gitignore）
│   ├── best/                     # HuggingFace 格式（model.safetensors + config.json）
│   └── checkpoint/               # PyTorch epoch checkpoint（epoch_N.pt）
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
│   ├── meta_encoder.py           # MetaEncoder：metadata + RAG → float32 特徵向量
│   ├── model.py                  # FusionMLP（modality projection + backbone + head）
│   └── train.py                  # 訓練 loop、mixed precision、early stopping、評估
│
├── utilities/
│   ├── config.py                 # YAML config 載入 + run_id 自動遞增
│   ├── evaluate.py               # Spearman / R² / MAE / log_MAE（popularity only）
│   └── summarize_experiments.py  # 掃描所有 run，彙整成 experiments_summary.csv
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
│   ├── fusion_config.yaml        # MLP 訓練設定（data 路徑、model、training 超參數）
│   └── rag_config.yaml           # Qdrant / encoder / path 設定
│
├── run_text_embedding.py         # 產生 embedding/text/text_embeddings_{split}.parquet
├── run_image_embedding.py        # 產生 embedding/image/image_embeddings_{split}.parquet
├── run_rag.py                    # 建 Qdrant collection + 查詢所有 split
├── run_train.py                  # 訓練 popularity / meanScore 模型
├── run_evaluate.py               # 最終 test set 評估（訓練完成後執行一次）
└── run_shap.py                   # SHAP feature importance 分析（modality gate + meta 特徵）
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
# ※ 需要 src/fussion_branch/model/best/model.safetensors

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

# Step 5：test set 評估（訓練完成後執行一次）
python -m src.fussion_branch.run_evaluate
python -m src.fussion_branch.run_evaluate --run-id 02
python -m src.fussion_branch.run_evaluate --run-id 02 --target meanScore

# 彙整所有實驗結果
python -m src.fussion_branch.utilities.summarize_experiments
# → .exp/fussion/experiments_summary.csv

# Step 6（選用）：SHAP feature importance 分析
python -m src.fussion_branch.run_shap --target popularity
python -m src.fussion_branch.run_shap --target meanScore
# → .exp/fussion/results/{run_id}/{target}/shap/
#     modality_importance.json   — 各 modality 貢獻佔比
#     meta_bar.png               — top-20 meta 特徵 bar chart
#     meta_beeswarm.png          — 特徵值高低 vs SHAP 方向
```

> Step 1 若跳過，Step 3 自動退回 **sparse-only** 模式。
> Step 2 若跳過，FusionDataset 自動補零向量（1024-dim）。

---

## 實驗管理

### 自動 run_id 遞增

`run_train.py` 執行時，若 config 的 `run_id` 對應資料夾已存在，會自動遞增到下一個可用編號並印出提示：

```
[run_id] '01' already exists → using '02'
```

### 切換資料集（全量 vs post-2000）

`fusion_config.yaml` 中 comment / uncomment 對應行即可，無需修改程式碼：

```yaml
# 全量資料集
# fusion_meta_dir:   data/fussion
# meta_encoder_path: .exp/fussion/meta_encoder.json

# 2000 年後資料集
fusion_meta_dir:   data/fussion/post2000
meta_encoder_path: .exp/fussion/meta_encoder_post2000.json
```

post-2000 版本的 CSV 位於 `data/fussion/post2000/`，text / image / RAG embedding **不需重新產生**（FusionDataset 以 `id` 取交集，自動過濾）。

---

## 模型架構

### FusionMLP

```
text_emb  (384)  ──→ text_proj  (Linear→LN→GELU, 128) ──→ × α_text  ─┐
image_emb (1024) ──→ image_proj (Linear→LN→GELU, 256) ──→ × α_image ──┤→ concat(448) → backbone → head
meta_rag   (65)  ──→ meta_proj  (Linear→LN→GELU,  64) ──→ × α_meta  ─┘

Modality Gate（各自獨立，語意對應）：
  α_text  = softmax( [Linear(128→1)(t), Linear(256→1)(img), Linear(64→1)(m)] )[0]
  α_image = softmax( ... )[1]
  α_meta  = softmax( ... )[2]

backbone: Dropout → [Linear→LN→GELU→Dropout] × 3（256→128→64）
head:     Linear(64→1)
```

**設計重點：**
- 各模態先獨立 projection，解決 image（1024-dim）對梯度的主導問題
- Modality Gate：每個 gate 只看自己的 projection（語意對應有保證），softmax 確保三者加總 = 1
- Gate 是 input-dependent：每部動畫動態決定哪個 modality 較重要，而非固定比例
- LayerNorm 取代 BatchNorm（inference 時行為和 training 一致）
- GELU 取代 ReLU（平滑梯度）
- Projection 層無 Dropout（參數量小，主要正則化在 backbone）

**Gate 查看（inference）：**
```python
gates = model.get_gates(x.to(device)).mean(dim=0)
print(f"text={gates[0]:.3f}  image={gates[1]:.3f}  meta={gates[2]:.3f}")
```

### 訓練設定

| 項目 | 設定 |
|------|------|
| Loss | HuberLoss（delta=1.0，對離群值穩健） |
| Optimizer | AdamW（lr=5e-4, weight_decay=1e-3） |
| LR Schedule | warmup 5 epochs → ReduceLROnPlateau（patience=5, factor=0.5） |
| Early stopping | patience=20 |
| Mixed precision | FP16 autocast + GradScaler（RTX GPU） |

---

## 評估指標

| 指標 | 說明 | 適用 |
|------|------|------|
| Spearman ρ | 排名相關性（主要指標）| 兩個 target |
| R² | 解釋變異量（診斷 distribution shift）| 兩個 target |
| MAE | 原始尺度平均絕對誤差 | 兩個 target |
| log_MAE | log1p 空間 MAE（scale-free，對應訓練目標）| popularity only |

**popularity 特別說明：** 採用 `log1p` + z-score 訓練，AMP 模式下 model 輸出為 float16；`denormalize()` 強制轉 float64 並在 normalized 空間 clip ±5σ 後再執行 `expm1`，避免數值溢位（float16 max ≈ 65504）。

---

## 特徵維度

| 來源 | 維度 | 說明 |
|------|------|------|
| Text embedding | 384 | all-MiniLM-L6-v2，description |
| Image embedding | 1024 | Swin-base pooler_output（缺失補零）|
| MetaEncoder | 65 | 見下表 |
| **合計** | **1473** | |

### MetaEncoder 特徵明細（65 dims）

| 類型 | 欄位 | 維度 |
|------|------|------|
| 標準化 | release_year, episodes, duration, startDate_day, prequel_count, prequel_meanScore_mean | 6 |
| log1p + 標準化 | prequel_popularity_mean | 1 |
| Cyclical sin/cos | release_quarter（period=4）, startDate_month（period=12）| 4 |
| One-hot | format（7）, source（7）, countryOfOrigin（4）| 18 |
| Binary | isAdult, is_sequel, has_sequel | 3 |
| Multi-hot | genres（19 類）| 19 |
| Studio Target Encoding | 此動畫製作公司的歷史 mean_log1p_popularity, mean_score | 2 |
| Voice Actor Target Encoding | 此動畫聲優群的歷史 mean_log1p_popularity, mean_score | 2 |
| RAG 標準化 | rag_popularity（log1p）, rag_score, rag_release_year, rag_episodes | 4 |
| RAG binary | rag_found | 1 |
| Overlap Scalar | studio_match（binary）, genre_overlap（Jaccard）, format_match（binary）| 3 |
| RAG Studio Target Encoding | RAG 結果製作社 mean_popularity, mean_score | 2 |

### Target Encoding 說明

**Studio / Voice Actor TE：**
fit 階段從訓練集統計每個製作公司 / 聲優的歷史 `mean_log1p_popularity` 和 `mean_score`（popularity 先 log1p 再平均，與訓練目標單位一致），transform 時以該動畫的 studio 或 voice_actor 查表並取平均，最後 z-score 標準化。
未見過的 studio / va → 訓練集全體均值（標準化後 ≈ 0）。

`rag_popularity` 同樣先做 log1p 再標準化，與 meta 的 popularity 單位對齊。

**Overlap Scalar：**
| 欄位 | 計算方式 |
|------|---------|
| `studio_match` | meta studios ∩ RAG studios 有交集 → 1，否則 → 0 |
| `genre_overlap` | \|meta genres ∩ RAG genres\| / \|meta genres ∪ RAG genres\|（Jaccard） |
| `format_match` | meta format == RAG format → 1，否則 → 0 |

---

## 訓練目標轉換

| Target | 轉換 | 反轉 |
|--------|------|------|
| `popularity` | `log1p` → z-score | 反標準化 → `expm1` |
| `meanScore` | z-score（直接）| 反標準化 |

mean/std 僅從訓練集計算，再套用到 val/test。

---

## 輸出檔案

```
.exp/fussion/
├── meta_encoder.json                    ← 全量資料集的 MetaEncoder
├── meta_encoder_post2000.json           ← post-2000 資料集的 MetaEncoder
├── experiments_summary.csv              ← 所有 run 的 metrics 彙整
└── results/{run_id}/{target}/
    ├── best_model.pt        ← 最佳 val checkpoint（state dict）
    ├── model_config.json    ← 架構參數（用於 inference 重建模型）
    ├── target_scaler.json   ← 標準化參數（mean, std, log_transform）
    ├── training_log.jsonl   ← 每 epoch 的 train_loss / val_MAE / lr
    ├── metrics_val.json     ← 最終 val 評估指標
    ├── metrics_test.json    ← 最終 test 評估指標
    └── shap/                ← run_shap.py 產生
        ├── modality_importance.json
        ├── meta_bar.png
        └── meta_beeswarm.png
```

---

## RAG 查詢模式

| 條件 | 查詢方式 |
|------|---------|
| text_embeddings 存在 | **Hybrid**：sparse + dense → RRF fusion |
| text_embeddings 不存在 | **Sparse-only**：genre+studio+voice_actor+source |

- Sparse 向量：genre、studio、voice_actor、source token
- 時間過濾：Qdrant server-side filter（`release_year/quarter < target` + self-exclusion）
- 結果：top-1 的 payload 提取 popularity、score、studios 等數值

---

## 已知限制

### 1. meanScore 時序 Distribution Shift

資料採時序切分（train → val → test），而 AniList 社群的評分中位數隨時間系統性上升：

| Split | 年份範圍 | meanScore 中位數 |
|-------|---------|----------------|
| Train | –2018 | 60 |
| Val | 2018–2022 | 62 |
| Test | 2022–2026 | 66 |

**2022 年後出現約 +4 的跳升**，主要原因推測為：
- AniList 社群於 2020–2022 爆發性成長，新用戶評分行為與舊用戶不同
- 季播文化：邊看邊評分的觀眾對當季作品給分較寬鬆
- 近期作品評分尚未穩定（資料截止 2026 年 4 月，高峰期尚未退去）

此現象導致 test 集 R² 偏低（~0.077），為本模型的固有限制。

可視化圖表：`.exp/fussion/meanscore_distribution_over_time.png`

### 2. popularity AMP 數值溢位

`popularity` 採用 `log1p` + z-score 訓練，FP16 AMP 模式下模型輸出為 float16（max ≈ 65504）。若直接執行 `expm1`，在極端預測值下會產生 `Inf`。

**解法**：`denormalize()` 在反標準化前強制轉換為 float64，並在 normalized 空間 clip ±5σ，再執行 `expm1`。

### 3. Pre-release 特徵限制

模型只能使用播出前已知的特徵（封面圖、類型、聲優、製作公司、集數等），無法使用 `averageScore`、`favourites`、`trending` 等播出後才產生的數據。這從根本上限制了預測上限，尤其對 meanScore 的預測精度影響顯著。

---

## 資料來源

| 資料 | 路徑 |
|------|------|
| Fusion metadata（全量）| `data/fussion/fusion_meta_clean_{split}.csv` |
| Fusion metadata（post-2000）| `data/fussion/post2000/fusion_meta_clean_{split}.csv` |
| Text embedding | `src/fussion_branch/embedding/text/text_embeddings_{split}.parquet` |
| Image embedding | `src/fussion_branch/embedding/image/image_embeddings_{split}.parquet` |
| RAG features | `src/fussion_branch/RAG/return/rag_features_{split}.parquet` |
| Swin-base checkpoint | `src/fussion_branch/model/best/` |
| MetaEncoder（全量）| `.exp/fussion/meta_encoder.json` |
| MetaEncoder（post-2000）| `.exp/fussion/meta_encoder_post2000.json` |
| 訓練 checkpoint | `.exp/fussion/results/{run_id}/{target}/` |
| 實驗統計 | `.exp/fussion/experiments_summary.csv` |
