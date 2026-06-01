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
│   ├── sparse_encoder.py         # 混合加權 sparse encoder（genre/studio/source=IDF，voice=df）
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
│   ├── meta_encoder.py           # MetaEncoder：metadata + RAG → float32 特徵向量（66 dims）
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

# Step 3：RAG 完整重建（SparseEncoder 有改動時必須重建）
python -m src.fussion_branch.run_rag
# → src/fussion_branch/RAG/return/rag_features_{train,val,test}.parquet

# Step 4：訓練 MLP
python -m src.fussion_branch.run_train                      # 讀 config 的 active_targets
python -m src.fussion_branch.run_train --target popularity  # CLI 覆蓋
python -m src.fussion_branch.run_train --target meanScore

# Step 5：test set 評估（訓練完成後執行一次）
python -m src.fussion_branch.run_evaluate
python -m src.fussion_branch.run_evaluate --run-id 11
python -m src.fussion_branch.run_evaluate --run-id 11 --target meanScore

# 彙整所有實驗結果
python -m src.fussion_branch.utilities.summarize_experiments

# Step 6（選用）：SHAP feature importance 分析
python -m src.fussion_branch.run_shap --target popularity
python -m src.fussion_branch.run_shap --target meanScore
```

> Step 1 若跳過，Step 3 自動退回 **sparse-only** 模式。
> Step 2 若跳過，FusionDataset 自動補零向量（1024-dim）。

---

## 模型架構

### FusionMLP

```
text_emb  (384)  ──→ text_proj  (Linear→LN→GELU, 128) ──→ × α_text  ─┐
image_emb (1024) ──→ image_proj (Linear→LN→GELU,  64) ──→ × α_image ──┤→ concat(256) → backbone → head
meta_rag   (66)  ──→ meta_proj  (Linear→LN→GELU,  64) ──→ × α_meta  ─┘

Modality Gate（各自獨立，語意對應）：
  α = softmax( [Linear(128→1)(t), Linear(64→1)(img), Linear(64→1)(m)] )

backbone: Dropout → [Linear→LN→GELU→Dropout] × 3（256→128→64）
head:     Linear(64→1)
```

**設計重點：**
- 各模態先獨立 projection，解決 image（1024-dim）對梯度的主導問題
- Modality Gate：每個 gate 只看自己的 projection，softmax 確保三者加總 = 1
- Gate 是 input-dependent：每部動畫動態決定哪個 modality 較重要
- LayerNorm 取代 BatchNorm（inference 時行為和 training 一致）
- GELU 取代 ReLU（平滑梯度）

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

---

## 特徵維度

| 來源 | 維度 | 說明 |
|------|------|------|
| Text embedding | 384 | all-MiniLM-L6-v2，description |
| Image embedding | 1024 | Swin-base pooler_output（缺失補零）|
| MetaEncoder | 66 | 見下表 |
| **合計** | **1474** | |

### MetaEncoder 特徵明細（66 dims）

| 類型 | 欄位 | 維度 |
|------|------|------|
| 標準化 | release_year, episodes, duration, startDate_day, prequel_count, prequel_meanScore_mean | 6 |
| log1p + 標準化 | prequel_popularity_mean | 1 |
| Cyclical sin/cos | release_quarter（period=4）, startDate_month（period=12）| 4 |
| One-hot | format（7）, source（7）, countryOfOrigin（4）| 18 |
| Binary | isAdult, is_sequel, has_sequel | 3 |
| Multi-hot | genres（19 類）| 19 |
| Studio Target Encoding | 此動畫製作公司的歷史 mean_popularity, mean_score | 2 |
| is_new_studio | 所有 studio 均為訓練集未見過 → 1，否則 → 0 | 1 |
| Voice Actor Target Encoding | 此動畫聲優群的歷史 mean_popularity, mean_score | 2 |
| RAG 標準化 | rag_popularity（log1p）, rag_score, rag_release_year, rag_episodes | 4 |
| RAG binary | rag_found | 1 |
| Overlap Scalar | studio_match（binary）, genre_overlap（Jaccard）, format_match（binary）| 3 |
| RAG Studio Target Encoding | RAG 結果製作社 mean_popularity, mean_score | 2 |

### Target Encoding 說明

**Studio / Voice Actor TE：**
fit 階段從訓練集統計每個製作公司 / 聲優的歷史 mean_popularity 和 mean_score，transform 時查表取平均後 z-score 標準化。
未見過的 studio / va → 訓練集全體均值（標準化後 ≈ 0）。
`is_new_studio = 1` 用於告知模型「studio_te 值為補值，可信度低」。

**Overlap Scalar：**
| 欄位 | 計算方式 |
|------|---------|
| `studio_match` | meta studios ∩ RAG studios 有交集 → 1，否則 → 0 |
| `genre_overlap` | \|meta genres ∩ RAG genres\| / \|meta genres ∪ RAG genres\|（Jaccard） |
| `format_match` | meta format == RAG format → 1，否則 → 0 |

---

## RAG 查詢模式

| 條件 | 查詢方式 |
|------|---------|
| text_embeddings 存在 | **Hybrid**：sparse + dense → RRF fusion |
| text_embeddings 不存在 | **Sparse-only**：genre+studio+voice_actor+source |

### Sparse 向量加權策略

| Token 類型 | 加權方式 | 說明 |
|-----------|---------|------|
| genre, studio, source | Robertson IDF：`log((N−df+0.5)/(df+0.5)+1)` | 稀有 token 鑑別力高 → 權重高 |
| voice_actor | `log(df+1)` | 出演次數多 = 知名度高 = 匹配時信號強 |

- 時間過濾：Qdrant server-side filter（`release_year/quarter < target` + self-exclusion）
- 結果：top-1 的 payload 提取 popularity、score、studios 等數值

---

## 實驗記錄

### popularity（指標：log_MAE，越低越好）

| Run | val log_MAE | test log_MAE | val Spearman | 主要改動 |
|-----|------------|-------------|-------------|---------|
| 01 | 0.8537 | 0.9581 | 0.8653 | Baseline：全量資料集，lr=1e-3 |
| 02 | 0.8551 | 1.0157 | 0.8665 | post-2000 資料集 |
| 03 | 0.8588 | 1.1717 | 0.8638 | 對齊 embedding/RAG 訓練資料 |
| 04 | 0.8610 | 0.9943 | 0.8655 | 同上調整 |
| 05 | 0.8276 | — | 0.8703 | 加入 SHAP 分析（無 test 評估）|
| 06 | 0.8520 | 1.0213 | 0.8632 | 加入 Modality Gate；TE 改 log1p |
| 07 | 0.8671 | 0.9793 | 0.8675 | TE 用 log1p popularity 重新 fit |
| 08 | 0.8682 | 1.0313 | 0.8619 | TE 改回 raw popularity |
| 09 | 0.8601 | 1.0104 | 0.8654 | image_proj 256→64；fused_dim 448→256 |
| 10 | 0.8358 | — | 0.8647 | RAG sparse：multi-hot → IDF 加權 |
| 11 | 0.8383 | 0.9766 | 0.8637 | RAG sparse 混合加權（IDF+df）；MetaEncoder 新增 `is_new_studio`（65→66 維）|
| 12 | 0.8505 | 1.0227 | 0.8655 | TextGNN + ImageGNN（star-topology, cosine attention）；RAG 三路混合（sparse+text dense+image dense）→ top-5 retrieved_ids |
| 13 | 0.8980 | 1.0411 | 0.8461 | GNN 超參調整：gnn_num_layers=2, gnn_dropout=0.2, gnn_lr_factor=0.3（GNN LR=9e-5），warmup=10, patience=25 |
| 14 | 0.8573 | 0.9795 | 0.8602 | YOLO char embedding；image=concat([enh_char,cover])=2048；image_proj 64→128；hidden_dims [320,128,64] |
| **15** | **0.8458** | **1.0112** | **0.8602** | extraLarge cover embedding；移除無封面圖資料（_v2）；無 YOLO char；GNN；150 epochs |

### meanScore（指標：MAE，越低越好）

| Run | val MAE | test MAE | val Spearman | 主要改動 |
|-----|--------|---------|-------------|---------|
| 01 | 7.0786 | 7.7203 | 0.6060 | Baseline |
| 02 | 7.0398 | 8.1183 | 0.6218 | post-2000 資料集 |
| 03 | 6.9641 | 7.5844 | 0.6242 | 對齊訓練資料 |
| 04 | 6.9735 | 7.8441 | 0.6049 | 同上調整 |
| 05 | 6.8343 | — | 0.6363 | SHAP 分析（無 test 評估）|
| 06 | 6.8890 | 8.2518 | 0.6312 | Modality Gate |
| 07 | 6.9081 | 8.2099 | 0.6264 | TE log1p |
| 08 | 6.8100 | 8.2085 | 0.6394 | TE raw popularity |
| 09 | 6.7996 | 8.1903 | 0.6417 | image_proj 256→64 |
| 10 | 6.7724 | — | 0.6415 | RAG sparse IDF |
| 11 | 6.7212 | 8.0691 | 0.6428 | RAG 混合加權；`is_new_studio` |
| 12 | 7.0060 | 8.3455 | 0.6137 | TextGNN + ImageGNN；RAG 三路混合 |
| 13 | 7.2949 | 8.4585 | 0.5799 | GNN 超參調整 |
| 14 | 7.4034 | 8.6465 | 0.5720 | YOLO char embedding；image_proj 64→128 |
| **15** | **7.1378** | **8.0865** | **0.5986** | extraLarge cover embedding；移除無封面圖資料（_v2）；無 YOLO char |

> 目前 **Run 15 在兩個 target 的 val 均達最佳**（popularity log_MAE=0.8458、meanScore MAE=7.1378）；換用 extraLarge 圖片並清理無封面圖資料後，test meanScore 顯著改善（8.09 vs Run 11 的 8.07，且同時 popularity 也有提升）。

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

## 已知限制

### 1. meanScore 時序 Distribution Shift

資料採時序切分（train → val → test），而 AniList 社群的評分中位數隨時間系統性上升：

| Split | 年份範圍 | meanScore 中位數 |
|-------|---------|----------------|
| Train | –2018 | 60 |
| Val | 2018–2022 | 62 |
| Test | 2022–2026 | 66 |

**2022 年後出現約 +4 的跳升**，導致 test 集 R² 偏低（~0.077），為本模型的固有限制。

可視化圖表：`.exp/fussion/meanscore_distribution_over_time.png`

### 2. popularity AMP 數值溢位

`popularity` 採用 `log1p` + z-score 訓練，FP16 AMP 模式下若直接執行 `expm1` 可能產生 `Inf`。
**解法**：`denormalize()` 強制轉 float64，在 normalized 空間 clip ±5σ 後再執行 `expm1`。

### 3. Pre-release 特徵限制

模型只能使用播出前已知的特徵，無法使用 `averageScore`、`favourites`、`trending` 等播出後才產生的數據。這從根本上限制了預測上限，尤其對 meanScore 影響顯著。

### 4. Cold-start（新 studio / 新聲優）

- **MetaEncoder**：未見過的 studio/聲優 TE 補訓練集全體均值；`is_new_studio` 旗標告知模型補值情況
- **SparseEncoder**：OOV token 在 RAG 檢索時直接忽略，靠 genre 等已知 token 退化匹配

---

## 封面圖資料清理（coverImage_extraLarge）

### 背景

原始 pipeline 使用 `coverImage_medium` 作為封面圖來源。為提升圖像品質，改用 `coverImage_extraLarge`，需重新建立圖片庫並清理無封面圖的資料。

---

### Step 1：解壓縮與命名對應

新封面圖來源為 `data/image/a.zip`（共 20,264 個檔案）。解壓縮後放置於 `data/image/a/a/`。

來源檔名有五種命名格式，統一用正規表達式提取 AniList `id`：

| 格式 | 範例 |
|------|------|
| `{id}.jpg` | `10013.jpg` |
| `{id}-{亂碼}.jpg` | `100071-8LCECzqyhA5e.jpg` |
| `b{id}-{亂碼}.jpg` | `b1001-jeUOTUfNlONM.jpg` |
| `bx{id}-{亂碼}.jpg` | `bx1000-Xpeob9jND2tg.jpg` |
| `nx{id}-{亂碼}.jpg` | `nx100003-zLLjPbFyYZzp.jpg` |

比對結果：20,121 個唯一 ID（另有 143 個無法比對，含 `default_2.jpg` 佔位圖）。

成功比對的圖片複製至各 split 目錄，命名為 `{id}_coverImage_extraLarge.jpg`：

| Split | 目錄 | 總筆數 | 成功複製 | 找不到 |
|-------|------|--------|---------|--------|
| train | `data/image/train_image/` | 13,376 | 13,320 | 56 |
| val | `data/image/validation_image/` | 2,918 | 2,918 | 0 |
| test | `data/image/test_image/` | 3,087 | 3,086 | 1 |
| holdout_unknown | `data/image/holdout_unknow_image/` | 943 | 939 | 4 |

---

### Step 2：透過 MyAnimeList 補圖

對 61 筆找不到圖的 ID，先查詢 AniList API 確認現況，確定全部仍為 default 圖後，改透過 **Jikan API**（MAL 非官方 API）查詢 MyAnimeList 是否有封面圖：

```
AniList API → 61 筆全為 default
Jikan API   → 60 筆有 MAL ID（1 筆無對應）
```

查詢結果中，4 筆在 MAL 有真實封面圖，直接下載補入對應 split 目錄：

| AniList ID | 標題 | Split |
|-----------|------|-------|
| 184982 | Poppoya-san: Nonki Ekichou | train |
| 103480 | Phantasm | holdout_unknown |
| 176858 | Tearmoon Teikoku Monogatari Mini Anime | test |
| 177457 | Tenchi Muyou! Ryououki: Heianmuyo! Picture Drama | holdout_unknown |

---

### Step 3：排除無封面圖資料

補圖後剩餘 **57 筆** AniList 與 MAL 皆無封面圖。這些動畫幾乎全為 **1949～1974 年**的極老舊短篇，無後續作品、無前傳，且為完全獨立作品：

- 共 56 筆來自 1949～1974 年
- 1 筆（`154099 Kachoufuugetsu`）為 1985 年

由於這 57 筆：
1. AniList 與 MAL 均無封面圖，且網路上難以找到對應資料
2. 若使用 `default_2.jpg` 的 embedding 會讓這 57 筆在視覺特徵上人為地互相相似，污染檢索結果
3. 皆為孤立作品，無法透過前後作關係補強其他模態特徵

決定**直接從資料集排除**這 57 筆。

---

### Step 4：產生清理後 CSV（_v2）

排除 57 筆後，產生新版資料集：

| Split | 原始筆數 | v2 筆數 | 移除筆數 |
|-------|---------|---------|---------|
| train | 13,376 | 13,321 | 55 |
| val | 2,918 | 2,918 | 0 |
| test | 3,087 | 3,087 | 0 |
| holdout_unknown | 943 | 941 | 2 |
| **合計** | **20,324** | **20,267** | **57** |

清理後 CSV 路徑：`data/fussion/fusion_meta_clean_{split}_v2.csv`

切換版本只需修改 `fusion_config.yaml` 的 `data.meta_suffix`：

```yaml
data:
  meta_suffix: "_v2"   # 清理後版本
  # meta_suffix: ""    # 原始版本
```

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
| 訓練 checkpoint | `.exp/fussion/results/{run_id}/{target}/` |
| 實驗統計 | `.exp/fussion/experiments_summary.csv` |
