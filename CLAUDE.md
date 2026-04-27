# CLAUDE.md — 動漫預測專案

## 專案概述
在動畫播出前，透過多模態融合（文字 + 圖片 + 元資料 + RAG）預測動漫的 `popularity`（人氣）與 `meanScore`（評分）。
兩個迴歸目標各自獨立訓練。
環境：`conda activate animeprediction`（Python、PyTorch CUDA 12.8、RTX 5070 Ti）。

---

## 指令

```bash
# 環境
conda activate animeprediction

# 資料 Pipeline（依序執行）
python scripts/build_interim_dataset.py
python scripts/build_processed_dataset.py
python scripts/export_multimodal_inputs.py

# 文字 Embedding
python -m src.text_branch.run_text_embedding_pipeline

# 圖片 Embedding（需要 data/image/ 內有圖片，以及 results/01/best/ 的 checkpoint）
python -m src.image_branch.run_fetch       # 下載封面圖片
python util/split_images_by_split.py       # 將圖片依 split 分入子資料夾
python -m src.image_branch.run_train       # 微調 Swin Transformer
python -m src.image_branch.run_predict     # 推論 → .exp/image_embedding/image_embeddings.parquet

# Fusion RAG Pipeline
python -m src.fussion_branch.run_rag       # 建立 Qdrant + 查詢所有 split

# Fusion MLP 訓練
python -m src.fussion_branch.run_train                     # 訓練兩個目標
python -m src.fussion_branch.run_train --target popularity # 只訓練 popularity
python -m src.fussion_branch.run_train --target meanScore  # 只訓練 meanScore

# 快速測試（1000 筆 smoke test）
python test_train.py
```

---

## 架構

```
src/
  text_branch/        all-MiniLM-L6-v2，輸出 artifacts/text_embeddings_{split}.parquet（384 維）
  image_branch/       Swin-base 對比學習微調，輸出 .exp/image_embedding/image_embeddings.parquet（1024 維）
  fussion_branch/     RAG（Qdrant sparse）+ MLP 融合模型
    text_components/  text_branch 元件本地副本（TextPreprocessor、EmbeddingGenerator）
    image_components/ image_branch 元件本地副本（image_process、configs）

data/
  interim/            清洗後中間層 CSV
  processed/          特徵工程 CSV + 各 split CSV（anilist_anime_data_processed_v1_{split}.csv）
  fussion/
    fusion_meta_{train,val,test,holdout_unknown}.csv        — 33 欄（原始，未清理）
    fusion_meta_clean_{train,val,test,holdout_unknown}.csv  — 24 欄（模型可用，見下方說明）
  image/
    train_image/          train split 的封面 + banner 圖片
    validation_image/     val split 的封面 + banner 圖片
    test_image/           test split 的封面 + banner 圖片
    holdout_unknow_image/ holdout_unknown 的封面 + banner 圖片

artifacts/
  text_embeddings_{split}.parquet   欄位：id, emb_0..emb_383, popularity, meanScore, split（text_branch 產出）

src/fussion_branch/RAG/             （RAG 中間產物）
  sparse_encoder.json               genre+studio 稀疏向量詞表（1483 tokens：19 genres + 1464 studios）
  rag_features_{split}.parquet      欄位：id, rag_title_romaji, rag_popularity, rag_score, rag_release_year, rag_studios, rag_found

.exp/
  image_embedding/image_embeddings.parquet   平坦格式 id + img_0..img_1023（1024 維 Swin-base）
  fussion/
    meta_encoder.json               已擬合的 MetaEncoder 詞表與 scaler
    results/{run_id}/{target}/      訓練 checkpoint、metrics、target_scaler.json

qdrant_db/            本地 Qdrant 向量資料庫（已 gitignore，執行 run_rag 自動重建）
results/              image branch checkpoint（已 gitignore）
.exp/                 所有 branch 的推論/訓練產出（已 gitignore）
```

---

## fusion_meta_clean 欄位（24 欄）

| 類別 | 欄位 | 編碼方式 |
|------|------|---------|
| 識別 | `id`, `title_romaji`, `title_english`, `title_native` | 不編碼（join / 顯示用） |
| 識別 | `description` | 不編碼（text embedding 原始文字，null 約 4–10%） |
| 動畫規格 | `format`（7 類） | one-hot |
| 動畫規格 | `episodes`, `duration` | 標準化 |
| 時間 | `release_year` | 標準化 |
| 時間 | `release_quarter`（4）、`startDate_month`（12） | cyclical sin/cos |
| 時間 | `startDate_day` | 標準化 |
| 內容屬性 | `source`（7 類，含 UNKNOWN_SOURCE）、`countryOfOrigin`（4 類） | one-hot |
| 內容屬性 | `isAdult` | binary 0/1 |
| 標籤 | `genres`（19 類） | multi-hot |
| 標籤 | `studios`（top-50） | multi-hot |
| 續集 | `is_sequel`、`has_sequel` | binary 0/1 |
| 續集 | `prequel_count` | 標準化 |
| 續集 | `prequel_popularity_mean` | log1p + 標準化 |
| 續集 | `prequel_meanScore_mean` | 標準化 |
| 目標 | `popularity` | log1p + 標準化（訓練目標） |
| 目標 | `meanScore` | 標準化（訓練目標） |

**已移除欄位：** `type`（zero variance）、`season` / `seasonYear` / `startDate_year`（重複時間資訊）、`release_date` / `release_quarter_key`（重複）、`voice_actor_names`（缺值率 40%，v4 改 multi-hot）、`popularity_quarter_pct` / `popularity_quarter_bucket`（target leakage）、`is_source_missing`（冗餘，`source='UNKNOWN_SOURCE'` 由 one-hot 自然處理）

---

## Fusion 模型

```
輸入：text(384) + image(1024，缺失時補零) + meta_rag(~158) ≈ 1566 維
架構：Linear → BN → ReLU → Dropout × 3 層（hidden_dims: [512, 256, 128]，dropout: 0.4）
輸出：scalar（每個 target 獨立）
```

### 評估指標

| 指標 | 說明 | 適用 |
|------|------|------|
| MAE / RMSE | 原始尺度誤差 | 兩個 target |
| Spearman ρ | 排名相關性 | 兩個 target |
| MAPE | 百分比誤差（scale-free） | popularity |
| log_MAE | log 空間誤差（對應訓練目標） | popularity |
| bucket_accuracy | 訓練集四分位分類準確率 | popularity |

---

## 程式碼慣例

- 所有模組從專案根目錄以 `python -m src.<branch>.run_*` 執行
- import 使用 `from src.<branch>.<module> import ...`，不使用相對 import
- Config 為 YAML 檔，放在 `src/<branch>/configs/`
- Embedding 輸出使用 Parquet 格式；元資料使用 CSV
- `id`（AniList 整數）為所有資料集的 join key
- CSV 中 `studios` 欄位為 JSON 字串 — 用 `json.loads()` 解析，不要用 `ast.literal_eval()`（含 JSON boolean）
- CSV 中 `genres` 欄位為 Python list 字串 — 用 `ast.literal_eval()` 解析

---

## 資料規則

- **時間序 split**：train = 較早季度，val/test = 較晚季度（時序切分，非隨機）
- **holdout_unknown**（943 筆，`release_quarter` 缺失）不參與任何模型訓練與評估
- **Pre-release 限制**：只允許播出前已知的特徵；排除 `averageScore`、`favourites`、`trending`、`status`
- **RAG leakage 防護**：Qdrant 只收錄 training set；Python 端 post-filter 移除 `release_year >= target`
- **Target leakage**：`popularity_quarter_pct` 和 `popularity_quarter_bucket` 絕對不能作為模型輸入特徵

---

## 常見操作

**新增 fusion metadata 欄位：**
1. 修改 source processed CSV 或 `fusion_data_exploration.ipynb`
2. 透過 notebook 或 script 重新產生 `fusion_meta_clean_{split}.csv`
3. 重新執行 `python -m src.fussion_branch.run_rag` 以更新 Qdrant payload 欄位

**重建文字 Embedding：**
```bash
python -m src.text_branch.run_text_embedding_pipeline
```
覆寫 `artifacts/text_embeddings_{train,val,test}.parquet`。

**產生圖片 Embedding：**
```bash
python -m src.image_branch.run_predict
```
輸出 `.exp/image_embedding/image_embeddings.parquet`（平坦格式 `id + img_0..img_1023`）。
若檔案不存在，FusionDataset 自動補零向量（1024 維），訓練可正常進行。

**重置 Qdrant collection：**
每次執行 `run_rag.py` 都會自動重建 `qdrant_db/`，直接刪除資料夾或重跑腳本即可。

---

## 注意事項

- 執行訓練指令前，必須確認 GPU 可用（`torch.cuda.is_available()`）
- 不得將後製特徵（`averageScore`、`favourites`、`trending`）加入 fusion_meta — 會造成 data leakage
- `rag_features` 中的 `rag_title_romaji` 僅供可解釋性使用 — 不得輸入 MLP
- `rag_features` 中的 `rag_studios` 為 JSON 字串；multi-hot encoding 在模型訓練時進行
- Image embedding 維度為 **1024**（Swin-base `pooler_output`）；不是 768（那是 Swin-tiny）
- `startDate_month` 補值：使用 season 對應月份（SPRING→4、SUMMER→7、FALL→10、WINTER→1）
- holdout_unknown 的 `startDate_month` / `release_quarter` / `season` 全部為 null，無法補值

---

## TODO（v4）

- `voice_actor_names` → top-K 聲優 multi-hot encoding（同 studios 邏輯，建議先測 top-50）
