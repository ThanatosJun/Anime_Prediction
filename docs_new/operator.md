# 部署操作手冊（v2）

所有指令皆從 **專案根目錄** 執行。

---

## 前置條件

### 環境
```bash
conda activate animeprediction   # 或對應的 Python 環境
```

### 必要資料（需手動準備）

| 路徑 | 說明 |
|------|------|
| `src_2/data/dataset/fusion_meta_clean_{split}_v2.csv` | 各 split 的 metadata（train / val / test / holdout_unknown） |
| `src_2/data/image/{split}/` | 動畫封面與 banner 圖片 |
| `src_2/component_image/model-image/best/` | Fine-tuned Swin-B（`config.json` + `model.safetensors`） |

### 必要服務
- **Docker**（Qdrant 用）— Step 4、5（RAG 建置/查詢）與 Step 10（推論）需運行

### 可解釋性 / VLM 額外依賴
- Step 9（explain）：`captum` / `shap`（已列入 `requirements.txt`）
- `component_image_text_description/`（VLM 圖片描述，探索中）：`transformers` + `accelerate`

---

## Step 1：生成 Text Embeddings

```bash
python src_2/RAG/run_build_embeddings.py
```

| 輸出 | 說明 |
|------|------|
| `src_2/embedding/text/text_embeddings_train.parquet` | e5-base-v2，768-dim |
| `src_2/embedding/text/text_embeddings_val.parquet` | |
| `src_2/embedding/text/text_embeddings_test.parquet` | |
| `src_2/embedding/text/text_embeddings_holdout_unknown.parquet` | |
| `src_2/embedding/image_rag/image_embeddings_train.parquet` | Swin-B 1024-dim，無 YOLO（RAG 知識庫用） |

---

## Step 2：YOLO Crop 圖片

```bash
python src_2/component_image/run_yolo_crop.py --splits train val test holdout_unknown
```

| 輸出 | 說明 |
|------|------|
| `src_2/data/image/crops/{split}/{id}_crop_{n}.jpg` | YOLO 裁切後的人物／臉部圖片 |

偵測模式在 `image_encoder_config.yaml` 的 `yolo_detection.detect_mode` 設定（`both` = 人物 + 臉部）。

---

## Step 3：生成 Fusion Image Embeddings

```bash
python src_2/component_image/run_swin_embedding.py --splits train val test holdout_unknown
```

| 輸出 | 說明 |
|------|------|
| `src_2/embedding/image/image_embeddings_{split}.parquet` | 欄位：`id, yolo_0…1023, cover_0…1023, banner_0…1023, has_yolo, has_cover, has_banner` |

---

## Step 4：啟動 Qdrant 並建立 Collection

```bash
# 啟動 Qdrant（Docker）
bash src_2/RAG/start_qdrant.sh

# 建立 collection 並寫入 embeddings
python src_2/RAG/rag_builder.py
```

| 輸出 | 說明 |
|------|------|
| `src_2/RAG/qdrant_storage/` | Qdrant 本地儲存（collection: `anime_rag_v2`） |
| `src_2/RAG/sparse_encoder.json` | genres / studios / voice actors 的 sparse encoder 詞彙表 |

> **注意**：Qdrant 需在後續步驟執行期間持續運行。確認服務正常：
> ```bash
> curl http://localhost:6333/healthz
> ```

---

## Step 5：查詢 RAG Features

```bash
python src_2/RAG/rag_query.py --splits train val test holdout_unknown
```

| 輸出 | 說明 |
|------|------|
| `src_2/RAG/return/rag_features_{split}.parquet` | 欄位：`id, rag_popularity, rag_score, rag_release_year, rag_episodes, rag_found, retrieved_ids` |

`rag_found` 表示是否找到符合時間條件的相似動畫，`retrieved_ids` 為 top-5 相似動畫 ID（Cross Attention 輸入）。

---

## Step 6：訓練 Fusion Model

訓練前確認 `src_2/fussion_configs.yaml` 的 `output.run_id` 和 `output.notes`，每次新實驗應更新。

```bash
# 訓練兩個 target（popularity + meanScore）
python src_2/train.py

# 或只訓練單一 target
python src_2/train.py --target popularity
python src_2/train.py --target meanScore
```

### 關鍵設定（`fussion_configs.yaml`）

| 欄位 | 預設 | 說明 |
|------|------|------|
| `image_mode` | `cover_banner_yolo` | `cover` / `cover_banner` / `cover_banner_yolo` |
| `use_rag` | `true` | `false` 則不使用 Cross Attention |
| `training.mixed_precision` | `true` | AMP（float16 forward，float32 gradient） |
| `output.run_id` | `"01"` | 實驗編號，結果存於 `src_2/runs/{run_id}/` |
| `output.notes` | `"..."` | 實驗描述，會寫入所有輸出 JSON |

### 訓練輸出

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/{target}/best_model.pt` | 最佳 val loss 的 checkpoint |
| `src_2/runs/{run_id}/{target}/target_scaler.json` | 目標值正規化參數（center / scale / log_transform） |
| `src_2/runs/{run_id}/{target}/history.json` | 每 epoch 的 train_loss / val_loss / val_mae（原始 scale）/ lr |
| `src_2/runs/{run_id}/{target}/final_metrics.json` | 訓練結束後 train + val 的完整 metrics（test 由 Step 7 merge 進來） |
| `src_2/fussion_training/meta_encoder.json` | 訓練集 fit 的 MetaEncoder（自動生成，只需 fit 一次） |

### final_metrics.json 格式（popularity 範例，含 test）

```json
{
  "run_id": "07",
  "target": "popularity",
  "notes": "...",
  "best_epoch": 8,
  "train": { "spearman_rho": 0.93, "log_R2": 0.87, "MAE": 4957, "log_MAE": 0.56, "factor_acc_2x": 0.59 },
  "val":   { "spearman_rho": 0.88, "log_R2": 0.81, "MAE": 9295, "log_MAE": 0.79, "factor_acc_2x": 0.53 },
  "test":  { "spearman_rho": 0.85, "log_R2": 0.76, "MAE": 9499, "log_MAE": 0.89, "factor_acc_2x": 0.49 }
}
```

> meanScore 的指標欄位為 `spearman_rho / R2 / MAE / acc_within_10pt`（無 log 版，見 Step 7）。

---

## Step 7：Test Set 評估

訓練完成後，對 test set 評估。結果會 merge 進 `final_metrics.json`。

```bash
# 兩個 target 一起跑
python src_2/evaluate.py --split test

# 或單一 target
python src_2/evaluate.py --target popularity --split test
```

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/{target}/final_metrics.json` | 新增 `"test"` key，包含完整 metrics |
| `src_2/runs/{run_id}/{target}/pred_test.csv` | 欄位：`id, pred, target`（原始 scale） |

### Metrics 說明

| Metric | Target | 說明 |
|--------|--------|------|
| `spearman_rho` | 兩者 | 排名相關係數，主要指標 |
| `log_R2` | popularity | log1p 空間的 R²，匹配訓練目標（原始 R² 會被少數爆紅動畫綁架） |
| `R2` | meanScore | 原始 scale 的 R²（0–100 線性，不需 log） |
| `MAE` | 兩者 | 原始 scale 的平均絕對誤差 |
| `log_MAE` | popularity | log1p 空間的 MAE，scale-free（0=完美，naive≈2.0） |
| `factor_acc_2x` | popularity | 預測落在真實值 [0.5×, 2×] 內的比例（乘法尺度準確率，0~1） |
| `acc_within_10pt` | meanScore | 預測誤差 < 10 分的比例（加法尺度準確率，0~1；facc_2x 對 0–100 分無鑑別力） |

---

## Step 8：Holdout 推論

Holdout 無真實標籤，只輸出預測值，不計算 metrics。

```bash
python src_2/evaluate.py --split holdout_unknown
```

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/{target}/pred_holdout_unknown.csv` | 欄位：`id, pred`（target 欄為 0，無標籤） |

---

## Step 9：可解釋性分析

> explain **不需要 Qdrant**（讀預存的 `rag_features` parquet）。`captum` / `shap` 已列入 `requirements.txt`。

### 指定要解釋的 run

explain 腳本從 config 的 `run_id` 決定載哪個 checkpoint + 輸出位置。最佳 run 為 pop=07 / score=02，需用對應 run_id 的 config：

```bash
python3 -c "import yaml; c=yaml.safe_load(open('src_2/fussion_configs.yaml')); c['output']['run_id']='07'; yaml.dump(c, open('/tmp/run07.yaml','w'), allow_unicode=True)"
python3 -c "import yaml; c=yaml.safe_load(open('src_2/fussion_configs.yaml')); c['output']['run_id']='02'; yaml.dump(c, open('/tmp/run02.yaml','w'), allow_unicode=True)"
```

### RAG Attention Heatmap

顯示模型對各 retrieved anime 及各 modality（meta / text / image）的注意力權重。

```bash
python src_2/explain/rag_heatmap.py --config /tmp/run07.yaml --target popularity --n 5
python src_2/explain/rag_heatmap.py --config /tmp/run02.yaml --target meanScore  --n 5

# 指定特定 anime ID
python src_2/explain/rag_heatmap.py --config /tmp/run07.yaml --target popularity --ids 12345 67890
```

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/explain/{target}/rag/{id}_attn.png` | heatmap：x = retrieved anime（顯示動畫名稱），y = modality，顏色 = attention weight |

### Captum + SHAP

```bash
# Captum（modality 貢獻）+ SHAP（meta feature 貢獻），各抽 20 筆
python src_2/explain/feature_attr.py --config /tmp/run07.yaml --target popularity --n 20 --background 50

# 只跑其中一個
python src_2/explain/feature_attr.py --config /tmp/run07.yaml --target popularity --skip_shap
python src_2/explain/feature_attr.py --config /tmp/run07.yaml --target popularity --skip_captum
```

| 輸出 | 說明 |
|------|------|
| `explain/{target}/feature/captum_modality.csv` | 每筆 sample 的各 modality 歸一化重要性 |
| `explain/{target}/feature/captum_modality.png` | 平均 modality 重要性長條圖（8 模態） |
| `explain/{target}/feature/shap_values.npy` | raw SHAP values `[n, 56]` |
| `explain/{target}/feature/shap_summary.png` | top-k meta feature 重要性（`prequel_meanScore_mean`, `va_te_*` 等） |

| 分析方法 | 說明 |
|----------|------|
| **Captum IG** | Integrated Gradients，計算 8 模態貢獻（image_yolo / image_cover / image_banner / text / meta / rag_meta / rag_text / rag_image） |
| **SHAP GradientExplainer** | 固定 image/text/rag，對 meta 的 56 個可解釋維度計算貢獻（用 GradientExplainer 而非 DeepExplainer，對 attention/LayerNorm 較穩健） |

---

## Step 10：推論 Pipeline（新動畫即時預測）

`src_2/inference.py`：給定一部新動畫（封面圖 + metadata + 描述），即時走完 YOLO → Swin → e5 → RAG → FusionModel。**需 Qdrant 運行**（RAG 用）。

```bash
bash src_2/RAG/start_qdrant.sh        # 確認 Qdrant 運行中

# metadata 用單列 CSV（欄位同訓練 schema，可無 popularity/meanScore）
python src_2/inference.py \
    --cover  path/to/cover.jpg \
    --banner path/to/banner.jpg \
    --meta   path/to/new_anime.csv \
    --description "動畫劇情描述..."

# 驗證模式：用既有 test 動畫，對照 pred_test.csv
python src_2/inference.py --anime-id 21294 --split test --verify
```

| 項目 | 說明 |
|------|------|
| 輸出 | stdout 印出 `popularity` / `meanScore` / `retrieved_ids` |
| 最佳 checkpoint | popularity → `runs/07/...`；meanScore → `runs/02/...`（架構相同，僅超參不同） |
| RAG 檢索 | 預設 `rag_use_image=False`（image_rag 僅 train，val/test 為 sparse+text，對齊驗證指標） |

---

## 快速重跑新實驗

1. 修改 `src_2/fussion_configs.yaml`：更新 `run_id` 和 `notes`
2. 跑訓練：`python src_2/train.py`
3. 跑評估：`python src_2/evaluate.py --split test`

若只修改模型架構或訓練超參數（不動 embeddings 和 RAG），只需重跑 Step 6–7。

---

## 各步驟依賴關係

```
Step 1（Text Emb）──────────────────────────────┐
Step 2（YOLO Crop）→ Step 3（Image Emb）─────────┤
Step 1 + Step 3 → Step 4（Qdrant）→ Step 5（RAG）┤
                                                  ↓
                          Step 6（Train）→ Step 7（Eval）→ Step 8（Holdout）
                                  ↓                ↓
                          Step 9（Explain）   Step 10（Inference，新動畫，需 Qdrant）
```

- Steps 1–5 只需執行一次（資料不變時）。Steps 6–8 每次新實驗重跑。
- Step 9（explain）/ Step 10（inference）用已訓練 checkpoint，隨需執行。
- **Qdrant 需運行的步驟**：Step 4、5（建置/查詢）與 Step 10（推論）。Step 6–9 不需 Qdrant（讀預存 parquet）。

---

## 超參數範圍（注意）

per-target 可調：loss（Huber/LogCosh，由 target 決定）、`log_transform`、`winsor_pct`、`trend_head`/`temporal_weight` 的 `apply_to`。
**全域共用**（兩 target 同一組）：`dropout`、`weight_decay`、`lr`、`batch_size`。
→ 兩 target 各自最佳超參（pop: dropout=0.3,wd=1e-3；score: dropout=0.5,wd=5e-4）目前靠 `--target` 分開跑、各帶不同 config 達成。
