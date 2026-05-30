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
- **Docker**（Qdrant 用）

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
| `src_2/runs/{run_id}/{target}/history.json` | 每 epoch 的 train_loss / val_loss / val_mae / lr |
| `src_2/runs/{run_id}/{target}/final_metrics.json` | 訓練結束後 train + val 的完整 metrics |
| `src_2/fussion_training/meta_encoder.json` | 訓練集 fit 的 MetaEncoder（自動生成，只需 fit 一次） |

### final_metrics.json 格式

```json
{
  "run_id": "01",
  "target": "popularity",
  "notes": "...",
  "best_epoch": 7,
  "train": { "spearman_rho": 0.95, "log_R2": 0.90, "MAE": 4849, "log_MAE": 0.49 },
  "val":   { "spearman_rho": 0.89, "log_R2": 0.81, "MAE": 9924, "log_MAE": 0.78 }
}
```

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
| `log_R2` | popularity | log1p 空間的 R²，匹配訓練目標 |
| `R2` | meanScore | 原始 scale 的 R² |
| `MAE` | 兩者 | 原始 scale 的平均絕對誤差 |
| `log_MAE` | popularity | log1p 空間的 MAE，scale-free |

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

### 前置安裝
```bash
pip install captum shap matplotlib
```

### RAG Attention Heatmap

顯示模型對各 retrieved anime 及各 modality（meta / text / image）的注意力權重。

```bash
# 從 val set 隨機抽 5 筆
python src_2/explain/rag_heatmap.py --target popularity --n 5

# 指定特定 anime ID
python src_2/explain/rag_heatmap.py --target popularity --ids 12345 67890
```

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/{target}/explain/rag/{id}_attn.png` | heatmap：x = retrieved anime，y = modality，顏色 = attention weight |

### Captum + SHAP

```bash
# Captum（modality 貢獻）+ SHAP（meta feature 貢獻），各抽 20 筆
python src_2/explain/feature_attr.py --target popularity --n 20

# 只跑其中一個
python src_2/explain/feature_attr.py --target popularity --skip_shap
python src_2/explain/feature_attr.py --target popularity --skip_captum
```

| 輸出 | 說明 |
|------|------|
| `explain/feature/captum_modality.csv` | 每筆 sample 的各 modality 歸一化重要性 |
| `explain/feature/captum_modality.png` | 平均 modality 重要性長條圖 |
| `explain/feature/shap_values.npy` | raw SHAP values `[n, 56]` |
| `explain/feature/shap_summary.png` | top-k meta feature 重要性（`release_year`, `genre_Action` 等） |

| 分析方法 | 說明 |
|----------|------|
| **Captum IG** | Integrated Gradients，計算各 modality 對最終預測的貢獻（image_yolo / image_cover / image_banner / text / meta / rag） |
| **SHAP** | 固定 image/text/rag，對 meta 的 56 個可解釋維度計算 Shapley value（哪些 metadata 欄位影響預測最大） |

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
```

Steps 1–5 只需執行一次（資料不變時）。Steps 6–8 每次新實驗重跑。
